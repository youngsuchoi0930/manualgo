"""답변 품질 평가 — LLM-as-judge로 충실성(환각률)·인용 정확성·정답성을 측정한다.

지금까지의 평가는 전부 '검색'(Recall@k/MRR)이었고 생성 답변은 미측정이었다.
이 모듈은 실제 서비스 경로(Agentic 검색 → 생성)로 답변을 만든 뒤, 심판 LLM이
(a) 충실성: 답변의 주장이 '검색된 근거'로 뒷받침되는가 (hallucinated = 환각)
(b) 인용 정확성: 표기한 (출처: N쪽)이 실제 근거 쪽과 일치하는가
(c) 정답성: '정답 페이지 원문' 기준으로 사실이 맞는가
를 채점한다.

비용: 질문당 generate_content 2회(답변+심판) — 무료 20회/일이므로 소표본 스폿체크용.
429(쿼터 소진) 시 그때까지의 부분 결과를 저장한다.
한계: 심판이 생성기와 같은 모델 계열(Gemini)이라 자기-심판 편향 가능. 표본도 작음(방향 지표).

실행: python -m evaluation.answer_quality [n문항]   (기본 8, 유형별 라운드로빈 샘플)
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

JUDGE_PROMPT = """당신은 RAG 답변 품질 심판입니다. 아래 자료로 '생성된 답변'을 평가하세요.

[질문]
{question}

[시스템이 검색해 답변 생성에 사용한 근거]
{contexts}

[생성된 답변]
{answer}

[정답 페이지 원문] ({gold_ref} — 평가셋이 정한 정답 위치)
{gold_text}

JSON 하나로만 답하세요:
{{"faithfulness": "faithful" | "minor_unsupported" | "hallucinated",
  "citation_ok": true/false,
  "correct": true/false,
  "reason": "판정 근거 한 문장"}}

기준:
- faithfulness: 답변의 사실 주장들이 [근거] 텍스트로 뒷받침되면 faithful.
  근거에 없는 사실을 지어냈으면 hallucinated. 사소한 표현 확장/일반화는 minor_unsupported.
- citation_ok: 답변에 표기된 (출처: N쪽)의 쪽 번호가, 그 내용이 실제로 적힌 근거의 쪽과 일치하면 true.
  출처 표기가 없거나 엉뚱한 쪽이면 false.
- correct: [정답 페이지 원문] 기준으로 답의 내용이 사실과 맞으면 true.
"""


def _sample(evalset: list[dict], n: int) -> list[dict]:
    """유형별 라운드로빈으로 n개 뽑는다 (재현 가능·유형 커버)."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for it in evalset:
        by_type[it.get("type", "?")].append(it)
    order = sorted(by_type)
    picked, i = [], 0
    while len(picked) < n and any(by_type[t] for t in order):
        t = order[i % len(order)]
        if by_type[t]:
            picked.append(by_type[t].pop(0))
        i += 1
    return picked


def _fmt_contexts(chunks) -> str:
    return "\n\n".join(f"(근거{i + 1}: {c.manual_id} {c.page}쪽)\n{c.text}" for i, c in enumerate(chunks))


def run(n: int = 8, evalset_path: str = "data/eval/evalset.jsonl",
        results_dir: str = "evaluation/results") -> None:
    from dotenv import load_dotenv

    load_dotenv()
    from google import genai
    from google.genai import types

    from evaluation.generate_evalset import _load_pages
    from rag.generation.generator import AnswerGenerator
    from rag.retrieval.agentic import AgenticRetriever

    evalset = [json.loads(l) for l in open(evalset_path, encoding="utf-8") if l.strip()]
    sample = _sample(evalset, n)

    # 이어하기: 이미 채점된 질문은 건너뛴다 (쿼터 절약)
    out_path = Path(results_dir) / "answer_quality.json"
    prev_rows: list[dict] = []
    if out_path.exists():
        try:
            prev_rows = json.loads(out_path.read_text(encoding="utf-8")).get("rows", [])
        except json.JSONDecodeError:
            pass
    done_qs = {r["question"] for r in prev_rows}
    todo = [it for it in sample if it["question"] not in done_qs]
    print(f"[*] {len(todo)}문항 채점 (기존 {len(prev_rows)}개 유지, 호출 {2 * len(todo)}회 예상)", flush=True)

    retriever = AgenticRetriever()
    generator = AnswerGenerator()
    judge = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    judge_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    gold_pages = _load_pages("data/index")

    def _gen_with_retry(fn, attempts: int = 3):
        """503(과부하)·RPM성 429는 대기 후 재시도, 일일 한도(PerDay)만 즉시 포기."""
        last = None
        for a in range(attempts):
            try:
                return fn()
            except Exception as e:
                last, msg = e, str(e)
                if "PerDay" in msg:
                    raise
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(30 * (a + 1))
                elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(60)
                else:
                    raise
        raise last

    rows = list(prev_rows)
    for i, it in enumerate(todo, 1):
        q, typ = it["question"], it.get("type", "?")
        gold_key = (it["manual_id"], int(it["page"]))
        try:
            chunks = retriever.retrieve(q, top_k=5)
            gen = _gen_with_retry(lambda: generator.generate(q, chunks))  # 호출 1
            answer_text = " ".join(filter(None, [gen.get("headline"), gen.get("answer")]))
            prompt = JUDGE_PROMPT.format(
                question=q,
                contexts=_fmt_contexts(chunks)[:8000],
                answer=answer_text,
                gold_ref=f"{gold_key[0]} {gold_key[1]}쪽",
                gold_text=(gold_pages.get(gold_key) or "(원문 없음)")[:3000],
            )
            time.sleep(3)
            resp = _gen_with_retry(lambda: judge.models.generate_content(  # 호출 2
                model=judge_model, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
            ))
            verdict = json.loads(resp.text)
            rows.append({
                "type": typ, "question": q, "gold": list(gold_key),
                "answer": answer_text,
                "faithfulness": verdict.get("faithfulness"),
                "citation_ok": bool(verdict.get("citation_ok")),
                "correct": bool(verdict.get("correct")),
                "reason": verdict.get("reason", ""),
            })
            print(f"  [{i}/{len(todo)}] {typ:15} faith={verdict.get('faithfulness'):18} "
                  f"cite={verdict.get('citation_ok')} correct={verdict.get('correct')}", flush=True)
            time.sleep(3)
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print(f"  [!] 쿼터 소진 - 지금까지의 부분 결과로 집계합니다", flush=True)
                break
            print(f"  [{i}] 실패({typ}): {msg[:120]}", flush=True)

    if not rows:
        print("[!] 채점된 문항이 없습니다.")
        return

    n_done = len(rows)
    halluc = sum(r["faithfulness"] == "hallucinated" for r in rows)
    minor = sum(r["faithfulness"] == "minor_unsupported" for r in rows)
    cite = sum(r["citation_ok"] for r in rows)
    correct = sum(r["correct"] for r in rows)

    print("\n=== 답변 품질 (n=%d, LLM-as-judge) ===" % n_done)
    print(f"  환각률(hallucinated)      : {halluc}/{n_done} ({halluc / n_done:.0%})")
    print(f"  경미한 비근거(minor)      : {minor}/{n_done} ({minor / n_done:.0%})")
    print(f"  인용 정확률(citation_ok)  : {cite}/{n_done} ({cite / n_done:.0%})")
    print(f"  정답률(correct)           : {correct}/{n_done} ({correct / n_done:.0%})")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    print("  --- 유형별 (환각/인용ok/정답 / n) ---")
    for t, rs in sorted(by_type.items()):
        print(f"  {t:16} {sum(x['faithfulness'] == 'hallucinated' for x in rs)}/"
              f"{sum(x['citation_ok'] for x in rs)}/{sum(x['correct'] for x in rs)} / {len(rs)}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(results_dir) / "answer_quality.json"
    summary = {"n": n_done, "hallucinated": halluc, "minor_unsupported": minor,
               "citation_ok": cite, "correct": correct, "judge_model": judge_model,
               "caveat": "small-n spot check; judge shares model family with generator"}
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    import sys

    run(n=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
