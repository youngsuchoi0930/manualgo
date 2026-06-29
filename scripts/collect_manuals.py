"""매뉴얼 수집 — URL 목록을 받아 PDF를 data/raw/manuals/ 에 다운로드한다.

사용법:
  1) data/raw/manual_urls.txt 에 한 줄에 하나씩 적는다:
       name,https://.../manual.pdf      (name 생략 시 URL에서 파일명 추론)
       # 으로 시작하는 줄은 주석
  2) python scripts/collect_manuals.py
  3) python -m rag.indexing.build_index

주의: 직접 PDF 링크여야 한다(뷰어/HTML 페이지는 실패). 받은 파일은 %PDF 인지 검사한다.
저작권/ToS: 개인·로컬 인덱싱 용도로만. robots.txt·약관 존중, 호출 간격 유지, 재배포 금지.
"""
from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URLS_FILE = ROOT / "data" / "raw" / "manual_urls.txt"
OUT_DIR = ROOT / "data" / "raw" / "manuals"
HEADERS = {"User-Agent": "manualgo/0.1 (personal manual collector)"}
DELAY = 2.0  # 요청 간격(초) — 서버 예의


def _entries(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            name, url = line.split(",", 1)
        elif "\t" in line:
            name, url = line.split("\t", 1)
        else:
            name, url = "", line
        name, url = name.strip(), url.strip()
        if not name:
            name = Path(url.split("?")[0]).stem or "manual"
        yield name, url


def _safe_name(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_.()[] ").strip().replace(" ", "_")
    return cleaned or "manual"


def main() -> None:
    import requests

    if not URLS_FILE.exists():
        print(f"[!] URL 목록이 없습니다: {URLS_FILE}")
        print("    한 줄에 하나씩 'name,url' 또는 'url' 형식으로 적은 뒤 다시 실행하세요.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = list(_entries(URLS_FILE))
    if not entries:
        print("[!] 받을 URL이 없습니다 (비었거나 전부 주석).")
        return

    ok = skip = fail = 0
    for i, (name, url) in enumerate(entries):
        target = OUT_DIR / f"{_safe_name(name)}.pdf"
        if target.exists():
            print(f"[skip] {target.name} (이미 있음)")
            skip += 1
            continue
        try:
            if i:
                time.sleep(DELAY)  # 서버 예의
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            content = resp.content
            if not content.startswith(b"%PDF"):
                ctype = resp.headers.get("content-type", "?")
                print(f"[fail] {name}: PDF 아님 (content-type={ctype}) — 직접 PDF 링크인지 확인")
                fail += 1
                continue
            target.write_bytes(content)
            print(f"[ok]   {target.name}  ({len(content) // 1024} KB)")
            ok += 1
        except Exception as e:
            print(f"[fail] {name}: {e}")
            fail += 1

    print(f"\n완료 — 받음 {ok} · 건너뜀 {skip} · 실패 {fail}  (위치: {OUT_DIR})")
    if ok:
        print("다음: python -m rag.indexing.build_index")


if __name__ == "__main__":
    main()
