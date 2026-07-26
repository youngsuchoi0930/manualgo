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

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URLS_FILE = ROOT / "data" / "raw" / "manual_urls.txt"
OUT_DIR = ROOT / "data" / "raw" / "manuals"
HEADERS = {"User-Agent": "manualgo/0.1 (personal manual collector)"}
DELAY = 2.0  # 요청 간격(초) — 서버 예의
MAX_BYTES = 120 * 1024 * 1024  # 파일당 상한 (0이면 무제한) — 이상치 폭주 방지


def _entries(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # 줄 끝 인라인 주석 제거 (' # 25MB' 같은 메모가 URL에 붙어 404 나는 것 방지).
        # URL 프래그먼트('...#page=3')는 공백 없이 붙으므로 영향 없다.
        line = line.split(" #", 1)[0].split("\t#", 1)[0].strip()
        if not line:
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

    # 파이프/파일로 출력할 때 cp949 기본 인코딩이면 한글·em dash에서 UnicodeEncodeError가 난다
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
    total_bytes = 0
    for i, (name, url) in enumerate(entries):
        target = OUT_DIR / f"{_safe_name(name)}.pdf"
        if target.exists():
            print(f"[skip] {target.name} (이미 있음)", flush=True)
            skip += 1
            continue
        part = target.with_suffix(".pdf.part")
        try:
            if i:
                time.sleep(DELAY)  # 서버 예의
            # 스트리밍 — 대용량(수십 MB) 매뉴얼을 메모리에 전부 올리지 않는다
            with requests.get(url, headers=HEADERS, timeout=120, stream=True) as resp:
                resp.raise_for_status()
                got = 0
                first = b""
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        if not first:
                            first = chunk[:5]
                            if not first.startswith(b"%PDF"):  # 앞바이트로 조기 중단
                                ctype = resp.headers.get("content-type", "?")
                                raise ValueError(f"PDF 아님 (content-type={ctype}, 앞바이트={first!r})")
                        f.write(chunk)
                        got += len(chunk)
                        if MAX_BYTES and got > MAX_BYTES:
                            raise ValueError(f"용량 초과 (>{MAX_BYTES // 1024 // 1024}MB) — 건너뜀")
                # 본문이 비었으면 %PDF 검사가 한 번도 안 돌아간다 → 0바이트 파일 승격 방지
                if not first.startswith(b"%PDF"):
                    raise ValueError(f"PDF 아님 (본문 {got}바이트, 매직바이트 확인 불가)")
            part.replace(target)
            total_bytes += got
            print(f"[ok]   {target.name}  ({got / 1024 / 1024:.1f} MB)", flush=True)
            ok += 1
        except Exception as e:
            part.unlink(missing_ok=True)  # 부분 파일 정리
            print(f"[fail] {name}: {e}", flush=True)
            fail += 1

    print(f"\n완료 — 받음 {ok} ({total_bytes / 1024 / 1024:.0f} MB) · 건너뜀 {skip} · 실패 {fail}"
          f"  (위치: {OUT_DIR})", flush=True)
    if ok:
        print("다음: python -m rag.indexing.build_index")


if __name__ == "__main__":
    main()
