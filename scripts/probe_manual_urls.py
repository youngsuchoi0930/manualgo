"""매뉴얼 URL 프로브 — 받기 전에 '진짜 PDF인지'만 싸게 확인한다.

Range 요청으로 앞부분만 받아 %PDF 매직바이트·content-type·크기를 보고한다(디스크 저장 없음).
직링크가 아닌 뷰어/HTML, 만료된 링크, 세션이 필요한 download.jsp를 미리 걸러낸다.
lge.co.kr의 동적 download.jsp는 Referer가 없으면 HTML을 주기도 해서 Referer 유무를 둘 다 시도한다.

실행:
  python scripts/probe_manual_urls.py                    # data/raw/manual_urls.txt 전체
  python scripts/probe_manual_urls.py candidates.txt     # 다른 목록 파일
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DELAY = 1.5
UA = "manualgo/0.1 (personal manual collector)"


def probe(url: str, referer: str | None = None) -> tuple[bool, str]:
    """(ok, 설명) — 앞 1KB만 받아 PDF 여부를 판정한다."""
    import requests

    headers = {"User-Agent": UA, "Range": "bytes=0-1023"}
    if referer:
        headers["Referer"] = referer
    try:
        # stream=True — Range를 무시하고 200+전체본문을 주는 서버에서도 앞부분만 읽고 끊는다
        with requests.get(url, headers=headers, timeout=30, allow_redirects=True, stream=True) as r:
            head = next(r.iter_content(chunk_size=1024), b"")[:5]
            ctype = r.headers.get("content-type", "?").split(";")[0]
            size = r.headers.get("content-range") or r.headers.get("content-length") or "?"
            status = r.status_code
    except Exception as e:
        return False, f"요청실패 {type(e).__name__}: {e}"
    if head.startswith(b"%PDF"):
        return True, f"PDF ok (status={status}, type={ctype}, size={size})"
    return False, f"PDF아님 (status={status}, type={ctype}, 앞바이트={head!r})"


def main() -> None:
    from scripts.collect_manuals import _entries

    try:  # 파이프/파일 출력 시 cp949 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "manual_urls.txt"
    if not path.exists():
        print(f"[!] 목록 없음: {path}")
        return
    entries = list(_entries(path))
    print(f"[*] {len(entries)}개 URL 프로브 (저장하지 않음)\n", flush=True)

    ok_list, bad_list = [], []
    for i, (name, url) in enumerate(entries):
        if i:
            time.sleep(DELAY)
        ok, msg = probe(url)
        if not ok and "download.jsp" in url:  # Referer 붙여 재시도
            time.sleep(DELAY)
            ok2, msg2 = probe(url, referer="https://www.lge.co.kr/")
            if ok2:
                ok, msg = True, msg2 + "  [Referer 필요]"
            else:
                msg = f"{msg} | Referer 재시도: {msg2}"
        print(f"[{'ok ' if ok else 'x  '}] {name}: {msg}", flush=True)
        (ok_list if ok else bad_list).append(name)

    print(f"\n요약 — 유효 {len(ok_list)} · 무효 {len(bad_list)}", flush=True)
    if bad_list:
        print("무효:", ", ".join(bad_list), flush=True)


if __name__ == "__main__":
    main()
