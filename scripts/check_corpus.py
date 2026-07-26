"""코퍼스 점검 — 인덱싱 전에 '텍스트 레이어가 살아있는지'와 OCR 비용을 미리 본다.

각 PDF의 쪽수·직접추출 한글량·텍스트레이어 판정(파서 vs OCR 폴백)을 표로 보여준다.
OCR 폴백은 쪽당 2~5초라, 몇 쪽이 OCR로 갈지 미리 알면 인덱싱 시간을 예측할 수 있다.
텍스트가 거의 없는 PDF(폰트를 벡터로 변환한 매뉴얼 등)는 여기서 드러난다.

실행: python scripts/check_corpus.py [manuals_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIN_PAGE_CHARS = 30  # parse_pdf의 쪽 단위 OCR 폴백 기준과 동일


def _hangul(s: str) -> int:
    return sum(1 for ch in s if "가" <= ch <= "힣")


def main() -> None:
    import fitz

    from rag.indexing.pdf_parser import _doc_text_usable

    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/manuals")
    pdfs = sorted(d.glob("*.pdf"))
    print(f"[*] {len(pdfs)}개 PDF 점검 (OCR 실행 안 함)\n", flush=True)

    hdr = f"{'manual_id':<44}{'쪽':>5}{'한글':>9}{'OCR쪽':>7}  판정"
    print(hdr)
    print("-" * len(hdr))

    tot_pages = ocr_pages = 0
    rows = []
    for p in pdfs:
        try:
            doc = fitz.open(str(p))
        except Exception as e:
            print(f"{p.stem:<44}{'?':>5}  열기 실패: {e}")
            continue
        usable = _doc_text_usable(doc)
        texts = [pg.get_text("text") for pg in doc]
        hg = sum(_hangul(t) for t in texts)
        # 문서 레이어가 깨졌으면 전 쪽 OCR, 정상이면 빈약한 쪽만 OCR
        need = len(texts) if not usable else sum(1 for t in texts if len(t.strip()) < MIN_PAGE_CHARS)
        verdict = "파서" if usable else "OCR 전체"
        if usable and need:
            verdict = f"파서(+{need}쪽 OCR)"
        rows.append((p.stem, len(texts), hg, need, verdict, usable))
        tot_pages += len(texts)
        ocr_pages += need
        doc.close()

    for stem, n, hg, need, verdict, _ in rows:
        print(f"{stem:<44}{n:>5}{hg:>9,}{need:>7}  {verdict}")

    full_ocr = [r for r in rows if not r[5]]
    lowtext = [r for r in rows if r[5] and r[2] < 200 * max(1, r[1]) // 10]
    print(f"\n총 {len(rows)}종 · {tot_pages:,}쪽 · 직접추출 한글 {sum(r[2] for r in rows):,}자")
    print(f"OCR 예정 {ocr_pages:,}쪽 → 예상 {ocr_pages * 3 / 60:.0f}~{ocr_pages * 5 / 60:.0f}분 (쪽당 3~5초)")
    print(f"문서 전체 OCR 대상 {len(full_ocr)}종: " + (", ".join(r[0] for r in full_ocr[:8]) or "없음"))
    if lowtext:
        print(f"[!] 텍스트 빈약(파서 판정이나 한글 적음) {len(lowtext)}종: "
              + ", ".join(r[0] for r in lowtext[:8]))


if __name__ == "__main__":
    main()
