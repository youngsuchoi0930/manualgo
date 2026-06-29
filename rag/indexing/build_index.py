"""오프라인 인덱싱 엔트리포인트 (사전 1회 실행).

매뉴얼 PDF → OCR 파싱 → 청킹 → 임베딩 → 벡터 DB 저장.
``data/raw/manuals/`` 의 모든 PDF를 인덱싱한다. 파일명(확장자 제외)이 manual_id가 된다.

실행: ``python -m rag.indexing.build_index``
"""
from __future__ import annotations

from pathlib import Path


def build_index(manuals_dir: str = "data/raw/manuals", index_dir: str = "data/index") -> None:
    """디렉토리의 모든 매뉴얼 PDF를 인덱싱한다."""
    from rag.indexing.chunker import chunk_pages
    from rag.indexing.embedder import Embedder
    from rag.indexing.pdf_parser import TesseractOcrEngine, parse_pdf
    from rag.vectorstore.store import ChromaStore

    pdfs = sorted(Path(manuals_dir).glob("*.pdf"))
    if not pdfs:
        print(f"[!] PDF가 없습니다: {manuals_dir}")
        return

    print(f"[*] {len(pdfs)}개 PDF 인덱싱 (Tesseract OCR + Gemini 임베딩, torch 미사용·무제한)...", flush=True)
    ocr = TesseractOcrEngine()
    embedder = Embedder()

    # 먼저 메모리에 전부 수집 → 데이터가 확보된 뒤에만 인덱스를 덮어쓴다
    # (OCR이 도중에 실패해도 기존 인덱스를 파괴하지 않도록)
    ids: list[str] = []
    vectors: list[list[float]] = []
    metas: list[dict] = []
    docs: list[str] = []

    for pdf in pdfs:
        manual_id = pdf.stem
        print(f"[parse] {pdf.name} — Tesseract OCR 중 (로컬, DPI 300)...", flush=True)
        pages = parse_pdf(str(pdf), manual_id, ocr=ocr, dpi=300, skip_errors=True)
        empty = sum(1 for p in pages if not p.text.strip())
        chunks = chunk_pages(pages)
        note = f" (OCR 실패 {empty}쪽 제외)" if empty else ""
        print(f"        페이지 {len(pages)}개{note} → 청크 {len(chunks)}개 → 임베딩 중...", flush=True)
        if not chunks:
            print(f"        [!] {pdf.name}: 인덱싱할 청크 없음, 건너뜀", flush=True)
            continue
        vecs = embedder.encode([c.text for c in chunks])
        ids += [c.chunk_id for c in chunks]
        vectors += vecs
        metas += [{"manual_id": c.manual_id, "page": c.page, "section": c.section or ""} for c in chunks]
        docs += [c.text for c in chunks]
        print(f"        ✓ {len(chunks)}개 청크 준비 완료", flush=True)

    if not ids:
        print("[!] 인덱싱할 데이터가 없습니다 (OCR 전부 실패?). 기존 인덱스를 그대로 둡니다.", flush=True)
        return

    # 데이터 확보 후에만 컬렉션을 재구축
    store = ChromaStore(persist_dir=index_dir)
    store.reset()
    store.add(ids=ids, vectors=vectors, metadatas=metas, documents=docs)
    print(f"[완료] 저장소 총 청크 수: {store.count()}  (위치: {index_dir})", flush=True)


if __name__ == "__main__":
    build_index()
