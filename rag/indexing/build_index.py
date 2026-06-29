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
    from rag.indexing.pdf_parser import EasyOcrEngine, parse_pdf
    from rag.vectorstore.store import ChromaStore

    pdfs = sorted(Path(manuals_dir).glob("*.pdf"))
    if not pdfs:
        print(f"[!] PDF가 없습니다: {manuals_dir}")
        return

    print(f"[*] {len(pdfs)}개 PDF 인덱싱 시작 — OCR/임베딩 모델 로드 중...")
    ocr = EasyOcrEngine()        # 모델은 한 번만 로드해 재사용
    embedder = Embedder()
    store = ChromaStore(persist_dir=index_dir)

    for pdf in pdfs:
        manual_id = pdf.stem
        print(f"[parse] {pdf.name} — OCR 중...")
        pages = parse_pdf(str(pdf), manual_id, ocr=ocr)
        chunks = chunk_pages(pages)
        print(f"        페이지 {len(pages)}개 → 청크 {len(chunks)}개 → 임베딩 중...")
        vectors = embedder.encode([c.text for c in chunks])
        store.add(
            ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            metadatas=[
                {"manual_id": c.manual_id, "page": c.page, "section": c.section or ""}
                for c in chunks
            ],
            documents=[c.text for c in chunks],
        )
        print(f"        ✓ {len(chunks)}개 청크 인덱싱 완료")

    print(f"[완료] 저장소 총 청크 수: {store.count()}  (위치: {index_dir})")


if __name__ == "__main__":
    build_index()
