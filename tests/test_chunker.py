"""청킹 단위 테스트 (순수 파이썬, 무거운 의존성 없음)."""
from dataclasses import dataclass

from rag.indexing.chunker import chunk_pages


@dataclass
class _Page:
    manual_id: str
    page: int
    text: str
    section: str | None = None


def test_short_page_is_one_chunk():
    pages = [_Page("sew650", 3, "전원버튼을 누르면 켜짐/꺼짐이 반복됩니다.")]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "sew650_p3_c0"
    assert chunks[0].page == 3
    assert chunks[0].manual_id == "sew650"


def test_empty_page_skipped():
    pages = [_Page("m", 1, "   "), _Page("m", 2, "내용 있음")]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_long_page_splits_and_keeps_page():
    long_text = " ".join([f"문장{i}." for i in range(400)])  # > max_chars
    pages = [_Page("m", 7, long_text)]
    chunks = chunk_pages(pages, max_chars=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.page == 7 for c in chunks)              # 페이지 추적 유지
    assert [c.chunk_id for c in chunks] == [f"m_p7_c{i}" for i in range(len(chunks))]
