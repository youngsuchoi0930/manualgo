"""리랭커 스모크 + 레이턴시 테스트.

(1) 관련 문장이 무관 문장보다 높은 점수를 받는지 (2) 20후보 재정렬 1회 지연(데모 프록시).
첫 실행 시 ONNX 모델(bge-reranker-v2-m3, int8)을 HF에서 캐시한다.
실행: conda run -n manual python scripts/check_reranker.py
"""
import time

from rag.retrieval.reranker import Reranker


def main() -> None:
    t0 = time.perf_counter()
    r = Reranker()
    load_s = time.perf_counter() - t0

    q = "예약 세탁은 최대 몇 시간까지 돼?"
    s = r.scores(q, ["예약 버튼을 누르면 최대 24시간까지 예약 가능합니다", "정수기 필터는 6개월마다 교체하세요"])
    print(f"[판별] 관련 {s[0]:.3f} | 무관 {s[1]:.3f}  ->  {'OK' if s[0] > s[1] else 'FAIL'}")
    print(f"[로드] {load_s:.1f}s")

    # 데모 프록시: 20후보 재정렬 지연 (워밍업 1회 후 3회 평균)
    passages = [f"세탁기 사용 설명 {i}. " + "예약 헹굼 탈수 코스 온도 조절 동작 일시정지 " * 20 for i in range(20)]
    r.scores(q, passages)  # 워밍업
    times = []
    for _ in range(3):
        t = time.perf_counter()
        r.scores(q, passages)
        times.append(time.perf_counter() - t)
    print(f"[지연] 20후보 재정렬: 평균 {sum(times) / len(times) * 1000:.0f}ms (min {min(times) * 1000:.0f}ms)")


if __name__ == "__main__":
    main()
