"""
Validation Module for Q-Sorting
Mirror Test + Flat-lining Check + Self-Check
"""
from typing import Dict, List, Tuple
import numpy as np
from utils.llm_client import generate_json, generate_embedding
from utils.similarity import calculate_cosine_similarity


def mirror_test(
    sorting_data: Dict[str, int], 
    contradiction_pairs: List[Tuple[str, str]],
    threshold: int = 3
) -> Tuple[bool, List[Dict]]:
    """
    Mirror Test: 상충 문항에 대한 논리적 모순 검증
    
    Logic: 만약 페르소나가 "돈이 최고다"(A)에 +5를 주고, 
           "돈은 중요하지 않다"(B)에 +4를 준다면 논리적 모순
    
    Args:
        sorting_data: 문항 ID → 점수 딕셔너리
        contradiction_pairs: 상충 문항 쌍 리스트 [(id_a, id_b), ...]
        threshold: 양쪽 모두 이 값 이상이면 모순으로 판정
    
    Returns:
        is_valid: 유효성 여부
        violations: 위반 사항 리스트
    """
    violations = []
    
    for id_a, id_b in contradiction_pairs:
        score_a = sorting_data.get(id_a, 0)
        score_b = sorting_data.get(id_b, 0)
        
        # 둘 다 높은 점수(>=threshold)면 모순
        if score_a >= threshold and score_b >= threshold:
            violations.append({
                "type": "both_high",
                "pair": (id_a, id_b),
                "scores": (score_a, score_b),
                "message": f"상충 문항 {id_a}({score_a})와 {id_b}({score_b}) 모두 높은 동의"
            })
        
        # 둘 다 낮은 점수(<=-threshold)도 모순일 수 있음
        if score_a <= -threshold and score_b <= -threshold:
            violations.append({
                "type": "both_low",
                "pair": (id_a, id_b),
                "scores": (score_a, score_b),
                "message": f"상충 문항 {id_a}({score_a})와 {id_b}({score_b}) 모두 강한 비동의"
            })
    
    is_valid = len(violations) == 0
    
    if not is_valid:
        print(f"[VALIDATION] Mirror Test 실패: {len(violations)}개 모순 발견", flush=True)
    
    return is_valid, violations


def flatline_check(
    sorting_data: Dict[str, int],
    min_std: float = 1.5,
    neutral_threshold: int = 2
) -> Tuple[bool, Dict]:
    """
    Flat-lining Check: 답변이 중립에 몰리는 경우 검출
    
    Args:
        sorting_data: 문항 ID → 점수 딕셔너리
        min_std: 최소 표준편차 (이하이면 플랫라인)
        neutral_threshold: 중립으로 간주하는 점수 범위 (-n ~ +n)
    
    Returns:
        is_valid: 유효성 여부
        stats: 통계 정보
    """
    scores = list(sorting_data.values())
    
    if len(scores) == 0:
        return False, {"error": "No scores provided"}
    
    scores_array = np.array(scores)
    std = np.std(scores_array)
    mean = np.mean(scores_array)
    
    # 중립 범위 내 점수 비율
    neutral_count = sum(1 for s in scores if abs(s) <= neutral_threshold)
    neutral_ratio = neutral_count / len(scores)
    
    # 극단값 비율
    extreme_count = sum(1 for s in scores if abs(s) >= 4)
    extreme_ratio = extreme_count / len(scores)
    
    stats = {
        "std": float(std),
        "mean": float(mean),
        "neutral_ratio": float(neutral_ratio),
        "extreme_ratio": float(extreme_ratio),
        "total_items": len(scores)
    }
    
    # 표준편차가 너무 낮으면 플랫라인
    is_valid = std >= min_std
    
    if not is_valid:
        print(f"[VALIDATION] Flat-line 감지: SD={std:.2f} (기준: {min_std})", flush=True)
    
    return is_valid, stats


def validate_sorting(
    sorting_data: Dict[str, int],
    contradiction_pairs: List[Tuple[str, str]],
    mirror_threshold: int = 3,
    min_std: float = 1.5
) -> Tuple[bool, Dict]:
    """
    전체 Q-Sorting 유효성 검증
    
    Returns:
        is_valid: 전체 유효성
        report: 상세 검증 리포트
    """
    mirror_valid, mirror_violations = mirror_test(
        sorting_data, contradiction_pairs, mirror_threshold
    )
    
    flatline_valid, flatline_stats = flatline_check(sorting_data, min_std)
    
    is_valid = mirror_valid and flatline_valid
    
    report = {
        "is_valid": is_valid,
        "mirror_test": {
            "passed": mirror_valid,
            "violations": mirror_violations
        },
        "flatline_check": {
            "passed": flatline_valid,
            "stats": flatline_stats
        }
    }
    
    status = "✅ 통과" if is_valid else "❌ 실패"
    print(f"[VALIDATION] 전체 검증 {status}", flush=True)
    
    return is_valid, report


def check_forced_distribution(
    sorting_data: Dict[str, int],
    expected_distribution: Dict[int, int]
) -> Tuple[bool, Dict]:
    """
    강제 분포 준수 여부 확인
    
    Args:
        sorting_data: 문항 ID → 점수 딕셔너리
        expected_distribution: 점수 → 개수 (예: {-5: 2, -4: 3, ...})
    
    Returns:
        is_valid: 분포 준수 여부
        difference: 예상 vs 실제 차이
    """
    actual_distribution = {}
    for score in sorting_data.values():
        actual_distribution[score] = actual_distribution.get(score, 0) + 1
    
    differences = {}
    is_valid = True
    
    for score, expected_count in expected_distribution.items():
        actual_count = actual_distribution.get(score, 0)
        if actual_count != expected_count:
            differences[score] = {
                "expected": expected_count,
                "actual": actual_count,
                "diff": actual_count - expected_count
            }
            is_valid = False
    
    if not is_valid:
        print(f"[VALIDATION] 강제 분포 위반: {differences}", flush=True)
    
    return is_valid, {"is_valid": is_valid, "differences": differences}


def self_check_sorting(
    persona: Dict,
    sorting_data: Dict[str, int],
    q_set: List[Dict],
    similarity_threshold: float = 0.6
) -> Tuple[bool, Dict]:
    """
    Self-Check: 페르소나의 +5 선택 이유가 프로필과 일치하는지 검증
    
    Process:
    1. 페르소나에게 "+5를 준 문항들을 왜 선택했는지" 물음
    2. 답변을 페르소나 프로필과 비교 (Semantic Similarity)
    3. 유사도 < 0.6이면 "Hallucination"으로 판정하고 폐기
    
    Args:
        persona: 페르소나 정보
        sorting_data: 문항 ID → 점수 딕셔너리
        q_set: Q-Set 문항 리스트
        similarity_threshold: 최소 유사도 (기본 0.6)
    
    Returns:
        is_valid: 일관성 충족 여부
        report: 검증 상세 리포트
    """
    # +5 문항 추출
    top_items = []
    for q in q_set:
        if sorting_data.get(q["id"], 0) >= 4:
            top_items.append(q["text"])
    
    if not top_items:
        return True, {"skip": "No +4/+5 items found"}
    
    # 페르소나 프로필 텍스트 구성
    psycho = persona.get("psychographics", {})
    profile_text = " ".join([
        " ".join(psycho.get("core_values", [])),
        " ".join(psycho.get("fears", [])),
        persona.get("brief_description", ""),
        persona.get("internal_conflict", "")
    ])
    
    # LLM에게 선택 이유 물어보기
    prompt = f"""당신은 {persona.get('name', 'Unknown')}입니다.
프로필: {persona.get('brief_description', '')}
핵심 가치: {psycho.get('core_values', [])}

Q-Sorting에서 다음 문항들에 +4 또는 +5를 주었습니다:
{chr(10).join([f"• {item}" for item in top_items[:5]])}

왜 이 문항들을 가장 높게 평가했는지 1인칭으로 설명해주세요.
(2-3문장으로 간결하게)
"""
    
    reasoning_result = generate_json(f'{{"instruction": "{prompt}", "response_format": {{"reasoning": "설명"}}}}')
    reasoning_text = reasoning_result.get("reasoning", "")
    
    if not reasoning_text:
        return True, {"skip": "No reasoning generated"}
    
    # 임베딩 비교
    profile_embedding = generate_embedding(profile_text)
    reasoning_embedding = generate_embedding(reasoning_text)
    
    similarity = calculate_cosine_similarity(profile_embedding, reasoning_embedding)
    
    is_valid = similarity >= similarity_threshold
    
    report = {
        "persona_name": persona.get("name", "Unknown"),
        "similarity": float(similarity),
        "threshold": similarity_threshold,
        "is_valid": is_valid,
        "reasoning_excerpt": reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
    }
    
    if not is_valid:
        print(f"[VALIDATION] Self-Check 실패: {persona.get('name', 'Unknown')} (유사도: {similarity:.2f} < {similarity_threshold})", flush=True)
    else:
        print(f"[VALIDATION] Self-Check 통과: {persona.get('name', 'Unknown')} (유사도: {similarity:.2f})", flush=True)
    
    return is_valid, report


if __name__ == "__main__":
    # 테스트
    test_sorting = {"A": 5, "B": 4, "C": -5, "D": -4, "E": 3, "F": -3}
    test_pairs = [("A", "B")]  # A와 B는 상충
    
    valid, report = validate_sorting(test_sorting, test_pairs)
    print(f"Valid: {valid}")
    print(f"Report: {report}")
