"""
Polarity Decomposer
Factor → Positive Type + Negative Type 분리
귀추법적 해석 (생존 본능, 방어 기제 중심)
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from utils.llm_client import generate_json


def decompose_factor_to_types(
    factor_scores: np.ndarray,
    q_set: List[Dict],
    factor_index: int,
    topic_info: Dict
) -> Tuple[Dict, Dict]:
    """
    하나의 Factor를 Positive Type과 Negative Type으로 분리
    
    Args:
        factor_scores: Q-Set 문항별 요인 점수 (shape: [n_items])
        q_set: Q-Set 문항 리스트
        factor_index: 요인 인덱스 (1, 2, 3, ...)
        topic_info: 주제 정보
    
    Returns:
        positive_type: 동의 기반 유형
        negative_type: 비동의 기반 유형
    """
    # 정렬하여 상위/하위 문항 추출
    sorted_indices = np.argsort(factor_scores)
    
    # 상위 10개 (가장 동의하는 문항)
    top_indices = sorted_indices[-10:][::-1]
    top_statements = [q_set[i]["text"] for i in top_indices if i < len(q_set)]
    top_scores = [float(factor_scores[i]) for i in top_indices if i < len(factor_scores)]
    
    # 하위 10개 (가장 비동의하는 문항)
    bottom_indices = sorted_indices[:10]
    bottom_statements = [q_set[i]["text"] for i in bottom_indices if i < len(q_set)]
    bottom_scores = [float(factor_scores[i]) for i in bottom_indices if i < len(factor_scores)]
    
    # LLM을 통한 귀추법적 해석
    positive_type = interpret_type(
        topic_info, 
        factor_index, 
        "positive",
        top_statements, 
        top_scores,
        bottom_statements
    )
    
    negative_type = interpret_type(
        topic_info, 
        factor_index, 
        "negative",
        bottom_statements, 
        bottom_scores,
        top_statements
    )
    
    return positive_type, negative_type


def interpret_type(
    topic_info: Dict,
    factor_index: int,
    polarity: str,
    defining_statements: List[str],
    defining_scores: List[float],
    contrasting_statements: List[str]
) -> Dict:
    """
    Mirror Logic 기반 유형 해석
    
    ⚠️ CRITICAL MIRROR RULE:
    - Type A (Positive): 동의 문항으로 정의 (정상 해석)
    - Type B (Negative): Type A가 거부한 문항을 믿는 사람으로 해석
      → Type B는 단순히 "Type A가 아닌 것"이 아님
      → Type B는 Type A가 -5를 준 문항에 +5를 주는 별개의 캐릭터
    """
    topic = topic_info.get("final_topic", topic_info.get("topic", ""))
    group = topic_info.get("group", "참여자")
    
    if polarity == "positive":
        # Type A: 정상적인 동의 기반 해석
        prompt = f"""당신은 Q방법론 심리 분석 전문가입니다.

주제: {topic}
집단: {group}
요인: Factor {factor_index} - **Type A (Positive Pole)**

이 유형을 정의하는 핵심 문항들 (가장 강하게 동의, Z > +1.0):
{chr(10).join([f"• ✅ {s} (z={sc:.2f})" for s, sc in zip(defining_statements[:7], defining_scores[:7])])}

이 유형이 강하게 거부하는 문항들 (Z < -1.0):
{chr(10).join([f"• ❌ {s}" for s in contrasting_statements[:5]])}

다음 관점에서 해석해주세요:
1. **생존 본능 (Survival Instinct)**: 핵심 생존 전략
2. **방어 기제 (Defense Mechanism)**: 심리적 방어 방식
3. **숨겨진 두려움 (Hidden Fear)**: 표면 아래 불안
4. **자기 정당화 로직 (Self-Justification)**: 합리화 방식

JSON 형식:
{{
  "type_name": "짧고 직관적인 유형명 (한글)",
  "short_description": "한 문장 요약",
  "survival_instinct": "생존 본능",
  "defense_mechanism": "방어 기제",
  "hidden_fear": "숨겨진 두려움",
  "self_justification": "자기 정당화",
  "core_values": ["핵심 가치 1", "가치 2", "가치 3"],
  "trigger_phrases": ["자극 트리거 1", "트리거 2"],
  "action_plan": ["행동 지침 1", "지침 2", "지침 3"]
}}
"""
    else:
        # Type B: ⚠️ MIRROR RULE - Type A가 거부한 것을 믿는 사람으로 해석
        prompt = f"""당신은 Q방법론 심리 분석 전문가입니다.

주제: {topic}
집단: {group}
요인: Factor {factor_index} - **Type B (Negative Pole)**

⚠️ **MIRROR RULE 적용** ⚠️
Type A가 강하게 거부한 다음 문항들을 Type B는 **강하게 믿습니다**:
{chr(10).join([f"• 🔥 \"{s}\" ← Type B는 이것을 진심으로 믿음" for s in contrasting_statements[:7]])}

반대로, Type A가 믿는 다음 문항들을 Type B는 **거부합니다**:
{chr(10).join([f"• ❌ \"{s}\"" for s in defining_statements[:5]])}

중요: Type B를 해석할 때:
- ❌ 단순히 "Type A가 아닌 사람"으로 해석하지 마세요
- ✅ Type A가 거부한 것을 **적극적으로 믿고 실천하는** 별개의 캐릭터로 해석하세요
- ✅ 예: Type A가 "돈이 최고다"에 -5를 줬다면, Type B는 돈을 최우선시하는 "물질주의자"

이 관점에서 Type B를 완전히 독립된 유형으로 정의해주세요:
1. **생존 본능**: Type A가 거부한 가치를 핵심으로 삼는 이유는?
2. **방어 기제**: 이 믿음을 지키기 위해 어떤 심리적 방어를 하는가?
3. **숨겨진 두려움**: 이 믿음 뒤에 있는 불안은?
4. **자기 정당화**: 이 가치관을 어떻게 정당화하는가?

JSON 형식:
{{
  "type_name": "Type A와 이념적으로 반대되는 유형명 (한글)",
  "short_description": "Type A와 대조되는 한 문장 정체성",
  "survival_instinct": "Type A와 반대되는 생존 전략",
  "defense_mechanism": "이 믿음을 지키는 방어 기제",
  "hidden_fear": "이 캐릭터만의 숨겨진 두려움",
  "self_justification": "자기 정당화 로직",
  "core_values": ["Type A와 반대되는 핵심 가치 1", "가치 2", "가치 3"],
  "trigger_phrases": ["이 유형을 자극하는 말 1", "트리거 2"],
  "action_plan": ["행동 지침 1", "지침 2", "지침 3"],
  "mirror_belief": "Type A가 거부한 것 중 이 유형이 가장 믿는 신념"
}}
"""
    
    result = generate_json(prompt)
    
    # 메타데이터 추가
    result["factor"] = f"Factor {factor_index}"
    result["polarity"] = polarity
    result["bias"] = polarity
    result["defining_statements"] = defining_statements[:5]
    result["defining_scores"] = defining_scores[:5]
    result["mirror_contrasts"] = contrasting_statements[:5] if polarity == "negative" else []
    
    type_label = "Type A" if polarity == "positive" else "Type B (Mirror)"
    print(f"[POLARITY] Factor {factor_index} {type_label}: {result.get('type_name', 'Unknown')}", flush=True)
    
    return result


def generate_six_types(
    factor_scores_matrix: np.ndarray,
    q_set: List[Dict],
    topic_info: Dict,
    n_factors: int = 3
) -> List[Dict]:
    """
    DEPRECATED: Use conditional_decompose_factors instead
    """
    return conditional_decompose_factors(factor_scores_matrix, q_set, topic_info, None, n_factors)


def conditional_decompose_factors(
    factor_scores_matrix: np.ndarray,
    q_set: List[Dict],
    topic_info: Dict,
    factor_loadings: Optional[np.ndarray] = None,
    n_factors: int = 3,
    loading_threshold: float = 0.4
) -> List[Dict]:
    """
    Conditional Mirror Logic으로 Factor 분해
    
    Decision Rule:
    - CASE 1: Significant Negative Loaders Exist (< -0.4) → Bipolar Factor (Type A + Type B)
    - CASE 2: NO Significant Negative Loaders → Unipolar Factor (Type A only)
    
    Args:
        factor_scores_matrix: 요인 점수 행렬 (shape: [n_items, n_factors])
        q_set: Q-Set 문항 리스트
        topic_info: 주제 정보
        factor_loadings: 요인 적재량 행렬 (shape: [n_personas, n_factors])
        n_factors: 사용할 요인 수
        loading_threshold: 유의미한 적재량 임계값
    
    Returns:
        유형 리스트 (Bipolar: Type A + B, Unipolar: Type A only)
    """
    print(f"\n[POLARITY] === Conditional Factor Decomposition (Kaiser Rule) ===", flush=True)
    
    all_types = []
    factor_info = []
    
    for i in range(min(n_factors, factor_scores_matrix.shape[1])):
        factor_scores = factor_scores_matrix[:, i]
        
        # Check if Bipolar or Unipolar
        is_bipolar = False
        negative_loaders_count = 0
        positive_loaders_count = 0
        
        if factor_loadings is not None and i < factor_loadings.shape[1]:
            loadings = factor_loadings[:, i]
            negative_loaders_count = np.sum(loadings < -loading_threshold)
            positive_loaders_count = np.sum(loadings > loading_threshold)
            is_bipolar = negative_loaders_count > 0
        else:
            # factor_loadings가 없으면 factor_scores의 분포로 추정
            is_bipolar = np.min(factor_scores) < -0.5 and np.max(factor_scores) > 0.5
        
        factor_type = "Bipolar" if is_bipolar else "Unipolar"
        print(f"[POLARITY] Factor {i+1}: {factor_type} (Positive: {positive_loaders_count}, Negative: {negative_loaders_count})", flush=True)
        
        # 정렬하여 상위/하위 문항 추출
        sorted_indices = np.argsort(factor_scores)
        
        top_indices = sorted_indices[-10:][::-1]
        top_statements = [q_set[j]["text"] for j in top_indices if j < len(q_set)]
        top_scores = [float(factor_scores[j]) for j in top_indices if j < len(factor_scores)]
        
        bottom_indices = sorted_indices[:10]
        bottom_statements = [q_set[j]["text"] for j in bottom_indices if j < len(q_set)]
        bottom_scores = [float(factor_scores[j]) for j in bottom_indices if j < len(factor_scores)]
        
        # Type A는 항상 생성
        positive_type = interpret_type(
            topic_info, i + 1, "positive",
            top_statements, top_scores, bottom_statements
        )
        positive_type["factor_type"] = factor_type
        positive_type["is_consensus"] = not is_bipolar
        all_types.append(positive_type)
        
        # Type B는 Bipolar일 때만 생성
        if is_bipolar:
            negative_type = interpret_type(
                topic_info, i + 1, "negative",
                bottom_statements, bottom_scores, top_statements
            )
            negative_type["factor_type"] = factor_type
            negative_type["is_consensus"] = False
            all_types.append(negative_type)
        else:
            # Unipolar: Type B 없음 - 합의 항목으로 표시
            print(f"[POLARITY] Factor {i+1}: Unipolar - Type B 생략 (Universal Agreement)", flush=True)
        
        factor_info.append({
            "factor": i + 1,
            "type": factor_type,
            "positive_loaders": positive_loaders_count,
            "negative_loaders": negative_loaders_count
        })
    
    bipolar_count = sum(1 for f in factor_info if f["type"] == "Bipolar")
    unipolar_count = sum(1 for f in factor_info if f["type"] == "Unipolar")
    
    print(f"[POLARITY] === 총 {len(all_types)}개 유형 생성 (Bipolar: {bipolar_count}, Unipolar: {unipolar_count}) ===\n", flush=True)
    
    return all_types


def analyze_internal_conflict(types: List[Dict], topic_info: Dict) -> Dict:
    """
    Single Group Mode: 내부 갈등 분석
    같은 집단 내에서 왜 하위 유형으로 분화되는지
    """
    topic = topic_info.get("final_topic", "")
    group = topic_info.get("group", "참여자")
    
    type_summaries = "\n".join([
        f"- {t.get('type_name', 'Unknown')}: {t.get('short_description', '')}"
        for t in types
    ])
    
    prompt = f"""당신은 조직 심리학 전문가입니다.

주제: {topic}
집단: {group}

이 집단에서 다음 6가지 하위 유형이 발견되었습니다:
{type_summaries}

귀추법적 분석을 통해 다음을 설명해주세요:

1. **분화의 원인**: 왜 같은 집단이 이렇게 다른 유형으로 나뉘는가?
2. **공통 기반**: 이들이 공유하는 근본적인 불안이나 욕구는?
3. **잠재적 갈등**: 어떤 유형 간에 충돌이 예상되는가?
4. **내부 조화 전략**: 이 하위 유형들이 공존하려면?

JSON 형식:
{{
  "fragmentation_cause": "분화의 근본 원인",
  "shared_anxiety": "공통 불안/욕구",
  "conflict_pairs": [
    {{"type_a": "유형명", "type_b": "유형명", "conflict_reason": "갈등 원인"}}
  ],
  "harmony_strategies": ["전략 1", "전략 2", "전략 3"]
}}
"""
    
    result = generate_json(prompt)
    result["analysis_mode"] = "single_group"
    result["group"] = group
    
    print(f"[POLARITY] 내부 갈등 분석 완료", flush=True)
    
    return result


if __name__ == "__main__":
    # 테스트
    print("Polarity Decomposer Module Loaded")
