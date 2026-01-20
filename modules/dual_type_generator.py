"""
Step 6: Dual-Type Generation Module (유형의 이원화)
각 Factor에 대해 긍정 편향 유형과 부정 편향 유형을 분리합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from utils.llm_client import generate_json
from modules.factor_analysis import get_factor_interpretation_data


def generate_dual_types(
    factor_scores: pd.DataFrame,
    q_set: list[str],
    topic_info: dict,
    significant_loadings: dict
) -> list[dict]:
    """
    각 Factor에 대해 긍정/부정 이원화 유형을 생성합니다.
    
    Args:
        factor_scores: 요인별 문항 Z-score DataFrame
        q_set: Q-Set 문항 리스트
        topic_info: 연구 주제 정보
        significant_loadings: 요인별 유의미한 참여자 정보
    
    Returns:
        생성된 유형 리스트
    """
    print("\n" + "="*60)
    print("🔀 유형 이원화 (Dual-Type Generation)")
    print("="*60)
    
    interpretation_data = get_factor_interpretation_data(factor_scores, q_set, top_n=7)
    all_types = []
    
    for factor_name, data in interpretation_data.items():
        print(f"\n📌 {factor_name} 이원화 중...")
        
        # 긍정 편향 유형 생성
        positive_type = generate_type(
            factor_name=factor_name,
            bias="positive",
            key_items=data["top_items"],
            topic_info=topic_info,
            significant_participants=[
                p for p in significant_loadings.get(factor_name, [])
                if p["direction"] == "positive"
            ]
        )
        positive_type["factor"] = factor_name
        positive_type["bias"] = "positive"
        positive_type["key_statements"] = data["top_items"]
        all_types.append(positive_type)
        print(f"   ✅ {positive_type.get('type_name', 'N/A')} (긍정)")
        
        # 부정 편향 유형 생성
        negative_type = generate_type(
            factor_name=factor_name,
            bias="negative",
            key_items=data["bottom_items"],
            topic_info=topic_info,
            significant_participants=[
                p for p in significant_loadings.get(factor_name, [])
                if p["direction"] == "negative"
            ]
        )
        negative_type["factor"] = factor_name
        negative_type["bias"] = "negative"
        negative_type["key_statements"] = data["bottom_items"]
        all_types.append(negative_type)
        print(f"   ✅ {negative_type.get('type_name', 'N/A')} (부정)")
    
    print(f"\n📊 총 {len(all_types)}개 유형 생성 완료")
    
    return all_types


def generate_type(
    factor_name: str,
    bias: str,
    key_items: list[dict],
    topic_info: dict,
    significant_participants: list[dict]
) -> dict:
    """
    단일 유형을 생성합니다.
    
    Args:
        factor_name: 요인 이름
        bias: "positive" 또는 "negative"
        key_items: 핵심 문항 리스트
        topic_info: 연구 주제 정보
        significant_participants: 해당 유형의 대표 참여자들
    
    Returns:
        생성된 유형 정보
    """
    bias_label = "긍정 편향" if bias == "positive" else "부정 편향"
    
    items_text = "\n".join([
        f"- {item['statement']} (Z-score: {item['z_score']:.2f})"
        for item in key_items
    ])
    
    participants_text = ""
    if significant_participants:
        participants_text = "대표 참여자: " + ", ".join([
            f"{p['name']} (적재량: {p['loading']:.2f})"
            for p in significant_participants[:3]
        ])
    
    prompt = f"""
Q방법론 연구에서 도출된 유형을 분석하고 설명해주세요.

## 연구 주제
{topic_info.get('final_topic', '')}

## 유형 정보
- 요인: {factor_name}
- 편향: {bias_label}
{participants_text}

## 핵심 문항 (Z-score 순)
{items_text}

## 유형 명명 규칙 (★ 매우 중요)
1. **다른 유형과 명확히 구분되는 이름**: 비슷한 이름 금지! 각 유형의 가장 두드러진 특징을 담아야 함
2. **형식**: "정식 이름 (Raw Voice)"
   - 정식 이름: 학술적이고 전문적인 명칭 (4~8자)
   - Raw Voice: 그 유형 사람이 실제로 할 법한 한마디 (5~15자)
3. **예시**:
   - "냉소적 생존주의자 (결국 살아남는 게 이기는 거야)"
   - "열정적 가치추구형 (내 신념은 타협 못 해)"
   - "무관심 방관자 (남 일에 감정 쏟지 마)"
   - "현실주의 적응가 (안 맞아도 맞춰야지 뭐)"

## 요청사항
이 유형의 특성을 심층 분석하여 다음 정보를 JSON으로 제공해주세요.
특히 '생존 본능', '방어 기제', '숨겨진 두려움', '자기 정당화' 부분은 날것 그대로의 심리를 파헤쳐주세요.

{{
    "type_name": "정식 이름 (Raw Voice 한마디)",
    "short_description": "15단어 이내의 핵심 특징 요약 (다른 유형과 차별점 강조)",
    "survival_instinct": "이 유형이 무의식적으로 추구하는 생존 전략. 왜 이런 태도를 가지게 되었는지 본능적 관점에서 분석 (50자 이상)",
    "defense_mechanism": "이 유형이 사용하는 심리적 방어 기제. 어떤 상황에서 어떻게 자신을 보호하는지 (50자 이상)",
    "hidden_fear": "이 유형이 겉으로 드러내지 않지만 내면에 가진 두려움과 불안 (50자 이상)",
    "self_justification": "이 유형이 자신의 태도와 행동을 정당화하는 내면의 논리와 말투 (50자 이상)",
    "core_values": ["핵심가치1", "핵심가치2", "핵심가3"],
    "trigger_phrases": ["이 유형을 자극하는 말1", "자극하는 말2", "자극하는 말3"],
    "action_plan": ["이 유형에게 효과적인 접근법1", "접근법2", "접근법3"]
}}
"""
    
    return generate_json(prompt, temperature=0.7)


def create_type_summary(types: list[dict]) -> str:
    """
    모든 유형의 요약을 생성합니다.
    
    Args:
        types: 유형 리스트
    
    Returns:
        마크다운 형식의 요약
    """
    summary = "# 유형 요약\n\n"
    
    for i, t in enumerate(types):
        bias_emoji = "➕" if t.get("bias") == "positive" else "➖"
        summary += f"""
## {i+1}. {t.get('type_name', f'유형 {i+1}')} {bias_emoji}

**요인**: {t.get('factor', 'N/A')} | **편향**: {"긍정" if t.get('bias') == 'positive' else "부정"}

**핵심 특징**: {t.get('short_description', 'N/A')}

**심리 분석**: {t.get('psychology_analysis', 'N/A')}

**핵심 가치**: {', '.join(t.get('core_values', []))}

---
"""
    
    return summary


if __name__ == "__main__":
    # 테스트
    test_factor_scores = pd.DataFrame({
        'Factor1': [1.5, 1.2, 0.8, -0.5, -1.0, -1.5],
        'Factor2': [-1.0, 0.5, 1.0, 1.5, -0.5, -1.2]
    }, index=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'])
    
    test_q_set = [
        "나는 A가 중요하다",
        "나는 B를 선호한다",
        "나는 C를 추구한다",
        "나는 D가 필요하다",
        "나는 E를 원한다",
        "나는 F를 지지한다"
    ]
    
    test_topic = {"final_topic": "테스트 주제"}
    test_loadings = {"Factor1": [], "Factor2": []}
    
    types = generate_dual_types(test_factor_scores, test_q_set, test_topic, test_loadings)
    print(create_type_summary(types))
