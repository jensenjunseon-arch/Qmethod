"""
Step 4: Q-Sorting Simulation Module (모의 분류)
각 페르소나가 Q-Set을 강제 분포에 따라 분류하는 과정을 시뮬레이션합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils.llm_client import generate_json
import config


def get_forced_distribution_slots() -> list[int]:
    """
    강제 분포에 따른 슬롯 리스트를 반환합니다.
    
    Returns:
        각 슬롯에 배치할 점수 리스트 (예: [-5, -5, -4, -4, -4, ...])
    """
    slots = []
    for score, count in sorted(config.FORCED_DISTRIBUTION.items()):
        slots.extend([score] * count)
    return slots


def simulate_single_sorting(
    persona: dict,
    q_set: list[str],
    topic_info: dict
) -> dict[int, int]:
    """
    단일 페르소나의 Q-Sorting을 시뮬레이션합니다.
    
    Args:
        persona: 페르소나 정보
        q_set: Q-Set 문항 리스트
        topic_info: 연구 주제 정보
    
    Returns:
        {문항_인덱스: 점수} 딕셔너리
    """
    statements_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(q_set)])
    distribution_desc = ", ".join([f"{score}점: {count}개" for score, count in sorted(config.FORCED_DISTRIBUTION.items())])
    
    prompt = f"""
다음 페르소나의 관점에서 Q-Set 문항들을 분류해주세요.

## 연구 주제
{topic_info.get('final_topic', '')}

## 페르소나 정보
- 이름: {persona.get('name', 'N/A')}
- 나이: {persona.get('age', 'N/A')}
- 직업: {persona.get('occupation', 'N/A')}
- 성격: {', '.join(persona.get('personality_traits', []))}
- 가치관: {', '.join(persona.get('values', []))}
- 주제에 대한 태도: {persona.get('attitude_toward_topic', 'N/A')}
- 의사결정 스타일: {persona.get('decision_making_style', 'N/A')}

## Q-Set 문항 ({len(q_set)}개)
{statements_text}

## 강제 분포 규칙
점수 범위: -5 (가장 비동의) ~ +5 (가장 동의)
각 점수별 배치 문항 수: {distribution_desc}
총 {sum(config.FORCED_DISTRIBUTION.values())}개 문항

## 분류 지침
1. 이 페르소나의 성격, 가치관, 태도를 고려하여 각 문항에 대한 동의/비동의 정도를 판단합니다.
2. 강제 분포 규칙을 정확히 따라야 합니다 (각 점수별 문항 수 준수).
3. 페르소나의 관점에서 일관성 있게 분류합니다.

JSON 형식으로 응답해주세요:
{{
    "sorting": {{
        "1": 점수,
        "2": 점수,
        ...
        "{len(q_set)}": 점수
    }},
    "reasoning": "분류 시 고려한 핵심 요소들 간단 설명"
}}
"""
    
    result = generate_json(prompt, temperature=0.6)
    sorting = result.get("sorting", {})
    
    # 문자열 키를 정수로 변환하고, 값도 정수로 변환 (None 값은 0으로 처리)
    parsed = {}
    for k, v in sorting.items():
        try:
            key = int(k)
            value = int(v) if v is not None else 0
            # 값이 -5~+5 범위 내에 있는지 확인
            value = max(-5, min(5, value))
            parsed[key] = value
        except (ValueError, TypeError):
            # 변환 실패 시 기본값 0
            try:
                parsed[int(k)] = 0
            except:
                pass
    return parsed


def validate_and_adjust_sorting(sorting: dict[int, int]) -> dict[int, int]:
    """
    분류 결과가 강제 분포 규칙을 따르는지 검증하고 강제로 조정합니다.
    
    Args:
        sorting: 원본 분류 결과
    
    Returns:
        조정된 분류 결과 (반드시 강제 분포를 따름)
    """
    target_dist = config.FORCED_DISTRIBUTION.copy()
    n_items = len(sorting)
    
    # 점수 순서대로 슬롯 생성 (예: [-5, -5, -4, -4, -4, ...])
    target_slots = []
    for score in sorted(target_dist.keys()):
        target_slots.extend([score] * target_dist[score])
    
    # 슬롯 수 확인
    if len(target_slots) != n_items:
        print(f"⚠️ 슬롯 수({len(target_slots)})와 문항 수({n_items})가 다름, 비율 조정 중...", flush=True)
        # 비율에 맞게 슬롯 수 조정
        total = sum(target_dist.values())
        adjusted_slots = []
        for score in sorted(target_dist.keys()):
            count = round(target_dist[score] * n_items / total)
            adjusted_slots.extend([score] * count)
        # 부족하거나 초과하면 0점으로 조정
        while len(adjusted_slots) < n_items:
            adjusted_slots.append(0)
        while len(adjusted_slots) > n_items:
            adjusted_slots.pop()
        target_slots = adjusted_slots
    
    # 원본 점수를 기준으로 문항 정렬 (높은 점수순)
    sorted_items = sorted(sorting.items(), key=lambda x: x[1], reverse=True)
    
    # 높은 target_slots부터 할당
    target_slots_sorted = sorted(target_slots, reverse=True)
    
    adjusted = {}
    for (item_idx, _), new_score in zip(sorted_items, target_slots_sorted):
        adjusted[item_idx] = new_score
    
    return adjusted


def simulate_all_sortings(
    personas: list[dict],
    q_set: list[str],
    topic_info: dict
) -> pd.DataFrame:
    """
    모든 페르소나의 Q-Sorting을 시뮬레이션하고 데이터 매트릭스를 생성합니다.
    
    Args:
        personas: 페르소나 리스트
        q_set: Q-Set 문항 리스트
        topic_info: 연구 주제 정보
    
    Returns:
        20 x 60 데이터 매트릭스 (DataFrame)
    """
    print("\n" + "="*60)
    print("📊 Q-Sorting 시뮬레이션")
    print("="*60)
    
    all_sortings = []
    
    for i, persona in enumerate(personas):
        print(f"\n🎯 {persona.get('name', f'페르소나{i+1}')} Q-Sorting 중... ({i+1}/{len(personas)})")
        
        # 분류 시뮬레이션
        sorting = simulate_single_sorting(persona, q_set, topic_info)
        
        # 강제 분포 검증 및 조정
        sorting = validate_and_adjust_sorting(sorting)
        
        # 리스트 형태로 변환 (1-indexed에서 0-indexed로)
        row = [sorting.get(j+1, 0) for j in range(len(q_set))]
        all_sortings.append(row)
        
        # 분포 확인
        score_counts = {s: row.count(s) for s in sorted(config.FORCED_DISTRIBUTION.keys())}
        print(f"   분포: {score_counts}")
    
    # DataFrame 생성
    columns = [f"Q{i+1}" for i in range(len(q_set))]
    index = [p.get('name', f'P{i+1}') for i, p in enumerate(personas)]
    
    df = pd.DataFrame(all_sortings, columns=columns, index=index)
    
    print(f"\n✅ Q-Sorting 완료: {df.shape[0]} 참여자 × {df.shape[1]} 문항")
    
    return df


def display_sorting_matrix(df: pd.DataFrame) -> str:
    """
    Q-Sorting 매트릭스를 마크다운 형식으로 표시합니다.
    
    Args:
        df: Q-Sorting 데이터프레임
    
    Returns:
        마크다운 테이블 문자열
    """
    return df.to_markdown()


if __name__ == "__main__":
    # 테스트
    test_personas = [
        {
            "name": "김철수",
            "age": 28,
            "occupation": "IT 개발자",
            "personality_traits": ["분석적", "내향적", "완벽주의"],
            "values": ["개인 성장", "효율성"],
            "attitude_toward_topic": "워라밸을 중요하게 생각하지만 커리어 성장도 포기할 수 없다",
            "decision_making_style": "논리적"
        }
    ]
    
    test_q_set = [
        "나는 현재 상황이 개선될 거라고 믿는다.",
        "변화를 위해서는 개인의 노력이 가장 중요하다.",
        "사회 구조가 바뀌지 않으면 개인의 노력은 한계가 있다.",
    ]
    
    test_topic = {"final_topic": "MZ세대의 워라밸 인식"}
    
    sorting = simulate_single_sorting(test_personas[0], test_q_set, test_topic)
    print("테스트 결과:", sorting)
