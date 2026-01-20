"""
Match/Mismatch Matrix
Group A ↔ Group B 상성 분석
"""
from typing import Dict, List, Tuple
import numpy as np
from utils.llm_client import generate_json


def calculate_type_compatibility(
    type_a: Dict, 
    type_b: Dict
) -> Tuple[float, str]:
    """
    두 유형 간의 호환성 점수 계산
    
    Returns:
        compatibility_score: -1.0 (최악) ~ +1.0 (최고)
        chemistry: 'synergy' | 'neutral' | 'conflict'
    """
    # 핵심 가치 비교
    values_a = set(type_a.get("core_values", []))
    values_b = set(type_b.get("core_values", []))
    
    value_overlap = len(values_a & values_b)
    value_total = len(values_a | values_b) if values_a | values_b else 1
    value_compatibility = value_overlap / value_total
    
    # 방어 기제 충돌 확인
    defense_a = type_a.get("defense_mechanism", "").lower()
    defense_b = type_b.get("defense_mechanism", "").lower()
    
    # 충돌하는 방어 기제 패턴
    conflicting_defenses = [
        ("회피", "직면"), ("공격", "회피"), ("통제", "자유")
    ]
    
    defense_conflict = 0
    for d1, d2 in conflicting_defenses:
        if (d1 in defense_a and d2 in defense_b) or (d2 in defense_a and d1 in defense_b):
            defense_conflict = -0.3
            break
    
    # 두려움 상호 자극 확인
    fear_a = type_a.get("hidden_fear", "").lower()
    trigger_b = " ".join(type_b.get("trigger_phrases", [])).lower()
    
    fear_triggered = -0.2 if any(word in trigger_b for word in fear_a.split()) else 0
    
    # 최종 점수 계산
    compatibility_score = value_compatibility + defense_conflict + fear_triggered
    compatibility_score = max(-1.0, min(1.0, compatibility_score))
    
    # 화학 작용 판정
    if compatibility_score >= 0.3:
        chemistry = "synergy"
    elif compatibility_score <= -0.2:
        chemistry = "conflict"
    else:
        chemistry = "neutral"
    
    return compatibility_score, chemistry


def generate_match_matrix(
    types_a: List[Dict],
    types_b: List[Dict],
    topic_info: Dict
) -> Dict:
    """
    Group A와 Group B 유형 간 전체 상성 매트릭스 생성
    """
    print(f"\n[MATRIX] === Match/Mismatch Matrix 생성 ===", flush=True)
    
    matrix = []
    
    for type_a in types_a:
        row = []
        for type_b in types_b:
            score, chemistry = calculate_type_compatibility(type_a, type_b)
            row.append({
                "type_a": type_a.get("type_name", "Unknown A"),
                "type_b": type_b.get("type_name", "Unknown B"),
                "score": score,
                "chemistry": chemistry
            })
        matrix.append(row)
    
    # 최고/최악 매칭 찾기
    all_pairs = [cell for row in matrix for cell in row]
    best_match = max(all_pairs, key=lambda x: x["score"])
    worst_match = min(all_pairs, key=lambda x: x["score"])
    
    result = {
        "matrix": matrix,
        "best_match": best_match,
        "worst_match": worst_match,
        "group_a": topic_info.get("group_a", "Group A"),
        "group_b": topic_info.get("group_b", "Group B")
    }
    
    print(f"[MATRIX] 최고 매칭: {best_match['type_a']} ↔ {best_match['type_b']} ({best_match['score']:.2f})", flush=True)
    print(f"[MATRIX] 최악 매칭: {worst_match['type_a']} ↔ {worst_match['type_b']} ({worst_match['score']:.2f})", flush=True)
    
    return result


def generate_communication_scripts(
    match_info: Dict,
    topic_info: Dict
) -> Dict:
    """
    최고/최악 매칭에 대한 커뮤니케이션 스크립트 생성
    """
    best = match_info["best_match"]
    worst = match_info["worst_match"]
    
    prompt = f"""당신은 그룹 간 커뮤니케이션 전문가입니다.

주제: {topic_info.get('final_topic', '')}
Group A: {match_info.get('group_a', 'A')}
Group B: {match_info.get('group_b', 'B')}

## 최고 시너지 매칭
- {best['type_a']} ↔ {best['type_b']} (점수: {best['score']:.2f})

## 최악 갈등 매칭  
- {worst['type_a']} ↔ {worst['type_b']} (점수: {worst['score']:.2f})

각 케이스에 대해 실제 사용할 수 있는 커뮤니케이션 스크립트를 작성해주세요.

JSON 형식:
{{
  "best_match_scripts": {{
    "opening_line": "첫 대화를 여는 문장",
    "appreciation_phrases": ["인정/감사 표현 1", "표현 2"],
    "collaboration_prompts": ["협업 제안 문장 1", "제안 2"],
    "do_list": ["해야 할 것 1", "해야 할 것 2"],
    "dont_list": ["피해야 할 것 1", "피해야 할 것 2"]
  }},
  "worst_match_scripts": {{
    "warning": "이 매칭의 위험성 경고 문구",
    "opening_line": "조심스럽게 대화를 여는 문장",
    "defusing_phrases": ["갈등 완화 문장 1", "문장 2"],
    "absolute_donts": ["절대 해서는 안 되는 것 1", "안 되는 것 2"],
    "exit_strategies": ["상황 악화 시 탈출 전략 1", "전략 2"]
  }},
  "general_tips": ["일반 팁 1", "일반 팁 2", "일반 팁 3"]
}}
"""
    
    result = generate_json(prompt)
    result["best_match"] = best
    result["worst_match"] = worst
    
    print(f"[MATRIX] 커뮤니케이션 스크립트 생성 완료", flush=True)
    
    return result


def generate_risk_warnings(
    matrix: Dict,
    risk_threshold: float = -0.3
) -> List[Dict]:
    """
    위험 매칭 경고 생성
    """
    warnings = []
    
    all_pairs = [cell for row in matrix["matrix"] for cell in row]
    
    for pair in all_pairs:
        if pair["score"] <= risk_threshold:
            risk_level = "critical" if pair["score"] <= -0.5 else "warning"
            
            # 이탈 확률 추정 (단순 휴리스틱)
            churn_probability = min(95, int(abs(pair["score"]) * 100 + 30))
            
            warnings.append({
                "type_a": pair["type_a"],
                "type_b": pair["type_b"],
                "score": pair["score"],
                "risk_level": risk_level,
                "churn_probability": churn_probability,
                "warning_message": f"⚠️ 경고: [{pair['type_a']}]에게 [{pair['type_b']}]를 매칭하면 이탈 확률이 {churn_probability}% 증가합니다."
            })
    
    print(f"[MATRIX] {len(warnings)}개 위험 매칭 경고 생성", flush=True)
    
    return warnings


def analyze_dual_group_dynamics(
    types_a: List[Dict],
    types_b: List[Dict],
    topic_info: Dict
) -> Dict:
    """
    Dual Group Mode: 전체 그룹 간 역학 분석
    """
    # 매트릭스 생성
    matrix = generate_match_matrix(types_a, types_b, topic_info)
    
    # 커뮤니케이션 스크립트
    scripts = generate_communication_scripts(matrix, topic_info)
    
    # 위험 경고
    warnings = generate_risk_warnings(matrix)
    
    # LLM을 통한 종합 분석
    group_a = topic_info.get("group_a", "Group A")
    group_b = topic_info.get("group_b", "Group B")
    
    prompt = f"""당신은 집단 심리와 상호작용 전문가입니다.

{group_a}(6개 유형)와 {group_b}(6개 유형) 간의 상성 분석 결과:

최고 시너지: {matrix['best_match']['type_a']} ↔ {matrix['best_match']['type_b']}
최악 갈등: {matrix['worst_match']['type_a']} ↔ {matrix['worst_match']['type_b']}
위험 매칭 수: {len(warnings)}개

두 그룹이 서로를 오해하는 근본 원인과 해결책을 분석해주세요.

JSON:
{{
  "misunderstanding_root_cause": "상호 오해의 근본 원인",
  "group_a_perspective": "{group_a}가 {group_b}를 보는 시각",
  "group_b_perspective": "{group_b}가 {group_a}를 보는 시각", 
  "bridge_strategies": ["다리 놓기 전략 1", "전략 2", "전략 3"],
  "quick_wins": ["즉시 실행 가능한 개선 1", "개선 2"]
}}
"""
    
    dynamics = generate_json(prompt)
    
    return {
        "analysis_mode": "dual_group",
        "match_matrix": matrix,
        "communication_scripts": scripts,
        "risk_warnings": warnings,
        "dynamics_analysis": dynamics
    }


if __name__ == "__main__":
    print("Match/Mismatch Matrix Module Loaded")
