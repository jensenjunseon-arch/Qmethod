"""
Step 7: Report Generator Module (결과 리포트 생성)
분석 결과를 마크다운 형식의 리포트로 생성합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
import pandas as pd
import config


def generate_report(
    topic_info: dict,
    q_set: list[str],
    personas: list[dict],
    sorting_matrix: pd.DataFrame,
    factor_result: dict,
    types: list[dict],
    output_path: str = None
) -> str:
    """
    전체 분석 결과를 마크다운 리포트로 생성합니다.
    
    Args:
        topic_info: 연구 주제 정보
        q_set: Q-Set 문항 리스트
        personas: 페르소나 리스트
        sorting_matrix: Q-Sorting 데이터 매트릭스
        factor_result: 요인 분석 결과
        types: 생성된 유형 리스트
        output_path: 저장 경로 (None이면 자동 생성)
    
    Returns:
        리포트 파일 경로
    """
    report = []
    
    # 헤더
    report.append(f"""# Q방법론 연구 통찰 리포트

**생성일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}

---

## 1. 연구 개요

### 📌 연구 주제
**{topic_info.get('final_topic', 'N/A')}**

| 항목 | 내용 |
|------|------|
| 연구 질문 | {topic_info.get('research_question', 'N/A')} |
| 대상 집단 | {topic_info.get('target_population', 'N/A')} |
| 연구 맥락 | {topic_info.get('context', 'N/A')} |
| 핵심 키워드 | {', '.join(topic_info.get('keywords', []))} |

---

## 2. Q-Set 문항 ({len(q_set)}개)

""")
    
    # Q-Set 문항 목록
    for i, stmt in enumerate(q_set):
        report.append(f"{i+1}. {stmt}")
    
    report.append(f"""

---

## 3. 참여자 (P-Set) 정보 ({len(personas)}명)

""")
    
    # 페르소나 요약
    for i, p in enumerate(personas):
        report.append(f"""
### {i+1}. {p.get('name', f'참여자 {i+1}')}
- **프로필**: {p.get('age', '?')}세, {p.get('gender', '?')}, {p.get('occupation', '?')}
- **성격 특성**: {', '.join(p.get('personality_traits', []))}
- **핵심 가치관**: {', '.join(p.get('values', []))}
- **주제 태도**: {p.get('attitude_toward_topic', 'N/A')[:100]}...
""")
    
    report.append(f"""

---

## 4. 요인 분석 결과

### 📊 추출된 요인 수: {factor_result.get('n_factors', 'N/A')}

""")
    
    # 분산 설명력
    variance = factor_result.get('variance', {})
    if variance:
        report.append("| 요인 | Eigenvalue | 분산 설명력 | 누적 분산 |")
        report.append("|------|------------|-------------|-----------|")
        for i in range(factor_result.get('n_factors', 0)):
            report.append(f"| Factor {i+1} | {variance['ss_loadings'][i]:.2f} | {variance['proportion_var'][i]:.1%} | {variance['cumulative_var'][i]:.1%} |")
    
    # 요인 적재량 표
    loadings_df = factor_result.get('loadings_df')
    if loadings_df is not None:
        report.append("\n### 참여자별 요인 적재량\n")
        report.append(loadings_df.to_markdown())
    
    report.append(f"""

---

## 5. 도출된 유형 분석 ({len(types)}개 유형)

""")
    
    # 각 유형 상세 분석
    for i, t in enumerate(types):
        bias_emoji = "🔵" if t.get("bias") == "positive" else "🔴"
        bias_label = "긍정 편향" if t.get("bias") == "positive" else "부정 편향"
        
        # 핵심 문항
        key_statements = t.get('key_statements', [])
        statements_text = ""
        for item in key_statements[:5]:
            score_sign = "+" if item['z_score'] > 0 else ""
            statements_text += f"   - {item['statement']} (Z: {score_sign}{item['z_score']:.2f})\n"
        
        report.append(f"""
### {bias_emoji} 유형 {i+1}: {t.get('type_name', f'유형 {i+1}')}

> **{t.get('short_description', 'N/A')}**

| 속성 | 정보 |
|------|------|
| 기반 요인 | {t.get('factor', 'N/A')} |
| 편향 방향 | {bias_label} |

#### 📖 심리 분석
{t.get('psychology_analysis', 'N/A')}

#### 🎯 핵심 가치
{', '.join(t.get('core_values', ['N/A']))}

#### 📋 행동 패턴
""")
        for pattern in t.get('behavioral_patterns', []):
            report.append(f"- {pattern}")
        
        report.append(f"""

#### 💪 강점
""")
        for strength in t.get('strengths', []):
            report.append(f"- {strength}")
        
        report.append(f"""

#### ⚠️ 도전 과제
""")
        for challenge in t.get('challenges', []):
            report.append(f"- {challenge}")
        
        report.append(f"""

#### 💡 실용적 조언
""")
        for advice in t.get('practical_advice', []):
            report.append(f"1. {advice}")
        
        report.append(f"""

#### 🚀 권장 행동
""")
        for action in t.get('recommended_actions', []):
            report.append(f"- {action}")
        
        report.append(f"""

#### 📌 핵심 문항
{statements_text}

---
""")
    
    # 결론 및 요약
    report.append(f"""

## 6. 결론 및 요약

### 📊 유형 분포 요약

| 유형명 | 요인 | 편향 | 핵심 특징 |
|--------|------|------|----------|
""")
    
    for t in types:
        bias = "긍정" if t.get("bias") == "positive" else "부정"
        report.append(f"| {t.get('type_name', 'N/A')} | {t.get('factor', 'N/A')} | {bias} | {t.get('short_description', 'N/A')} |")
    
    report.append(f"""

### 🔍 주요 통찰

본 Q방법론 연구를 통해 **{topic_info.get('final_topic', '연구 주제')}**에 대한 다양한 관점을 탐색하였습니다.

- **추출된 요인 수**: {factor_result.get('n_factors', 0)}개
- **도출된 유형 수**: {len(types)}개 (요인당 긍정/부정 2개씩)
- **분석 대상 문항**: {len(q_set)}개
- **가상 참여자**: {len(personas)}명

각 유형은 연구 주제에 대한 독특한 주관적 관점을 반영하며, 실무에서 타겟 집단별 맞춤형 전략 수립에 활용할 수 있습니다.

---

*이 리포트는 Q-Methodology Research Insight Generator에 의해 자동 생성되었습니다.*
""")
    
    # 파일 저장
    report_content = "\n".join(report)
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"q_report_{timestamp}.md"
        output_path = os.path.join(config.OUTPUT_DIR, filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ 리포트 저장 완료: {output_path}")
    
    return output_path


def save_data_artifacts(
    topic_info: dict,
    q_population: list[str],
    q_set: list[str],
    personas: list[dict],
    sorting_matrix: pd.DataFrame,
    output_dir: str = None
) -> dict:
    """
    분석에 사용된 데이터를 JSON/CSV 형식으로 저장합니다.
    
    Args:
        topic_info: 연구 주제 정보
        q_population: Q-Population 문항
        q_set: Q-Set 문항
        personas: 페르소나
        sorting_matrix: Q-Sorting 매트릭스
        output_dir: 저장 디렉토리
    
    Returns:
        저장된 파일 경로 딕셔너리
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    paths = {}
    
    # 연구 주제 정보
    topic_path = os.path.join(output_dir, f"topic_{timestamp}.json")
    with open(topic_path, 'w', encoding='utf-8') as f:
        json.dump(topic_info, f, ensure_ascii=False, indent=2)
    paths['topic'] = topic_path
    
    # Q-Population
    qpop_path = os.path.join(output_dir, f"q_population_{timestamp}.json")
    with open(qpop_path, 'w', encoding='utf-8') as f:
        json.dump(q_population, f, ensure_ascii=False, indent=2)
    paths['q_population'] = qpop_path
    
    # Q-Set
    qset_path = os.path.join(output_dir, f"q_set_{timestamp}.json")
    with open(qset_path, 'w', encoding='utf-8') as f:
        json.dump(q_set, f, ensure_ascii=False, indent=2)
    paths['q_set'] = qset_path
    
    # 페르소나
    personas_path = os.path.join(output_dir, f"personas_{timestamp}.json")
    with open(personas_path, 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    paths['personas'] = personas_path
    
    # Q-Sorting 매트릭스
    matrix_path = os.path.join(output_dir, f"sorting_matrix_{timestamp}.csv")
    sorting_matrix.to_csv(matrix_path, encoding='utf-8')
    paths['sorting_matrix'] = matrix_path
    
    print(f"\n📁 데이터 아티팩트 저장 완료: {output_dir}")
    
    return paths


if __name__ == "__main__":
    # 테스트
    test_topic = {
        "final_topic": "테스트 연구 주제",
        "research_question": "테스트 질문",
        "target_population": "테스트 대상",
        "context": "테스트 맥락",
        "keywords": ["키워드1", "키워드2"]
    }
    
    test_types = [
        {
            "factor": "Factor1",
            "bias": "positive",
            "type_name": "테스트 유형",
            "short_description": "테스트 설명",
            "psychology_analysis": "심리 분석 내용",
            "core_values": ["가치1", "가치2"],
            "behavioral_patterns": ["패턴1"],
            "strengths": ["강점1"],
            "challenges": ["도전1"],
            "practical_advice": ["조언1"],
            "recommended_actions": ["행동1"],
            "key_statements": [{"statement": "테스트 문항", "z_score": 1.5}]
        }
    ]
    
    print("테스트 리포트 생성...")
