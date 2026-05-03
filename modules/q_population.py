"""
Step 2: Q-Population & Q-Set Construction Module (문항 생성 및 선정)
연구 주제를 바탕으로 Q-Population을 생성하고 Q-Set을 선정합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import generate_json
from utils.similarity import find_most_dissimilar, calculate_text_similarity_matrix
from utils.localization import get_cultural_context
import config


def generate_q_population(topic_info: dict) -> list[str]:
    """
    연구 주제를 바탕으로 Q-Population (100개 문항)을 생성합니다.
    
    Args:
        topic_info: 구조화된 연구 주제 정보
    
    Returns:
        100개의 Q-Population 문항 리스트
    """
    final_topic = topic_info.get("final_topic", "")
    research_question = topic_info.get("research_question", "")
    target_population = topic_info.get("target_population", "")
    context = topic_info.get("context", "")
    keywords = topic_info.get("keywords", [])
    language = topic_info.get("language", "ko")
    
    cul_ctx = get_cultural_context(language)
    
    prompt = f"""
Q방법론 연구를 위한 콘코스(Concourse) 문항을 생성합니다.

## 연구 정보 (★ 이 주제에 집중하여 문항 생성)
- 연구 주제: {final_topic}
- 연구 질문: {research_question}
- 대상 집단: {target_population}
- 연구 맥락: {context}
- 핵심 키워드: {', '.join(keywords)}

⚠️ **중요**: 위 연구 주제와 직접적으로 관련된 문항만 생성하세요.

{cul_ctx['statement_rules']}
반드시 **{cul_ctx['report_language']}** 언어로 생성해야 합니다.

---

## 🎯 3-Track 소스 시뮬레이션 (★ 핵심)

실제 Q방법론 연구에서 콘코스를 구성하는 3가지 출처를 시뮬레이션합니다.
각 Track의 언어적 특성을 반드시 반영하세요.

### Track 1: Naturalistic (심층 인터뷰형) - 80개 (40%)
**출처**: 1:1 인터뷰, FGI(포커스 그룹 인터뷰)에서 나온 '날것의 언어'
**언어적 특징**:
- 반드시 **1인칭(나, 저 / I, me)** 주어로 시작
- 구어체, 감정 표현 포함
- 개인적 경험과 감정이 녹아있음

### Track 2: Ready-made (문헌/전문가형) - 60개 (30%)
**출처**: 학술 논문, 신문 칼럼, 전문가 기고문, 정책 보고서
**언어적 특징**:
- 3인칭 또는 일반화된 표현
- 논리적, 분석적 어조
- 구조적/제도적 관점 포함

### Track 3: Realism (커뮤니티/SNS형) - 60개 (30%)
**출처**: 블라인드, 트위터, 커뮤니티 댓글, 온라인 게시판 (또는 Reddit, Twitter 등)
**언어적 특징**:
- 짧고 강렬한 문장
- 냉소적, 풍자적, 직설적 표현
- 숨겨진 본심, 불편한 진실
- 비격식체, 때로는 신조어 사용 가능

---

## ✍️ Q문항 작성 원칙 (Writing Principles)

### 원칙 1: 일물일어 (One Idea per Item)
한 문장에 **하나의 핵심 아이디어**만 담습니다.

### 원칙 2: 자기 참조적 (Self-Referent) ★중요★
응답자가 **'나의 이야기'**로 느낄 수 있도록 작성합니다.

### 원칙 3: 양극성 자극 (Polarity) ★중요★
뻔하거나 미지근한 문장은 금지입니다.
찬성/반대할 때 **감정적 동요가 일어날 수 있는 강한 표현**을 사용합니다.

### 원칙 4: 전문 용어 배제
연구 대상자가 이해할 수 있는 **일상적인 언어**로 작성합니다.

### 원칙 5: 입장별 균형 (편향 방지)
- 긍정적/찬성 입장 (~30%)
- 부정적/반대 입장 (~30%)
- 중립/양가 입장 (~40%)

---

## 📝 출력 형식 (JSON)
JSON 형식: {{"statements": ["문항1", "문항2", ..., "문항{config.Q_POPULATION_SIZE}"]}}
"""
    
    result = generate_json(prompt, system_prompt=cul_ctx["system_prompt"], temperature=0.8)
    statements = result.get("statements", result.get("items", []))
    if not statements and isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                statements = v
                break
    
    # 100개 미만이면 추가 생성
    retry_count = 0
    while len(statements) < config.Q_POPULATION_SIZE and retry_count < 5:
        retry_count += 1
        additional_prompt = f"""
기존에 생성된 {len(statements)}개의 문항에 추가로 {config.Q_POPULATION_SIZE - len(statements)}개의 문항을 더 생성해주세요.
주제: {final_topic}

기존 문항들과 중복되지 않는 새로운 관점의 문항을 생성합니다.

JSON 형식: {{"statements": ["추가문항1", ...]}}
"""
        additional = generate_json(additional_prompt, temperature=0.9)
        new_stmts = additional.get("statements", additional.get("items", []))
        if not new_stmts and isinstance(additional, dict):
            for v in additional.values():
                if isinstance(v, list):
                    new_stmts = v
                    break
        statements.extend(new_stmts)
    
    return statements[:config.Q_POPULATION_SIZE]


def filter_q_set(q_population: list[str], target_count: int = None) -> list[str]:
    """
    Q-Population에서 가장 차별적인 문항들을 선정하여 Q-Set을 구성합니다.
    
    Args:
        q_population: Q-Population 문항 리스트
        target_count: 선정할 문항 수 (기본값: config.Q_SET_SIZE)
    
    Returns:
        선정된 Q-Set 문항 리스트
    """
    if target_count is None:
        target_count = config.Q_SET_SIZE
    
    print(f"\n🔍 {len(q_population)}개 문항 중 {target_count}개 선정 중...")
    
    # 가장 다양한 문항들 선정
    selected_indices = find_most_dissimilar(q_population, target_count)
    
    q_set = [q_population[i] for i in selected_indices]
    
    print(f"✅ {len(q_set)}개 Q-Set 문항 선정 완료")
    
    return q_set


def validate_q_set(q_set: list[str], topic_info: dict) -> dict:
    """
    Q-Set의 품질을 검증합니다.
    
    Args:
        q_set: Q-Set 문항 리스트
        topic_info: 연구 주제 정보
    
    Returns:
        검증 결과
    """
    statements_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(q_set)])
    
    prompt = f"""
다음 Q-Set 문항들의 품질을 검증해주세요.

연구 주제: {topic_info.get('final_topic', '')}

Q-Set 문항들:
{statements_text}

다음 기준으로 평가해주세요:
1. 주제 관련성: 모든 문항이 연구 주제와 관련되어 있는가?
2. 다양성: 다양한 관점이 포함되어 있는가?
3. 균형성: 긍정/부정/중립 의견이 균형잡혀 있는가?
4. 명확성: 문항들이 이해하기 쉽고 명확한가?
5. 변별력: 참여자들 간 의견 차이를 드러낼 수 있는가?

JSON 형식으로 응답해주세요:
{{
    "overall_score": 1-10,
    "relevance_score": 1-10,
    "diversity_score": 1-10,
    "balance_score": 1-10,
    "clarity_score": 1-10,
    "discrimination_score": 1-10,
    "feedback": "전반적인 피드백",
    "suggestions": ["개선 제안1", ...]
}}
"""
    return generate_json(prompt)


def construct_q_set(topic_info: dict) -> tuple[list[str], list[str]]:
    """
    Q-Population을 생성하고 Q-Set을 선정하는 전체 프로세스를 수행합니다.
    
    Args:
        topic_info: 구조화된 연구 주제 정보
    
    Returns:
        (Q-Population, Q-Set)
    """
    print("\n" + "="*60)
    print("📝 Q-Population & Q-Set 생성")
    print("="*60)
    
    # Q-Population 생성
    print(f"\n💭 {config.Q_POPULATION_SIZE}개 Q-Population 문항 생성 중...")
    q_population = generate_q_population(topic_info)
    print(f"✅ {len(q_population)}개 문항 생성 완료")
    
    # Q-Set 선정
    q_set = filter_q_set(q_population, config.Q_SET_SIZE)
    
    # 검증
    print("\n🔬 Q-Set 품질 검증 중...")
    validation = validate_q_set(q_set, topic_info)
    print(f"📊 품질 점수: {validation.get('overall_score', 'N/A')}/10")
    
    if validation.get('feedback'):
        print(f"💬 피드백: {validation.get('feedback')}")
    
    return q_population, q_set


if __name__ == "__main__":
    # 테스트
    test_topic = {
        "final_topic": "MZ세대의 워라밸에 대한 인식",
        "research_question": "MZ세대는 일과 삶의 균형을 어떻게 인식하는가?",
        "target_population": "20-35세 직장인",
        "context": "한국 기업 환경",
        "keywords": ["워라밸", "MZ세대", "직장", "삶의 질"]
    }
    
    q_pop, q_set = construct_q_set(test_topic)
    
    print("\n\n===== Q-Set 문항 =====")
    for i, stmt in enumerate(q_set):
        print(f"{i+1}. {stmt}")
