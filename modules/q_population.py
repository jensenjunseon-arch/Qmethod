"""
Step 2: Q-Population & Q-Set Construction Module (문항 생성 및 선정)
연구 주제를 바탕으로 Q-Population을 생성하고 Q-Set을 선정합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import generate_json
from utils.similarity import find_most_dissimilar, calculate_text_similarity_matrix
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
    
    prompt = f"""
Q방법론 연구를 위한 콘코스(Concourse) 문항을 생성합니다.

## 연구 정보 (★ 이 주제에 집중하여 문항 생성)
- 연구 주제: {final_topic}
- 연구 질문: {research_question}
- 대상 집단: {target_population}
- 연구 맥락: {context}
- 핵심 키워드: {', '.join(keywords)}

⚠️ **중요**: 위 연구 주제와 직접적으로 관련된 문항만 생성하세요. 
연구 주제와 관련 없는 일반적인 직장/회사 관련 문항은 생성하지 마세요.

---

## Q문항 생성 필수 규칙 (★ 엄격히 준수)

### 규칙 1: 주관적 진술문만 허용
- 사실 진술 ❌ → 태도/가치/판단이 드러나는 문장 ✅
- 피험자가 "동의/비동의" 정도를 표현할 수 있어야 함
- 연구 주제에 대한 개인의 태도, 신념, 가치관을 반영

### 규칙 2: 하나의 핵심 아이디어만
- 이중/복합 진술 절대 금지
- 한 문장에 하나의 명확한 의견/태도만 포함

### 규칙 3: 간결하되 여지를 남김
- 15~40자 권장
- 피험자가 자신의 의미를 조금 투영할 여지

### 규칙 4: 자연스러운 언어
- 전문용어 최소화, 일상적 표현 사용
- 대상 집단({target_population})이 실제 사용하는 말투 반영

---

## 관점 분배 기준 ({config.Q_POPULATION_SIZE}개)

### 연구 주제 "{final_topic}"에 맞는 다양한 차원에서 문항 생성:

1. **인지/판단 차원** (~{config.Q_POPULATION_SIZE // 5}개): 이 주제에 대한 사실 인식, 원인 분석, 판단
2. **감정/태도 차원** (~{config.Q_POPULATION_SIZE // 5}개): 이 주제에 대한 감정적 반응, 기본 태도
3. **가치/신념 차원** (~{config.Q_POPULATION_SIZE // 5}개): 이 주제와 관련된 핵심 가치관, 신념
4. **행동/의도 차원** (~{config.Q_POPULATION_SIZE // 5}개): 이 주제 관련 행동 의향, 대응 방식
5. **사회/맥락 차원** (~{config.Q_POPULATION_SIZE // 5}개): 사회적 영향, 외부 요인에 대한 인식

### 입장별 균형 (편향 방지)
- 긍정적/찬성 입장 (~30%): 주제에 대해 긍정적이거나 낙관적인 표현
- 부정적/반대 입장 (~30%): 주제에 대해 부정적이거나 비판적인 표현  
- 중립/양가 입장 (~40%): 상황에 따라 다르거나 혼합된 입장

### 관점 다양화
- 대상 집단 내 다양한 하위 집단의 시각
- 다양한 이해관계자의 입장 반영

---

## 피해야 할 문항 유형 (★ 강력 금지)
❌ 사실 진술: 객관적 사실만 나열한 문장
❌ 이중 질문: 두 가지 이상의 내용을 담은 문장
❌ 모호한 표현: "그럭저럭", "별로다", "좋은 것 같다" 등
❌ 너무 긴 문장: (40자 초과)
❌ 전문 용어: 대상 집단이 모를 수 있는 학술 용어
❌ 피상적 진술: 너무 일반적이고 당연한 말
❌ 연구 주제와 무관한 내용

## 깊이 있는 문항의 특징 (★ 반드시 이 수준으로)
✅ **구체적 상황**: 연구 주제와 관련된 구체적 맥락에서의 생각
✅ **내면의 갈등**: 이 주제에 대해 느끼는 복잡한 감정이나 딜레마
✅ **양가감정**: 찬성과 반대 사이에서 느끼는 갈등
✅ **숨겨진 본심**: 공개적으로 말하기 어려운 솔직한 생각
✅ **날카로운 통찰**: 이 주제의 본질을 파고드는 의견
✅ **현실적 고민**: 이상과 현실 사이의 괴리

## 품질 체크리스트
- [ ] "{final_topic}"에 직접적으로 관련된 문항인가?
- [ ] 읽는 사람이 공감할 수 있는가?
- [ ] 특정 입장을 가진 사람은 강하게 동의/비동의할 수 있는가?
- [ ] 연구자에게 인사이트를 줄 수 있을 만큼 구체적인가?

---

JSON 형식: {{"statements": ["문항1", "문항2", ..., "문항{config.Q_POPULATION_SIZE}"]}}
"""
    
    result = generate_json(prompt, temperature=0.8)
    statements = result.get("statements", [])
    
    # 100개 미만이면 추가 생성
    while len(statements) < config.Q_POPULATION_SIZE:
        additional_prompt = f"""
기존에 생성된 {len(statements)}개의 문항에 추가로 {config.Q_POPULATION_SIZE - len(statements)}개의 문항을 더 생성해주세요.
주제: {final_topic}

기존 문항들과 중복되지 않는 새로운 관점의 문항을 생성합니다.

JSON 형식: {{"statements": ["추가문항1", ...]}}
"""
        additional = generate_json(additional_prompt, temperature=0.9)
        statements.extend(additional.get("statements", []))
    
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
