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

## 연구 정보
- 연구 주제: {final_topic}
- 연구 질문: {research_question}
- 대상 집단: {target_population}
- 연구 맥락: {context}
- 핵심 키워드: {', '.join(keywords)}

---

## Q문항 생성 필수 규칙 (★ 엄격히 준수)

### 규칙 1: 주관적 진술문만 허용
- 사실 진술 ❌ → 태도/가치/판단이 드러나는 문장 ✅
- 피험자가 "동의/비동의" 정도를 표현할 수 있어야 함
- ❌ "조직에는 다양한 가치관이 존재한다" (사실)
- ✅ "다른 가치관을 가진 사람과 일하는 게 불편하다" (태도)

### 규칙 2: 하나의 핵심 아이디어만
- 이중/복합 진술 절대 금지
- ❌ "회사가 좋고 동료도 좋아서 다닐 만하다" (두 가지)
- ✅ "나는 동료들과의 관계가 좋다" (하나)

### 규칙 3: 간결하되 여지를 남김
- 15~35자 권장 (너무 길면 정렬 곤란)
- 피험자가 자신의 의미를 조금 투영할 여지
- ✅ "결국 돈이 제일 중요하다"
- ✅ "내 가치관은 타협할 수 없다"

### 규칙 4: 자연스러운 언어
- 전문용어 최소화, 일상적 표현 사용
- 대상 집단이 실제 사용하는 말투 반영

---

## 관점 분배 기준 ({config.Q_POPULATION_SIZE}개)

### 차원별 분배 (각 차원에서 고르게)
1. **제도/조직 차원** (~40개): 조직문화, 정책, 경영진, 제도
2. **개인/심리 차원** (~40개): 가치관, 정체성, 감정, 동기
3. **관계/사회 차원** (~40개): 동료, 상사, 팀, 소속감
4. **현실/타협 차원** (~40개): 생계, 현실적 선택, 자기합리화
5. **성장/미래 차원** (~40개): 이직, 커리어, 변화 기대

### 입장별 균형 (편향 방지)
- 긍정적 입장 (~30%): 조직/가치에 동조하는 표현
- 부정적 입장 (~30%): 조직/가치에 갈등을 느끼는 표현  
- 중립/양가 입장 (~40%): 상황에 따라 다르거나 혼합된 감정

### 이해관계자 관점 다양화
- 신입 직원, 경력 직원, 관리자, 전문가 등 다양한 위치
- 열정적/냉소적/방관적 등 다양한 태도

---

## 피해야 할 문항 유형 (★ 강력 금지)
❌ 사실 진술: "회사에는 다양한 부서가 있다"
❌ 이중 질문: "상사도 좋고 업무도 좋다"
❌ 모호한 표현: "그럭저럭 괜찮다", "별로다", "좋은 것 같다"
❌ 너무 긴 문장: (35자 초과)
❌ 전문 용어: "조직몰입도", "심리적 계약" 등
❌ 피상적 진술: "일하는 게 힘들다", "회사 다니기 싫다" (너무 일반적)
❌ 당연한 말: "월급은 중요하다", "일이 많으면 힘들다"

## 깊이 있는 문항의 특징 (★ 반드시 이 수준으로)
✅ **구체적 상황**: "야근 후 집에 가면서 '이게 맞나' 싶을 때가 많다"
✅ **내면의 갈등**: "회사 욕하면서도 떠나지 못하는 내가 한심하다"
✅ **양가감정**: "인정받고 싶지만, 그만큼 희생하고 싶진 않다"
✅ **숨겨진 본심**: "사실 동료들 성공하면 내심 불편하다"
✅ **날카로운 통찰**: "결국 윗사람한테 잘 보이는 게 실력보다 중요하다"
✅ **뼈아픈 현실**: "내 가치관? 월급 앞에선 다 타협하게 된다"

## 품질 체크리스트 (모든 문항이 아래 중 하나 이상 충족)
- [ ] 읽는 사람이 "아, 이런 생각 나도 했는데" 할 정도로 공감되는가?
- [ ] 쉽게 말 못하는 속마음을 건드리는가?
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
