"""
Step 1: Topic Refinement Module (주제 구체화)
사용자로부터 연구 주제를 입력받고 명확히 구조화합니다.
"""
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import generate_text, generate_json
from utils.localization import get_cultural_context
import config


def ask_clarifying_question(topic: str, iteration: int, previous_context: str = "") -> dict:
    """
    주제를 명확화하기 위한 후속 질문을 생성합니다.
    
    Args:
        topic: 현재 연구 주제
        iteration: 현재 반복 횟수
        previous_context: 이전 대화 맥락
    
    Returns:
        {"question": 질문, "aspect": 질문이 다루는 측면}
    """
    prompt = f"""
다음 연구 주제를 더 구체화하기 위한 질문을 생성해주세요.

현재 연구 주제: {topic}

이전 대화 맥락:
{previous_context if previous_context else "없음"}

반복 횟수: {iteration}/{config.MAX_TOPIC_REFINEMENT_ITERATIONS}

질문을 생성할 때 다음 측면들을 고려하세요:
1. 왜 이 연구가 필요한가? (연구의 필요성)
2. 구체적으로 어떤 대상을 타겟하는가? (연구 대상)
3. 어떤 맥락/상황에서의 연구인가? (연구 맥락)
4. 기대하는 결과는 무엇인가? (연구 목표)

JSON 형식으로 응답해주세요:
{{"question": "질문 내용", "aspect": "이 질문이 다루는 측면"}}
"""
    return generate_json(prompt)


def evaluate_topic_clarity(topic: str, context: str = "") -> dict:
    """
    주제의 명확성을 평가합니다.
    
    Args:
        topic: 연구 주제
        context: 추가 맥락
    
    Returns:
        {"is_clear": bool, "score": 1-10, "missing_aspects": [], "refined_topic": str}
    """
    prompt = f"""
다음 Q방법론 연구 주제의 명확성을 평가해주세요.

연구 주제: {topic}

추가 맥락:
{context if context else "없음"}

다음 기준으로 평가해주세요:
1. 연구 대상이 명확한가?
2. 연구 맥락/상황이 구체적인가?
3. 연구 목적이 분명한가?
4. Q방법론에 적합한 주제인가? (주관성 탐구에 적합한지)

JSON 형식으로 응답해주세요:
{{
    "is_clear": true/false (충분히 명확한지),
    "score": 1-10 (명확성 점수),
    "missing_aspects": ["부족한 측면1", ...],
    "refined_topic": "명확화된 주제 (한 문장)"
}}
"""
    return generate_json(prompt)


def structure_final_topic(topic: str, context: str, language: str = 'ko') -> dict:
    """
    최종 연구 주제를 구조화합니다.
    """
    cul_ctx = get_cultural_context(language)
    
    prompt = f"""
다음 정보를 바탕으로 Q방법론 연구의 최종 주제를 구조화해주세요.

연구 주제: {topic}

대화 맥락:
{context}

JSON 형식으로 다음 정보를 포함해주세요:
{{
    "final_topic": "최종 확정된 연구 주제 (한 문장)",
    "research_question": "핵심 연구 질문",
    "target_population": "연구 대상 집단 (텍스트 설명)",
    "context": "연구 맥락/상황",
    "expected_outcomes": "기대하는 결과/통찰",
    "keywords": ["핵심", "키워드", "목록"],
    "demographic_constraints": {{
        "age_min": 연구 대상의 최소 연령 (숫자, 제약 없으면 null),
        "age_max": 연구 대상의 최대 연령 (숫자, 제약 없으면 null),
        "gender": "연구 대상 성별 (남성/여성/null - 제약 없으면 null)",
        "occupation_types": ["직업군1", "직업군2"] 또는 null (제약 없으면 null),
        "other_requirements": ["기타 필수조건1", "기타 필수조건2"] 또는 []
    }}
}}

demographic_constraints 작성 시 주의사항:
- "20대" → age_min: 20, age_max: 29
- "MZ세대" → age_min: 20, age_max: 44 (1980-2005년생 기준)
- "직장인" → occupation_types: ["사무직", "전문직", "기술직", "서비스직", "관리직"]
- "대학생" → occupation_types: ["대학생", "대학원생"]
- 성별 언급이 없으면 gender: null
"""
    result = generate_json(prompt, system_prompt=cul_ctx["system_prompt"])
    result['language'] = language
    return result


def refine_topic_interactive() -> dict:
    """
    대화형으로 연구 주제를 구체화합니다.
    
    Returns:
        구조화된 최종 연구 주제
    """
    print("\n" + "="*60)
    print("📚 Q방법론 연구 주제 구체화")
    print("="*60 + "\n")
    
    # 초기 주제 입력
    topic = input("연구하고 싶은 주제를 입력해주세요: ").strip()
    
    if not topic:
        raise ValueError("연구 주제가 입력되지 않았습니다.")
    
    conversation_context = f"초기 주제: {topic}\n"
    
    # 주제 명확성 평가
    evaluation = evaluate_topic_clarity(topic)
    
    iteration = 0
    while not evaluation.get("is_clear", False) and iteration < config.MAX_TOPIC_REFINEMENT_ITERATIONS:
        iteration += 1
        
        # 후속 질문 생성
        question_data = ask_clarifying_question(topic, iteration, conversation_context)
        question = question_data.get("question", "이 연구가 왜 필요한지 설명해주시겠어요?")
        
        print(f"\n💡 추가 질문 ({iteration}/{config.MAX_TOPIC_REFINEMENT_ITERATIONS}):")
        print(f"   {question}")
        
        answer = input("\n답변: ").strip()
        
        if answer:
            conversation_context += f"\n질문 {iteration}: {question}\n답변: {answer}\n"
            
            # 다시 평가
            evaluation = evaluate_topic_clarity(topic, conversation_context)
            
            if evaluation.get("refined_topic"):
                topic = evaluation["refined_topic"]
                print(f"\n✨ 주제 업데이트: {topic}")
    
    # 최종 구조화
    final_topic = structure_final_topic(topic, conversation_context)
    
    print("\n" + "="*60)
    print("✅ 최종 연구 주제 확정")
    print("="*60)
    print(f"\n📌 {final_topic.get('final_topic', topic)}")
    print(f"❓ 연구 질문: {final_topic.get('research_question', 'N/A')}")
    print(f"👥 대상: {final_topic.get('target_population', 'N/A')}")
    print(f"🎯 맥락: {final_topic.get('context', 'N/A')}")
    
    return final_topic


def refine_topic_from_string(initial_topic: str, language: str = 'ko') -> dict:
    """
    주어진 주제 문자열로부터 직접 주제를 구조화합니다. (비대화형)
    
    Args:
        initial_topic: 초기 연구 주제
    
    Returns:
        구조화된 연구 주제
    """
    evaluation = evaluate_topic_clarity(initial_topic)
    return structure_final_topic(initial_topic, f"초기 주제: {initial_topic}", language)


if __name__ == "__main__":
    # 테스트
    result = refine_topic_interactive()
    print("\n최종 결과:")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
