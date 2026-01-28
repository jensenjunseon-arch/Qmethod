"""
Step 3: P-Set Generation Module (참여자 페르소나 생성)
연구 주제와 관련된 가상 참여자 페르소나를 생성합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import generate_json, generate_embedding
from utils.similarity import check_diversity, calculate_embedding_similarity_matrix
import config
import numpy as np


def generate_single_persona(topic_info: dict, persona_index: int, existing_personas: list[dict] = None) -> dict:
    """
    단일 페르소나를 생성합니다.
    
    Args:
        topic_info: 연구 주제 정보
        persona_index: 페르소나 인덱스
        existing_personas: 기존 생성된 페르소나들 (다양성 확보를 위해)
    
    Returns:
        생성된 페르소나 정보
    """
    existing_desc = ""
    if existing_personas:
        existing_desc = "기존 페르소나들:\n"
        for i, p in enumerate(existing_personas):
            existing_desc += f"- {p.get('name', f'페르소나{i+1}')}: {p.get('brief_description', '')}\n"
    
    prompt = f"""
Q방법론 연구를 위한 가상 참여자 페르소나를 생성해주세요.

연구 주제: {topic_info.get('final_topic', '')}
대상 집단: {topic_info.get('target_population', '')}
연구 맥락: {topic_info.get('context', '')}

페르소나 번호: {persona_index + 1}/{config.P_SET_SIZE}

{existing_desc}

다음 조건을 충족하는 새로운 페르소나를 생성해주세요:
1. 기존 페르소나들과 명확하게 다른 성격, 배경, 가치관을 가져야 합니다.
2. 연구 주제에 대해 독특하고 일관된 관점을 가져야 합니다.
3. 현실적이고 구체적인 배경 스토리가 있어야 합니다.

JSON 형식으로 응답해주세요:
{{
    "name": "이름 (가상)",
    "age": 나이,
    "gender": "성별",
    "occupation": "직업",
    "education": "학력",
    "personality_traits": ["성격특성1", "성격특성2", "성격특성3"],
    "values": ["핵심가치1", "핵심가치2"],
    "life_experiences": ["주요경험1", "주요경험2"],
    "attitude_toward_topic": "연구 주제에 대한 기본 태도 (상세 설명)",
    "brief_description": "한 문장 요약",
    "decision_making_style": "의사결정 스타일",
    "social_orientation": "사회적 성향 (개인주의/집단주의 등)"
}}
"""
    return generate_json(prompt, temperature=0.9)


def generate_all_personas(topic_info: dict, max_retries: int = 3) -> list[dict]:
    """
    모든 페르소나를 생성하고 다양성을 검증합니다.
    
    Args:
        topic_info: 연구 주제 정보
        max_retries: 다양성 미달 시 최대 재시도 횟수
    
    Returns:
        페르소나 리스트
    """
    print("\n" + "="*60)
    print("👥 P-Set (참여자 페르소나) 생성")
    print("="*60)
    
    personas = []
    
    for i in range(config.P_SET_SIZE):
        print(f"\n🧑 페르소나 {i+1}/{config.P_SET_SIZE} 생성 중...")
        persona = generate_single_persona(topic_info, i, personas)
        personas.append(persona)
        print(f"   ✅ {persona.get('name', f'페르소나{i+1}')} - {persona.get('brief_description', '')[:40]}...")
    
    # 다양성 검증 (임베딩 실패 시 건너뜀)
    print("\n🔍 페르소나 다양성 검증 중...")
    
    try:
        for retry in range(max_retries):
            embeddings = []
            for p in personas:
                # 페르소나 설명을 임베딩
                desc = f"{p.get('personality_traits', [])} {p.get('values', [])} {p.get('attitude_toward_topic', '')}"
                embeddings.append(generate_embedding(desc))
            
            is_diverse, violations = check_diversity(embeddings, config.PERSONA_SIMILARITY_THRESHOLD)
            
            if is_diverse:
                print(f"✅ 다양성 검증 통과! (모든 페르소나 쌍의 유사도 < {config.PERSONA_SIMILARITY_THRESHOLD})")
                break
            else:
                print(f"⚠️  다양성 미달: {len(violations)}개 쌍이 임계값 초과")
                
                if retry < max_retries - 1:
                    # 가장 유사한 쌍 중 하나 재생성
                    violations.sort(key=lambda x: x[2], reverse=True)
                    idx_to_replace = violations[0][1]  # 두 번째 인덱스 교체
                    
                    print(f"   🔄 페르소나 {idx_to_replace + 1} 재생성 중...")
                    new_persona = generate_single_persona(
                        topic_info, 
                        idx_to_replace, 
                        [p for i, p in enumerate(personas) if i != idx_to_replace]
                    )
                    personas[idx_to_replace] = new_persona
                    print(f"   ✅ {new_persona.get('name', f'페르소나{idx_to_replace+1}')} - {new_persona.get('brief_description', '')[:40]}...")
    except Exception as e:
        print(f"⚠️  다양성 검증 건너뜀 (임베딩 에러): {str(e)[:100]}")
        # 다양성 검증 실패해도 20명의 페르소나는 정상 반환
    
    return personas


def describe_personas(personas: list[dict]) -> str:
    """
    페르소나들의 요약 설명을 생성합니다.
    
    Args:
        personas: 페르소나 리스트
    
    Returns:
        요약 설명 문자열
    """
    summary = []
    for i, p in enumerate(personas):
        summary.append(f"""
### 페르소나 {i+1}: {p.get('name', 'N/A')}
- **나이/성별**: {p.get('age', 'N/A')}세 / {p.get('gender', 'N/A')}
- **직업**: {p.get('occupation', 'N/A')}
- **성격**: {', '.join(p.get('personality_traits', []))}
- **가치관**: {', '.join(p.get('values', []))}
- **주제 태도**: {p.get('attitude_toward_topic', 'N/A')[:100]}...
""")
    return "\n".join(summary)


if __name__ == "__main__":
    # 테스트
    test_topic = {
        "final_topic": "MZ세대의 워라밸에 대한 인식",
        "research_question": "MZ세대는 일과 삶의 균형을 어떻게 인식하는가?",
        "target_population": "20-35세 직장인",
        "context": "한국 기업 환경",
        "keywords": ["워라밸", "MZ세대", "직장", "삶의 질"]
    }
    
    personas = generate_all_personas(test_topic)
    print("\n\n" + describe_personas(personas))
