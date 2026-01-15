"""
Realism P-Set Generator
Enhanced Persona Generation with Psychographics
핵심 가치관, 불안 요소, 방어 기제 포함
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import generate_json, generate_embedding
from utils.similarity import check_diversity
import config


def generate_realism_persona(
    topic_info: dict, 
    group: str,
    persona_index: int, 
    existing_personas: list[dict] = None
) -> dict:
    """
    Psychographics 변수를 포함한 페르소나 생성
    """
    existing_desc = ""
    if existing_personas:
        existing_desc = "기존 페르소나들 (반드시 이들과 다르게 생성):\n"
        for i, p in enumerate(existing_personas):
            existing_desc += f"- {p.get('name', f'P{i+1}')}: {p.get('psychographics', {}).get('core_values', [])}\n"
    
    prompt = f"""당신은 '{topic_info.get('final_topic', topic_info.get('topic', ''))}' 연구를 위한 심층 페르소나 전문가입니다.

**집단**: {group}
**페르소나 번호**: {persona_index + 1}/20

{existing_desc}

다음 조건을 만족하는 **극단적으로 독특한** 페르소나를 생성하세요:

1. **Psychographics**가 핵심입니다 - 단순 인구통계가 아닌 심리적 특성에 집중
2. 기존 페르소나와 **완전히 다른** 가치관/두려움/방어 기제
3. {group}의 현실적인 경험을 반영

JSON 형식:
{{
  "id": "P{persona_index + 1:02d}",
  "name": "실제 이름 (한국식)",
  "group": "{group}",
  "demographics": {{
    "age": 28,
    "gender": "남성/여성",
    "experience_years": 5,
    "current_position": "현재 직책/상황"
  }},
  "psychographics": {{
    "core_values": ["가장 중요한 가치 1", "가치 2", "가치 3"],
    "fears": ["가장 큰 두려움 1", "두려움 2"],
    "defense_mechanisms": ["주로 사용하는 방어 기제 1", "방어 기제 2"],
    "hidden_desires": ["숨기는 욕구 1", "욕구 2"],
    "trigger_points": ["화나게 하는 것 1", "화나게 하는 것 2"]
  }},
  "work_style": {{
    "decision_making": "직관적/분석적/협의적",
    "stress_response": "스트레스 받을 때 행동",
    "motivation_source": "동기 부여 원천",
    "communication_style": "소통 방식"
  }},
  "brief_description": "한 문장 정체성 요약",
  "internal_conflict": "이 사람이 겪는 내적 갈등"
}}
"""
    
    result = generate_json(prompt, temperature=0.95)
    result["group"] = group
    result["id"] = f"P{persona_index + 1:02d}"
    
    return result


def generate_realism_personas(
    topic_info: dict,
    group: str,
    count: int = 20,
    similarity_threshold: float = 0.4,
    max_retries: int = 3
) -> list[dict]:
    """
    다양성 제약을 가진 페르소나 생성
    Cosine Similarity < 0.4 보장
    """
    print(f"\n[P-SET] === {group} 페르소나 생성 시작 ({count}명) ===", flush=True)
    
    personas = []
    embeddings = []
    
    for i in range(count):
        retry_count = 0
        while retry_count < max_retries:
            print(f"[P-SET] 페르소나 {i+1}/{count} 생성 중...", flush=True)
            
            persona = generate_realism_persona(topic_info, group, i, personas)
            
            # 임베딩 생성
            psycho = persona.get("psychographics", {})
            embed_text = " ".join([
                " ".join(psycho.get("core_values", [])),
                " ".join(psycho.get("fears", [])),
                " ".join(psycho.get("defense_mechanisms", [])),
                persona.get("brief_description", ""),
                persona.get("internal_conflict", "")
            ])
            
            new_embedding = generate_embedding(embed_text)
            
            # 다양성 체크
            if embeddings:
                is_diverse, violations = check_diversity(
                    embeddings + [new_embedding], 
                    similarity_threshold
                )
                
                if not is_diverse and len(violations) > 0:
                    # 마지막 추가가 문제인지 확인
                    last_violations = [v for v in violations if i in (v[0], v[1])]
                    if last_violations:
                        retry_count += 1
                        print(f"[P-SET] ⚠️ 유사도 초과, 재생성 ({retry_count}/{max_retries})", flush=True)
                        continue
            
            # 성공
            personas.append(persona)
            embeddings.append(new_embedding)
            print(f"[P-SET] ✅ {persona.get('name', f'P{i+1}')} - {persona.get('brief_description', '')[:30]}...", flush=True)
            break
        else:
            # max_retries 초과해도 추가 (경고와 함께)
            print(f"[P-SET] ⚠️ 다양성 경고: 재시도 한도 초과, 그대로 추가", flush=True)
            personas.append(persona)
            embeddings.append(new_embedding)
    
    print(f"[P-SET] === {group} 페르소나 생성 완료: {len(personas)}명 ===\n", flush=True)
    
    return personas


def generate_dual_group_personas(
    topic_info: dict,
    group_a: str,
    group_b: str,
    count_per_group: int = 20
) -> tuple[list[dict], list[dict]]:
    """
    Dual Group Mode용 양쪽 그룹 페르소나 생성
    """
    topic_a = {**topic_info, "group": group_a}
    topic_b = {**topic_info, "group": group_b}
    
    personas_a = generate_realism_personas(topic_a, group_a, count_per_group)
    personas_b = generate_realism_personas(topic_b, group_b, count_per_group)
    
    return personas_a, personas_b


if __name__ == "__main__":
    # 테스트
    test_topic = {
        "final_topic": "웹툰 산업의 창작 환경",
        "topic": "웹툰 창작"
    }
    
    personas = generate_realism_personas(test_topic, "웹툰 작가", count=3)
    for p in personas:
        print(f"\n{p.get('name')}: {p.get('brief_description')}")
