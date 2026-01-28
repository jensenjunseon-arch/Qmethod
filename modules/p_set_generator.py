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
import random


def generate_demographic_slots(constraints: dict, count: int = 20) -> list[dict]:
    """
    제약조건 내에서 인구통계 슬롯을 균등 분배로 생성합니다.
    
    Args:
        constraints: demographic_constraints 딕셔너리
            - age_min, age_max: 연령 범위
            - gender: 성별 제약 (None이면 남/여 균등 분배)
            - occupation_types: 직업군 리스트 (None이면 자유)
        count: 생성할 슬롯 수
    
    Returns:
        각 페르소나에 할당할 인구통계 슬롯 리스트
    """
    slots = []
    
    # 연령 범위 설정 (기본값: 20-60세)
    age_min = constraints.get('age_min') or 20
    age_max = constraints.get('age_max') or 60
    
    # 연령 구간 생성 (가능한 균등 분배)
    age_range = age_max - age_min + 1
    if age_range >= 10:
        # 5세 단위로 구간 나누기
        age_bins = []
        for start in range(age_min, age_max + 1, 5):
            end = min(start + 4, age_max)
            age_bins.append((start, end))
    else:
        # 범위가 좁으면 1세 단위
        age_bins = [(age, age) for age in range(age_min, age_max + 1)]
    
    # 성별 설정
    gender_constraint = constraints.get('gender')
    if gender_constraint:
        genders = [gender_constraint] * count
    else:
        # 남녀 균등 분배
        genders = ['남성'] * (count // 2) + ['여성'] * (count - count // 2)
        random.shuffle(genders)
    
    # 직업군 설정
    occupation_types = constraints.get('occupation_types')
    if not occupation_types:
        occupation_types = None  # 자유 (프롬프트에서 다양하게)
    
    # 슬롯 생성
    for i in range(count):
        # 연령 구간 라운드로빈 할당
        age_bin = age_bins[i % len(age_bins)]
        
        slot = {
            'index': i + 1,
            'age_range': f"{age_bin[0]}-{age_bin[1]}세" if age_bin[0] != age_bin[1] else f"{age_bin[0]}세",
            'age_min': age_bin[0],
            'age_max': age_bin[1],
            'gender': genders[i],
        }
        
        # 직업군 라운드로빈 할당 (있는 경우)
        if occupation_types:
            slot['occupation_hint'] = occupation_types[i % len(occupation_types)]
        else:
            slot['occupation_hint'] = None
        
        slots.append(slot)
    
    # 슬롯 섞기 (같은 특성이 연속되지 않도록)
    random.shuffle(slots)
    
    # 인덱스 재할당
    for i, slot in enumerate(slots):
        slot['index'] = i + 1
    
    return slots


def validate_persona_constraints(persona: dict, slot: dict, constraints: dict) -> tuple[bool, str]:
    """
    생성된 페르소나가 제약조건을 준수하는지 검증합니다.
    
    Returns:
        (준수 여부, 위반 사유)
    """
    issues = []
    
    # 연령 검증
    age = persona.get('age', 0)
    if age < slot['age_min'] or age > slot['age_max']:
        issues.append(f"연령 불일치: {age}세 (기대: {slot['age_range']})")
    
    # 성별 검증
    gender_constraint = constraints.get('gender')
    if gender_constraint and persona.get('gender') != gender_constraint:
        issues.append(f"성별 불일치: {persona.get('gender')} (기대: {gender_constraint})")
    
    # 직업군 검증 (힌트가 있는 경우)
    # 직업은 유연하게 처리 (경고만)
    
    return (len(issues) == 0, "; ".join(issues) if issues else "")


def generate_single_persona(topic_info: dict, persona_index: int, existing_personas: list[dict] = None, demographic_slot: dict = None) -> dict:
    """
    단일 페르소나를 생성합니다.
    
    Args:
        topic_info: 연구 주제 정보
        persona_index: 페르소나 인덱스
        existing_personas: 기존 생성된 페르소나들 (다양성 확보를 위해)
        demographic_slot: 할당된 인구통계 슬롯 (연령, 성별, 직업 힌트)
    
    Returns:
        생성된 페르소나 정보
    """
    existing_desc = ""
    if existing_personas:
        existing_desc = "기존 페르소나들:\n"
        for i, p in enumerate(existing_personas):
            existing_desc += f"- {p.get('name', f'페르소나{i+1}')}: {p.get('brief_description', '')}\n"
    
    # 인구통계 제약 프롬프트 생성
    demographic_instruction = ""
    if demographic_slot:
        demographic_instruction = f"""
⚠️ 인구통계 필수 조건 (반드시 준수):
- 연령: {demographic_slot['age_range']} 범위 내에서 선택
- 성별: {demographic_slot['gender']}
"""
        if demographic_slot.get('occupation_hint'):
            demographic_instruction += f"- 직업군 힌트: {demographic_slot['occupation_hint']} (이 분야 관련 직업 선택)\n"
    
    prompt = f"""
Q방법론 연구를 위한 가상 참여자 페르소나를 생성해주세요.

연구 주제: {topic_info.get('final_topic', '')}
대상 집단: {topic_info.get('target_population', '')}
연구 맥락: {topic_info.get('context', '')}

페르소나 번호: {persona_index + 1}/{config.P_SET_SIZE}
{demographic_instruction}
{existing_desc}

다음 조건을 충족하는 새로운 페르소나를 생성해주세요:
1. 위의 인구통계 필수 조건을 반드시 따라야 합니다.
2. 기존 페르소나들과 명확하게 다른 성격, 배경, 가치관을 가져야 합니다.
3. 연구 주제에 대해 독특하고 일관된 관점을 가져야 합니다.
4. 현실적이고 구체적인 배경 스토리가 있어야 합니다.

JSON 형식으로 응답해주세요:
{{
    "name": "이름 (가상)",
    "age": 나이 (숫자, 인구통계 필수 조건 범위 내),
    "gender": "성별 (인구통계 필수 조건과 일치)",
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
        topic_info: 연구 주제 정보 (demographic_constraints 포함)
        max_retries: 다양성 미달 시 최대 재시도 횟수
    
    Returns:
        페르소나 리스트
    """
    print("\n" + "="*60)
    print("👥 P-Set (참여자 페르소나) 생성")
    print("="*60)
    
    # 인구통계 제약 추출 및 슬롯 생성
    constraints = topic_info.get('demographic_constraints', {}) or {}
    slots = generate_demographic_slots(constraints, config.P_SET_SIZE)
    
    # 슬롯 분포 출력
    print(f"\n📊 인구통계 슬롯 분포:")
    age_ranges = {}
    gender_counts = {}
    for slot in slots:
        ar = slot['age_range']
        age_ranges[ar] = age_ranges.get(ar, 0) + 1
        g = slot['gender']
        gender_counts[g] = gender_counts.get(g, 0) + 1
    
    print(f"   연령: {', '.join([f'{k}({v}명)' for k, v in sorted(age_ranges.items())])}")
    print(f"   성별: {', '.join([f'{k}({v}명)' for k, v in gender_counts.items()])}")
    if constraints.get('occupation_types'):
        print(f"   직업군: {', '.join(constraints['occupation_types'])}")
    
    personas = []
    
    for i in range(config.P_SET_SIZE):
        slot = slots[i]
        print(f"\n🧑 페르소나 {i+1}/{config.P_SET_SIZE} 생성 중... [{slot['age_range']}, {slot['gender']}]")
        
        persona = generate_single_persona(topic_info, i, personas, demographic_slot=slot)
        
        # 제약 준수 검증
        is_valid, issues = validate_persona_constraints(persona, slot, constraints)
        if not is_valid:
            print(f"   ⚠️ 제약 위반 감지: {issues}")
            # 재시도 (최대 2회)
            for retry in range(2):
                print(f"   🔄 재생성 시도 {retry + 1}...")
                persona = generate_single_persona(topic_info, i, personas, demographic_slot=slot)
                is_valid, issues = validate_persona_constraints(persona, slot, constraints)
                if is_valid:
                    break
            if not is_valid:
                print(f"   ⚠️ 재시도 후에도 제약 위반. 계속 진행합니다.")
        
        personas.append(persona)
        print(f"   ✅ {persona.get('name', f'페르소나{i+1}')} ({persona.get('age')}세 {persona.get('gender')}) - {persona.get('brief_description', '')[:40]}...")
    
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
                        [p for i, p in enumerate(personas) if i != idx_to_replace],
                        demographic_slot=slots[idx_to_replace]
                    )
                    personas[idx_to_replace] = new_persona
                    print(f"   ✅ {new_persona.get('name', f'페르소나{idx_to_replace+1}')} - {new_persona.get('brief_description', '')[:40]}...")
    except Exception as e:
        print(f"⚠️  다양성 검증 건너뜀 (임베딩 에러): {str(e)[:100]}")
        # 다양성 검증 실패해도 20명의 페르소나는 정상 반환
    
    # 최종 인구통계 분포 출력
    print("\n📊 최종 P-Set 인구통계 분포:")
    final_ages = {}
    final_genders = {}
    for p in personas:
        age = p.get('age', 0)
        age_group = f"{(age // 10) * 10}대"
        final_ages[age_group] = final_ages.get(age_group, 0) + 1
        g = p.get('gender', '미상')
        final_genders[g] = final_genders.get(g, 0) + 1
    
    print(f"   연령: {', '.join([f'{k}({v}명)' for k, v in sorted(final_ages.items())])}")
    print(f"   성별: {', '.join([f'{k}({v}명)' for k, v in final_genders.items()])}")
    
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
