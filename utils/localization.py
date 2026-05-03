"""
Deep Cultural Localization Utility
"""
from typing import Dict

def get_cultural_context(lang: str) -> Dict[str, str]:
    """
    선택된 언어에 따른 깊은 문화적 맥락(Cultural Context) 프롬프트를 반환합니다.
    """
    if lang == 'en':
        return {
            "system_prompt": "You are an expert in Q-methodology and a cultural psychologist. ALL your responses MUST be in rich, fluent English. Your analysis must deeply reflect Western societal norms, cultural nuances, individualism, and diversity.",
            "persona_rules": "Create personas with typical Western names (e.g., Emily, Michael). Their demographics, occupations, and core values must be deeply rooted in Western culture and societal structures.",
            "statement_rules": "Generate Q-statements that resonate strongly with Western emotional and cultural sensibilities. Use idiomatic English and reflect Western worldviews.",
            "report_language": "English",
            "terms": {
                "core_values": "Core Values",
                "survival_instinct": "Survival Instinct",
                "defense_mechanism": "Defense Mechanism",
                "hidden_fear": "Hidden Fear",
                "self_justification": "Self-Justification",
                "trigger_phrases": "Trigger Phrases",
                "action_plan": "Action Plan",
                "consensus": "Universal Consensus (Z-score Agreement)",
                "type": "Type",
                "psychology_profile": "Psychological Profile",
                "positive": "Positive",
                "negative": "Negative",
                "bipolar": "Bipolar Factor",
                "unipolar": "Unipolar Factor",
                "agree": "Strongly Agree",
                "disagree": "Strongly Disagree",
                "neutral": "Neutral"
            }
        }
    else:
        return {
            "system_prompt": "당신은 Q방법론 전문가이자 문화 심리학자입니다. 모든 응답은 자연스럽고 풍부한 한국어로 작성하세요. 한국 사회의 특수성(정서, 눈치, 체면, 집단주의, 한국식 워라밸, 사회적 압력 등)과 한국인 고유의 심리를 깊이 있게 반영해야 합니다.",
            "persona_rules": "전형적인 한국 이름(예: 김지수, 박민준)을 사용하세요. 한국의 사회적 계층, 세대 특성(MZ세대, 86세대 등), 한국적 직장 문화가 짙게 밴 페르소나를 생성하세요.",
            "statement_rules": "한국인들이 일상적으로 공감하고 느끼는 정서, 한국 사회 특유의 갈등과 가치관이 담긴 Q-문항을 생성하세요. 번역투가 아닌 자연스러운 한국어 구어체를 사용하세요.",
            "report_language": "Korean",
            "terms": {
                "core_values": "핵심 가치",
                "survival_instinct": "생존 본능",
                "defense_mechanism": "방어 기제",
                "hidden_fear": "숨겨진 두려움",
                "self_justification": "자기 정당화",
                "trigger_phrases": "트리거 표현",
                "action_plan": "행동 지침",
                "consensus": "보편적 합의 (Z-score 일치 문항)",
                "type": "유형",
                "psychology_profile": "심층 심리 프로파일",
                "positive": "긍정",
                "negative": "부정",
                "bipolar": "양극화 요인",
                "unipolar": "단일 요인",
                "agree": "동의",
                "disagree": "비동의",
                "neutral": "중립"
            }
        }
