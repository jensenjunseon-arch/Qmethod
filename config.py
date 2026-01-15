"""
Q-Methodology Research Insight Generator
Configuration settings
"""
import os

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"  # 사용할 모델

# Google Gemini API Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"  # 사용할 모델

# LLM Provider Selection: "openai" or "gemini" (auto-detect if not set)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto")

# Q-Methodology Configuration
Q_POPULATION_SIZE = 200  # Q-Population 문항 수 (200개 생성 후 60개 선별)
Q_SET_SIZE = 60  # 최종 Q-Set 문항 수
P_SET_SIZE = 20  # 페르소나 수
MAX_TOPIC_REFINEMENT_ITERATIONS = 3  # 주제 구체화 최대 반복 횟수
PERSONA_SIMILARITY_THRESHOLD = 0.4  # 페르소나 유사도 임계값

# Forced Distribution for Q-Sorting (-5 to +5)
# 정규분포 형태의 강제 분포
FORCED_DISTRIBUTION = {
    -5: 2,
    -4: 3,
    -3: 5,
    -2: 7,
    -1: 8,
    0: 10,
    1: 8,
    2: 7,
    3: 5,
    4: 3,
    5: 2
}

# Factor Analysis Configuration
EIGENVALUE_THRESHOLD = 1.0  # Eigenvalue 임계값
MIN_FACTOR_LOADING = 0.4  # 최소 요인 적재량

# Output Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
