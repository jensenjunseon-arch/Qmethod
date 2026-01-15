"""
LLM API Wrapper for Q-Methodology application
Supports both OpenAI and Google Gemini APIs
"""
import json
import time
from typing import Optional
import config

# Provider detection
def get_provider() -> str:
    """현재 사용할 LLM 프로바이더를 결정합니다."""
    if config.LLM_PROVIDER == "openai":
        return "openai"
    elif config.LLM_PROVIDER == "gemini":
        return "gemini"
    else:  # auto-detect
        if config.GOOGLE_API_KEY:
            return "gemini"
        elif config.OPENAI_API_KEY:
            return "openai"
        else:
            raise ValueError("API 키가 설정되지 않았습니다. OPENAI_API_KEY 또는 GOOGLE_API_KEY를 설정하세요.")


# ============== OpenAI ==============
def get_openai_client():
    """OpenAI 클라이언트 인스턴스를 반환합니다."""
    from openai import OpenAI
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def generate_text_openai(prompt: str, system_prompt: str, temperature: float, max_retries: int) -> str:
    client = get_openai_client()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI] 오류 (시도 {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAI API 호출 실패: {e}")


def generate_json_openai(prompt: str, system_prompt: str, temperature: float, max_retries: int) -> dict:
    client = get_openai_client()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[OpenAI] JSON 파싱 오류 (시도 {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                continue
            raise RuntimeError(f"JSON 파싱 실패: {e}")
        except Exception as e:
            print(f"[OpenAI] API 오류 (시도 {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAI API 호출 실패: {e}")


def generate_embedding_openai(text: str) -> list[float]:
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ============== Google Gemini ==============
def get_gemini_client():
    """Gemini 클라이언트를 초기화합니다."""
    import google.generativeai as genai
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    return genai


def generate_text_gemini(prompt: str, system_prompt: str, temperature: float, max_retries: int) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=system_prompt
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] 오류 (시도 {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini API 호출 실패: {e}")


def generate_json_gemini(prompt: str, system_prompt: str, temperature: float, max_retries: int) -> dict:
    genai = get_gemini_client()
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=system_prompt + "\n\n반드시 유효한 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요."
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json"
                )
            )
            content = response.text.strip()
            # JSON 블록 추출 (```json ... ``` 형태인 경우)
            if content.startswith("```"):
                lines = content.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```json"):
                        in_json = True
                        continue
                    elif line.startswith("```"):
                        in_json = False
                        continue
                    if in_json:
                        json_lines.append(line)
                content = "\n".join(json_lines)
            result = json.loads(content)
            # Handle case where Gemini returns a list instead of dict
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    result = result[0]  # Take first element if it's a list of dicts
                else:
                    result = {"items": result}  # Wrap in dict
            return result
        except json.JSONDecodeError as e:
            print(f"[Gemini] JSON 파싱 오류 (시도 {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                continue
            raise RuntimeError(f"JSON 파싱 실패: {e}")
        except Exception as e:
            print(f"[Gemini] API 오류 (시도 {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini API 호출 실패: {e}")


def generate_embedding_gemini(text: str) -> list[float]:
    """Gemini 임베딩 생성"""
    genai = get_gemini_client()
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result['embedding']


# ============== Unified Interface ==============
def generate_text(
    prompt: str,
    system_prompt: str = "당신은 Q방법론 연구를 돕는 전문 연구 보조원입니다. 모든 응답은 한국어로 합니다.",
    max_retries: int = 3,
    temperature: float = 0.7,
) -> str:
    """텍스트를 생성합니다."""
    provider = get_provider()
    print(f"[LLM] 텍스트 생성 시작... (프로바이더: {provider})", flush=True)
    
    if provider == "openai":
        result = generate_text_openai(prompt, system_prompt, temperature, max_retries)
    else:
        result = generate_text_gemini(prompt, system_prompt, temperature, max_retries)
    
    print(f"[LLM] 텍스트 생성 완료 ({len(result)} chars)", flush=True)
    return result


def generate_json(
    prompt: str,
    system_prompt: str = "당신은 Q방법론 연구를 돕는 전문 연구 보조원입니다. 모든 응답은 JSON 형식으로 합니다.",
    max_retries: int = 3,
    temperature: float = 0.7,
) -> dict:
    """JSON 응답을 생성하고 파싱합니다."""
    provider = get_provider()
    print(f"[LLM] JSON 생성 시작... (프로바이더: {provider})", flush=True)
    
    if provider == "openai":
        result = generate_json_openai(prompt, system_prompt, temperature, max_retries)
    else:
        result = generate_json_gemini(prompt, system_prompt, temperature, max_retries)
    
    print(f"[LLM] JSON 생성 완료", flush=True)
    return result


def generate_embedding(text: str) -> list[float]:
    """텍스트의 임베딩 벡터를 생성합니다."""
    provider = get_provider()
    
    if provider == "openai":
        return generate_embedding_openai(text)
    else:
        return generate_embedding_gemini(text)
