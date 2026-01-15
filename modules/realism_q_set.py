"""
Realism Q-Set Generator
Expansion (200+) → Reduction (60) → Blind Shuffle
"""
import random
from typing import List, Dict, Tuple, Optional
from utils.llm_client import generate_json
from utils.similarity import compute_tfidf_matrix, find_most_dissimilar_items


def generate_raw_statements(topic: str, group: str, count: int = 200) -> List[Dict]:
    """
    Phase 1: Expansion - 생존 본능, 독성 사고, 모순에 초점을 맞춘 200+ 문항 생성
    """
    prompt = f"""당신은 '{topic}' 분야에서 '{group}'의 심리를 분석하는 Q방법론 전문가입니다.

다음 카테고리별로 총 {count}개의 날것의, 필터링되지 않은 문장을 생성하세요:

1. **Survival (생존 본능)** - 40개: 직업적 생존, 경쟁, 불안정성에 대한 문장
2. **Toxic Thoughts (독성 사고)** - 40개: 숨기고 싶지만 진짜 느끼는 부정적 생각
3. **Contradictions (실제 모순)** - 40개: 말과 행동이 다른 상황, 이중성
4. **Values (가치관)** - 40개: 핵심 가치, 신념, 동기
5. **Fears (두려움)** - 40개: 직업적/심리적 불안 요소

각 문장은:
- 1인칭 시점 ("나는...")
- 구체적이고 솔직한 표현
- 논쟁적이거나 도발적일 수 있음
- 상반되는 문장 쌍 포함 (contradiction_pair 필드로 연결)

JSON 형식:
{{
  "statements": [
    {{
      "id": "S001",
      "text": "나는 마감에 쫓기면 창작이 아니라 생존이라고 느낀다",
      "category": "Survival",
      "group": "{group}",
      "contradiction_pair": null,
      "intensity": "high"
    }},
    {{
      "id": "S002", 
      "text": "나는 마감 압박이 오히려 창작의 원동력이 된다",
      "category": "Contradictions",
      "group": "{group}",
      "contradiction_pair": "S001",
      "intensity": "high"
    }}
  ]
}}

주제: {topic}
대상 집단: {group}
"""
    
    result = generate_json(prompt)
    statements = result.get("statements", [])
    
    # 그룹 필드 확인
    for stmt in statements:
        stmt["group"] = group
    
    print(f"[Q-SET] {group}: {len(statements)}개 원시 문항 생성", flush=True)
    return statements


def reduce_to_final_set(
    statements: List[Dict], 
    target_count: int = 60
) -> List[Dict]:
    """
    Phase 2: Reduction - NLP 기반 60개 필터링
    TF-IDF를 사용하여 가장 변별력 있는 문항 선정
    """
    if len(statements) <= target_count:
        return statements
    
    # TF-IDF 행렬 계산
    texts = [s["text"] for s in statements]
    tfidf_matrix = compute_tfidf_matrix(texts)
    
    # 가장 변별력 있는 문항 선정
    selected_indices = find_most_dissimilar_items(tfidf_matrix, target_count)
    
    # 상충 쌍 보존 확인
    selected = [statements[i] for i in selected_indices]
    selected_ids = {s["id"] for s in selected}
    
    # 상충 쌍이 있는 경우 쌍도 포함
    for stmt in list(selected):
        pair_id = stmt.get("contradiction_pair")
        if pair_id and pair_id not in selected_ids:
            pair_stmt = next((s for s in statements if s["id"] == pair_id), None)
            if pair_stmt and len(selected) < target_count + 5:
                selected.append(pair_stmt)
                selected_ids.add(pair_id)
    
    # 최종 개수 맞추기
    final = selected[:target_count]
    
    print(f"[Q-SET] {len(statements)}개 → {len(final)}개로 축소", flush=True)
    return final


def blind_shuffle(statements: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Phase 3: Blind Shuffle - 카테고리 태그 제거 및 무작위 섞기
    
    Returns:
        shuffled_statements: 셔플된 문항 (카테고리 숨김)
        category_map: 원본 ID → 카테고리 매핑 (분석용)
    """
    # 카테고리 매핑 저장
    category_map = {s["id"]: s.get("category", "Unknown") for s in statements}
    
    # 깊은 복사 후 카테고리 제거
    shuffled = []
    for s in statements:
        blind_stmt = {
            "id": s["id"],
            "text": s["text"],
            "group": s.get("group", "unknown"),
            # category는 제거
        }
        shuffled.append(blind_stmt)
    
    # 무작위 섞기
    random.shuffle(shuffled)
    
    # 새로운 표시 순서 부여
    for i, stmt in enumerate(shuffled, 1):
        stmt["display_order"] = i
    
    print(f"[Q-SET] Blind Shuffle 완료: {len(shuffled)}개 문항", flush=True)
    return shuffled, category_map


def get_contradiction_pairs(statements: List[Dict]) -> List[Tuple[str, str]]:
    """
    Mirror Test용 상충 쌍 추출
    """
    pairs = []
    seen = set()
    
    for stmt in statements:
        pair_id = stmt.get("contradiction_pair")
        if pair_id:
            pair_key = tuple(sorted([stmt["id"], pair_id]))
            if pair_key not in seen:
                pairs.append((stmt["id"], pair_id))
                seen.add(pair_key)
    
    print(f"[Q-SET] 상충 쌍 {len(pairs)}개 추출", flush=True)
    return pairs


def generate_q_set(
    topic: str, 
    group: str, 
    expansion_count: int = 200,
    final_count: int = 60
) -> Tuple[List[Dict], Dict[str, str], List[Tuple[str, str]]]:
    """
    전체 Q-Set 생성 파이프라인
    
    Returns:
        q_set: 최종 셔플된 Q-Set
        category_map: ID → 카테고리 매핑
        contradiction_pairs: 상충 쌍 리스트
    """
    print(f"\n[Q-SET] === {group} Q-Set 생성 시작 ===", flush=True)
    
    # Phase 1: Expansion
    raw_statements = generate_raw_statements(topic, group, expansion_count)
    
    # Phase 2: Reduction
    reduced = reduce_to_final_set(raw_statements, final_count)
    
    # 상충 쌍 추출 (Reduction 후)
    contradiction_pairs = get_contradiction_pairs(reduced)
    
    # Phase 3: Blind Shuffle
    q_set, category_map = blind_shuffle(reduced)
    
    print(f"[Q-SET] === {group} Q-Set 생성 완료 ===\n", flush=True)
    
    return q_set, category_map, contradiction_pairs


if __name__ == "__main__":
    # 테스트
    q_set, categories, pairs = generate_q_set("웹툰 창작", "작가", 20, 10)
    print(f"Q-Set: {len(q_set)}개")
    print(f"Categories: {len(categories)}")
    print(f"Contradiction Pairs: {pairs}")
