"""
Similarity calculation utilities for Q-Methodology application
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    두 벡터 간의 코사인 유사도를 계산합니다.
    
    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터
    
    Returns:
        코사인 유사도 (-1.0 ~ 1.0)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def calculate_text_similarity_matrix(texts: list[str]) -> np.ndarray:
    """
    TF-IDF 기반으로 텍스트 리스트의 유사도 매트릭스를 계산합니다.
    
    Args:
        texts: 텍스트 리스트
    
    Returns:
        유사도 매트릭스 (n x n)
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)


def compute_tfidf_matrix(texts: list[str]) -> np.ndarray:
    """
    텍스트 리스트의 TF-IDF 행렬을 계산합니다.
    
    Args:
        texts: 텍스트 리스트
    
    Returns:
        TF-IDF 행렬
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()


def find_most_dissimilar_items(
    tfidf_matrix: np.ndarray,
    target_count: int
) -> list[int]:
    """
    TF-IDF 행렬에서 가장 변별력 있는(서로 다른) 항목들의 인덱스를 찾습니다.
    
    Args:
        tfidf_matrix: TF-IDF 행렬
        target_count: 선택할 항목 수
    
    Returns:
        선택된 항목들의 인덱스 리스트
    """
    similarity_matrix = cosine_similarity(tfidf_matrix)
    n = len(tfidf_matrix)
    selected = [0]  # 첫 번째 항목으로 시작
    
    while len(selected) < target_count and len(selected) < n:
        min_max_similarity = float('inf')
        best_candidate = None
        
        for i in range(n):
            if i in selected:
                continue
            
            # 이미 선택된 항목들과의 최대 유사도 계산
            max_sim_to_selected = max(similarity_matrix[i][j] for j in selected)
            
            # 최대 유사도가 가장 낮은 후보 선택
            if max_sim_to_selected < min_max_similarity:
                min_max_similarity = max_sim_to_selected
                best_candidate = i
        
        if best_candidate is not None:
            selected.append(best_candidate)
        else:
            break
    
    return selected


def find_most_dissimilar(
    texts: list[str],
    target_count: int,
    existing_indices: Optional[list[int]] = None
) -> list[int]:
    """
    주어진 텍스트 중 가장 서로 다른(비유사한) 항목들의 인덱스를 찾습니다.
    Greedy selection 알고리즘을 사용합니다.
    
    Args:
        texts: 텍스트 리스트
        target_count: 선택할 항목 수
        existing_indices: 이미 선택된 인덱스들
    
    Returns:
        선택된 항목들의 인덱스 리스트
    """
    similarity_matrix = calculate_text_similarity_matrix(texts)
    n = len(texts)
    selected = list(existing_indices) if existing_indices else []
    
    # 첫 번째 항목이 없으면 무작위 선택
    if not selected:
        selected.append(0)
    
    while len(selected) < target_count and len(selected) < n:
        min_max_similarity = float('inf')
        best_candidate = None
        
        for i in range(n):
            if i in selected:
                continue
            
            # 이미 선택된 항목들과의 최대 유사도 계산
            max_sim_to_selected = max(similarity_matrix[i][j] for j in selected)
            
            # 최대 유사도가 가장 낮은 후보 선택
            if max_sim_to_selected < min_max_similarity:
                min_max_similarity = max_sim_to_selected
                best_candidate = i
        
        if best_candidate is not None:
            selected.append(best_candidate)
        else:
            break
    
    return selected


def calculate_embedding_similarity_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """
    임베딩 벡터들의 유사도 매트릭스를 계산합니다.
    
    Args:
        embeddings: 임베딩 벡터 리스트
    
    Returns:
        유사도 매트릭스 (n x n)
    """
    embeddings_array = np.array(embeddings)
    return cosine_similarity(embeddings_array)


def check_diversity(embeddings: list[list[float]], threshold: float = 0.4) -> tuple[bool, list[tuple[int, int, float]]]:
    """
    임베딩들의 다양성을 검증합니다.
    
    Args:
        embeddings: 임베딩 벡터 리스트
        threshold: 유사도 임계값 (이 값 미만이어야 다양성 충족)
    
    Returns:
        (다양성 충족 여부, 임계값 초과 쌍 리스트)
    """
    similarity_matrix = calculate_embedding_similarity_matrix(embeddings)
    n = len(embeddings)
    violations = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                violations.append((i, j, similarity_matrix[i][j]))
    
    return len(violations) == 0, violations
