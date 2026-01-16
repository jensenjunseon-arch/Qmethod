"""
Step 5: Statistical Analysis Module (통계적 분석)
Factor Analysis를 수행하여 유의미한 요인을 추출합니다.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
import config


def perform_pca_analysis(df: pd.DataFrame) -> dict:
    """
    PCA 분석을 수행하여 주요 요인을 식별합니다.
    
    Args:
        df: Q-Sorting 데이터 매트릭스 (참여자 x 문항)
    
    Returns:
        PCA 분석 결과
    """
    # 데이터 전치 (Q방법론에서는 참여자를 변수로, 문항을 관측치로 처리)
    data_transposed = df.T.values
    
    # ★ 중요: 상관행렬 기반 PCA를 위해 데이터 표준화 (Z-score)
    # 이렇게 해야 Eigenvalue가 변수 수 기준으로 계산됨 (합계 = 변수 수)
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_transposed)
    
    # PCA 수행 (모든 컴포넌트)
    pca = PCA()
    pca.fit(data_standardized)
    
    # ★ 상관행렬 기반 Eigenvalue 계산
    # 표준화된 데이터에서 explained_variance_는 상관행렬의 Eigenvalue와 동일
    eigenvalues = pca.explained_variance_
    n_factors = sum(1 for ev in eigenvalues if ev >= config.EIGENVALUE_THRESHOLD)
    
    print(f"[PCA] 데이터 shape: {data_transposed.shape}", flush=True)
    print(f"[PCA] 총 Eigenvalue 합계: {sum(eigenvalues):.2f} (변수 수와 동일해야 함)", flush=True)
    print(f"[PCA] Eigenvalue >= 1.0인 요인 수: {n_factors}", flush=True)
    
    return {
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_factors": max(n_factors, 2),  # 최소 2개 요인
        "components": pca.components_
    }


def perform_factor_analysis(
    df: pd.DataFrame,
    n_factors: int = None,
    rotation: str = "varimax"
) -> dict:
    """
    요인 분석을 수행합니다.
    scikit-learn 호환성을 위해 PCA 직접 사용
    
    Args:
        df: Q-Sorting 데이터 매트릭스
        n_factors: 추출할 요인 수 (None이면 PCA로 결정)
        rotation: 회전 방법 (varimax)
    
    Returns:
        요인 분석 결과
    """
    print("\n" + "="*60, flush=True)
    print("📈 통계적 분석 (Factor Analysis)", flush=True)
    print("="*60, flush=True)
    
    # 데이터 전치
    data_transposed = df.T.values
    
    # ★ 상관행렬 기반 분석을 위해 데이터 표준화
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_transposed)
    
    # PCA로 요인 수 결정
    if n_factors is None:
        pca_result = perform_pca_analysis(df)
        n_factors = pca_result["n_factors"]
        eigenvalues = pca_result["eigenvalues"]
        print(f"\n🔢 Eigenvalue > 1.0 기준 요인 수: {n_factors}", flush=True)
        print(f"   Eigenvalues: {[f'{ev:.2f}' for ev in eigenvalues[:n_factors+2]]}", flush=True)
    
    # PCA 직접 수행 (factor_analyzer 대신)
    try:
        # 먼저 factor_analyzer 시도 (표준화된 데이터 사용)
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='principal')
        fa.fit(data_standardized)  # ★ 표준화된 데이터 사용
        loadings = fa.loadings_
        variance = fa.get_factor_variance()
        ss_loadings = variance[0].tolist()
        proportion_var = variance[1].tolist()
        cumulative_var = variance[2].tolist()
    except Exception as e:
        print(f"⚠️ factor_analyzer 오류, PCA로 대체: {e}", flush=True)
        # PCA 직접 사용 (표준화된 데이터)
        pca = PCA(n_components=n_factors)
        pca.fit(data_standardized)  # ★ 표준화된 데이터 사용
        loadings = pca.components_.T  # Transpose to get (n_features, n_components)
        
        # Varimax 회전 수동 적용
        if rotation == "varimax":
            loadings = varimax_rotation(loadings)
        
        # 분산 계산 (상관행렬 기반 PCA에서는 총 분산 = 변수 수)
        n_vars = data_standardized.shape[1]  # 변수(참여자) 수
        # PCA의 explained_variance_ratio_ 사용
        proportion_var = list(pca.explained_variance_ratio_)
        ss_loadings = list(pca.explained_variance_)  # Eigenvalues
        cumulative_var = np.cumsum(proportion_var).tolist()
    
    # 각 참여자의 요인 점수 계산
    factor_scores = calculate_factor_scores(df, loadings)
    
    # 유의미한 적재량을 가진 참여자 식별
    significant_loadings = identify_significant_loadings(
        loadings, 
        df.index.tolist(),
        threshold=config.MIN_FACTOR_LOADING
    )
    
    print(f"\n📊 분석 결과:", flush=True)
    print(f"   추출된 요인 수: {n_factors}", flush=True)
    print(f"   총 설명 분산: {sum(proportion_var):.2%}", flush=True)
    
    for i in range(n_factors):
        print(f"   - Factor {i+1}: {proportion_var[i]:.2%} (SS Loading: {ss_loadings[i]:.2f})", flush=True)
    
    return {
        "n_factors": n_factors,
        "eigenvalues": eigenvalues,  # ★ 실제 Eigenvalue 추가
        "loadings": loadings,
        "loadings_df": pd.DataFrame(
            loadings,
            index=df.index,
            columns=[f"Factor{i+1}" for i in range(n_factors)]
        ),
        "variance": {
            "ss_loadings": ss_loadings,
            "proportion_var": proportion_var,
            "cumulative_var": cumulative_var
        },
        "factor_scores": factor_scores,
        "significant_loadings": significant_loadings
    }


def varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    """
    Varimax 회전 수동 구현
    """
    n_vars, n_factors = loadings.shape
    rotation_matrix = np.eye(n_factors)
    
    for _ in range(max_iter):
        old_rotation = rotation_matrix.copy()
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                # Varimax criterion
                x = loadings[:, i]
                y = loadings[:, j]
                
                u = x**2 - y**2
                v = 2 * x * y
                
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = np.sum(2 * u * v)
                
                num = D - 2 * A * B / n_vars
                den = C - (A**2 - B**2) / n_vars
                
                phi = 0.25 * np.arctan2(num, den)
                
                # Rotation
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                loadings[:, i] = x * cos_phi + y * sin_phi
                loadings[:, j] = -x * sin_phi + y * cos_phi
        
        # Check convergence
        if np.allclose(rotation_matrix, old_rotation, atol=tol):
            break
    
    return loadings



def calculate_factor_scores(df: pd.DataFrame, loadings: np.ndarray) -> pd.DataFrame:
    """
    각 문항의 요인별 Z-score를 계산합니다.
    
    Args:
        df: Q-Sorting 데이터 매트릭스
        loadings: 요인 적재량 매트릭스
    
    Returns:
        요인별 문항 Z-score DataFrame
    """
    n_factors = loadings.shape[1]
    n_items = df.shape[1]
    
    factor_scores = np.zeros((n_items, n_factors))
    
    for factor_idx in range(n_factors):
        # 해당 요인에 유의미하게 적재된 참여자들의 가중 평균
        factor_loadings = loadings[:, factor_idx]
        
        # 유의미한 적재량을 가진 참여자만 선택
        significant_mask = np.abs(factor_loadings) >= config.MIN_FACTOR_LOADING
        
        # ★ 유의미한 참여자가 없으면, 상위 3명이라도 사용
        if not np.any(significant_mask):
            # 적재량 절대값 기준 상위 3명 선택
            top_indices = np.argsort(np.abs(factor_loadings))[-3:]
            significant_mask = np.zeros(len(factor_loadings), dtype=bool)
            significant_mask[top_indices] = True
        
        weights = factor_loadings[significant_mask]
        weighted_data = df.iloc[significant_mask].values
        
        # 가중 평균 계산
        if len(weights) > 0 and np.sum(np.abs(weights)) > 0:
            weighted_sum = np.average(weighted_data, axis=0, weights=np.abs(weights))
            
            # Z-score 변환 (std=0 방지)
            std_val = np.std(weighted_sum)
            if std_val > 0:
                factor_scores[:, factor_idx] = (weighted_sum - np.mean(weighted_sum)) / std_val
            else:
                # 표준편차 0이면 모든 값이 동일 → 중앙화만
                factor_scores[:, factor_idx] = weighted_sum - np.mean(weighted_sum)
    
    return pd.DataFrame(
        factor_scores,
        index=[f"Q{i+1}" for i in range(n_items)],
        columns=[f"Factor{i+1}" for i in range(n_factors)]
    )


def identify_significant_loadings(
    loadings: np.ndarray,
    participant_names: list[str],
    threshold: float = 0.4
) -> dict:
    """
    유의미한 요인 적재량을 가진 참여자를 식별합니다.
    
    Args:
        loadings: 요인 적재량 매트릭스
        participant_names: 참여자 이름 리스트
        threshold: 유의미한 적재량 임계값
    
    Returns:
        요인별 유의미한 참여자 딕셔너리
    """
    n_factors = loadings.shape[1]
    result = {}
    
    for factor_idx in range(n_factors):
        factor_name = f"Factor{factor_idx + 1}"
        factor_loadings = loadings[:, factor_idx]
        
        significant = []
        for i, (name, loading) in enumerate(zip(participant_names, factor_loadings)):
            if abs(loading) >= threshold:
                significant.append({
                    "name": name,
                    "loading": loading,
                    "direction": "positive" if loading > 0 else "negative"
                })
        
        # 적재량 크기로 정렬
        significant.sort(key=lambda x: abs(x["loading"]), reverse=True)
        result[factor_name] = significant
    
    return result


def identify_consensus_statements(
    factor_scores: pd.DataFrame,
    q_set: list[str],
    threshold: float = 0.5
) -> list[dict]:
    """
    합의 문항(Consensus Statements)을 식별합니다.
    모든 Factor에서 비슷한 Z-score를 받은 문항들입니다.
    
    Args:
        factor_scores: 요인별 문항 Z-score DataFrame
        q_set: Q-Set 문항 리스트
        threshold: Z-score 차이 임계값 (이 이하면 합의로 판단)
    
    Returns:
        합의 문항 리스트
    """
    consensus = []
    n_factors = len(factor_scores.columns)
    
    for idx in factor_scores.index:
        item_num = int(idx.replace("Q", "")) - 1
        scores = factor_scores.loc[idx].values
        
        # 모든 Factor 간 Z-score 차이 계산
        max_diff = max(scores) - min(scores)
        avg_score = np.mean(scores)
        
        # 차이가 임계값 이하면 합의 문항
        if max_diff <= threshold:
            consensus.append({
                "item_number": item_num + 1,
                "statement": q_set[item_num] if item_num < len(q_set) else f"Q{item_num+1}",
                "avg_z_score": float(avg_score),
                "max_difference": float(max_diff),
                "factor_scores": {col: float(factor_scores.loc[idx, col]) for col in factor_scores.columns}
            })
    
    # 평균 Z-score 절대값으로 정렬 (강한 합의가 먼저)
    consensus.sort(key=lambda x: abs(x["avg_z_score"]), reverse=True)
    
    return consensus


def identify_distinguishing_statements(
    factor_scores: pd.DataFrame,
    q_set: list[str],
    threshold: float = 1.0
) -> dict:
    """
    구분 문항(Distinguishing Statements)을 식별합니다.
    특정 Factor에서만 높거나 낮은 Z-score를 보이는 문항들입니다.
    
    Args:
        factor_scores: 요인별 문항 Z-score DataFrame
        q_set: Q-Set 문항 리스트
        threshold: 다른 Factor와의 Z-score 차이 임계값
    
    Returns:
        Factor별 구분 문항 딕셔너리
    """
    distinguishing = {}
    
    for col in factor_scores.columns:
        other_cols = [c for c in factor_scores.columns if c != col]
        dist_items = []
        
        for idx in factor_scores.index:
            item_num = int(idx.replace("Q", "")) - 1
            this_score = factor_scores.loc[idx, col]
            other_scores = [factor_scores.loc[idx, c] for c in other_cols]
            
            # 다른 모든 Factor보다 현저히 높거나 낮은 경우
            min_diff = min([abs(this_score - other) for other in other_scores])
            
            if min_diff >= threshold:
                dist_items.append({
                    "item_number": item_num + 1,
                    "statement": q_set[item_num] if item_num < len(q_set) else f"Q{item_num+1}",
                    "z_score": float(this_score),
                    "min_diff_from_others": float(min_diff),
                    "direction": "high" if this_score > 0 else "low"
                })
        
        # Z-score 차이로 정렬
        dist_items.sort(key=lambda x: x["min_diff_from_others"], reverse=True)
        distinguishing[col] = dist_items
    
    return distinguishing


def get_factor_interpretation_data(
    factor_scores: pd.DataFrame,
    q_set: list[str],
    top_n: int = 5
) -> dict:
    """
    각 요인 해석을 위한 데이터를 준비합니다.
    
    Args:
        factor_scores: 요인별 문항 Z-score
        q_set: Q-Set 문항 리스트
        top_n: 상위/하위 문항 수
    
    Returns:
        요인별 해석 데이터
    """
    result = {}
    
    for col in factor_scores.columns:
        factor_scores_sorted = factor_scores[col].sort_values(ascending=False)
        
        # 상위 문항 (가장 동의)
        top_items = []
        for idx in factor_scores_sorted.head(top_n).index:
            item_num = int(idx.replace("Q", "")) - 1
            top_items.append({
                "item_number": item_num + 1,
                "statement": q_set[item_num],
                "z_score": factor_scores_sorted[idx]
            })
        
        # 하위 문항 (가장 비동의)
        bottom_items = []
        for idx in factor_scores_sorted.tail(top_n).index:
            item_num = int(idx.replace("Q", "")) - 1
            bottom_items.append({
                "item_number": item_num + 1,
                "statement": q_set[item_num],
                "z_score": factor_scores_sorted[idx]
            })
        
        result[col] = {
            "top_items": top_items,
            "bottom_items": bottom_items[::-1],  # 가장 낮은 것부터
            "mean_score": factor_scores[col].mean(),
            "std_score": factor_scores[col].std()
        }
    
    return result


if __name__ == "__main__":
    # 테스트용 더미 데이터
    np.random.seed(42)
    dummy_data = pd.DataFrame(
        np.random.randint(-5, 6, size=(20, 60)),
        index=[f"P{i+1}" for i in range(20)],
        columns=[f"Q{i+1}" for i in range(60)]
    )
    
    result = perform_factor_analysis(dummy_data)
    print("\n요인 적재량:")
    print(result["loadings_df"])
