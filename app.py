#!/usr/bin/env python3
"""
Q-Methodology Research Insight Generator - Web Application
Flask 기반 웹 인터페이스
"""
import sys
import os
from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경변수 로드
import json
import threading
import uuid
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.topic_refiner import refine_topic_from_string, evaluate_topic_clarity, ask_clarifying_question, structure_final_topic
from modules.q_population import construct_q_set
from modules.p_set_generator import generate_all_personas
from modules.q_sorting import simulate_all_sortings
from modules.factor_analysis import perform_factor_analysis
from modules.dual_type_generator import generate_dual_types
from modules.report_generator import generate_report, save_data_artifacts

app = Flask(__name__, template_folder='templates', static_folder='static')

# 세션별 진행 상태 저장 (TTL 기반 정리)
sessions = {}
SESSION_TTL_SECONDS = 1800  # 30분


def _cleanup_expired_sessions():
    """만료된 세션을 메모리에서 제거합니다."""
    now = time.time()
    expired = [sid for sid, s in sessions.items()
               if now - s.get('created_at', now) > SESSION_TTL_SECONDS]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"[CLEANUP] {len(expired)}개 만료 세션 제거 (남은 세션: {len(sessions)}개)", flush=True)


def _get_factor_scores_summary(factor_scores_df, q_set, top_n=5):
    """각 요인별 상위/하위 Z-score 문항 요약"""
    if factor_scores_df is None:
        return {}
    
    summary = {}
    import pandas as pd
    
    for col in factor_scores_df.columns:
        scores = factor_scores_df[col].sort_values(ascending=False)
        
        # 상위 5개 (가장 동의)
        top_items = []
        for idx in scores.head(top_n).index:
            item_num = int(idx.replace("Q", "")) - 1
            if item_num < len(q_set):
                top_items.append({
                    'q_num': item_num + 1,
                    'text': q_set[item_num][:50] + '...' if len(q_set[item_num]) > 50 else q_set[item_num],
                    'z_score': round(float(scores[idx]), 2)
                })
        
        # 하위 5개 (가장 비동의)
        bottom_items = []
        for idx in scores.tail(top_n).index:
            item_num = int(idx.replace("Q", "")) - 1
            if item_num < len(q_set):
                bottom_items.append({
                    'q_num': item_num + 1,
                    'text': q_set[item_num][:50] + '...' if len(q_set[item_num]) > 50 else q_set[item_num],
                    'z_score': round(float(scores[idx]), 2)
                })
        
        summary[col] = {
            'top_agree': top_items,
            'top_disagree': bottom_items[::-1]
        }
    
    return summary


def _get_consensus_statements(factor_scores_df, q_set, threshold=0.5):
    """합의 문항 식별 - 모든 Factor에서 비슷한 Z-score를 받은 문항"""
    if factor_scores_df is None:
        return []
    
    import numpy as np
    consensus = []
    
    for idx in factor_scores_df.index:
        item_num = int(idx.replace("Q", "")) - 1
        scores = factor_scores_df.loc[idx].values
        
        max_diff = max(scores) - min(scores)
        avg_score = np.mean(scores)
        
        if max_diff <= threshold:
            consensus.append({
                'item_number': item_num + 1,
                'statement': q_set[item_num] if item_num < len(q_set) else f"Q{item_num+1}",
                'avg_z_score': round(float(avg_score), 2),
                'max_difference': round(float(max_diff), 2),
                'interpretation': '동의' if avg_score > 0.3 else ('비동의' if avg_score < -0.3 else '중립')
            })
    
    consensus.sort(key=lambda x: abs(x['avg_z_score']), reverse=True)
    return consensus[:10]  # 상위 10개만


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_analysis():
    """분석 시작"""
    data = request.json
    topic = data.get('topic', '')
    api_key = data.get('api_key', '').strip()
    
    if not topic:
        return jsonify({'error': '연구 주제를 입력해주세요.'}), 400
    
    # If no API key provided in request, check server configuration
    if not api_key:
        if config.OPENAI_API_KEY:
            api_key = config.OPENAI_API_KEY
        elif config.GOOGLE_API_KEY:
            api_key = config.GOOGLE_API_KEY
    
    if not api_key:
        return jsonify({'error': '서버에 API Key가 설정되어 있지 않습니다. 관리자에게 문의하세요.'}), 400
    
    # 만료된 세션 정리
    _cleanup_expired_sessions()
    
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'status': 'started',
        'topic': topic,
        'api_key': api_key,
        'progress': 0,
        'current_step': '시작',
        'logs': [],
        'result': None,
        'created_at': time.time()
    }
    
    # 백그라운드에서 분석 실행
    print(f"\n[API] 분석 시작 요청 - 세션: {session_id}", flush=True)
    print(f"[API] 스레드 생성 중...", flush=True)
    thread = threading.Thread(target=run_analysis_background, args=(session_id, topic, api_key), daemon=True)
    thread.start()
    print(f"[API] 스레드 시작됨 - Thread ID: {thread.ident}", flush=True)
    
    return jsonify({'session_id': session_id})


def run_analysis_background(session_id: str, topic: str, api_key: str):
    """백그라운드에서 분석 실행"""
    import sys
    print(f"\n[THREAD] 백그라운드 스레드 시작: {session_id}", flush=True)
    print(f"[THREAD] 주제: {topic[:50]}...", flush=True)
    print(f"[THREAD] API Key: {'서버 설정 사용' if not api_key else api_key[:10] + '...'}", flush=True)
    sys.stdout.flush()
    
    # API 키가 직접 전달된 경우에만 config 업데이트 (서버 시작 시 .env에서 이미 로드됨)
    # 주의: --workers 1 환경에서만 안전. 멀티 워커 시 thread-local 사용 필요
    if api_key and api_key != config.GOOGLE_API_KEY and api_key != config.OPENAI_API_KEY:
        if api_key.startswith("AIza"):
            config.GOOGLE_API_KEY = api_key
            config.LLM_PROVIDER = "gemini"
        elif api_key.startswith("sk-"):
            config.OPENAI_API_KEY = api_key
            config.LLM_PROVIDER = "openai"
    
    provider = config.LLM_PROVIDER if config.LLM_PROVIDER != "auto" else ("gemini" if config.GOOGLE_API_KEY else "openai")
    print(f"[THREAD] LLM Provider: {provider}", flush=True)
    
    session = sessions[session_id]
    
    try:
        # Step 1: 주제 구체화
        print("[THREAD] Step 1 시작: 주제 구체화")
        update_session(session_id, 1, "주제 구체화 중...", 5)
        topic_info = refine_topic_from_string(topic)
        update_session(session_id, 1, f"주제 확정: {topic_info.get('final_topic', topic)}", 15)
        print(f"[THREAD] Step 1 완료: {topic_info.get('final_topic', topic)}")
        
        # Step 2: Q-Set 구성
        update_session(session_id, 2, "Q-Population 생성 중 (200개 문항)...", 20)
        q_population, q_set = construct_q_set(topic_info)
        update_session(session_id, 2, f"Q-Set 선정 완료: {len(q_set)}개 문항", 30)
        
        # Step 3: 페르소나 생성
        update_session(session_id, 3, "가상 참여자 페르소나 생성 중...", 35)
        personas = generate_all_personas(topic_info)
        update_session(session_id, 3, f"페르소나 생성 완료: {len(personas)}명", 45)
        
        # Step 4: Q-Sorting
        update_session(session_id, 4, "Q-Sorting 시뮬레이션 중...", 50)
        sorting_matrix = simulate_all_sortings(personas, q_set, topic_info)
        update_session(session_id, 4, f"Q-Sorting 완료: {sorting_matrix.shape}", 60)
        
        # Step 5: 요인 분석
        update_session(session_id, 5, "통계 분석 (Factor Analysis) 중...", 65)
        factor_result = perform_factor_analysis(sorting_matrix)
        update_session(session_id, 5, f"요인 분석 완료: {factor_result['n_factors']}개 요인", 75)
        
        # Step 6: 유형 이원화
        update_session(session_id, 6, "유형 이원화 생성 중...", 80)
        types = generate_dual_types(
            factor_result['factor_scores'],
            q_set,
            topic_info,
            factor_result['significant_loadings']
        )
        update_session(session_id, 6, f"유형 생성 완료: {len(types)}개 유형", 90)
        
        # Step 7: 리포트 생성
        update_session(session_id, 7, "리포트 생성 중...", 95)
        report_path = generate_report(
            topic_info, q_set, personas, sorting_matrix, factor_result, types
        )
        
        # 결과 저장 - 상세 데이터 포함
        session['result'] = {
            'topic_info': topic_info,
            # Q-Set 문항 (id, text)
            'q_set': [{'id': i+1, 'text': q} for i, q in enumerate(q_set)],
            # 페르소나 상세 정보
            'personas': [{
                'id': i+1,
                'name': p.get('name', f'P{i+1}'),
                'age': p.get('age', 'N/A'),
                'gender': p.get('gender', 'N/A'),
                'occupation': p.get('occupation', 'N/A'),
                'personality': p.get('personality_traits', []),
                'values': p.get('values', []),
                'attitude': p.get('attitude_toward_topic', '')[:100],
                'brief': p.get('brief_description', '')
            } for i, p in enumerate(personas)],
            # Factor 분석 통계
            'factor_stats': {
                'n_factors': factor_result['n_factors'],
                'eigenvalues': factor_result.get('eigenvalues', [])[:factor_result['n_factors']],  # ★ 실제 Eigenvalue 사용
                'explained_variance': factor_result['variance'].get('proportion_var', []),
                'cumulative_variance': factor_result['variance'].get('cumulative_var', []),
                'total_variance': sum(factor_result['variance'].get('proportion_var', []))
            },
            # 요인별 Z-score 상위/하위 문항
            'factor_scores_summary': _get_factor_scores_summary(factor_result.get('factor_scores'), q_set),
            # 합의 문항 (모든 Factor에서 비슷한 점수)
            'consensus_statements': _get_consensus_statements(factor_result.get('factor_scores'), q_set),
            # 유형 정보
            'types': types,
            'n_types': len(types),
            'report_path': report_path
        }
        
        update_session(session_id, 7, "분석 완료!", 100)
        session['status'] = 'completed'
        
    except Exception as e:
        print(f"\n[ERROR] 분석 중 오류 발생: {e}", flush=True)
        session['status'] = 'error'
        session['error'] = str(e)
        session['logs'].append(f"❌ 오류: {e}")


def update_session(session_id: str, step: int, message: str, progress: int):
    """세션 상태 업데이트"""
    session = sessions.get(session_id)
    if session:
        session['current_step'] = f"Step {step}: {message}"
        session['progress'] = progress
        session['logs'].append(f"[Step {step}] {message}")


@app.route('/api/status/<session_id>')
def get_status(session_id):
    """진행 상태 조회"""
    session = sessions.get(session_id)
    if not session:
        return jsonify({'error': '세션을 찾을 수 없습니다.'}), 404
    
    return jsonify({
        'status': session['status'],
        'progress': session['progress'],
        'current_step': session['current_step'],
        'logs': session['logs'][-10:],  # 최근 10개 로그
        'result': session.get('result'),
        'error': session.get('error')
    })


@app.route('/api/result/<session_id>')
def get_result(session_id):
    """분석 결과 조회"""
    session = sessions.get(session_id)
    if not session:
        return jsonify({'error': '세션을 찾을 수 없습니다.'}), 404
    
    if session['status'] != 'completed':
        return jsonify({'error': '분석이 아직 완료되지 않았습니다.'}), 400
    
    return jsonify(session['result'])


if __name__ == '__main__':
    # 템플릿 폴더 생성
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 Q-Methodology Web Application")
    print("="*60)
    print(f"\n   URL: http://localhost:8080")
    print(f"   API Key: {'✅ 설정됨' if (config.OPENAI_API_KEY or config.GOOGLE_API_KEY) else '❌ 미설정'}\n")
    
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False, threaded=True)
