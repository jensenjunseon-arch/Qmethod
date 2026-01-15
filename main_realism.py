#!/usr/bin/env python3
"""
Realism Q - Main CLI Entry Point
Hybrid Single/Dual Group Q-Methodology Analysis
"""
import argparse
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.realism_q_set import generate_q_set, get_contradiction_pairs
from modules.realism_p_set import generate_realism_personas, generate_dual_group_personas
from modules.q_sorting import simulate_single_sorting
from modules.validation import validate_sorting
from modules.factor_analysis import perform_factor_analysis
from modules.polarity_decomposer import generate_six_types, analyze_internal_conflict
from modules.match_matrix import analyze_dual_group_dynamics
from modules.realism_report import generate_realism_report, save_realism_report


def run_single_group_analysis(topic: str, group_a: str) -> dict:
    """
    Single Group Deep-Dive Mode
    집단 내 분화와 갈등하는 생존 전략 분석
    """
    print("\n" + "="*70)
    print("🔬 REALISM Q - Single Group Deep-Dive Mode")
    print("="*70)
    print(f"📌 Topic: {topic}")
    print(f"👥 Group: {group_a}")
    print("="*70 + "\n")
    
    topic_info = {
        "topic": topic,
        "final_topic": topic,
        "group": group_a,
        "group_a": group_a,
        "analysis_mode": "single"
    }
    
    # Step 1: Q-Set 생성 (Expansion → Reduction → Blind Shuffle)
    print("\n📝 Step 1: Q-Set Construction (200 → 60)")
    q_set, category_map, contradiction_pairs = generate_q_set(topic, group_a, 200, 60)
    
    # Step 2: 페르소나 생성 (Psychographics 포함)
    print("\n👥 Step 2: Persona Generation (20 AI Personas)")
    personas = generate_realism_personas(topic_info, group_a, 20)
    
    # Step 3: Q-Sorting 시뮬레이션 + 검증
    print("\n🎯 Step 3: Q-Sorting Simulation with Validation")
    valid_sortings = []
    sorting_matrix = []
    
    for persona in personas:
        max_attempts = 3
        for attempt in range(max_attempts):
            sorting = simulate_single_sorting(persona, q_set, topic_info)
            
            # 검증
            sorting_dict = {q_set[i]["id"]: score for i, score in enumerate(sorting)}
            is_valid, report = validate_sorting(sorting_dict, contradiction_pairs)
            
            if is_valid:
                valid_sortings.append(sorting_dict)
                sorting_matrix.append(sorting)
                break
            elif attempt < max_attempts - 1:
                print(f"[SORT] {persona.get('name', 'Unknown')} 재시뮬레이션 ({attempt+2}/{max_attempts})", flush=True)
        else:
            # 검증 실패해도 추가 (경고와 함께)
            valid_sortings.append(sorting_dict)
            sorting_matrix.append(sorting)
    
    import numpy as np
    sorting_matrix = np.array(sorting_matrix)
    
    # Step 4: Factor Analysis (PCA + Varimax)
    print("\n📊 Step 4: Factor Analysis (PCA + Varimax Rotation)")
    factor_result = perform_factor_analysis(sorting_matrix)
    
    # Step 5: Polarity Decomposition (6 Types)
    print("\n🎭 Step 5: Polarity Decomposition (3 Factors × 2 Poles)")
    types = generate_six_types(
        factor_result["factor_scores"],
        q_set,
        topic_info,
        n_factors=3
    )
    
    # Step 6: Internal Conflict Analysis
    print("\n🔗 Step 6: Internal Conflict Analysis")
    internal_conflict = analyze_internal_conflict(types, topic_info)
    
    # Step 7: Generate Report
    print("\n📄 Step 7: Report Generation")
    report_content = generate_realism_report(
        topic_info,
        types,
        "single",
        q_set,
        internal_conflict=internal_conflict
    )
    
    report_path = save_realism_report(report_content, topic_info)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print(f"📄 Report saved to: {report_path}")
    print("="*70 + "\n")
    
    return {
        "mode": "single",
        "topic_info": topic_info,
        "q_set": q_set,
        "personas": personas,
        "types": types,
        "internal_conflict": internal_conflict,
        "report_path": report_path
    }


def run_dual_group_analysis(topic: str, group_a: str, group_b: str) -> dict:
    """
    Dual Group Dynamics Mode
    두 집단 간 갈등과 화학 작용 분석
    """
    print("\n" + "="*70)
    print("🔬 REALISM Q - Dual Group Dynamics Mode")
    print("="*70)
    print(f"📌 Topic: {topic}")
    print(f"👥 Group A: {group_a}")
    print(f"👥 Group B: {group_b}")
    print("="*70 + "\n")
    
    topic_info = {
        "topic": topic,
        "final_topic": topic,
        "group_a": group_a,
        "group_b": group_b,
        "analysis_mode": "dual"
    }
    
    results = {"mode": "dual", "topic_info": topic_info}
    
    # Group A 분석
    print(f"\n{'='*30} GROUP A: {group_a} {'='*30}")
    q_set_a, cat_map_a, pairs_a = generate_q_set(topic, group_a, 200, 60)
    personas_a = generate_realism_personas(topic_info, group_a, 20)
    
    sorting_matrix_a = []
    for persona in personas_a:
        sorting = simulate_single_sorting(persona, q_set_a, topic_info)
        sorting_matrix_a.append(sorting)
    
    import numpy as np
    sorting_matrix_a = np.array(sorting_matrix_a)
    factor_result_a = perform_factor_analysis(sorting_matrix_a)
    types_a = generate_six_types(factor_result_a["factor_scores"], q_set_a, topic_info, 3)
    
    # Group B 분석
    print(f"\n{'='*30} GROUP B: {group_b} {'='*30}")
    q_set_b, cat_map_b, pairs_b = generate_q_set(topic, group_b, 200, 60)
    personas_b = generate_realism_personas(topic_info, group_b, 20)
    
    sorting_matrix_b = []
    for persona in personas_b:
        sorting = simulate_single_sorting(persona, q_set_b, topic_info)
        sorting_matrix_b.append(sorting)
    
    sorting_matrix_b = np.array(sorting_matrix_b)
    factor_result_b = perform_factor_analysis(sorting_matrix_b)
    types_b = generate_six_types(factor_result_b["factor_scores"], q_set_b, topic_info, 3)
    
    # Match/Mismatch Matrix
    print(f"\n{'='*30} MATCH MATRIX {'='*30}")
    match_analysis = analyze_dual_group_dynamics(types_a, types_b, topic_info)
    
    # Combined Types for report
    all_types = types_a + types_b
    
    # Generate Report
    print("\n📄 Report Generation")
    report_content = generate_realism_report(
        topic_info,
        all_types,
        "dual",
        q_set_a + q_set_b,
        match_matrix=match_analysis
    )
    
    report_path = save_realism_report(report_content, topic_info)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print(f"📄 Report saved to: {report_path}")
    print("="*70 + "\n")
    
    return {
        "mode": "dual",
        "topic_info": topic_info,
        "group_a": {"q_set": q_set_a, "personas": personas_a, "types": types_a},
        "group_b": {"q_set": q_set_b, "personas": personas_b, "types": types_b},
        "match_analysis": match_analysis,
        "report_path": report_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Realism Q - Advanced Q-Methodology Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single Group Analysis
  python main_realism.py --topic "웹툰 창작" --group-a "웹툰 작가"
  
  # Dual Group Analysis
  python main_realism.py --topic "웹툰 산업" --group-a "웹툰 작가" --group-b "웹툰 PD"
        """
    )
    
    parser.add_argument(
        "--topic", "-t",
        required=True,
        help="연구 주제 (Required)"
    )
    
    parser.add_argument(
        "--group-a", "-a",
        required=True,
        help="대상 집단 A (Required)"
    )
    
    parser.add_argument(
        "--group-b", "-b",
        default="",
        help="대상 집단 B (Optional: Leave blank for Single Group Analysis)"
    )
    
    args = parser.parse_args()
    
    # API Key 확인
    if not config.OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Mode 분기
    if args.group_b:
        result = run_dual_group_analysis(args.topic, args.group_a, args.group_b)
    else:
        result = run_single_group_analysis(args.topic, args.group_a)
    
    return result


if __name__ == "__main__":
    main()
