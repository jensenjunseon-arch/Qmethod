#!/usr/bin/env python3
"""
Q-Methodology Research Insight Generator
Q방법론 기반 연구 통찰 생성기

메인 애플리케이션 - 전체 워크플로우를 통합 실행합니다.
"""
import sys
import os
import json
import argparse

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.topic_refiner import refine_topic_interactive, refine_topic_from_string
from modules.q_population import construct_q_set
from modules.p_set_generator import generate_all_personas, describe_personas
from modules.q_sorting import simulate_all_sortings
from modules.factor_analysis import perform_factor_analysis, get_factor_interpretation_data
from modules.dual_type_generator import generate_dual_types
from modules.report_generator import generate_report, save_data_artifacts


def print_banner():
    """애플리케이션 배너를 출력합니다."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🔬 Q-Methodology Research Insight Generator               ║
║   Q방법론 기반 연구 통찰 생성기                              ║
║                                                              ║
║   Version: 1.0.0                                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def validate_api_key():
    """OpenAI API 키가 설정되어 있는지 확인합니다."""
    if not config.OPENAI_API_KEY:
        print("\n❌ 오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   다음 명령어로 API 키를 설정해주세요:")
        print("   export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    print("✅ OpenAI API 키 확인 완료")


def run_full_pipeline(initial_topic: str = None, interactive: bool = True) -> str:
    """
    Q방법론 전체 파이프라인을 실행합니다.
    
    Args:
        initial_topic: 초기 연구 주제 (None이면 대화형 입력)
        interactive: 대화형 모드 여부
    
    Returns:
        생성된 리포트 파일 경로
    """
    print_banner()
    validate_api_key()
    
    print("\n" + "="*60)
    print("🚀 Q방법론 연구 분석 시작")
    print("="*60)
    
    # Step 1: 주제 구체화
    print("\n📌 Step 1/7: 주제 구체화")
    print("-" * 40)
    
    if interactive and initial_topic is None:
        topic_info = refine_topic_interactive()
    else:
        if initial_topic:
            topic_info = refine_topic_from_string(initial_topic)
        else:
            raise ValueError("비대화형 모드에서는 initial_topic이 필요합니다.")
    
    print(f"\n✅ 확정된 주제: {topic_info.get('final_topic', 'N/A')}")
    
    # Step 2: Q-Set 구성
    print("\n📌 Step 2/7: Q-Population & Q-Set 구성")
    print("-" * 40)
    
    q_population, q_set = construct_q_set(topic_info)
    print(f"\n✅ Q-Set 완성: {len(q_set)}개 문항")
    
    # Step 3: P-Set 생성
    print("\n📌 Step 3/7: P-Set (페르소나) 생성")
    print("-" * 40)
    
    personas = generate_all_personas(topic_info)
    print(f"\n✅ 페르소나 생성 완료: {len(personas)}명")
    
    # Step 4: Q-Sorting 시뮬레이션
    print("\n📌 Step 4/7: Q-Sorting 시뮬레이션")
    print("-" * 40)
    
    sorting_matrix = simulate_all_sortings(personas, q_set, topic_info)
    print(f"\n✅ Q-Sorting 완료: {sorting_matrix.shape[0]} x {sorting_matrix.shape[1]} 매트릭스")
    
    # Step 5: 통계 분석
    print("\n📌 Step 5/7: 통계적 분석 (Factor Analysis)")
    print("-" * 40)
    
    factor_result = perform_factor_analysis(sorting_matrix)
    print(f"\n✅ 요인 분석 완료: {factor_result['n_factors']}개 요인 추출")
    
    # Step 6: 유형 이원화
    print("\n📌 Step 6/7: 유형 이원화 (Dual-Type Generation)")
    print("-" * 40)
    
    types = generate_dual_types(
        factor_result['factor_scores'],
        q_set,
        topic_info,
        factor_result['significant_loadings']
    )
    print(f"\n✅ 유형 생성 완료: {len(types)}개 유형")
    
    # Step 7: 리포트 생성
    print("\n📌 Step 7/7: 리포트 생성")
    print("-" * 40)
    
    # 데이터 아티팩트 저장
    data_paths = save_data_artifacts(
        topic_info,
        q_population,
        q_set,
        personas,
        sorting_matrix
    )
    
    # 리포트 생성
    report_path = generate_report(
        topic_info,
        q_set,
        personas,
        sorting_matrix,
        factor_result,
        types
    )
    
    # 완료 메시지
    print("\n" + "="*60)
    print("🎉 분석 완료!")
    print("="*60)
    print(f"\n📄 리포트 경로: {report_path}")
    print(f"📁 데이터 저장 위치: {config.OUTPUT_DIR}")
    
    # 유형 요약 출력
    print("\n📊 도출된 유형 요약:")
    print("-" * 40)
    for i, t in enumerate(types):
        bias = "➕" if t.get("bias") == "positive" else "➖"
        print(f"  {i+1}. {t.get('type_name', f'유형 {i+1}')} {bias}")
        print(f"     → {t.get('short_description', 'N/A')}")
    
    return report_path


def main():
    """메인 함수 - CLI 인터페이스를 제공합니다."""
    parser = argparse.ArgumentParser(
        description='Q-Methodology Research Insight Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py                           # 대화형 모드
  python main.py --topic "MZ세대의 워라밸" # 주제 직접 입력
  python main.py --non-interactive --topic "..."  # 비대화형 모드
        """
    )
    
    parser.add_argument(
        '--topic', '-t',
        type=str,
        help='초기 연구 주제 (생략 시 대화형으로 입력받음)'
    )
    
    parser.add_argument(
        '--non-interactive', '-n',
        action='store_true',
        help='비대화형 모드로 실행 (주제 구체화 단계 생략)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='출력 디렉토리 경로'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output:
        config.OUTPUT_DIR = args.output
    
    try:
        interactive = not args.non_interactive
        report_path = run_full_pipeline(
            initial_topic=args.topic,
            interactive=interactive
        )
        
        print(f"\n✨ 리포트가 성공적으로 생성되었습니다!")
        print(f"   경로: {report_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
