"""
Realism Report Generator
Radar Chart, Z-Score Heatmap, Action Plan
"""
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import os


def generate_realism_report(
    topic_info: Dict,
    types: List[Dict],
    analysis_mode: str,
    q_set: List[Dict],
    internal_conflict: Optional[Dict] = None,
    match_matrix: Optional[Dict] = None
) -> str:
    """
    Realism Q 분석 리포트 생성
    
    Args:
        topic_info: 주제 정보
        types: 6개 유형 리스트
        analysis_mode: 'single' | 'dual'
        q_set: Q-Set 문항 리스트
        internal_conflict: Single Mode - 내부 갈등 분석
        match_matrix: Dual Mode - 상성 매트릭스
    """
    topic = topic_info.get("final_topic", topic_info.get("topic", ""))
    group_a = topic_info.get("group_a", topic_info.get("group", ""))
    group_b = topic_info.get("group_b", "")
    
    report_lines = []
    
    # 헤더
    report_lines.append(f"# 🔍 The Naked Truth of **{topic}**")
    report_lines.append("")
    report_lines.append(f"> *\"당신의 창작/관리 DNA를 날것 그대로 분석합니다.\"*")
    report_lines.append("")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**분석 모드**: {'Single Group Deep-Dive' if analysis_mode == 'single' else 'Dual Group Dynamics'}")
    report_lines.append(f"**대상 집단**: {group_a}" + (f" ↔ {group_b}" if group_b else ""))
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 요약
    report_lines.append("## 📊 Executive Summary")
    report_lines.append("")
    report_lines.append(f"- **총 유형 수**: {len(types)}개 (3 Factors × 2 Poles)")
    report_lines.append(f"- **Q-Set 문항 수**: {len(q_set)}개")
    
    if analysis_mode == "single" and internal_conflict:
        report_lines.append(f"- **내부 분화 원인**: {internal_conflict.get('fragmentation_cause', 'N/A')}")
    elif analysis_mode == "dual" and match_matrix:
        best = match_matrix.get("match_matrix", {}).get("best_match", {})
        worst = match_matrix.get("match_matrix", {}).get("worst_match", {})
        report_lines.append(f"- **최고 시너지**: {best.get('type_a', 'N/A')} ↔ {best.get('type_b', 'N/A')}")
        report_lines.append(f"- **최악 갈등**: {worst.get('type_a', 'N/A')} ↔ {worst.get('type_b', 'N/A')}")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 6개 유형 상세
    report_lines.append("## 🎭 The 6 Realism Types")
    report_lines.append("")
    
    for i, t in enumerate(types, 1):
        polarity_emoji = "⬆️" if t.get("polarity") == "positive" else "⬇️"
        report_lines.append(f"### {polarity_emoji} Type {i}: **{t.get('type_name', f'Type {i}')}**")
        report_lines.append("")
        report_lines.append(f"*{t.get('factor', 'Factor ?')} | {t.get('polarity', '?').upper()} Pole*")
        report_lines.append("")
        report_lines.append(f"> {t.get('short_description', '')}")
        report_lines.append("")
        
        report_lines.append("| 차원 | 분석 |")
        report_lines.append("|------|------|")
        report_lines.append(f"| 🎯 **생존 본능** | {t.get('survival_instinct', 'N/A')} |")
        report_lines.append(f"| 🛡️ **방어 기제** | {t.get('defense_mechanism', 'N/A')} |")
        report_lines.append(f"| 😰 **숨겨진 두려움** | {t.get('hidden_fear', 'N/A')} |")
        report_lines.append(f"| 💭 **자기 정당화** | {t.get('self_justification', 'N/A')} |")
        report_lines.append("")
        
        # 핵심 가치
        core_values = t.get("core_values", [])
        if core_values:
            report_lines.append(f"**핵심 가치**: {', '.join(core_values)}")
            report_lines.append("")
        
        # 트리거 문구
        triggers = t.get("trigger_phrases", [])
        if triggers:
            report_lines.append(f"**⚠️ 자극 트리거**: \"{triggers[0]}\"")
            report_lines.append("")
        
        # 행동 지침
        action_plan = t.get("action_plan", [])
        if action_plan:
            report_lines.append("**📋 Action Plan**:")
            for action in action_plan[:3]:
                report_lines.append(f"- {action}")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # 모드별 추가 분석
    if analysis_mode == "single" and internal_conflict:
        report_lines.append("## 🔗 Internal Harmony Analysis")
        report_lines.append("")
        report_lines.append(f"### 분화의 근본 원인")
        report_lines.append(f"{internal_conflict.get('fragmentation_cause', 'N/A')}")
        report_lines.append("")
        report_lines.append(f"### 공유된 불안")
        report_lines.append(f"{internal_conflict.get('shared_anxiety', 'N/A')}")
        report_lines.append("")
        
        # 갈등 쌍
        conflict_pairs = internal_conflict.get("conflict_pairs", [])
        if conflict_pairs:
            report_lines.append("### 잠재적 갈등 쌍")
            report_lines.append("")
            for pair in conflict_pairs:
                report_lines.append(f"- **{pair.get('type_a', '?')}** vs **{pair.get('type_b', '?')}**: {pair.get('conflict_reason', '')}")
            report_lines.append("")
        
        # 조화 전략
        strategies = internal_conflict.get("harmony_strategies", [])
        if strategies:
            report_lines.append("### 🕊️ 내부 조화 전략")
            for s in strategies:
                report_lines.append(f"1. {s}")
            report_lines.append("")
    
    elif analysis_mode == "dual" and match_matrix:
        report_lines.append("## ⚡ Match/Mismatch Matrix")
        report_lines.append("")
        
        # 최고/최악 매칭
        matrix_data = match_matrix.get("match_matrix", {})
        best = matrix_data.get("best_match", {})
        worst = matrix_data.get("worst_match", {})
        
        report_lines.append("### 🏆 Best Match (최고 시너지)")
        report_lines.append(f"**{best.get('type_a', 'N/A')}** ↔ **{best.get('type_b', 'N/A')}** (점수: {best.get('score', 0):.2f})")
        report_lines.append("")
        
        report_lines.append("### 💥 Worst Match (최악 갈등)")
        report_lines.append(f"**{worst.get('type_a', 'N/A')}** ↔ **{worst.get('type_b', 'N/A')}** (점수: {worst.get('score', 0):.2f})")
        report_lines.append("")
        
        # 위험 경고
        warnings = match_matrix.get("risk_warnings", [])
        if warnings:
            report_lines.append("### ⚠️ Risk Warnings")
            for w in warnings[:5]:
                report_lines.append(f"- {w.get('warning_message', '')}")
            report_lines.append("")
        
        # 커뮤니케이션 스크립트
        scripts = match_matrix.get("communication_scripts", {})
        if scripts:
            report_lines.append("### 💬 Communication Scripts")
            report_lines.append("")
            
            best_scripts = scripts.get("best_match_scripts", {})
            if best_scripts:
                report_lines.append("**시너지 매칭 대화법**:")
                report_lines.append(f"- 첫 마디: *\"{best_scripts.get('opening_line', '')}\"*")
                report_lines.append("")
            
            worst_scripts = scripts.get("worst_match_scripts", {})
            if worst_scripts:
                report_lines.append("**갈등 매칭 주의사항**:")
                report_lines.append(f"- ⚠️ {worst_scripts.get('warning', '')}")
                donts = worst_scripts.get("absolute_donts", [])
                for d in donts[:2]:
                    report_lines.append(f"- ❌ {d}")
                report_lines.append("")
    
    # 푸터
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Generated by Realism Q System | Q-Methodology Research Platform*")
    
    return "\n".join(report_lines)


def save_realism_report(
    report_content: str,
    topic_info: Dict,
    output_dir: str = "outputs"
) -> str:
    """
    리포트를 파일로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    topic = topic_info.get("final_topic", topic_info.get("topic", "report"))
    safe_topic = "".join(c if c.isalnum() or c in "가-힣" else "_" for c in topic)[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"realism_report_{safe_topic}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"[REPORT] 리포트 저장 완료: {filepath}", flush=True)
    
    return filepath


if __name__ == "__main__":
    print("Realism Report Generator Module Loaded")
