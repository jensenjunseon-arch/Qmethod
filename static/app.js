// Q-Methodology Web App JavaScript

class QMethodApp {
    constructor() {
        this.sessionId = null;
        this.pollInterval = null;
        this.startTime = null;
        this.timerInterval = null;
        this.consecutiveErrors = 0;
        this.maxConsecutiveErrors = 15;
        this.currentLang = 'ko';
        
        this.translations = {
            'ko': {
                'app_title': 'Eigen Knot',
                'app_subtitle': 'Q + AI = 154 Knot',
                'input_title': '연구 주제 입력',
                'input_desc': '분석하고 싶은 연구 주제를 입력해주세요. AI가 Q방법론 분석을 자동으로 수행합니다.',
                'input_label': '연구 주제',
                'input_placeholder': '예: MZ세대의 워라밸에 대한 인식\n예: 원격근무에 대한 직장인들의 태도\n예: 환경 보호 정책에 대한 시민들의 인식',
                'start_btn': '분석 시작',
                'starting': '시작 중...',
                'alert_no_topic': '연구 주제를 입력해주세요.',
                'progress_title': '⏳ 분석 진행 중',
                'step_preparing': '시작 준비 중...',
                'step_1': '주제 구체화',
                'step_2': 'Q-Set 생성 (60문항)',
                'step_3': '페르소나 생성 (20명)',
                'step_4': 'Q-Sorting 시뮬레이션',
                'step_5': '요인 분석',
                'step_6': '유형 이원화',
                'step_7': '리포트 생성',
                'log_title': '📜 로그',
                'result_title': '분석 완료',
                'restart_btn': '새 분석 시작',
                'error_title': '❌ 오류 발생',
                'retry_btn': '다시 시도',
                'error_response': '서버 응답 오류',
                'error_timeout': '서버 연결 지연',
                // Result screen
                'result_summary_title': '분석 결과 요약',
                'result_topic': '연구 주제',
                'result_qset_items': 'Q-Set 문항',
                'result_participants': '가상 참여자',
                'result_factors': '추출된 요인',
                'result_variance': '총 설명력',
                'result_types': '도출된 유형',
                'result_count_suffix': '개',
                'result_ppl_suffix': '명',
                'result_stats_toggle': '📋 통계적 검증 살펴보기',
                'result_factor_stats': '📈 Factor 통계 (Eigenvalue ≥ 1.0)',
                'result_factor_col': 'Factor',
                'result_eigenvalue': 'Eigenvalue',
                'result_explained_var': '설명 분산',
                'result_cumulative_var': '누적 분산',
                'result_kaiser_note': '※ Kaiser Rule 적용: Eigenvalue ≥ 1.0인 Factor만 표시됩니다.',
                'result_personas_toggle': '👥 AI 페르소나 참여자',
                'result_qset_toggle': '📝 Q-Set 문항',
                'result_consensus_toggle': '🤝 합의 문항 (모든 유형이 동의/비동의)',
                'result_consensus_note': '모든 Factor에서 Z-score 차이가 0.5 이하인 문항들입니다.',
                'result_no_consensus': '합의 문항이 없습니다. 유형 간 구분이 명확합니다.',
                'result_type_label': '유형',
                'result_positive': '⬆️ Positive',
                'result_negative': '⬇️ Negative (Mirror)',
                'result_detail_btn': '🔍 상세 설명 보기',
                'result_survival': '🎯 생존 본능 (Survival Instinct)',
                'result_defense': '🛡️ 방어 기제 (Defense Mechanism)',
                'result_fear': '😰 숨겨진 두려움 (Hidden Fear)',
                'result_justification': '💭 자기 정당화 (Self-Justification)',
                'result_triggers': '⚡ 트리거 표현',
                'result_action': '📌 행동 지침 (Action Plan)',
                'result_key_stmts': '📊 대표 문항 및 Z-Score',
                'result_no_info': '정보 없음',
                'result_no_stmts': '문항 정보 없음',
                'result_age_suffix': '세',
                'result_preparing': '시작 준비 중...'
            },
            'en': {
                'app_title': 'Eigen Knot',
                'app_subtitle': 'Q + AI = 154 Knot',
                'input_title': 'Enter Research Topic',
                'input_desc': 'Enter a research topic you want to analyze. AI will automatically perform a Q-Methodology analysis.',
                'input_label': 'Research Topic',
                'input_placeholder': 'e.g., Gen Z\'s perception of work-life balance\ne.g., Office workers\' attitudes toward remote work\ne.g., Citizens\' awareness of environmental policies',
                'start_btn': 'Start Analysis',
                'starting': 'Starting...',
                'alert_no_topic': 'Please enter a research topic.',
                'progress_title': '⏳ Analysis in Progress',
                'step_preparing': 'Preparing to start...',
                'step_1': 'Refining Topic',
                'step_2': 'Generating Q-Set (60 items)',
                'step_3': 'Generating Personas (20 ppl)',
                'step_4': 'Q-Sorting Simulation',
                'step_5': 'Factor Analysis',
                'step_6': 'Dual-Type Generation',
                'step_7': 'Generating Report',
                'log_title': '📜 Logs',
                'result_title': 'Analysis Complete',
                'restart_btn': 'Start New Analysis',
                'error_title': '❌ Error Occurred',
                'retry_btn': 'Retry',
                'error_response': 'Server response error',
                'error_timeout': 'Server connection timeout',
                // Result screen
                'result_summary_title': 'Analysis Summary',
                'result_topic': 'Research Topic',
                'result_qset_items': 'Q-Set Items',
                'result_participants': 'Virtual Participants',
                'result_factors': 'Extracted Factors',
                'result_variance': 'Total Variance',
                'result_types': 'Derived Types',
                'result_count_suffix': '',
                'result_ppl_suffix': '',
                'result_stats_toggle': '📋 View Statistical Validation',
                'result_factor_stats': '📈 Factor Statistics (Eigenvalue ≥ 1.0)',
                'result_factor_col': 'Factor',
                'result_eigenvalue': 'Eigenvalue',
                'result_explained_var': 'Explained Var.',
                'result_cumulative_var': 'Cumulative Var.',
                'result_kaiser_note': '※ Kaiser Rule applied: Only Factors with Eigenvalue ≥ 1.0 are shown.',
                'result_personas_toggle': '👥 AI Persona Participants',
                'result_qset_toggle': '📝 Q-Set Items',
                'result_consensus_toggle': '🤝 Consensus Statements (All types agree/disagree)',
                'result_consensus_note': 'Statements with Z-score differences ≤ 0.5 across all Factors.',
                'result_no_consensus': 'No consensus statements. Clear differentiation between types.',
                'result_type_label': 'Type',
                'result_positive': '⬆️ Positive',
                'result_negative': '⬇️ Negative (Mirror)',
                'result_detail_btn': '🔍 View Details',
                'result_survival': '🎯 Survival Instinct',
                'result_defense': '🛡️ Defense Mechanism',
                'result_fear': '😰 Hidden Fear',
                'result_justification': '💭 Self-Justification',
                'result_triggers': '⚡ Trigger Phrases',
                'result_action': '📌 Action Plan',
                'result_key_stmts': '📊 Key Statements & Z-Scores',
                'result_no_info': 'No information',
                'result_no_stmts': 'No statement data',
                'result_age_suffix': '',
                'result_preparing': 'Preparing to start...'
            }
        };

        this.initElements();
        this.bindEvents();
        this.setLanguage('ko');
    }

    setLanguage(lang) {
        this.currentLang = lang;
        
        document.querySelectorAll('.lang-btn').forEach(btn => {
            if (btn.id === `lang-${lang}`) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (this.translations[lang][key]) {
                el.innerText = this.translations[lang][key];
            }
        });
        
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            if (this.translations[lang][key]) {
                el.placeholder = this.translations[lang][key];
            }
        });
    }

    initElements() {
        this.inputSection = document.getElementById('input-section');
        this.progressSection = document.getElementById('progress-section');
        this.resultSection = document.getElementById('result-section');
        this.errorSection = document.getElementById('error-section');

        this.topicInput = document.getElementById('topic-input');
        this.startBtn = document.getElementById('start-btn');
        this.restartBtn = document.getElementById('restart-btn');
        this.retryBtn = document.getElementById('retry-btn');
        
        this.langKoBtn = document.getElementById('lang-ko');
        this.langEnBtn = document.getElementById('lang-en');

        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.stepText = document.getElementById('step-text');
        this.elapsedTime = document.getElementById('elapsed-time');
        this.logsContainer = document.getElementById('logs');

        this.resultSummary = document.getElementById('result-summary');
        this.typesContainer = document.getElementById('types-container');
        this.errorMessage = document.getElementById('error-message');
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.startAnalysis());
        this.restartBtn.addEventListener('click', () => this.reset());
        this.retryBtn.addEventListener('click', () => this.reset());

        this.topicInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.startAnalysis();
            }
        });
        
        this.langKoBtn.addEventListener('click', () => this.setLanguage('ko'));
        this.langEnBtn.addEventListener('click', () => this.setLanguage('en'));
    }

    async startAnalysis() {
        const topic = this.topicInput.value.trim();

        if (!topic) {
            alert(this.translations[this.currentLang]['alert_no_topic']);
            this.topicInput.focus();
            return;
        }

        this.startBtn.disabled = true;
        this.startBtn.innerHTML = `<span class="btn-icon">⏳</span> ${this.translations[this.currentLang]['starting']}`;

        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic, language: this.currentLang })
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            this.sessionId = data.session_id;
            this.startTime = Date.now();
            this.showSection('progress');
            this.startPolling();
            this.startTimer();

        } catch (error) {
            this.showError(error.message);
        }
    }

    startPolling() {
        this.pollInterval = setInterval(() => this.checkStatus(), 2000);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        this.stopTimer();
    }

    startTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const mins = Math.floor(elapsed / 60);
            const secs = elapsed % 60;
            this.elapsedTime.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    async checkStatus() {
        try {
            const response = await fetch(`/api/status/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error(`${this.translations[this.currentLang]['error_response']} (${response.status})`);
            }
            
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // 성공 시 에러 카운터 초기화
            this.consecutiveErrors = 0;

            this.updateProgress(data);

            if (data.status === 'completed') {
                this.stopPolling();
                this.showResult(data.result);
            } else if (data.status === 'error') {
                this.stopPolling();
                this.showError(data.error || this.translations[this.currentLang]['error_response']);
            }

        } catch (error) {
            this.consecutiveErrors++;
            console.error(`Status check failed (${this.consecutiveErrors}/${this.maxConsecutiveErrors}):`, error);
            
            if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
                this.stopPolling();
                this.showError(this.translations[this.currentLang]['error_timeout']);
            }
        }
    }

    updateProgress(data) {
        // Update progress bar
        this.progressFill.style.width = `${data.progress}%`;
        this.progressText.textContent = `${data.progress}%`;

        // Update current step text
        this.stepText.textContent = data.current_step;

        // Update step indicators
        const currentStepMatch = data.current_step.match(/Step (\d)/);
        const currentStepNum = currentStepMatch ? parseInt(currentStepMatch[1]) : 0;

        document.querySelectorAll('.step').forEach(step => {
            const stepNum = parseInt(step.dataset.step);
            step.classList.remove('active', 'completed');

            if (stepNum < currentStepNum) {
                step.classList.add('completed');
            } else if (stepNum === currentStepNum) {
                step.classList.add('active');
            }
        });

        // Update logs
        this.logsContainer.innerHTML = data.logs
            .map(log => `<div class="log-entry">${log}</div>`)
            .join('');
        this.logsContainer.scrollTop = this.logsContainer.scrollHeight;
    }

    showResult(result) {
        this.showSection('result');
        const t = this.translations[this.currentLang];

        // Factor Stats
        const factorStats = result.factor_stats || {};
        const eigenvalues = factorStats.eigenvalues || [];
        const explainedVar = factorStats.explained_variance || [];
        const totalVar = factorStats.total_variance || 0;

        // Summary - compact with collapsible stats section
        this.resultSummary.innerHTML = `
            <h3>${t['result_summary_title']}</h3>
            <p><strong>${t['result_topic']}:</strong> ${result.topic_info?.final_topic || result.topic_info?.topic || 'N/A'}</p>
            <div class="summary-grid">
                <div class="summary-item"><span class="summary-label">${t['result_qset_items']}</span><span class="summary-value">${result.q_set?.length || 0}${t['result_count_suffix']}</span></div>
                <div class="summary-item"><span class="summary-label">${t['result_participants']}</span><span class="summary-value">${result.personas?.length || 0}${t['result_ppl_suffix']}</span></div>
                <div class="summary-item"><span class="summary-label">${t['result_factors']}</span><span class="summary-value">${factorStats.n_factors || 0}${t['result_count_suffix']}</span></div>
                <div class="summary-item"><span class="summary-label">${t['result_variance']}</span><span class="summary-value">${(totalVar * 100).toFixed(1)}%</span></div>
                <div class="summary-item"><span class="summary-label">${t['result_types']}</span><span class="summary-value">${result.n_types || result.types?.length || 0}${t['result_count_suffix']}</span></div>
            </div>
            
            <!-- Statistical Validation (Collapsible) -->
            <details class="stats-accordion">
                <summary class="accordion-trigger">${t['result_stats_toggle']}</summary>
                <div class="accordion-content">
                    <!-- Factor Stats Toggle -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">${t['result_factor_stats']}</summary>
                        <div class="nested-content">
                            <table class="stats-table">
                                <thead>
                                    <tr>
                                        <th>${t['result_factor_col']}</th>
                                        <th>${t['result_eigenvalue']}</th>
                                        <th>${t['result_explained_var']}</th>
                                        <th>${t['result_cumulative_var']}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${eigenvalues.map((ev, i) => {
            if (ev < 1.0) return '';
            return `
                                        <tr>
                                            <td>Factor ${i + 1}</td>
                                            <td>${ev.toFixed(2)}</td>
                                            <td>${(explainedVar[i] * 100).toFixed(1)}%</td>
                                            <td>${((factorStats.cumulative_variance?.[i] || 0) * 100).toFixed(1)}%</td>
                                        </tr>`;
        }).join('')}
                                </tbody>
                            </table>
                            <p class="stats-note">${t['result_kaiser_note']}</p>
                        </div>
                    </details>
                    
                    <!-- AI Persona Participants Toggle -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">${t['result_personas_toggle']} (${result.personas?.length || 0}${t['result_ppl_suffix']})</summary>
                        <div class="nested-content">
                            <div class="personas-grid">
                                ${(result.personas || []).map(p => `
                                    <div class="persona-card">
                                        <strong>${p.name || p}</strong>
                                        ${typeof p === 'object' ? `
                                            <span class="persona-meta">${p.age || ''}${t['result_age_suffix']} / ${p.gender || ''}</span>
                                            <span class="persona-occupation">${p.occupation || ''}</span>
                                            <span class="persona-brief">${(p.brief || '').substring(0, 80)}${(p.brief || '').length > 80 ? '...' : ''}</span>
                                        ` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </details>
                    
                    <!-- Q-Set Items Toggle -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">${t['result_qset_toggle']} (${result.q_set?.length || 0}${t['result_count_suffix']})</summary>
                        <div class="nested-content">
                            <ol class="qset-list">
                                ${(result.q_set || []).map(q => `
                                    <li>${typeof q === 'object' ? q.text : q}</li>
                                `).join('')}
                            </ol>
                        </div>
                    </details>
                    
                    <!-- Consensus Statements Toggle -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">${t['result_consensus_toggle']}</summary>
                        <div class="nested-content">
                            ${(result.consensus_statements && result.consensus_statements.length > 0) ? `
                                <p class="consensus-note">${t['result_consensus_note']}</p>
                                <ul class="consensus-list">
                                    ${result.consensus_statements.map(c => `
                                        <li class="consensus-item ${c.interpretation}">
                                            <span class="consensus-score ${c.avg_z_score >= 0 ? 'positive' : 'negative'}">${c.avg_z_score >= 0 ? '+' : ''}${c.avg_z_score}</span>
                                            <span class="consensus-badge">${c.interpretation}</span>
                                            <span class="consensus-text">${c.statement}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                            ` : `<p>${t['result_no_consensus']}</p>`}
                        </div>
                    </details>
                </div>
            </details>
        `;

        // Types with collapsible details
        this.typesContainer.innerHTML = result.types.map((type, index) => `
            <div class="type-card ${type.bias || type.polarity}">
                <div class="type-header">
                    <h4>${type.type_name || `${t['result_type_label']} ${index + 1}`}</h4>
                    <span class="type-badge ${type.bias || type.polarity}">
                        ${(type.bias === 'positive' || type.polarity === 'positive') ? t['result_positive'] : t['result_negative']} | ${type.factor || ''}
                    </span>
                </div>
                <p class="type-summary">${type.short_description || ''}</p>
                <div class="type-values">
                    ${(type.core_values || []).map(v => `<span>${v}</span>`).join('')}
                </div>
                
                <!-- Detail (Collapsible) -->
                <details class="type-accordion">
                    <summary class="accordion-trigger type-detail-btn">${t['result_detail_btn']}</summary>
                    <div class="accordion-content type-full-details">
                        <div class="detail-section">
                            <h5>${t['result_survival']}</h5>
                            <p>${type.survival_instinct || t['result_no_info']}</p>
                        </div>
                        <div class="detail-section">
                            <h5>${t['result_defense']}</h5>
                            <p>${type.defense_mechanism || t['result_no_info']}</p>
                        </div>
                        <div class="detail-section">
                            <h5>${t['result_fear']}</h5>
                            <p>${type.hidden_fear || t['result_no_info']}</p>
                        </div>
                        <div class="detail-section">
                            <h5>${t['result_justification']}</h5>
                            <p>${type.self_justification || t['result_no_info']}</p>
                        </div>
                        ${type.trigger_phrases ? `
                            <div class="detail-section">
                                <h5>${t['result_triggers']}</h5>
                                <div class="trigger-list">
                                    ${(type.trigger_phrases || []).map(tp => `<span class="trigger-tag">"${tp}"</span>`).join('')}
                                </div>
                            </div>
                        ` : ''}
                        ${type.action_plan ? `
                            <div class="detail-section">
                                <h5>${t['result_action']}</h5>
                                <ul class="action-list">
                                    ${(type.action_plan || []).map(a => `<li>${a}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        <div class="detail-section">
                            <h5>${t['result_key_stmts']}</h5>
                            <ul class="statement-list">
                                ${(type.key_statements || []).map(stmt => `
                                    <li>
                                        <span class="z-score ${stmt.z_score >= 0 ? 'positive' : 'negative'}">${typeof stmt.z_score === 'number' ? (stmt.z_score >= 0 ? '+' : '') + stmt.z_score.toFixed(2) : 'N/A'}</span>
                                        ${stmt.statement || stmt}
                                    </li>
                                `).join('')}
                            </ul>
                            ${(type.key_statements || []).length === 0 ? `<p>${t['result_no_stmts']}</p>` : ''}
                        </div>
                    </div>
                </details>
            </div>
        `).join('');
    }

    showError(message) {
        this.showSection('error');
        this.errorMessage.textContent = message;
    }

    showSection(section) {
        this.inputSection.classList.add('hidden');
        this.progressSection.classList.add('hidden');
        this.resultSection.classList.add('hidden');
        this.errorSection.classList.add('hidden');

        switch (section) {
            case 'input':
                this.inputSection.classList.remove('hidden');
                break;
            case 'progress':
                this.progressSection.classList.remove('hidden');
                break;
            case 'result':
                this.resultSection.classList.remove('hidden');
                break;
            case 'error':
                this.errorSection.classList.remove('hidden');
                break;
        }
    }

    reset() {
        this.stopPolling();
        this.sessionId = null;
        this.startTime = null;
        this.consecutiveErrors = 0;
        this.topicInput.value = '';
        this.startBtn.disabled = false;
        const t = this.translations[this.currentLang];
        this.startBtn.innerHTML = `${t['start_btn']}`;
        this.progressFill.style.width = '0%';
        this.progressText.textContent = '0%';
        this.stepText.textContent = t['result_preparing'];
        this.elapsedTime.textContent = '0:00';
        this.logsContainer.innerHTML = '';

        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active', 'completed');
        });

        this.showSection('input');
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new QMethodApp();
});
