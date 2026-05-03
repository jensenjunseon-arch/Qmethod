// Q-Methodology Web App JavaScript

class QMethodApp {
    constructor() {
        this.sessionId = null;
        this.pollInterval = null;
        this.startTime = null;
        this.timerInterval = null;
        this.consecutiveErrors = 0;
        this.maxConsecutiveErrors = 15;

        this.initElements();
        this.bindEvents();
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
    }

    async startAnalysis() {
        const topic = this.topicInput.value.trim();

        if (!topic) {
            alert('연구 주제를 입력해주세요.');
            this.topicInput.focus();
            return;
        }

        this.startBtn.disabled = true;
        this.startBtn.innerHTML = '<span class="btn-icon">⏳</span> 시작 중...';

        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic })
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
                throw new Error(`서버 응답 오류 (${response.status})`);
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
                this.showError(data.error || '알 수 없는 오류가 발생했습니다.');
            }

        } catch (error) {
            this.consecutiveErrors++;
            console.error(`Status check failed (${this.consecutiveErrors}/${this.maxConsecutiveErrors}):`, error);
            
            if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
                this.stopPolling();
                this.showError('서버와의 연결이 끊어졌습니다. 서버가 절전 모드에 들어갔거나 네트워크 문제가 발생했을 수 있습니다. 페이지를 새로고침한 후 다시 시도해주세요.');
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

        // Factor Stats
        const factorStats = result.factor_stats || {};
        const eigenvalues = factorStats.eigenvalues || [];
        const explainedVar = factorStats.explained_variance || [];
        const totalVar = factorStats.total_variance || 0;

        // Summary - compact with collapsible stats section
        this.resultSummary.innerHTML = `
            <h3>분석 결과 요약</h3>
            <p><strong>연구 주제:</strong> ${result.topic_info?.final_topic || result.topic_info?.topic || 'N/A'}</p>
            <div class="summary-grid">
                <div class="summary-item"><span class="summary-label">Q-Set 문항</span><span class="summary-value">${result.q_set?.length || 0}개</span></div>
                <div class="summary-item"><span class="summary-label">가상 참여자</span><span class="summary-value">${result.personas?.length || 0}명</span></div>
                <div class="summary-item"><span class="summary-label">추출된 요인</span><span class="summary-value">${factorStats.n_factors || 0}개</span></div>
                <div class="summary-item"><span class="summary-label">총 설명력</span><span class="summary-value">${(totalVar * 100).toFixed(1)}%</span></div>
                <div class="summary-item"><span class="summary-label">도출된 유형</span><span class="summary-value">${result.n_types || result.types?.length || 0}개</span></div>
            </div>
            
            <!-- 통계적 검증 살펴보기 (Collapsible) -->
            <details class="stats-accordion">
                <summary class="accordion-trigger">📋 통계적 검증 살펴보기</summary>
                <div class="accordion-content">
                    <!-- Factor 통계 토글 -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">📈 Factor 통계 (Eigenvalue ≥ 1.0)</summary>
                        <div class="nested-content">
                            <table class="stats-table">
                                <thead>
                                    <tr>
                                        <th>Factor</th>
                                        <th>Eigenvalue</th>
                                        <th>설명 분산</th>
                                        <th>누적 분산</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${eigenvalues.map((ev, i) => {
            if (ev < 1.0) return ''; // Kaiser Rule: Only show Eigenvalue >= 1.0
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
                            <p class="stats-note">※ Kaiser Rule 적용: Eigenvalue ≥ 1.0인 Factor만 표시됩니다.</p>
                        </div>
                    </details>
                    
                    <!-- AI 페르소나 참여자 토글 -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">👥 AI 페르소나 참여자 (${result.personas?.length || 0}명)</summary>
                        <div class="nested-content">
                            <div class="personas-grid">
                                ${(result.personas || []).map(p => `
                                    <div class="persona-card">
                                        <strong>${p.name || p}</strong>
                                        ${typeof p === 'object' ? `
                                            <span class="persona-meta">${p.age || ''}세 / ${p.gender || ''}</span>
                                            <span class="persona-occupation">${p.occupation || ''}</span>
                                            <span class="persona-brief">${(p.brief || '').substring(0, 80)}${(p.brief || '').length > 80 ? '...' : ''}</span>
                                        ` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </details>
                    
                    <!-- Q-Set 문항 토글 -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">📝 Q-Set 문항 (${result.q_set?.length || 0}개)</summary>
                        <div class="nested-content">
                            <ol class="qset-list">
                                ${(result.q_set || []).map(q => `
                                    <li>${typeof q === 'object' ? q.text : q}</li>
                                `).join('')}
                            </ol>
                        </div>
                    </details>
                    
                    <!-- 합의 문항 토글 -->
                    <details class="nested-accordion">
                        <summary class="nested-trigger">🤝 합의 문항 (모든 유형이 동의/비동의)</summary>
                        <div class="nested-content">
                            ${(result.consensus_statements && result.consensus_statements.length > 0) ? `
                                <p class="consensus-note">모든 Factor에서 Z-score 차이가 0.5 이하인 문항들입니다.</p>
                                <ul class="consensus-list">
                                    ${result.consensus_statements.map(c => `
                                        <li class="consensus-item ${c.interpretation}">
                                            <span class="consensus-score ${c.avg_z_score >= 0 ? 'positive' : 'negative'}">${c.avg_z_score >= 0 ? '+' : ''}${c.avg_z_score}</span>
                                            <span class="consensus-badge">${c.interpretation}</span>
                                            <span class="consensus-text">${c.statement}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                            ` : '<p>합의 문항이 없습니다. 유형 간 구분이 명확합니다.</p>'}
                        </div>
                    </details>
                </div>
            </details>
        `;

        // Types with collapsible details
        this.typesContainer.innerHTML = result.types.map((type, index) => `
            <div class="type-card ${type.bias || type.polarity}">
                <div class="type-header">
                    <h4>${type.type_name || `유형 ${index + 1}`}</h4>
                    <span class="type-badge ${type.bias || type.polarity}">
                        ${(type.bias === 'positive' || type.polarity === 'positive') ? '⬆️ Positive' : '⬇️ Negative (Mirror)'} | ${type.factor || ''}
                    </span>
                </div>
                <p class="type-summary">${type.short_description || ''}</p>
                <div class="type-values">
                    ${(type.core_values || []).map(v => `<span>${v}</span>`).join('')}
                </div>
                
                <!-- 상세 설명 (Collapsible) -->
                <details class="type-accordion">
                    <summary class="accordion-trigger type-detail-btn">🔍 상세 설명 보기</summary>
                    <div class="accordion-content type-full-details">
                        <div class="detail-section">
                            <h5>🎯 생존 본능 (Survival Instinct)</h5>
                            <p>${type.survival_instinct || '정보 없음'}</p>
                        </div>
                        <div class="detail-section">
                            <h5>🛡️ 방어 기제 (Defense Mechanism)</h5>
                            <p>${type.defense_mechanism || '정보 없음'}</p>
                        </div>
                        <div class="detail-section">
                            <h5>😰 숨겨진 두려움 (Hidden Fear)</h5>
                            <p>${type.hidden_fear || '정보 없음'}</p>
                        </div>
                        <div class="detail-section">
                            <h5>💭 자기 정당화 (Self-Justification)</h5>
                            <p>${type.self_justification || '정보 없음'}</p>
                        </div>
                        ${type.trigger_phrases ? `
                            <div class="detail-section">
                                <h5>⚡ 트리거 표현</h5>
                                <div class="trigger-list">
                                    ${(type.trigger_phrases || []).map(t => `<span class="trigger-tag">"${t}"</span>`).join('')}
                                </div>
                            </div>
                        ` : ''}
                        ${type.action_plan ? `
                            <div class="detail-section">
                                <h5>📌 행동 지침 (Action Plan)</h5>
                                <ul class="action-list">
                                    ${(type.action_plan || []).map(a => `<li>${a}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        <div class="detail-section">
                            <h5>📊 대표 문항 및 Z-Score</h5>
                            <ul class="statement-list">
                                ${(type.key_statements || []).map(stmt => `
                                    <li>
                                        <span class="z-score ${stmt.z_score >= 0 ? 'positive' : 'negative'}">${typeof stmt.z_score === 'number' ? (stmt.z_score >= 0 ? '+' : '') + stmt.z_score.toFixed(2) : 'N/A'}</span>
                                        ${stmt.statement || stmt}
                                    </li>
                                `).join('')}
                            </ul>
                            ${(type.key_statements || []).length === 0 ? '<p>문항 정보 없음</p>' : ''}
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
        this.startBtn.innerHTML = '<span class="btn-icon">🚀</span> 분석 시작';
        this.progressFill.style.width = '0%';
        this.progressText.textContent = '0%';
        this.stepText.textContent = '시작 준비 중...';
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
