/**
 * Ultra-Advanced Student Portal Logic
 * Interactive UI, Particle Engine, Cursor Spotlight, and SHAP Chart
 */

let currentStep = 1;
const totalSteps = 3;
let finalChart = null;

// ==========================================
// 1. PARTICLE ENGINE BACKGROUND
// ==========================================
class ParticleEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.numParticles = window.innerWidth < 768 ? 40 : 100;
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.init();
        this.animate();
    }
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    init() {
        this.particles = [];
        for (let i = 0; i < this.numParticles; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: Math.random() * 2 + 0.5,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                color: `rgba(${100 + Math.random() * 155}, ${100 + Math.random() * 155}, 255, ${Math.random() * 0.5 + 0.1})`
            });
        }
    }
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        for (let i = 0; i < this.particles.length; i++) {
            let p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;

            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.fill();

            // Connect nearby points
            for (let j = i + 1; j < this.particles.length; j++) {
                let p2 = this.particles[j];
                let dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                if (dist < 120) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.strokeStyle = `rgba(139, 92, 246, ${0.15 - dist / 800})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.stroke();
                }
            }
        }
        requestAnimationFrame(() => this.animate());
    }
}

// ==========================================
// 2. WIZARD INTERACTIVITY
// ==========================================
function updateVal(id) {
    const val = document.getElementById(`pred-${id}`).value;
    document.getElementById(`val-${id}`).textContent = val;
}

window.onload = () => {
    // Start particles
    new ParticleEngine('particle-canvas');

    ['sleep', 'study', 'late', 'stress', 'social', 'exercise', 'screen'].forEach(updateVal);
    updateProgress();

    const card = document.getElementById('wizard-card');

    // Spotlight Effect
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        card.style.setProperty('--x', `${x}px`);
        card.style.setProperty('--y', `${y}px`);
    });

    // 3D Tilt Effect
    document.addEventListener('mousemove', (e) => {
        if (card.classList.contains('analyzing') || window.innerWidth < 768) return;
        const xAxis = (window.innerWidth / 2 - e.pageX) / 60;
        const yAxis = (window.innerHeight / 2 - e.pageY) / 60;
        card.querySelector('.wizard-wrapper').style.transform = `perspective(1000px) rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
    });

    document.addEventListener('mouseleave', () => {
        card.querySelector('.wizard-wrapper').style.transform = `perspective(1000px) rotateY(0deg) rotateX(0deg)`;
    });
};

function updateProgress() {
    const fill = document.getElementById('progress-bar');
    const dots = document.querySelectorAll('.step-dot');

    fill.style.width = `${((currentStep - 1) / (totalSteps - 1)) * 100}%`;

    dots.forEach(dot => {
        const stepNum = parseInt(dot.getAttribute('data-step'));
        if (stepNum < currentStep) dot.className = 'step-dot completed';
        else if (stepNum === currentStep) dot.className = 'step-dot active';
        else dot.className = 'step-dot';
    });
}

function nextStep(step) {
    const currentEl = document.getElementById(`step-${step}`);
    const nextEl = document.getElementById(`step-${step + 1}`);

    currentEl.style.opacity = '0';
    currentEl.style.transform = 'translateY(20px) scale(0.95)';

    setTimeout(() => {
        currentEl.classList.remove('active');
        nextEl.classList.add('active');
        nextEl.style.opacity = '0';
        nextEl.style.transform = 'translateY(-20px) scale(1.05)';

        setTimeout(() => {
            nextEl.style.opacity = '1';
            nextEl.style.transform = 'translateY(0) scale(1)';
        }, 50);

        currentStep++;
        updateProgress();
    }, 400);
}

function prevStep(step) {
    const currentEl = document.getElementById(`step-${step}`);
    const prevEl = document.getElementById(`step-${step - 1}`);

    currentEl.style.opacity = '0';
    currentEl.style.transform = 'translateY(-20px) scale(1.05)';

    setTimeout(() => {
        currentEl.classList.remove('active');
        prevEl.classList.add('active');
        prevEl.style.opacity = '0';
        prevEl.style.transform = 'translateY(20px) scale(0.95)';

        setTimeout(() => {
            prevEl.style.opacity = '1';
            prevEl.style.transform = 'translateY(0) scale(1)';
        }, 50);

        currentStep--;
        updateProgress();
    }, 400);
}

// ==========================================
// 3. AI PREDICTION & CHART RENDERING
// ==========================================
async function submitPrediction() {
    const card = document.getElementById('wizard-card');
    const header = document.getElementById('form-header');
    const wrapper = document.querySelector('.wizard-wrapper');
    const loader = document.getElementById('predict-loader');
    const loadingSub = document.getElementById('loading-sub');
    const resultDiv = document.getElementById('prediction-result');
    const restartAct = document.getElementById('restart-action');

    card.classList.add('analyzing');
    wrapper.style.transform = 'none'; // reset tilt

    header.style.display = 'none';
    wrapper.style.display = 'none';
    loader.style.display = 'flex';
    resultDiv.classList.remove('active');

    // Fake loading steps for impressive UX
    const loaderMessages = [
        "Mapping parameters to latent space...",
        "Evaluating interconnected stress proxies...",
        "Computing multi-dimensional SHAP gradients...",
        "Finalizing personalized protocol..."
    ];

    let msgIdx = 0;
    const msgInterval = setInterval(() => {
        msgIdx = (msgIdx + 1) % loaderMessages.length;
        loadingSub.textContent = loaderMessages[msgIdx];
        loadingSub.classList.remove('fade-text');
        void loadingSub.offsetWidth; // trigger reflow
        loadingSub.classList.add('fade-text');
    }, 900);

    const payload = {
        sleep_hours: parseFloat(document.getElementById('pred-sleep').value) || 7,
        study_hours: parseFloat(document.getElementById('pred-study').value) || 5,
        late_submissions: parseInt(document.getElementById('pred-late').value) || 0,
        stress_level: parseInt(document.getElementById('pred-stress').value) || 5,
        social_activity_freq: parseInt(document.getElementById('pred-social').value) || 3,
        exercise_hours: parseFloat(document.getElementById('pred-exercise').value) || 2,
        screen_time: parseFloat(document.getElementById('pred-screen').value) || 6,
        emotional_state: document.getElementById('pred-emotion').value || 'Feeling okay',
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        setTimeout(() => {
            clearInterval(msgInterval);
            loader.style.display = 'none';
            renderResult(data);
            restartAct.style.display = 'block';
        }, 3600);

    } catch (err) {
        clearInterval(msgInterval);
        loader.style.display = 'none';
        resultDiv.innerHTML = `<div class="error-box">Prediction error: ${err.message}</div>`;
        resultDiv.classList.add('active');
        restartAct.style.display = 'block';
    }
}

function renderResult(data) {
    const resultDiv = document.getElementById('prediction-result');
    const riskClass = data.risk_level.toLowerCase();

    // Prepare interventions
    let interventionsHtml = data.interventions ? data.interventions.map(i => `<li>${i}</li>`).join('') : '';

    resultDiv.innerHTML = `
        <div class="result-header">
            <div>
                <h3 style="margin-bottom:0.3rem;">AI Analysis Complete</h3>
                <p style="color:var(--text-secondary); font-size:0.95rem;">Based on your inputs, our neural network suggests:</p>
            </div>
            <div class="risk-badge-large ${riskClass} drop-in">
                <span class="badge-label">Predicted State</span>
                <span class="badge-value">${data.risk_level} Risk</span>
            </div>
        </div>
        
        <div class="result-breakdown fade-in" style="animation-delay: 0.3s; margin-top: 2rem;">
            <div class="breakdown-col" style="position:relative;">
                <h4 class="section-title text-purple shimmer-text">SHAP Neural Impact Breakdown</h4>
                <div class="result-chart-container">
                    <canvas id="student-shap-chart"></canvas>
                </div>
            </div>
            <div class="breakdown-col">
                <h4 class="section-title text-green shimmer-text">Tailored protocol</h4>
                <ul class="interventions-list glow-list">${interventionsHtml}</ul>
            </div>
        </div>
    `;

    resultDiv.classList.add('active');

    // Render the Chart
    if (data.top_factors && data.top_factors.length > 0) {
        renderShapChart(data.top_factors);
    }
}

function renderShapChart(factors) {
    const ctx = document.getElementById('student-shap-chart').getContext('2d');

    Chart.defaults.color = '#9aa8c0';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    const labels = factors.map(f => {
        // Clean feature name dynamically
        let name = f.feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        // Truncate if too long
        return name.length > 18 ? name.substring(0, 18) + '...' : name;
    });

    const data = factors.map(f => f.shap_value);
    const colors = data.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)');
    const borders = data.map(v => v > 0 ? '#ef4444' : '#10b981');

    if (finalChart) finalChart.destroy();

    finalChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Contribution',
                data: data,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1,
                borderRadius: 4,
                barThickness: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10, 10, 26, 0.95)',
                    titleFont: { family: "'Outfit', sans-serif" },
                    padding: 12,
                    callbacks: {
                        label: (ctx) => `Impact: ${ctx.raw > 0 ? '+' : ''}${parseFloat(ctx.raw).toFixed(3)}`
                    }
                }
            }
        }
    });
}
