/**
 * Admin Dashboard Logic
 * Chart.js visualizations and API interaction for the staff portal
 */

Chart.defaults.color = '#8888aa';
Chart.defaults.borderColor = 'rgba(100, 100, 200, 0.1)';
Chart.defaults.font.family = "'Inter', sans-serif";

let allStudents = [];
let distributionChart, trendChart, featureChart;

document.addEventListener('DOMContentLoaded', () => {
    loadDashboard();
});

async function loadDashboard() {
    try {
        await Promise.all([
            loadOverview(),
            loadTrends(),
            loadFeatures(),
            loadStudentList(),
        ]);

        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.classList.add('hidden');
    } catch (err) {
        console.error('Dashboard load error:', err);
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.querySelector('.loading-text').textContent = 'Error loading data. Please ensure models are trained.';
        }
    }
}

async function loadOverview() {
    const res = await fetch('/api/overview');
    const data = await res.json();

    document.getElementById('stat-total').textContent = data.total_students;
    document.getElementById('stat-low').textContent = data.risk_counts.Low;
    document.getElementById('stat-medium').textContent = data.risk_counts.Medium;
    document.getElementById('stat-high').textContent = data.risk_counts.High;

    const ctx = document.getElementById('distribution-chart').getContext('2d');
    distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [data.risk_counts.Low, data.risk_counts.Medium, data.risk_counts.High],
                backgroundColor: ['#00e68a', '#ffc857', '#ff4757'],
                borderColor: ['rgba(0,230,138,0.3)', 'rgba(255,200,87,0.3)', 'rgba(255,71,87,0.3)'],
                borderWidth: 2,
                hoverOffset: 10,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, pointStyleWidth: 12, font: { size: 12, weight: '500' } } },
                tooltip: { backgroundColor: 'rgba(17, 17, 40, 0.95)', titleFont: { weight: '600' }, bodyFont: { size: 13 }, padding: 14, cornerRadius: 10, borderColor: 'rgba(100, 100, 200, 0.2)', borderWidth: 1, callbacks: { label: (ctx) => ` ${ctx.raw} students (${data.percentages[ctx.label.replace(' Risk', '')]}%)` } }
            }
        }
    });
}

async function loadTrends() {
    const res = await fetch('/api/trends');
    const data = await res.json();

    const weeks = Object.keys(data.trends).map(w => `Week ${w}`);
    const low = Object.values(data.trends).map(t => t.Low);
    const medium = Object.values(data.trends).map(t => t.Medium);
    const high = Object.values(data.trends).map(t => t.High);

    const ctx = document.getElementById('trend-chart').getContext('2d');
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: weeks,
            datasets: [
                { label: 'Low Risk %', data: low, borderColor: '#00e68a', backgroundColor: 'rgba(0, 230, 138, 0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2.5 },
                { label: 'Medium Risk %', data: medium, borderColor: '#ffc857', backgroundColor: 'rgba(255, 200, 87, 0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2.5 },
                { label: 'High Risk %', data: high, borderColor: '#ff4757', backgroundColor: 'rgba(255, 71, 87, 0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2.5 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: 'index' },
            scales: { y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' }, grid: { color: 'rgba(100, 100, 200, 0.06)' } }, x: { grid: { display: false } } },
            plugins: { legend: { position: 'top', labels: { padding: 16, usePointStyle: true, pointStyleWidth: 10, font: { size: 11 } } } }
        }
    });
}

async function loadFeatures() {
    const res = await fetch('/api/features');
    const data = await res.json();

    const top10 = data.features.slice(0, 10);
    const labels = top10.map(f => formatFeatureName(f.feature));
    const values = top10.map(f => f.importance);

    const colors = top10.map((_, i) => {
        const ratio = i / (top10.length - 1);
        const r = Math.round(74 + ratio * (124 - 74));
        const g = Math.round(127 + ratio * (92 - 127));
        const b = Math.round(255 + ratio * (252 - 255));
        return `rgba(${r}, ${g}, ${b}, 0.75)`;
    });

    const ctx = document.getElementById('feature-chart').getContext('2d');
    featureChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: 'SHAP Importance', data: values, backgroundColor: colors, borderColor: colors.map(c => c.replace('0.75', '1')), borderWidth: 1, borderRadius: 6, barThickness: 22 }] },
        options: {
            responsive: true, maintainAspectRatio: false, indexAxis: 'y',
            scales: { x: { beginAtZero: true, grid: { color: 'rgba(100, 100, 200, 0.06)' }, ticks: { font: { size: 11 } } }, y: { grid: { display: false }, ticks: { font: { size: 11, weight: '500' } } } },
            plugins: { legend: { display: false }, tooltip: { backgroundColor: 'rgba(17, 17, 40, 0.95)', padding: 12, cornerRadius: 10, borderColor: 'rgba(100, 100, 200, 0.2)', borderWidth: 1 } }
        }
    });
}

async function loadStudentList() {
    const res = await fetch('/api/students');
    const data = await res.json();
    allStudents = data.students;
    renderStudentTable(allStudents);
}

function renderStudentTable(students) {
    const tbody = document.getElementById('student-tbody');
    tbody.innerHTML = '';

    students.forEach(s => {
        const tr = document.createElement('tr');
        const riskClass = s.risk_level.toLowerCase();
        tr.innerHTML = `<td>${s.student_id}</td><td><span class="risk-dot ${riskClass}"></span>${s.risk_level}</td>`;
        tr.addEventListener('click', () => lookupStudent(s.student_id));
        tbody.appendChild(tr);
    });
}

function searchStudent() {
    const id = document.getElementById('student-search').value.trim();
    const filter = document.getElementById('risk-filter').value;

    if (id) lookupStudent(id);

    let filtered = allStudents;
    if (filter) filtered = allStudents.filter(s => s.risk_level === filter);
    if (id) filtered = filtered.filter(s => s.student_id.toLowerCase().includes(id.toLowerCase()));

    renderStudentTable(filtered);
}

async function lookupStudent(studentId) {
    const resultDiv = document.getElementById('student-result');

    try {
        const res = await fetch(`/api/student/${studentId}`);
        if (!res.ok) {
            resultDiv.innerHTML = `<p style="color: var(--accent-red);">Student not found.</p>`;
            resultDiv.classList.add('active');
            return;
        }
        const data = await res.json();
        const riskClass = data.risk_level.toLowerCase();

        let factorsHtml = data.top_factors ? data.top_factors.map(f => `<li>${f.description} <span class="factor-value">SHAP: ${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(3)}</span></li>`).join('') : '';
        let interventionsHtml = data.interventions ? data.interventions.map(i => `<li>${i}</li>`).join('') : '';

        resultDiv.innerHTML = `
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                <h3 style="margin:0;">${data.student_id}</h3>
                <span class="risk-badge ${riskClass}">${data.risk_level} Risk</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
                <div><h4 class="section-title text-purple">Top Contributing Factors</h4><ul class="factors-list">${factorsHtml}</ul></div>
                <div><h4 class="section-title text-green">Recommended Interventions</h4><ul class="interventions-list">${interventionsHtml}</ul></div>
            </div>
        `;
        resultDiv.classList.add('active');
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
        resultDiv.innerHTML = `<p style="color: var(--accent-red);">Error: ${err.message}</p>`;
        resultDiv.classList.add('active');
    }
}

function formatFeatureName(name) {
    const map = {
        'stress_level': 'Stress Level', 'sleep_hours': 'Sleep Hours', 'sleep_irregularity': 'Sleep Irregularity',
        'late_submissions': 'Late Submissions', 'study_hours': 'Study Hours', 'social_activity_freq': 'Social Activity',
        'social_isolation_score': 'Social Isolation', 'sentiment_polarity': 'Sentiment Polarity', 'sentiment_subjectivity': 'Sentiment Subjectivity',
        'has_negative_emotion': 'Negative Emotions', 'has_positive_emotion': 'Positive Emotions', 'procrastination_score': 'Procrastination',
        'negative_sentiment_trend': 'Neg. Sentiment Trend', 'stress_sleep_interaction': 'Stress × Sleep Deficit', 'study_overload': 'Study Overload',
        'screen_study_ratio': 'Screen/Study Ratio', 'exercise_hours': 'Exercise Hours', 'screen_time': 'Screen Time',
    };
    return map[name] || name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
