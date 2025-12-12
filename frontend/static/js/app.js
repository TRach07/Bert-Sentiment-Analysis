const API_BASE_URL = 'http://localhost:8000';

let currentFileId = null;
let pieChart = null;
let barChart = null;

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const modelSelect = document.getElementById('modelSelect');
    const textColumn = document.getElementById('textColumn');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadSpinner = document.getElementById('uploadSpinner');
    
    if (!fileInput.files[0]) {
        alert('Please select a file');
        return;
    }
    
    uploadBtn.disabled = true;
    uploadSpinner.classList.remove('d-none');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', modelSelect.value);
    if (textColumn.value.trim()) {
        formData.append('text_column', textColumn.value.trim());
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error during upload');
        }
        
        const result = await response.json();
        currentFileId = result.file_id;
        
        await loadResults(result.file_id);
        displayStats(result.stats);
        document.getElementById('resultsSection').classList.remove('d-none');
        
        alert(`✅ ${result.message}`);
        
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    } finally {
        uploadBtn.disabled = false;
        uploadSpinner.classList.add('d-none');
    }
}

function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length === 0) return [];
    
    const headers = [];
    let headerLine = lines[0];
    let inQuotes = false;
    let currentField = '';
    
    for (let i = 0; i < headerLine.length; i++) {
        const char = headerLine[i];
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            headers.push(currentField.trim());
            currentField = '';
        } else {
            currentField += char;
        }
    }
    headers.push(currentField.trim());
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (!line.trim()) continue;
        
        const row = {};
        let inQuotes = false;
        let currentField = '';
        let fieldIndex = 0;
        
        for (let j = 0; j < line.length; j++) {
            const char = line[j];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                if (fieldIndex < headers.length) {
                    row[headers[fieldIndex]] = currentField.trim();
                }
                currentField = '';
                fieldIndex++;
            } else {
                currentField += char;
            }
        }
        if (fieldIndex < headers.length) {
            row[headers[fieldIndex]] = currentField.trim();
        }
        
        data.push(row);
    }
    
    return data;
}

async function loadResults(fileId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/results/${fileId}/download`);
        const text = await response.text();
        
        const data = parseCSV(text);
        displayResults(data);
        
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

function displayStats(stats) {
    document.getElementById('statTotal').textContent = stats.total;
    document.getElementById('statNegative').textContent = stats.negative;
    document.getElementById('statNeutral').textContent = stats.neutral;
    document.getElementById('statPositive').textContent = stats.positive;
    
    updateCharts(stats);
}

function updateCharts(stats) {
    const ctxPie = document.getElementById('pieChart');
    const ctxBar = document.getElementById('barChart');
    
    if (pieChart) {
        pieChart.destroy();
    }
    pieChart = new Chart(ctxPie, {
        type: 'pie',
        data: {
            labels: ['Negative', 'Neutral', 'Positive'],
            datasets: [{
                data: [stats.negative, stats.neutral, stats.positive],
                backgroundColor: ['#dc3545', '#6c757d', '#28a745']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Sentiment Distribution'
                }
            }
        }
    });
    
    if (barChart) {
        barChart.destroy();
    }
    barChart = new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: ['Negative', 'Neutral', 'Positive'],
            datasets: [{
                label: 'Number of reviews',
                data: [stats.negative, stats.neutral, stats.positive],
                backgroundColor: ['#dc3545', '#6c757d', '#28a745']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Sentiment Breakdown'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function getSentimentClass(sentiment) {
    const normalized = sentiment.toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '');
    return normalized;
}

function getSentimentColor(sentiment) {
    const normalized = getSentimentClass(sentiment);
    if (normalized === 'negative') return '#dc3545';
    if (normalized === 'neutral') return '#6c757d';
    if (normalized === 'positive') return '#28a745';
    return '#6c757d';
}

function displayResults(data) {
    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = '';
    
    const textColumn = data[0] && Object.keys(data[0]).find(key => 
        !['sentiment', 'sentiment_id', 'confidence'].includes(key)
    );
    
    data.forEach((row) => {
        const tr = document.createElement('tr');
        
        const text = row[textColumn] || '';
        const sentiment = row.sentiment || 'N/A';
        const confidence = parseFloat(row.confidence || 0);
        const sentimentClass = getSentimentClass(sentiment);
        const sentimentColor = getSentimentColor(sentiment);
        const confidencePercent = Math.min(confidence * 100, 100);
        
        const escapedText = text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        
        tr.innerHTML = `
            <td style="max-width: 500px; word-wrap: break-word; white-space: normal;">
                <div style="max-height: 80px; overflow-y: auto; padding: 5px;">
                    ${escapedText}
                </div>
            </td>
            <td style="white-space: nowrap;">
                <span class="sentiment-badge sentiment-${sentimentClass}">
                    ${sentiment}
                </span>
            </td>
            <td style="font-weight: bold; white-space: nowrap;">${confidencePercent.toFixed(1)}%</td>
        `;
        
        tbody.appendChild(tr);
    });
    
    const searchInput = document.getElementById('searchInput');
    searchInput.oninput = null;
    searchInput.addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const rows = tbody.getElementsByTagName('tr');
        
        for (let row of rows) {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(searchTerm) ? '' : 'none';
        }
    });
}

function downloadResults() {
    if (currentFileId) {
        window.open(`${API_BASE_URL}/api/results/${currentFileId}/download`, '_blank');
    } else {
        alert('No file to download');
    }
}

async function analyzeRealtime() {
    const textInput = document.getElementById('realtimeText');
    const modelSelect = document.getElementById('realtimeModel');
    const resultDiv = document.getElementById('realtimeResult');
    
    if (!textInput.value.trim()) {
        alert('Please enter some text');
        return;
    }
    
    resultDiv.innerHTML = '<div class="spinner-border" role="status"></div> Loading...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict/single`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: textInput.value,
                model: modelSelect.value
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error during prediction');
        }
        
        const result = await response.json();
        const pred = result.prediction;
        
        resultDiv.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h5>Analysis Result</h5>
                    <p><strong>Sentiment:</strong> 
                        <span class="sentiment-badge sentiment-${getSentimentClass(pred.sentiment)}">
                            ${pred.sentiment}
                        </span>
                    </p>
                    <p><strong>Confidence:</strong> ${(pred.confidence * 100).toFixed(1)}%</p>
                    <h6>Probabilities:</h6>
                    <ul>
                        <li>Negative: ${(pred.probabilities.Negative * 100).toFixed(1)}%</li>
                        <li>Neutral: ${(pred.probabilities.Neutral * 100).toFixed(1)}%</li>
                        <li>Positive: ${(pred.probabilities.Positive * 100).toFixed(1)}%</li>
                    </ul>
                </div>
            </div>
        `;
        
    } catch (error) {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
}

window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const status = await response.json();
        console.log('API Status:', status);
    } catch (error) {
        console.error('API not available:', error);
        alert('⚠️ Backend API is not available. Make sure it is running on http://localhost:8000');
    }
});
