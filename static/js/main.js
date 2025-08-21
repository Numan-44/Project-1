const API_BASE = '/api';
let uploadedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('csvFile');
    const clusterSlider = document.getElementById('clusterCount');
    const clusterValue = document.getElementById('clusterValue');
    const runBtn = document.getElementById('runClustering');
    const backBtn = document.getElementById('backBtn');

    if (uploadBox) {
        uploadBox.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileUpload);
    }

    if (clusterSlider) {
        clusterSlider.addEventListener('input', (e) => {
            clusterValue.textContent = e.target.value;
        });
    }

    if (runBtn) {
        runBtn.addEventListener('click', runClustering);
    }

    if (backBtn) {
        backBtn.addEventListener('click', () => window.location.href = '/');
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showStatus('Uploading file...', 'info');

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            showStatus(`File uploaded successfully! ${result.total_rows} rows, ${result.valid_numeric_columns} numeric columns`, 'success');
            uploadedFile = file.name;
            await loadColumns();
            showConfigSection();
        } else {
            showStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Upload failed: ${error.message}`, 'error');
    }
}

async function loadColumns() {
    try {
        const response = await fetch(`${API_BASE}/columns`);
        const result = await response.json();

        if (response.ok) {
            populateColumnSelects(result.columns);
        } else {
            showStatus(`Error loading columns: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Failed to load columns: ${error.message}`, 'error');
    }
}

function populateColumnSelects(columns) {
    const xSelect = document.getElementById('xAxis');
    const ySelect = document.getElementById('yAxis');

    xSelect.innerHTML = '<option value="">Select X-Axis</option>';
    ySelect.innerHTML = '<option value="">Select Y-Axis</option>';

    columns.forEach(column => {
        xSelect.innerHTML += `<option value="${column}">${column}</option>`;
        ySelect.innerHTML += `<option value="${column}">${column}</option>`;
    });
}

function showConfigSection() {
    document.getElementById('configSection').style.display = 'block';
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.textContent = message;
    statusDiv.className = `status-message ${type}`;
    statusDiv.style.display = 'block';
}

async function runClustering() {
    const xAxis = document.getElementById('xAxis').value;
    const yAxis = document.getElementById('yAxis').value;
    const k = document.getElementById('clusterCount').value;

    if (!xAxis || !yAxis) {
        showStatus('Please select both X and Y axes', 'error');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/cluster`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x_column: xAxis,
                y_column: yAxis,
                k: parseInt(k)
            })
        });

        const result = await response.json();

        if (response.ok) {
            showStatus(`Clustering completed! ${result.iterations} iterations, SSE: ${result.sse.toFixed(2)}`, 'success');
            setTimeout(() => window.location.href = '/results', 1000);
        } else {
            showStatus(`Clustering failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
    document.getElementById('configSection').style.display = show ? 'none' : 'block';
}