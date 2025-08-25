// Global variables
let clusterChart, elbowChart;
let animationHistory = [];
let currentAnimationStep = 0;
let animationTimer = null;
let isAnimating = false;

// Edit mode variables
let editMode = false;
let selectedTargetCluster = 0;
let currentK = 3;
let plotData = null;

// Colors for clusters
const clusterColors = ["red","blue","green","purple","orange","brown","pink","cyan","magenta","yellow"];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    document.getElementById("uploadForm").addEventListener("submit", handleFileUpload);
});

// Helper function to draw crosses for centroids
function drawCentroidCrosses(ctx, centroids, color) {
    const scale = clusterChart.scales;
    if (!scale.x || !scale.y) return;
    
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    
    centroids.forEach(centroid => {
        const x = scale.x.getPixelForValue(centroid[0]);
        const y = scale.y.getPixelForValue(centroid[1]);
        const size = 8;
        
        // Draw a cross
        ctx.beginPath();
        ctx.moveTo(x - size, y);
        ctx.lineTo(x + size, y);
        ctx.moveTo(x, y - size);
        ctx.lineTo(x, y + size);
        ctx.stroke();
    });
    
    ctx.restore();
}

// File upload handler
async function handleFileUpload(e) {
    e.preventDefault();
    let file = document.getElementById("fileInput").files[0];
    if (!file) return;

    let formData = new FormData();
    formData.append("file", file);

    try {
        let res = await fetch("/upload", { method: "POST", body: formData });
        let data = await res.json();

        if (res.ok) {
            document.getElementById("uploadMsg").innerText = data.message;
            let xSel = document.getElementById("xColumn");
            let ySel = document.getElementById("yColumn");
            xSel.innerHTML = ""; ySel.innerHTML = "";
            data.columns.forEach(col => {
                xSel.innerHTML += `<option value="${col}">${col}</option>`;
                ySel.innerHTML += `<option value="${col}">${col}</option>`;
            });
            document.getElementById("columnCard").style.display = "block";
            // Hide results if previously shown
            document.getElementById("resultsCard").style.display = "none";
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert("Error uploading file: " + error.message);
    }
}

// Run clustering algorithm
async function runClustering() {
    let x = document.getElementById("xColumn").value;
    let y = document.getElementById("yColumn").value;
    let k = parseInt(document.getElementById("kValue").value);
    let init = document.getElementById("initMethod").value;

    try {
        let res = await fetch("/cluster", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ x_column: x, y_column: y, k: k, init: init })
        });
        let data = await res.json();

        if (res.ok) {
            const centroidsText = data.centroids.map(centroid => `${centroid[0].toFixed(4)}, ${centroid[1].toFixed(4)}`).join('\n');
            document.getElementById("centroids").innerText = centroidsText;
            document.getElementById("sse").innerText = data.sse.toFixed(2);
            currentK = k;
            document.getElementById("currentK").innerText = currentK;

            // Store animation history
            animationHistory = data.history;
            document.getElementById("totalSteps").innerText = animationHistory.length;
            document.getElementById("currentStep").innerText = "0";
            currentAnimationStep = 0;

            // Reset edit mode
            editMode = false;
            updateEditModeUI();

            // Now fetch plots
            await refreshPlots();
            
            document.getElementById("resultsCard").style.display = "block";
            document.getElementById("editingControls").style.display = "block";
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert("Error running clustering: " + error.message);
    }
}

// Refresh plots from server
async function refreshPlots() {
    try {
        let plotRes = await fetch("/plots");
        let data = await plotRes.json();
        if (plotRes.ok) {
            plotData = data;
            currentK = data.k || currentK;
            document.getElementById("currentK").innerText = currentK;
            showClusterPlot(data.points, data.centroids);
            showElbowPlot(data.elbow.k_values, data.elbow.sse_values);
            updateClusterButtons();
        }
    } catch (error) {
        alert("Error refreshing plots: " + error.message);
    }
}

// Display cluster plot
// Display cluster plot
function showClusterPlot(points, centroids) {
    let ctx = document.getElementById("clusterPlot").getContext("2d");
    if (clusterChart) clusterChart.destroy();
    
    let datasets = [];
    let grouped = {};
    
    // Group points by cluster
    points.forEach(p => {
        if (!grouped[p.cluster]) grouped[p.cluster] = [];
        grouped[p.cluster].push({x: p.x, y: p.y});
    });
    
    // Create datasets for each cluster
    Object.keys(grouped).forEach(c => {
        datasets.push({ 
            label: "Cluster " + c, 
            data: grouped[c], 
            pointRadius: editMode ? 8 : 5,
            pointHoverRadius: editMode ? 10 : 7,
            backgroundColor: clusterColors[c % clusterColors.length],
            pointBorderWidth: editMode ? 2 : 0,
            pointBorderColor: editMode ? '#000' : 'transparent'
        });
    });
    
    // Add centroids dataset (will be drawn as crosses)
    datasets.push({ 
        label: "Centroids", 
        data: centroids.map(c => ({x: c[0], y: c[1]})), 
        pointRadius: 0,  // Make points invisible (we'll draw crosses instead)
        backgroundColor: "transparent",
        showLine: false
    });
    
    clusterChart = new Chart(ctx, { 
        type: "scatter", 
        data: { datasets },
        options: {
            onClick: editMode ? handleChartClick : null,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === datasets.length - 1) {
                                return `Centroid ${context.dataIndex}: (${context.parsed.x.toFixed(2)}, ${context.parsed.y.toFixed(2)})`;
                            } else {
                                return `Point (${context.parsed.x.toFixed(2)}, ${context.parsed.y.toFixed(2)}) - Cluster ${context.datasetIndex}`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            },
            animation: {
                onComplete: function() {
                    drawCentroidCrosses(ctx, centroids, "black");
                }
            }
        }
    });
    
    // Draw crosses immediately if animation is disabled
    if (!clusterChart.options.animation || !clusterChart.options.animation.duration) {
        drawCentroidCrosses(ctx, centroids, "black");
    }
    
    // Update cursor style
    document.getElementById("clusterPlot").style.cursor = editMode ? "crosshair" : "default";
}

// Display elbow method plot
function showElbowPlot(kVals, sseVals) {
    let ctx = document.getElementById("elbowPlot").getContext("2d");
    if (elbowChart) elbowChart.destroy();
    elbowChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: kVals,
            datasets: [{ label: "SSE", data: sseVals, borderColor: "blue", fill: false }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Number of Clusters (k)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Sum of Squared Errors'
                    }
                }
            }
        }
    });
}

// Edit Mode Functions
function toggleEditMode() {
    editMode = !editMode;
    updateEditModeUI();
    if (plotData) {
        showClusterPlot(plotData.points, plotData.centroids);
    }
}

function updateEditModeUI() {
    const editBtn = document.getElementById("editModeBtn");
    const editStatus = document.getElementById("editModeStatus");
    const pointEditSection = document.getElementById("pointEditingSection");
    
    if (editMode) {
        editBtn.textContent = "Exit Edit Mode";
        editBtn.className = "btn btn-success";
        editStatus.innerHTML = '<span class="badge bg-warning text-dark">✏️ Edit Mode Active</span>';
        pointEditSection.style.display = "block";
        updateClusterButtons();
    } else {
        editBtn.textContent = "Enter Edit Mode";
        editBtn.className = "btn btn-warning";
        editStatus.innerHTML = '<span class="badge bg-secondary">👁️ View Mode</span>';
        pointEditSection.style.display = "none";
    }
}

function updateClusterButtons() {
    const buttonContainer = document.getElementById("clusterButtons");
    buttonContainer.innerHTML = "";
    
    // Get unique clusters from current data
    let uniqueClusters = new Set();
    if (plotData && plotData.points) {
        plotData.points.forEach(p => uniqueClusters.add(p.cluster));
    }
    
    uniqueClusters = Array.from(uniqueClusters).sort((a, b) => a - b);
    
    uniqueClusters.forEach(clusterId => {
        const btn = document.createElement("button");
        btn.className = `btn btn-outline-primary btn-sm me-1 mb-1 ${selectedTargetCluster === clusterId ? 'active' : ''}`;
        btn.style.backgroundColor = selectedTargetCluster === clusterId ? clusterColors[clusterId % clusterColors.length] : 'transparent';
        btn.style.borderColor = clusterColors[clusterId % clusterColors.length];
        btn.style.color = selectedTargetCluster === clusterId ? 'white' : clusterColors[clusterId % clusterColors.length];
        btn.textContent = `Cluster ${clusterId}`;
        btn.onclick = () => selectTargetCluster(clusterId);
        buttonContainer.appendChild(btn);
    });
}

function selectTargetCluster(clusterId) {
    selectedTargetCluster = clusterId;
    updateClusterButtons();
}

async function handleChartClick(event, elements) {
    if (!editMode || elements.length === 0) return;
    
    const element = elements[0];
    const datasetIndex = element.datasetIndex;
    
    // Don't edit centroids (last dataset)
    if (datasetIndex === clusterChart.data.datasets.length - 1) return;
    
    // Find the actual point index in our data
    const pointInCluster = element.index;
    let actualPointIndex = 0;
    let currentCluster = 0;
    
    // Count points until we reach the clicked point
    for (let i = 0; i < plotData.points.length; i++) {
        if (plotData.points[i].cluster === datasetIndex) {
            if (currentCluster === pointInCluster) {
                actualPointIndex = i;
                break;
            }
            currentCluster++;
        }
    }
    
    // Edit the point
    await editPoint(actualPointIndex, selectedTargetCluster);
}

async function editPoint(pointIndex, newCluster) {
    try {
        const response = await fetch("/edit-point", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                point_index: pointIndex,
                new_cluster: newCluster
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update display
            document.getElementById("centroids").innerText = JSON.stringify(data.centroids, null, 2);
            document.getElementById("sse").innerText = data.sse.toFixed(2);
            
            // Refresh plots
            await refreshPlots();
        } else {
            alert(data.error || "Failed to edit point");
        }
    } catch (error) {
        alert("Error editing point: " + error.message);
    }
}

// Merge Functions
function showMergeDialog() {
    const modal = new bootstrap.Modal(document.getElementById('mergeModal'));
    const cluster1Select = document.getElementById('mergeCluster1');
    const cluster2Select = document.getElementById('mergeCluster2');
    
    // Populate cluster options
    cluster1Select.innerHTML = "";
    cluster2Select.innerHTML = "";
    
    let uniqueClusters = new Set();
    if (plotData && plotData.points) {
        plotData.points.forEach(p => uniqueClusters.add(p.cluster));
    }
    
    Array.from(uniqueClusters).sort((a, b) => a - b).forEach(clusterId => {
        cluster1Select.innerHTML += `<option value="${clusterId}">Cluster ${clusterId}</option>`;
        cluster2Select.innerHTML += `<option value="${clusterId}">Cluster ${clusterId}</option>`;
    });
    
    modal.show();
}

async function executeMerge() {
    const cluster1 = parseInt(document.getElementById('mergeCluster1').value);
    const cluster2 = parseInt(document.getElementById('mergeCluster2').value);
    
    if (cluster1 === cluster2) {
        alert("Cannot merge a cluster with itself!");
        return;
    }
    
    try {
        const response = await fetch("/merge-clusters", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                cluster1: cluster1,
                cluster2: cluster2
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update display
            document.getElementById("centroids").innerText = JSON.stringify(data.centroids, null, 2);
            document.getElementById("sse").innerText = data.sse.toFixed(2);
            
            // Refresh plots
            await refreshPlots();
            
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById('mergeModal')).hide();
        } else {
            alert(data.error || "Failed to merge clusters");
        }
    } catch (error) {
        alert("Error merging clusters: " + error.message);
    }
}

// Split Functions
function showSplitDialog() {
    const modal = new bootstrap.Modal(document.getElementById('splitModal'));
    const splitSelect = document.getElementById('splitCluster');
    
    // Populate cluster options
    splitSelect.innerHTML = "";
    
    let uniqueClusters = new Set();
    if (plotData && plotData.points) {
        plotData.points.forEach(p => uniqueClusters.add(p.cluster));
    }
    
    Array.from(uniqueClusters).sort((a, b) => a - b).forEach(clusterId => {
        splitSelect.innerHTML += `<option value="${clusterId}">Cluster ${clusterId}</option>`;
    });
    
    modal.show();
}

async function executeSplit() {
    const clusterToSplit = parseInt(document.getElementById('splitCluster').value);
    
    try {
        const response = await fetch("/split-cluster", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                cluster_to_split: clusterToSplit
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update display
            document.getElementById("centroids").innerText = JSON.stringify(data.centroids, null, 2);
            document.getElementById("sse").innerText = data.sse.toFixed(2);
            currentK = data.new_k;
            document.getElementById("currentK").innerText = currentK;
            
            // Refresh plots
            await refreshPlots();
            
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById('splitModal')).hide();
        } else {
            alert(data.error || "Failed to split cluster");
        }
    } catch (error) {
        alert("Error splitting cluster: " + error.message);
    }
}

// Undo Function
async function undoLastEdit() {
    try {
        const response = await fetch("/undo-edit", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update display
            document.getElementById("centroids").innerText = JSON.stringify(data.centroids, null, 2);
            document.getElementById("sse").innerText = data.sse.toFixed(2);
            if (data.k) {
                currentK = data.k;
                document.getElementById("currentK").innerText = currentK;
            }
            
            // Refresh plots
            await refreshPlots();
            
            // Show feedback
            const status = document.getElementById("editModeStatus");
            const originalHTML = status.innerHTML;
            status.innerHTML = '<span class="badge bg-info">↶ Undone: ' + data.undone_action + '</span>';
            setTimeout(() => {
                status.innerHTML = originalHTML;
            }, 2000);
        } else {
            alert(data.error || "Nothing to undo");
        }
    } catch (error) {
        alert("Error undoing edit: " + error.message);
    }
}

// Reset Function
async function resetAllEdits() {
    if (!confirm("Are you sure you want to reset all manual edits?")) return;
    
    try {
        const response = await fetch("/reset-edits", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update display
            document.getElementById("centroids").innerText = JSON.stringify(data.centroids, null, 2);
            document.getElementById("sse").innerText = data.sse.toFixed(2);
            
            // Refresh plots
            await refreshPlots();
            
            // Show feedback
            const status = document.getElementById("editModeStatus");
            const originalHTML = status.innerHTML;
            status.innerHTML = '<span class="badge bg-success">🔄 All edits reset</span>';
            setTimeout(() => {
                status.innerHTML = originalHTML;
            }, 2000);
        } else {
            alert(data.error || "Failed to reset edits");
        }
    } catch (error) {
        alert("Error resetting edits: " + error.message);
    }
}

// Animation functions
function playAnimation() {
    if (animationHistory.length === 0) return;
    
    isAnimating = true;
    document.getElementById("playBtn").disabled = true;
    document.getElementById("pauseBtn").disabled = false;
    
    animationTimer = setInterval(() => {
        showAnimationStep(currentAnimationStep);
        currentAnimationStep++;
        
        if (currentAnimationStep >= animationHistory.length) {
            pauseAnimation();
            currentAnimationStep = animationHistory.length - 1;
        }
    }, 3500 - parseInt(document.getElementById("speedControl").value));
}

function pauseAnimation() {
    isAnimating = false;
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }
    document.getElementById("playBtn").disabled = false;
    document.getElementById("pauseBtn").disabled = true;
}

function resetAnimation() {
    pauseAnimation();
    currentAnimationStep = 0;
    document.getElementById("currentStep").innerText = "0";
    if (animationHistory.length > 0) {
        showAnimationStep(0);
    }
}

function showAnimationStep(step) {
    if (step >= animationHistory.length || step < 0) return;
    
    document.getElementById("currentStep").innerText = step + 1;
    
    let historyStep = animationHistory[step];
    let centroids = historyStep.centroids;
    let labels = historyStep.labels;
    
    // Get original data points from the last clustering result
    fetch("/plots").then(res => res.json()).then(plotData => {
        if (plotData.points) {
            // Create animated plot data
            let animatedPoints = plotData.points.map((point, i) => ({
                x: point.x,
                y: point.y,
                cluster: labels.length > i ? labels[i] : 0
            }));
            
            showAnimatedClusterPlot(animatedPoints, centroids, step);
        }
    });
}

function showAnimatedClusterPlot(points, centroids, step) {
    let ctx = document.getElementById("clusterPlot").getContext("2d");
    if (clusterChart) clusterChart.destroy();
    
    let datasets = [];
    let grouped = {};
    
    points.forEach(p => {
        if (!grouped[p.cluster]) grouped[p.cluster] = [];
        grouped[p.cluster].push({x: p.x, y: p.y});
    });
    
    Object.keys(grouped).forEach(c => {
        datasets.push({ 
            label: "Cluster " + c, 
            data: grouped[c], 
            pointRadius: 5, 
            backgroundColor: clusterColors[c % clusterColors.length],
            showLine: false
        });
    });
    
    // Add centroids dataset (will be drawn as crosses)
    const centroidColor = step === 0 ? "orange" : "black";
    datasets.push({ 
        label: "Centroids", 
        data: centroids.map(c => ({x: c[0], y: c[1]})), 
        pointRadius: 0,  // Make points invisible (we'll draw crosses instead)
        backgroundColor: "transparent",
        showLine: false
    });
    
    clusterChart = new Chart(ctx, { 
        type: "scatter", 
        data: { datasets },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: step === 0 ? "Initial Centroids" : `Iteration ${step}`
                }
            },
            animation: {
                onComplete: function() {
                    drawCentroidCrosses(ctx, centroids, centroidColor);
                }
            }
        }
    });
    
    // Draw crosses immediately if animation is disabled
    if (!clusterChart.options.animation || !clusterChart.options.animation.duration) {
        drawCentroidCrosses(ctx, centroids, centroidColor);
    }
}



Chart.register(crossPlugin);