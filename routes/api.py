from flask import Blueprint, request, jsonify, current_app
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from models.maths import KMeans, ElbowAnalyzer, ClusterCalculator
import copy

api_bp = Blueprint('api', __name__)
uploaded_data = {}


def serialize_history(history):
    """Serialize iteration history for JSON response.
    
    Args:
        history: List of iteration step dictionaries
        
    Returns:
        List of serialized iteration steps with centroids and labels
    """
    serialized = []
    for step in history:
        serialized.append({
            'centroids': (
                step['centroids'].tolist() 
                if hasattr(step['centroids'], "tolist") 
                else step['centroids']
            ),
            'labels': (
                step['labels'].tolist() 
                if hasattr(step['labels'], "tolist") 
                else step['labels']
            )
        })
    return serialized


@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for clustering data.
    
    Supports CSV and XLSX files with at least 2 numeric columns.
    
    Returns:
        JSON response with upload status and available columns
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    
    # Check file extension
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file.filename == '' or file_ext not in ['.csv', '.xlsx']:
        return jsonify({'error': 'Invalid file (must be CSV or XLSX)'}), 400

    path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(path)

    try:
        # Read file based on extension
        if file_ext == '.csv':
            df = pd.read_csv(path)
        else:  # .xlsx
            df = pd.read_excel(path)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    if df.empty:
        return jsonify({'error': 'Empty file'}), 400

    numeric_cols = df.select_dtypes(include=['number']).columns
    valid_cols = [c for c in numeric_cols if not (df[c] == 0).all()]

    if len(valid_cols) < 2:
        return jsonify({'error': 'Need at least 2 numeric columns'}), 400

    uploaded_data['dataframe'] = df
    uploaded_data['valid_columns'] = valid_cols
    # Clear any previous editing state
    uploaded_data.pop('edit_history', None)
    return jsonify({
        'message': 'File uploaded', 
        'columns': valid_cols
    }), 200


@api_bp.route('/columns', methods=['GET'])
def get_columns():
    """Get available numeric columns from uploaded file.
    
    Returns:
        JSON response with list of valid numeric columns
    """
    if 'valid_columns' not in uploaded_data:
        return jsonify({'error': 'No file uploaded'}), 400
    return jsonify({'columns': uploaded_data['valid_columns']}), 200


@api_bp.route('/cluster', methods=['POST'])
def run_clustering():
    """Perform K-Means clustering on uploaded data.
    
    Expects JSON with x_column, y_column, k (clusters), and init method.
    
    Returns:
        JSON response with clustering results including centroids, labels, and SSE
    """
    if 'dataframe' not in uploaded_data:
        return jsonify({'error': 'No file uploaded'}), 400

    data = request.get_json()
    x = data.get('x_column')
    y = data.get('y_column')
    k = data.get('k', 3)
    init_method = data.get('init', 'random')

    if (not x or not y 
            or x not in uploaded_data['valid_columns'] 
            or y not in uploaded_data['valid_columns']):
        return jsonify({'error': 'Invalid columns'}), 400
    if k < 1 or k > 10:
        return jsonify({'error': 'k must be between 1 and 10'}), 400
    if init_method not in ['random', 'kmeans++']:
        return jsonify({'error': 'init must be random or kmeans++'}), 400

    df = uploaded_data['dataframe']
    points = df[[x, y]].dropna().values
    if len(points) < k:
        return jsonify({'error': 'Not enough points'}), 400

    kmeans = KMeans(k=k, max_iters=100, init=init_method, max_history=30)
    kmeans.fit(points)

    # Store calculator instance for reuse
    uploaded_data['calculator'] = ClusterCalculator(k, precision=6)

    uploaded_data['cluster_results'] = {
        'data': points.tolist(),
        'labels': kmeans.labels_.tolist(),
        'centroids': kmeans.centroids_.tolist(),
        'sse': float(kmeans.inertia_),
        'x': x,
        'y': y,
        'k': k,
        'init': init_method,
        'history': serialize_history(kmeans.iteration_history_),
        'original_labels': kmeans.labels_.tolist(),  # Store original for reset
        'original_centroids': kmeans.centroids_.tolist()
    }
    
    # Initialize edit history
    uploaded_data['edit_history'] = []

    return jsonify({
        'centroids': kmeans.centroids_.tolist(),
        'labels': kmeans.labels_.tolist(),
        'sse': float(kmeans.inertia_),
        'history': serialize_history(kmeans.iteration_history_)
    })


@api_bp.route('/edit-point', methods=['POST'])
def edit_point():
    """Edit a single point's cluster assignment.
    
    Expects JSON with point_index and new_cluster.
    
    Returns:
        JSON response with updated clustering results
    """
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    data = request.get_json()
    point_index = data.get('point_index')
    new_cluster = data.get('new_cluster')
    
    if point_index is None or new_cluster is None:
        return jsonify({'error': 'Missing point_index or new_cluster'}), 400
    
    results = uploaded_data['cluster_results']
    calculator = uploaded_data.get('calculator')
    
    if not calculator:
        calculator = ClusterCalculator(results['k'], precision=6)
        uploaded_data['calculator'] = calculator
    
    # Store current state for undo
    if 'edit_history' not in uploaded_data:
        uploaded_data['edit_history'] = []
    
    uploaded_data['edit_history'].append({
        'action': 'edit_point',
        'labels': copy.deepcopy(results['labels']),
        'centroids': copy.deepcopy(results['centroids']),
        'sse': results['sse'],
        'point_index': point_index,
        'old_cluster': results['labels'][point_index],
        'new_cluster': new_cluster
    })
    
    # Update the point's cluster
    results['labels'][point_index] = new_cluster
    
    # Use ClusterCalculator for recalculation
    points_array = np.array(results['data'])
    labels_array = np.array(results['labels'])
    centroids_array = np.array(results['centroids'])
    
    # Recalculate centroids and SSE using the calculator
    new_centroids = calculator.recalculate_centroids(
        points_array, labels_array, centroids_array
    )
    new_sse = calculator.calculate_sse(
        points_array, labels_array, new_centroids
    )
    
    results['centroids'] = new_centroids.tolist()
    results['sse'] = float(new_sse)
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse']
    })


@api_bp.route('/merge-clusters', methods=['POST'])
def merge_clusters():
    """Merge two clusters into one.
    
    Expects JSON with cluster1 and cluster2 to merge.
    
    Returns:
        JSON response with updated clustering results after merge
    """
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    data = request.get_json()
    cluster1 = data.get('cluster1')
    cluster2 = data.get('cluster2')
    
    if (cluster1 is None or cluster2 is None 
            or cluster1 == cluster2):
        return jsonify({'error': 'Invalid cluster selection for merge'}), 400
    
    results = uploaded_data['cluster_results']
    calculator = uploaded_data.get('calculator')
    
    if not calculator:
        calculator = ClusterCalculator(results['k'], precision=6)
        uploaded_data['calculator'] = calculator
    
    # Store current state for undo
    if 'edit_history' not in uploaded_data:
        uploaded_data['edit_history'] = []
    
    uploaded_data['edit_history'].append({
        'action': 'merge_clusters',
        'labels': copy.deepcopy(results['labels']),
        'centroids': copy.deepcopy(results['centroids']),
        'sse': results['sse'],
        'cluster1': cluster1,
        'cluster2': cluster2
    })
    
    # Merge cluster2 into cluster1
    for i in range(len(results['labels'])):
        if results['labels'][i] == cluster2:
            results['labels'][i] = cluster1
    
    # Use ClusterCalculator for recalculation
    points_array = np.array(results['data'])
    labels_array = np.array(results['labels'])
    centroids_array = np.array(results['centroids'])
    
    # Recalculate centroids and SSE using the calculator
    new_centroids = calculator.recalculate_centroids(
        points_array, labels_array, centroids_array
    )
    new_sse = calculator.calculate_sse(
        points_array, labels_array, new_centroids
    )
    
    results['centroids'] = new_centroids.tolist()
    results['sse'] = float(new_sse)
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse']
    })


@api_bp.route('/split-cluster', methods=['POST'])
def split_cluster():
    """Split a cluster into two using k-means on the cluster points.
    
    Expects JSON with cluster_to_split.
    
    Returns:
        JSON response with updated clustering results after split
    """
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    data = request.get_json()
    cluster_to_split = data.get('cluster_to_split')
    
    if cluster_to_split is None:
        return jsonify({'error': 'No cluster specified for split'}), 400
    
    results = uploaded_data['cluster_results']
    calculator = uploaded_data.get('calculator')
    
    # Get points in the cluster to split
    cluster_points = []
    cluster_indices = []
    for i, label in enumerate(results['labels']):
        if label == cluster_to_split:
            cluster_points.append(results['data'][i])
            cluster_indices.append(i)
    
    if len(cluster_points) < 2:
        return jsonify({
            'error': 'Cannot split cluster with less than 2 points'
        }), 400
    
    # Store current state for undo
    if 'edit_history' not in uploaded_data:
        uploaded_data['edit_history'] = []
    
    uploaded_data['edit_history'].append({
        'action': 'split_cluster',
        'labels': copy.deepcopy(results['labels']),
        'centroids': copy.deepcopy(results['centroids']),
        'sse': results['sse'],
        'cluster_to_split': cluster_to_split,
        'k': results['k']
    })
    
    # Find next available cluster ID
    max_cluster = max(results['labels'])
    new_cluster_id = max_cluster + 1
    
    # Run k-means with k=2 on the cluster points using the KMeans class
    kmeans_split = KMeans(k=2, max_iters=100, init='random', precision=6)
    kmeans_split.fit(np.array(cluster_points))
    
    # Assign half of the points to the new cluster
    for i, split_label in enumerate(kmeans_split.labels_):
        if split_label == 1:  # Assign second cluster to new ID
            results['labels'][cluster_indices[i]] = new_cluster_id
    
    # Update k and create new calculator
    results['k'] = new_cluster_id + 1
    calculator = ClusterCalculator(results['k'], precision=6)
    uploaded_data['calculator'] = calculator
    
    # Use ClusterCalculator for recalculation
    points_array = np.array(results['data'])
    labels_array = np.array(results['labels'])
    centroids_array = np.array(results['centroids'])
    
    # Recalculate centroids and SSE using the calculator
    new_centroids = calculator.recalculate_centroids(
        points_array, labels_array, centroids_array
    )
    new_sse = calculator.calculate_sse(
        points_array, labels_array, new_centroids
    )
    
    results['centroids'] = new_centroids.tolist()
    results['sse'] = float(new_sse)
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse'],
        'new_k': results['k']
    })


@api_bp.route('/undo-edit', methods=['POST'])
def undo_edit():
    """Undo the last edit operation.
    
    Returns:
        JSON response with restored clustering state
    """
    if ('edit_history' not in uploaded_data 
            or len(uploaded_data['edit_history']) == 0):
        return jsonify({'error': 'No edits to undo'}), 400
    
    results = uploaded_data['cluster_results']
    
    # Get the last edit from history
    last_edit = uploaded_data['edit_history'].pop()
    
    # Restore the previous state
    results['labels'] = last_edit['labels']
    results['centroids'] = last_edit['centroids']
    results['sse'] = last_edit['sse']
    
    # Handle k changes from split operations and update calculator
    if last_edit['action'] == 'split_cluster':
        results['k'] = last_edit['k']
        uploaded_data['calculator'] = ClusterCalculator(results['k'], precision=6)
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse'],
        'k': results.get('k'),
        'undone_action': last_edit['action']
    })


@api_bp.route('/reset-edits', methods=['POST'])
def reset_edits():
    """Reset all manual edits and return to original clustering.
    
    Returns:
        JSON response with original clustering results
    """
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    results = uploaded_data['cluster_results']
    
    # Restore original clustering results
    results['labels'] = copy.deepcopy(results['original_labels'])
    results['centroids'] = copy.deepcopy(results['original_centroids'])
    
    # Get original k from the length of original centroids
    original_k = len(results['original_centroids'])
    results['k'] = original_k
    
    # Recreate calculator with original k
    calculator = ClusterCalculator(original_k, precision=6)
    uploaded_data['calculator'] = calculator
    
    # Recalculate SSE using the calculator
    points_array = np.array(results['data'])
    labels_array = np.array(results['labels'])
    centroids_array = np.array(results['centroids'])
    
    results['sse'] = float(calculator.calculate_sse(
        points_array, labels_array, centroids_array
    ))
    
    # Clear edit history
    uploaded_data['edit_history'] = []
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse']
    })


@api_bp.route('/animation', methods=['GET'])
def get_animation_steps():
    """Return history steps for animation (iteration by iteration).
    
    Returns:
        JSON response with clustering iteration history
    """
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    return jsonify({
        'history': uploaded_data['cluster_results']['history']
    }), 200


@api_bp.route('/plots', methods=['GET'])
def generate_plots():
    """Generate plot data for visualization.
    
    Returns:
        JSON response with points, centroids, and elbow analysis data
    """
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    r = uploaded_data['cluster_results']
    data = [{
        'x': p[0], 
        'y': p[1], 
        'cluster': r['labels'][i]
    } for i, p in enumerate(r['data'])]

    try:
        init_method = r.get('init', 'random')
        k_vals, sse_vals = ElbowAnalyzer.calculate_elbow_data(
            r['data'], 
            max_k=min(10, len(r['data'])), 
            init=init_method
        )
        elbow = {'k_values': k_vals, 'sse_values': sse_vals}
    except Exception:
        elbow = {}

    return jsonify({
        'points': data, 
        'centroids': r['centroids'], 
        'elbow': elbow,
        'k': r.get('k', 3)
    }), 200


@api_bp.route('/reset', methods=['POST'])
def reset_data():
    """Clear all uploaded data and clustering results.
    
    Returns:
        JSON confirmation of reset operation
    """
    uploaded_data.clear()
    return jsonify({'message': 'Reset done'}), 200