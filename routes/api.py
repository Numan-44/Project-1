from flask import Blueprint, request, jsonify, current_app
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from models.maths import KMeans, calculate_elbow_data
import copy

api_bp = Blueprint('api', __name__)
uploaded_data = {}

def serialize_history(history):
    serialized = []
    for step in history:
        serialized.append({
            'centroids': step['centroids'].tolist() if hasattr(step['centroids'], "tolist") else step['centroids'],
            'labels': step['labels'].tolist() if hasattr(step['labels'], "tolist") else step['labels']
        })
    return serialized

def recalculate_centroids(points, labels, k):
    """Recalculate centroids based on current point assignments"""
    centroids = []
    for cluster_id in range(k):
        cluster_points = [points[i] for i in range(len(points)) if labels[i] == cluster_id]
        if len(cluster_points) > 0:
            centroid = [np.mean([p[0] for p in cluster_points]), np.mean([p[1] for p in cluster_points])]
        else:
            # Keep previous centroid if cluster is empty
            if 'cluster_results' in uploaded_data and len(uploaded_data['cluster_results']['centroids']) > cluster_id:
                centroid = uploaded_data['cluster_results']['centroids'][cluster_id]
            else:
                centroid = [0, 0]  # Default fallback
        centroids.append(centroid)
    return centroids

def calculate_sse(points, labels, centroids):
    """Calculate Sum of Squared Errors"""
    sse = 0
    for i, point in enumerate(points):
        cluster_id = labels[i]
        centroid = centroids[cluster_id]
        sse += (point[0] - centroid[0])**2 + (point[1] - centroid[1])**2
    return sse

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Invalid file (must be CSV)'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(path)

    try:
        df = pd.read_csv(path)
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
    return jsonify({'message': 'File uploaded', 'columns': valid_cols}), 200

@api_bp.route('/columns', methods=['GET'])
def get_columns():
    if 'valid_columns' not in uploaded_data:
        return jsonify({'error': 'No file uploaded'}), 400
    return jsonify({'columns': uploaded_data['valid_columns']}), 200

@api_bp.route('/cluster', methods=['POST'])
def run_clustering():
    if 'dataframe' not in uploaded_data:
        return jsonify({'error': 'No file uploaded'}), 400

    data = request.get_json()
    x, y, k = data.get('x_column'), data.get('y_column'), data.get('k', 3)
    init_method = data.get('init', 'random')

    if not x or not y or x not in uploaded_data['valid_columns'] or y not in uploaded_data['valid_columns']:
        return jsonify({'error': 'Invalid columns'}), 400
    if k < 1 or k > 10:
        return jsonify({'error': 'k must be between 1 and 10'}), 400
    if init_method not in ['random', 'kmeans++']:
        return jsonify({'error': 'init must be random or kmeans++'}), 400

    df = uploaded_data['dataframe']
    points = df[[x, y]].dropna().values
    if len(points) < k:
        return jsonify({'error': 'Not enough points'}), 400

    kmeans = KMeans(k=k, max_iters=100, init=init_method)
    kmeans.fit(points)

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
    """Edit a single point's cluster assignment"""
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    data = request.get_json()
    point_index = data.get('point_index')
    new_cluster = data.get('new_cluster')
    
    if point_index is None or new_cluster is None:
        return jsonify({'error': 'Missing point_index or new_cluster'}), 400
    
    results = uploaded_data['cluster_results']
    
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
    
    # Recalculate centroids and SSE
    results['centroids'] = recalculate_centroids(results['data'], results['labels'], results['k'])
    results['sse'] = calculate_sse(results['data'], results['labels'], results['centroids'])
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse']
    })

@api_bp.route('/merge-clusters', methods=['POST'])
def merge_clusters():
    """Merge two clusters into one"""
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    data = request.get_json()
    cluster1 = data.get('cluster1')
    cluster2 = data.get('cluster2')
    
    if cluster1 is None or cluster2 is None or cluster1 == cluster2:
        return jsonify({'error': 'Invalid cluster selection for merge'}), 400
    
    results = uploaded_data['cluster_results']
    
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
    
    # Recalculate centroids and SSE
    results['centroids'] = recalculate_centroids(results['data'], results['labels'], results['k'])
    results['sse'] = calculate_sse(results['data'], results['labels'], results['centroids'])
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse']
    })

@api_bp.route('/split-cluster', methods=['POST'])
def split_cluster():
    """Split a cluster into two using k-means on the cluster points"""
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    data = request.get_json()
    cluster_to_split = data.get('cluster_to_split')
    
    if cluster_to_split is None:
        return jsonify({'error': 'No cluster specified for split'}), 400
    
    results = uploaded_data['cluster_results']
    
    # Get points in the cluster to split
    cluster_points = []
    cluster_indices = []
    for i, label in enumerate(results['labels']):
        if label == cluster_to_split:
            cluster_points.append(results['data'][i])
            cluster_indices.append(i)
    
    if len(cluster_points) < 2:
        return jsonify({'error': 'Cannot split cluster with less than 2 points'}), 400
    
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
    
    # Run k-means with k=2 on the cluster points
    kmeans_split = KMeans(k=2, max_iters=100, init='random')
    kmeans_split.fit(cluster_points)
    
    # Assign half of the points to the new cluster
    for i, split_label in enumerate(kmeans_split.labels_):
        if split_label == 1:  # Assign second cluster to new ID
            results['labels'][cluster_indices[i]] = new_cluster_id
    
    # Update k
    results['k'] = new_cluster_id + 1
    
    # Recalculate centroids and SSE
    results['centroids'] = recalculate_centroids(results['data'], results['labels'], results['k'])
    results['sse'] = calculate_sse(results['data'], results['labels'], results['centroids'])
    
    return jsonify({
        'success': True,
        'centroids': results['centroids'],
        'labels': results['labels'],
        'sse': results['sse'],
        'new_k': results['k']
    })

@api_bp.route('/undo-edit', methods=['POST'])
def undo_edit():
    """Undo the last edit operation"""
    if 'edit_history' not in uploaded_data or len(uploaded_data['edit_history']) == 0:
        return jsonify({'error': 'No edits to undo'}), 400
    
    results = uploaded_data['cluster_results']
    
    # Get the last edit from history
    last_edit = uploaded_data['edit_history'].pop()
    
    # Restore the previous state
    results['labels'] = last_edit['labels']
    results['centroids'] = last_edit['centroids']
    results['sse'] = last_edit['sse']
    
    # Handle k changes from split operations
    if last_edit['action'] == 'split_cluster':
        results['k'] = last_edit['k']
    
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
    """Reset all manual edits and return to original clustering"""
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    
    results = uploaded_data['cluster_results']
    
    # Restore original clustering results
    results['labels'] = copy.deepcopy(results['original_labels'])
    results['centroids'] = copy.deepcopy(results['original_centroids'])
    
    # Recalculate SSE
    results['sse'] = calculate_sse(results['data'], results['labels'], results['centroids'])
    
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
    """Return history steps for animation (iteration by iteration)."""
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    return jsonify({'history': uploaded_data['cluster_results']['history']}), 200

@api_bp.route('/plots', methods=['GET'])
def generate_plots():
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    r = uploaded_data['cluster_results']
    data = [{'x': p[0], 'y': p[1], 'cluster': r['labels'][i]} for i, p in enumerate(r['data'])]

    try:
        init_method = r.get('init', 'random')
        k_vals, sse_vals = calculate_elbow_data(r['data'], max_k=min(10, len(r['data'])), init=init_method)
        elbow = {'k_values': k_vals, 'sse_values': sse_vals}
    except:
        elbow = {}

    return jsonify({
        'points': data, 
        'centroids': r['centroids'], 
        'elbow': elbow,
        'k': r.get('k', 3)
    }), 200

@api_bp.route('/reset', methods=['POST'])
def reset_data():
    uploaded_data.clear()
    return jsonify({'message': 'Reset done'}), 200