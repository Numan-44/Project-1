from flask import Blueprint, request, jsonify, current_app
import os
import pandas as pd
from werkzeug.utils import secure_filename
from models.maths import KMeans, calculate_elbow_data

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
    init_method = data.get('init', 'random')  # Get initialization method

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
        'history': serialize_history(kmeans.iteration_history_)
    }

    return jsonify({
    'centroids': kmeans.centroids_.tolist() if hasattr(kmeans.centroids_, "tolist") else kmeans.centroids_,
    'labels': kmeans.labels_.tolist() if hasattr(kmeans.labels_, "tolist") else kmeans.labels_,
    'inertia': kmeans.inertia_,
    'history': serialize_history(kmeans.iteration_history_)
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

    return jsonify({'points': data, 'centroids': r['centroids'], 'elbow': elbow}), 200


@api_bp.route('/reset', methods=['POST'])
def reset_data():
    uploaded_data.clear()
    return jsonify({'message': 'Reset done'}), 200