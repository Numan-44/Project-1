from flask import Blueprint, request, jsonify, current_app
import os
import pandas as pd
from werkzeug.utils import secure_filename
from models.maths import KMeans, calculate_elbow_data

api_bp = Blueprint('api', __name__)
uploaded_data = {}

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV allowed'}), 400
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
    if not x or not y or x not in uploaded_data['valid_columns'] or y not in uploaded_data['valid_columns']:
        return jsonify({'error': 'Invalid columns'}), 400
    if k < 1 or k > 10:
        return jsonify({'error': 'k must be 1-10'}), 400
    df = uploaded_data['dataframe']
    points = df[[x, y]].dropna().values
    if len(points) < k:
        return jsonify({'error': 'Not enough points'}), 400
    kmeans = KMeans(k=k, max_iters=100)
    kmeans.fit(points)
    uploaded_data['cluster_results'] = {
        'data': points.tolist(),
        'labels': kmeans.labels_.tolist(),
        'centroids': kmeans.centroids_.tolist(),
        'sse': float(kmeans.inertia_),
        'x': x,
        'y': y,
        'k': k
    }
    return jsonify({'message': 'Clustering done', 'centroids': kmeans.centroids_.tolist(), 'sse': float(kmeans.inertia_)}), 200

@api_bp.route('/plots', methods=['GET'])
def generate_plots():
    if 'cluster_results' not in uploaded_data:
        return jsonify({'error': 'No clustering results'}), 400
    r = uploaded_data['cluster_results']
    data = [{'x': p[0], 'y': p[1], 'cluster': r['labels'][i]} for i, p in enumerate(r['data'])]
    try:
        k_vals, sse_vals = calculate_elbow_data(r['data'], max_k=min(10, len(r['data'])))
        elbow = {'k_values': k_vals, 'sse_values': sse_vals}
    except:
        elbow = {}
    return jsonify({'points': data, 'centroids': r['centroids'], 'elbow': elbow}), 200

@api_bp.route('/reset', methods=['POST'])
def reset_data():
    uploaded_data.clear()
    return jsonify({'message': 'Reset done'}), 200
