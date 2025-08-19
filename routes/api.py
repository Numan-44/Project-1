from flask import Blueprint, request, jsonify, current_app
import os
import pandas as pd
from werkzeug.utils import secure_filename
from models.maths import KMeans, calculate_elbow_data
import json

api_bp = Blueprint('api', __name__)

# Global variable to store uploaded data
uploaded_data = {}


@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and validation"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file format (CSV only)
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file only'}), 400

        # Secure filename and save
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

        # Create uploads directory if it doesn't exist
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)

        file.save(upload_path)

        # Read and validate CSV
        try:
            df = pd.read_csv(upload_path)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400

        # Check if file is empty
        if df.empty:
            return jsonify({'error': 'The uploaded file is empty. Please upload a file with data.'}), 400

        # Filter out zero columns (columns with all zeros)
        numeric_cols = df.select_dtypes(include=['number']).columns
        valid_columns = []

        for col in numeric_cols:
            if not (df[col] == 0).all():  # Keep columns that don't have all zeros
                valid_columns.append(col)

        # Check if any valid numeric columns remain
        if len(valid_columns) < 2:
            return jsonify({'error': 'File must have at least 2 numeric columns with non-zero values'}), 400

        # Store data globally for other endpoints
        uploaded_data['dataframe'] = df
        uploaded_data['filename'] = filename
        uploaded_data['valid_columns'] = valid_columns

        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'valid_numeric_columns': len(valid_columns)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@api_bp.route('/columns', methods=['GET'])
def get_columns():
    """Get available columns for selection"""
    try:
        if 'valid_columns' not in uploaded_data:
            return jsonify({'error': 'No file uploaded yet'}), 400

        return jsonify({
            'columns': uploaded_data['valid_columns']
        }), 200

    except Exception as e:
        return jsonify({'error': f'Error retrieving columns: {str(e)}'}), 500


@api_bp.route('/cluster', methods=['POST'])
def run_clustering():
    """Perform K-means clustering"""
    try:
        # Check if data is available
        if 'dataframe' not in uploaded_data:
            return jsonify({'error': 'No file uploaded yet'}), 400

        # Get request data
        data = request.get_json()
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        k = data.get('k', 3)

        # Validate input
        if not x_column or not y_column:
            return jsonify({'error': 'Please select both X and Y axis columns'}), 400

        if x_column not in uploaded_data['valid_columns'] or y_column not in uploaded_data['valid_columns']:
            return jsonify({'error': 'Invalid column selection'}), 400

        if k < 1 or k > 10:
            return jsonify({'error': 'Number of clusters must be between 1 and 10'}), 400

        # Extract selected columns data
        df = uploaded_data['dataframe']
        cluster_data = df[[x_column, y_column]].dropna().values

        # Check if enough data points for clustering
        if len(cluster_data) < k:
            return jsonify({'error': f'Not enough data points for {k} clusters. Available: {len(cluster_data)}'}), 400

        # Perform K-means clustering
        kmeans = KMeans(k=k, max_iters=100)
        kmeans.fit(cluster_data)

        # Store clustering results for plot generation
        uploaded_data['cluster_results'] = {
            'data': cluster_data.tolist(),
            'labels': kmeans.labels_.tolist(),
            'centroids': kmeans.centroids_.tolist(),
            'sse': float(kmeans.inertia_),
            'x_column': x_column,
            'y_column': y_column,
            'k': k
        }

        return jsonify({
            'message': 'Clustering completed successfully',
            'centroids': kmeans.centroids_.tolist(),
            'sse': float(kmeans.inertia_),
            'total_points': len(cluster_data),
            'clusters': k,
            'iterations': len(kmeans.iteration_history_)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Clustering error: {str(e)}'}), 500


@api_bp.route('/plots', methods=['GET'])
def generate_plots():
    """Generate plot data for visualization"""
    try:
        # Check if clustering results are available
        if 'cluster_results' not in uploaded_data:
            return jsonify({'error': 'No clustering results available. Run clustering first.'}), 400

        results = uploaded_data['cluster_results']
        data = results['data']
        labels = results['labels']
        centroids = results['centroids']

        # Prepare cluster plot data
        cluster_plot_data = {
            'data_points': [],
            'centroids': centroids,
            'x_label': results['x_column'],
            'y_label': results['y_column']
        }

        # Group data points by cluster
        for i, point in enumerate(data):
            cluster_plot_data['data_points'].append({
                'x': point[0],
                'y': point[1],
                'cluster': labels[i]
            })

        # Generate elbow method data
        try:
            original_data = [[point[0], point[1]] for point in data]
            k_values, sse_values = calculate_elbow_data(original_data, max_k=min(10, len(data)))

            elbow_plot_data = {
                'k_values': k_values,
                'sse_values': sse_values
            }
        except Exception as e:
            elbow_plot_data = {'error': f'Could not generate elbow plot: {str(e)}'}

        return jsonify({
            'cluster_plot': cluster_plot_data,
            'elbow_plot': elbow_plot_data
        }), 200

    except Exception as e:
        return jsonify({'error': f'Plot generation error: {str(e)}'}), 500


@api_bp.route('/reset', methods=['POST'])
def reset_data():
    """Reset uploaded data and clustering results"""
    try:
        global uploaded_data
        uploaded_data.clear()

        return jsonify({'message': 'Data reset successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'Reset error: {str(e)}'}), 500