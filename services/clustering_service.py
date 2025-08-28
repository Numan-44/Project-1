import os
import pandas as pd
import numpy as np
import copy
from flask import current_app
from werkzeug.utils import secure_filename
from models.maths import KMeans, ElbowAnalyzer, ClusterCalculator


class ClusteringService:
    """Service class handling all clustering-related business logic."""
    
    def __init__(self):
        self.uploaded_data = {}
    
    def _serialize_history(self, history):
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
    
    def handle_file_upload(self, files):
        """Handle file upload for clustering data.
        
        Args:
            files: Flask request.files object
            
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'file' not in files:
            return {'data': {'error': 'No file provided'}, 'status': 400}
        
        file = files['file']
        
        # Check file extension
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file.filename == '' or file_ext not in ['.csv', '.xlsx']:
            return {'data': {'error': 'Invalid file (must be CSV or XLSX)'}, 'status': 400}

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
            return {'data': {'error': str(e)}, 'status': 400}

        if df.empty:
            return {'data': {'error': 'Empty file'}, 'status': 400}

        numeric_cols = df.select_dtypes(include=['number']).columns
        valid_cols = [c for c in numeric_cols if not (df[c] == 0).all()]

        if len(valid_cols) < 2:
            return {'data': {'error': 'Need at least 2 numeric columns'}, 'status': 400}

        self.uploaded_data['dataframe'] = df
        self.uploaded_data['valid_columns'] = valid_cols
        # Clear any previous editing state
        self.uploaded_data.pop('edit_history', None)
        
        return {
            'data': {'message': 'File uploaded', 'columns': valid_cols}, 
            'status': 200
        }
    
    def get_available_columns(self):
        """Get available numeric columns from uploaded file.
        
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'valid_columns' not in self.uploaded_data:
            return {'data': {'error': 'No file uploaded'}, 'status': 400}
        
        return {
            'data': {'columns': self.uploaded_data['valid_columns']}, 
            'status': 200
        }
    
    def perform_clustering(self, request_data):
        """Perform K-Means clustering on uploaded data.
        
        Args:
            request_data: Dict with clustering parameters
            
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'dataframe' not in self.uploaded_data:
            return {'data': {'error': 'No file uploaded'}, 'status': 400}

        x = request_data.get('x_column')
        y = request_data.get('y_column')
        k = request_data.get('k', 3)
        init_method = request_data.get('init', 'random')

        # Validation
        if (not x or not y 
                or x not in self.uploaded_data['valid_columns'] 
                or y not in self.uploaded_data['valid_columns']):
            return {'data': {'error': 'Invalid columns'}, 'status': 400}
        
        if k < 1 or k > 10:
            return {'data': {'error': 'k must be between 1 and 10'}, 'status': 400}
        
        if init_method not in ['random', 'kmeans++']:
            return {'data': {'error': 'init must be random or kmeans++'}, 'status': 400}

        df = self.uploaded_data['dataframe']
        points = df[[x, y]].dropna().values
        
        if len(points) < k:
            return {'data': {'error': 'Not enough points'}, 'status': 400}

        # Perform clustering
        kmeans = KMeans(k=k, max_iters=100, init=init_method, max_history=30)
        kmeans.fit(points)

        # Store calculator instance for reuse
        self.uploaded_data['calculator'] = ClusterCalculator(k, precision=6)

        self.uploaded_data['cluster_results'] = {
            'data': points.tolist(),
            'labels': kmeans.labels_.tolist(),
            'centroids': kmeans.centroids_.tolist(),
            'sse': float(kmeans.inertia_),
            'x': x,
            'y': y,
            'k': k,
            'init': init_method,
            'history': self._serialize_history(kmeans.iteration_history_),
            'original_labels': kmeans.labels_.tolist(),  # Store original for reset
            'original_centroids': kmeans.centroids_.tolist()
        }
        
        # Initialize edit history
        self.uploaded_data['edit_history'] = []

        return {
            'data': {
                'centroids': kmeans.centroids_.tolist(),
                'labels': kmeans.labels_.tolist(),
                'sse': float(kmeans.inertia_),
                'history': self._serialize_history(kmeans.iteration_history_)
            },
            'status': 200
        }
    
    def edit_point_assignment(self, request_data):
        """Edit a single point's cluster assignment.
        
        Args:
            request_data: Dict with point_index and new_cluster
            
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'cluster_results' not in self.uploaded_data:
            return {'data': {'error': 'No clustering results'}, 'status': 400}
        
        point_index = request_data.get('point_index')
        new_cluster = request_data.get('new_cluster')
        
        if point_index is None or new_cluster is None:
            return {'data': {'error': 'Missing point_index or new_cluster'}, 'status': 400}
        
        results = self.uploaded_data['cluster_results']
        calculator = self.uploaded_data.get('calculator')
        
        if not calculator:
            calculator = ClusterCalculator(results['k'], precision=6)
            self.uploaded_data['calculator'] = calculator
        
        # Store current state for undo
        if 'edit_history' not in self.uploaded_data:
            self.uploaded_data['edit_history'] = []
        
        self.uploaded_data['edit_history'].append({
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
        
        # Recalculate using calculator
        points_array = np.array(results['data'])
        labels_array = np.array(results['labels'])
        centroids_array = np.array(results['centroids'])
        
        new_centroids = calculator.recalculate_centroids(
            points_array, labels_array, centroids_array
        )
        new_sse = calculator.calculate_sse(
            points_array, labels_array, new_centroids
        )
        
        results['centroids'] = new_centroids.tolist()
        results['sse'] = float(new_sse)
        
        return {
            'data': {
                'success': True,
                'centroids': results['centroids'],
                'labels': results['labels'],
                'sse': results['sse']
            },
            'status': 200
        }
    
    def merge_clusters(self, request_data):
        """Merge two clusters into one.
        
        Args:
            request_data: Dict with cluster1 and cluster2
            
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'cluster_results' not in self.uploaded_data:
            return {'data': {'error': 'No clustering results'}, 'status': 400}
        
        cluster1 = request_data.get('cluster1')
        cluster2 = request_data.get('cluster2')
        
        if (cluster1 is None or cluster2 is None or cluster1 == cluster2):
            return {'data': {'error': 'Invalid cluster selection for merge'}, 'status': 400}
        
        results = self.uploaded_data['cluster_results']
        calculator = self.uploaded_data.get('calculator')
        
        if not calculator:
            calculator = ClusterCalculator(results['k'], precision=6)
            self.uploaded_data['calculator'] = calculator
        
        # Store current state for undo
        if 'edit_history' not in self.uploaded_data:
            self.uploaded_data['edit_history'] = []
        
        self.uploaded_data['edit_history'].append({
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
        
        # Recalculate using calculator
        points_array = np.array(results['data'])
        labels_array = np.array(results['labels'])
        centroids_array = np.array(results['centroids'])
        
        new_centroids = calculator.recalculate_centroids(
            points_array, labels_array, centroids_array
        )
        new_sse = calculator.calculate_sse(
            points_array, labels_array, new_centroids
        )
        
        results['centroids'] = new_centroids.tolist()
        results['sse'] = float(new_sse)
        
        return {
            'data': {
                'success': True,
                'centroids': results['centroids'],
                'labels': results['labels'],
                'sse': results['sse']
            },
            'status': 200
        }
    
    def split_cluster(self, request_data):
        """Split a cluster into two using k-means on the cluster points.
        
        Args:
            request_data: Dict with cluster_to_split
            
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'cluster_results' not in self.uploaded_data:
            return {'data': {'error': 'No clustering results'}, 'status': 400}
        
        cluster_to_split = request_data.get('cluster_to_split')
        
        if cluster_to_split is None:
            return {'data': {'error': 'No cluster specified for split'}, 'status': 400}
        
        results = self.uploaded_data['cluster_results']
        
        # Get points in the cluster to split
        cluster_points = []
        cluster_indices = []
        for i, label in enumerate(results['labels']):
            if label == cluster_to_split:
                cluster_points.append(results['data'][i])
                cluster_indices.append(i)
        
        if len(cluster_points) < 2:
            return {
                'data': {'error': 'Cannot split cluster with less than 2 points'}, 
                'status': 400
            }
        
        # Store current state for undo
        if 'edit_history' not in self.uploaded_data:
            self.uploaded_data['edit_history'] = []
        
        self.uploaded_data['edit_history'].append({
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
        kmeans_split = KMeans(k=2, max_iters=100, init='random', precision=6)
        kmeans_split.fit(np.array(cluster_points))
        
        # Assign half of the points to the new cluster
        for i, split_label in enumerate(kmeans_split.labels_):
            if split_label == 1:  # Assign second cluster to new ID
                results['labels'][cluster_indices[i]] = new_cluster_id
        
        # Update k and create new calculator
        results['k'] = new_cluster_id + 1
        calculator = ClusterCalculator(results['k'], precision=6)
        self.uploaded_data['calculator'] = calculator
        
        # Recalculate using calculator
        points_array = np.array(results['data'])
        labels_array = np.array(results['labels'])
        centroids_array = np.array(results['centroids'])
        
        new_centroids = calculator.recalculate_centroids(
            points_array, labels_array, centroids_array
        )
        new_sse = calculator.calculate_sse(
            points_array, labels_array, new_centroids
        )
        
        results['centroids'] = new_centroids.tolist()
        results['sse'] = float(new_sse)
        
        return {
            'data': {
                'success': True,
                'centroids': results['centroids'],
                'labels': results['labels'],
                'sse': results['sse'],
                'new_k': results['k']
            },
            'status': 200
        }
    
    def undo_last_edit(self):
        """Undo the last edit operation.
        
        Returns:
            Dict with 'data' and 'status' keys
        """
        if ('edit_history' not in self.uploaded_data 
                or len(self.uploaded_data['edit_history']) == 0):
            return {'data': {'error': 'No edits to undo'}, 'status': 400}
        
        results = self.uploaded_data['cluster_results']
        
        # Get the last edit from history
        last_edit = self.uploaded_data['edit_history'].pop()
        
        # Restore the previous state
        results['labels'] = last_edit['labels']
        results['centroids'] = last_edit['centroids']
        results['sse'] = last_edit['sse']
        
        # Handle k changes from split operations
        if last_edit['action'] == 'split_cluster':
            results['k'] = last_edit['k']
            self.uploaded_data['calculator'] = ClusterCalculator(results['k'], precision=6)
        
        return {
            'data': {
                'success': True,
                'centroids': results['centroids'],
                'labels': results['labels'],
                'sse': results['sse'],
                'k': results.get('k'),
                'undone_action': last_edit['action']
            },
            'status': 200
        }
    
    def reset_all_edits(self):
        """Reset all manual edits and return to original clustering.
        
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'cluster_results' not in self.uploaded_data:
            return {'data': {'error': 'No clustering results'}, 'status': 400}
        
        results = self.uploaded_data['cluster_results']
        
        # Restore original clustering results
        results['labels'] = copy.deepcopy(results['original_labels'])
        results['centroids'] = copy.deepcopy(results['original_centroids'])
        
        # Get original k from the length of original centroids
        original_k = len(results['original_centroids'])
        results['k'] = original_k
        
        # Recreate calculator with original k
        calculator = ClusterCalculator(original_k, precision=6)
        self.uploaded_data['calculator'] = calculator
        
        # Recalculate SSE
        points_array = np.array(results['data'])
        labels_array = np.array(results['labels'])
        centroids_array = np.array(results['centroids'])
        
        results['sse'] = float(calculator.calculate_sse(
            points_array, labels_array, centroids_array
        ))
        
        # Clear edit history
        self.uploaded_data['edit_history'] = []
        
        return {
            'data': {
                'success': True,
                'centroids': results['centroids'],
                'labels': results['labels'],
                'sse': results['sse']
            },
            'status': 200
        }
    
    def get_animation_history(self):
        """Return history steps for animation.
        
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'cluster_results' not in self.uploaded_data:
            return {'data': {'error': 'No clustering results'}, 'status': 400}
        
        return {
            'data': {'history': self.uploaded_data['cluster_results']['history']},
            'status': 200
        }
    
    def generate_plot_data(self):
        """Generate plot data for visualization.
        
        Returns:
            Dict with 'data' and 'status' keys
        """
        if 'cluster_results' not in self.uploaded_data:
            return {'data': {'error': 'No clustering results'}, 'status': 400}
        
        r = self.uploaded_data['cluster_results']
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

        return {
            'data': {
                'points': data, 
                'centroids': r['centroids'], 
                'elbow': elbow,
                'k': r.get('k', 3)
            },
            'status': 200
        }
    
    def reset_all_data(self):
        """Clear all uploaded data and clustering results.
        
        Returns:
            Dict with 'data' and 'status' keys
        """
        self.uploaded_data.clear()
        return {
            'data': {'message': 'Reset done'},
            'status': 200
        }