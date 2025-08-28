from flask import Blueprint, request, jsonify
from services.clustering_service import ClusteringService

api_bp = Blueprint('api', __name__)
clustering_service = ClusteringService()


@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for clustering data."""
    try:
        result = clustering_service.handle_file_upload(request.files)
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/columns', methods=['GET'])
def get_columns():
    """Get available numeric columns from uploaded file."""
    try:
        result = clustering_service.get_available_columns()
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/cluster', methods=['POST'])
def run_clustering():
    """Perform K-Means clustering on uploaded data."""
    try:
        data = request.get_json()
        result = clustering_service.perform_clustering(data)
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/edit-point', methods=['POST'])
def edit_point():
    """Edit a single point's cluster assignment."""
    try:
        data = request.get_json()
        result = clustering_service.edit_point_assignment(data)
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/merge-clusters', methods=['POST'])
def merge_clusters():
    """Merge two clusters into one."""
    try:
        data = request.get_json()
        result = clustering_service.merge_clusters(data)
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/split-cluster', methods=['POST'])
def split_cluster():
    """Split a cluster into two using k-means on the cluster points."""
    try:
        data = request.get_json()
        result = clustering_service.split_cluster(data)
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/undo-edit', methods=['POST'])
def undo_edit():
    """Undo the last edit operation."""
    try:
        result = clustering_service.undo_last_edit()
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/reset-edits', methods=['POST'])
def reset_edits():
    """Reset all manual edits and return to original clustering."""
    try:
        result = clustering_service.reset_all_edits()
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/animation', methods=['GET'])
def get_animation_steps():
    """Return history steps for animation (iteration by iteration)."""
    try:
        result = clustering_service.get_animation_history()
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/plots', methods=['GET'])
def generate_plots():
    """Generate plot data for visualization."""
    try:
        result = clustering_service.generate_plot_data()
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/reset', methods=['POST'])
def reset_data():
    """Clear all uploaded data and clustering results."""
    try:
        result = clustering_service.reset_all_data()
        return jsonify(result['data']), result['status']
    except Exception as e:
        return jsonify({'error': str(e)}), 500