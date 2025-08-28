from flask import Blueprint, request, jsonify
from controller import ClusteringController

# Create API Blueprint
api_bp = Blueprint('api', __name__)

# Initialize controller
clustering_controller = ClusteringController()


@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for clustering data."""
    result = clustering_controller.handle_file_upload(request.files)
    return jsonify(result['data']), result['status']


@api_bp.route('/columns', methods=['GET'])
def get_columns():
    """Get available numeric columns from uploaded file."""
    result = clustering_controller.get_available_columns()
    return jsonify(result['data']), result['status']


@api_bp.route('/cluster', methods=['POST'])
def run_clustering():
    """Perform K-Means clustering on uploaded data."""
    data = request.get_json() or {}
    result = clustering_controller.execute_clustering(data)
    return jsonify(result['data']), result['status']


@api_bp.route('/edit-point', methods=['POST'])
def edit_point():
    """Edit a single point's cluster assignment."""
    data = request.get_json() or {}
    result = clustering_controller.edit_point_cluster(data)
    return jsonify(result['data']), result['status']


@api_bp.route('/merge-clusters', methods=['POST'])
def merge_clusters():
    """Merge two clusters into one."""
    data = request.get_json() or {}
    result = clustering_controller.merge_cluster_pair(data)
    return jsonify(result['data']), result['status']


@api_bp.route('/split-cluster', methods=['POST'])
def split_cluster():
    """Split a cluster into two using k-means on the cluster points."""
    data = request.get_json() or {}
    result = clustering_controller.split_single_cluster(data)
    return jsonify(result['data']), result['status']


@api_bp.route('/undo-edit', methods=['POST'])
def undo_edit():
    """Undo the last edit operation."""
    result = clustering_controller.undo_last_operation()
    return jsonify(result['data']), result['status']


@api_bp.route('/reset-edits', methods=['POST'])
def reset_edits():
    """Reset all manual edits and return to original clustering."""
    result = clustering_controller.reset_all_edits()
    return jsonify(result['data']), result['status']


@api_bp.route('/animation', methods=['GET'])
def get_animation_steps():
    """Return history steps for animation (iteration by iteration)."""
    result = clustering_controller.get_clustering_animation()
    return jsonify(result['data']), result['status']


@api_bp.route('/plots', methods=['GET'])
def generate_plots():
    """Generate plot data for visualization."""
    result = clustering_controller.get_plot_data()
    return jsonify(result['data']), result['status']


@api_bp.route('/reset', methods=['POST'])
def reset_data():
    """Clear all uploaded data and clustering results."""
    result = clustering_controller.reset_application_data()
    return jsonify(result['data']), result['status']


@api_bp.route('/status', methods=['GET'])
def get_status():
    """Get current service status."""
    result = clustering_controller.get_service_status()
    return jsonify(result['data']), result['status']


# Error handlers
@api_bp.errorhandler(400)
def bad_request(error):
    """Handle bad requests."""
    return jsonify({'error': 'Bad request'}), 400


@api_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return jsonify({'error': 'Internal server error'}), 500