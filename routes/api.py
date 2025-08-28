from flask import Blueprint, request, jsonify
from controller import ClusteringController

api_bp = Blueprint('api', __name__)
controller = ClusteringController()

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    result = controller.call_service('handle_file_upload', request.files)
    return jsonify(result['data']), result['status']

@api_bp.route('/columns', methods=['GET'])
def get_columns():
    result = controller.call_service('get_available_columns')
    return jsonify(result['data']), result['status']

@api_bp.route('/cluster', methods=['POST'])
def run_clustering():
    result = controller.call_service('perform_clustering', request.get_json() or {})
    return jsonify(result['data']), result['status']

@api_bp.route('/edit-point', methods=['POST'])
def edit_point():
    result = controller.call_service('edit_point_assignment', request.get_json() or {})
    return jsonify(result['data']), result['status']

@api_bp.route('/merge-clusters', methods=['POST'])
def merge_clusters():
    result = controller.call_service('merge_clusters', request.get_json() or {})
    return jsonify(result['data']), result['status']

@api_bp.route('/split-cluster', methods=['POST'])
def split_cluster():
    result = controller.call_service('split_cluster', request.get_json() or {})
    return jsonify(result['data']), result['status']

@api_bp.route('/undo-edit', methods=['POST'])
def undo_edit():
    result = controller.call_service('undo_last_edit')
    return jsonify(result['data']), result['status']

@api_bp.route('/reset-edits', methods=['POST'])
def reset_edits():
    result = controller.call_service('reset_all_edits')
    return jsonify(result['data']), result['status']

@api_bp.route('/animation', methods=['GET'])
def get_animation_steps():
    result = controller.call_service('get_animation_history')
    return jsonify(result['data']), result['status']

@api_bp.route('/plots', methods=['GET'])
def generate_plots():
    result = controller.call_service('generate_plot_data')
    return jsonify(result['data']), result['status']

@api_bp.route('/reset', methods=['POST'])
def reset_data():
    result = controller.call_service('reset_all_data')
    return jsonify(result['data']), result['status']

@api_bp.route('/status', methods=['GET'])
def get_status():
    result = controller.call_service('get_service_status')
    return jsonify(result['data']), result['status']
