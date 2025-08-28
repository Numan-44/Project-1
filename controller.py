from clustering_service import ClusteringService
from werkzeug.exceptions import BadRequest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringController:
    """
    Controller class that orchestrates clustering operations.
    
    This controller acts as the middleman between API routes and the clustering service,
    handling request validation, business logic coordination, and response formatting.
    """
    
    def __init__(self):
        self.clustering_service = ClusteringService()
    
    def handle_file_upload(self, files):
        """
        Handle file upload request and coordinate with clustering service.
        
        Args:
            files: Flask request.files object
            
        Returns:
            Dict with response data and HTTP status code
            
        Raises:
            BadRequest: If file validation fails
        """
        logger.info("Processing file upload request")
        
        try:
            # Delegate to service layer
            result = self.clustering_service.handle_file_upload(files)
            
            if result['status'] == 200:
                logger.info(f"File uploaded successfully with {len(result['data']['columns'])} valid columns")
            else:
                logger.warning(f"File upload failed: {result['data'].get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error during file upload: {str(e)}")
            return {
                'data': {'error': 'Internal server error during file upload'}, 
                'status': 500
            }
    
    def get_available_columns(self):
        """
        Retrieve available columns from uploaded data.
        
        Returns:
            Dict with response data and HTTP status code
        """
        logger.info("Retrieving available columns")
        
        try:
            result = self.clustering_service.get_available_columns()
            
            if result['status'] == 200:
                logger.info(f"Retrieved {len(result['data']['columns'])} available columns")
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving columns: {str(e)}")
            return {
                'data': {'error': 'Failed to retrieve columns'}, 
                'status': 500
            }
    
    def execute_clustering(self, request_data):
        """
        Execute K-Means clustering with given parameters.
        
        Args:
            request_data: Dict containing clustering parameters
            
        Returns:
            Dict with response data and HTTP status code
        """
        logger.info(f"Executing clustering with parameters: {request_data}")
        
        # Validate required parameters
        required_params = ['x_column', 'y_column']
        missing_params = [param for param in required_params if not request_data.get(param)]
        
        if missing_params:
            error_msg = f"Missing required parameters: {', '.join(missing_params)}"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        # Set defaults for optional parameters
        clustering_params = {
            'x_column': request_data.get('x_column'),
            'y_column': request_data.get('y_column'),
            'k': max(1, min(10, request_data.get('k', 3))),  # Clamp k between 1-10
            'init': request_data.get('init', 'random') if request_data.get('init') in ['random', 'kmeans++'] else 'random'
        }
        
        try:
            result = self.clustering_service.perform_clustering(clustering_params)
            
            if result['status'] == 200:
                logger.info(f"Clustering completed successfully with SSE: {result['data']['sse']}")
            else:
                logger.warning(f"Clustering failed: {result['data'].get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error during clustering: {str(e)}")
            return {
                'data': {'error': 'Internal server error during clustering'}, 
                'status': 500
            }
    
    def edit_point_cluster(self, request_data):
        """
        Edit a single point's cluster assignment.
        
        Args:
            request_data: Dict containing point_index and new_cluster
            
        Returns:
            Dict with response data and HTTP status code
        """
        point_index = request_data.get('point_index')
        new_cluster = request_data.get('new_cluster')
        
        logger.info(f"Editing point {point_index} to cluster {new_cluster}")
        
        # Validate parameters
        if point_index is None or new_cluster is None:
            error_msg = "Both point_index and new_cluster are required"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        if not isinstance(point_index, int) or point_index < 0:
            error_msg = "point_index must be a non-negative integer"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        if not isinstance(new_cluster, int) or new_cluster < 0:
            error_msg = "new_cluster must be a non-negative integer"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        try:
            result = self.clustering_service.edit_point_assignment(request_data)
            
            if result['status'] == 200:
                logger.info(f"Point edit completed successfully, new SSE: {result['data']['sse']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error editing point assignment: {str(e)}")
            return {
                'data': {'error': 'Failed to edit point assignment'}, 
                'status': 500
            }
    
    def merge_cluster_pair(self, request_data):
        """
        Merge two clusters into one.
        
        Args:
            request_data: Dict containing cluster1 and cluster2
            
        Returns:
            Dict with response data and HTTP status code
        """
        cluster1 = request_data.get('cluster1')
        cluster2 = request_data.get('cluster2')
        
        logger.info(f"Merging clusters {cluster1} and {cluster2}")
        
        # Validate parameters
        if cluster1 is None or cluster2 is None:
            error_msg = "Both cluster1 and cluster2 are required"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        if cluster1 == cluster2:
            error_msg = "Cannot merge a cluster with itself"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        try:
            result = self.clustering_service.merge_clusters(request_data)
            
            if result['status'] == 200:
                logger.info(f"Cluster merge completed successfully, new SSE: {result['data']['sse']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error merging clusters: {str(e)}")
            return {
                'data': {'error': 'Failed to merge clusters'}, 
                'status': 500
            }
    
    def split_single_cluster(self, request_data):
        """
        Split a cluster into two using K-Means.
        
        Args:
            request_data: Dict containing cluster_to_split
            
        Returns:
            Dict with response data and HTTP status code
        """
        cluster_to_split = request_data.get('cluster_to_split')
        
        logger.info(f"Splitting cluster {cluster_to_split}")
        
        # Validate parameters
        if cluster_to_split is None:
            error_msg = "cluster_to_split is required"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        if not isinstance(cluster_to_split, int) or cluster_to_split < 0:
            error_msg = "cluster_to_split must be a non-negative integer"
            logger.warning(error_msg)
            return {
                'data': {'error': error_msg}, 
                'status': 400
            }
        
        try:
            result = self.clustering_service.split_cluster(request_data)
            
            if result['status'] == 200:
                logger.info(f"Cluster split completed successfully, new SSE: {result['data']['sse']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error splitting cluster: {str(e)}")
            return {
                'data': {'error': 'Failed to split cluster'}, 
                'status': 500
            }
    
    def undo_last_operation(self):
        """
        Undo the last edit operation.
        
        Returns:
            Dict with response data and HTTP status code
        """
        logger.info("Undoing last operation")
        
        try:
            result = self.clustering_service.undo_last_edit()
            
            if result['status'] == 200:
                logger.info(f"Undo completed successfully, action: {result['data'].get('undone_action')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error undoing operation: {str(e)}")
            return {
                'data': {'error': 'Failed to undo operation'}, 
                'status': 500
            }
    
    def reset_all_edits(self):
        """
        Reset all manual edits and return to original clustering.
        
        Returns:
            Dict with response data and HTTP status code
        """
        logger.info("Resetting all edits to original clustering")
        
        try:
            result = self.clustering_service.reset_all_edits()
            
            if result['status'] == 200:
                logger.info("All edits reset successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error resetting edits: {str(e)}")
            return {
                'data': {'error': 'Failed to reset edits'}, 
                'status': 500
            }
    
    def get_clustering_animation(self):
        """
        Get iteration history for clustering animation.
        
        Returns:
            Dict with response data and HTTP status code
        """
        logger.info("Retrieving clustering animation data")
        
        try:
            result = self.clustering_service.get_animation_history()
            
            if result['status'] == 200:
                history_length = len(result['data']['history'])
                logger.info(f"Retrieved animation data with {history_length} steps")
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving animation data: {str(e)}")
            return {
                'data': {'error': 'Failed to retrieve animation data'}, 
                'status': 500
            }
    
    def get_plot_data(self):
        """
        Generate plot data for visualization including elbow analysis.
        
        Returns:
            Dict with response data and HTTP status code
        """
        logger.info("Generating plot data")
        
        try:
            result = self.clustering_service.generate_plot_data()
            
            if result['status'] == 200:
                points_count = len(result['data']['points'])
                centroids_count = len(result['data']['centroids'])
                logger.info(f"Generated plot data: {points_count} points, {centroids_count} centroids")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating plot data: {str(e)}")
            return {
                'data': {'error': 'Failed to generate plot data'}, 
                'status': 500
            }
    
    def reset_application_data(self):
        """
        Clear all uploaded data and clustering results.
        
        Returns:
            Dict with response data and HTTP status code
        """
        logger.info("Resetting all application data")
        
        try:
            result = self.clustering_service.reset_all_data()
            
            if result['status'] == 200:
                logger.info("Application data reset successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error resetting application data: {str(e)}")
            return {
                'data': {'error': 'Failed to reset application data'}, 
                'status': 500
            }
    
    def get_service_status(self):
        """
        Get current status of the clustering service.
        
        Returns:
            Dict with service status information
        """
        has_data = bool(self.clustering_service.uploaded_data.get('dataframe') is not None)
        has_results = bool(self.clustering_service.uploaded_data.get('cluster_results') is not None)
        edit_count = len(self.clustering_service.uploaded_data.get('edit_history', []))
        
        status_info = {
            'has_uploaded_data': has_data,
            'has_clustering_results': has_results,
            'edit_operations_count': edit_count,
            'available_columns': self.clustering_service.uploaded_data.get('valid_columns', [])
        }
        
        logger.info(f"Service status: {status_info}")
        
        return {
            'data': status_info,
            'status': 200
        }