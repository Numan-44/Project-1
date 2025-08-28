import logging
from clustering_service import ClusteringService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringController:
    def __init__(self):
        self.service = ClusteringService()

    def call_service(self, method_name, *args, **kwargs):
        """Generic wrapper with logging & error handling."""
        logger.info(f"Calling service method: {method_name}")
        try:
            method = getattr(self.service, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {method_name}: {e}")
            return {'data': {'error': 'Internal server error'}, 'status': 500}
