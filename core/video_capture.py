import cv2
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class VideoCapture:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.cap = None
        self.camera_connected = False
        self.connection_retry_count = 0
        self.max_retry_attempts = self.settings.get('max_retry_attempts', 3)
        self.retry_delay = self.settings.get('base_retry_delay', 1)
        self.rtsp_link = self.settings.get('rtsp_link')
        self.video_path = self.settings.get('video_path')
        self.target_width = self.settings.get('target_width', 1280)
        self.target_height = self.settings.get('target_height', 720)

    def initialize(self) -> bool:
        """Initialize camera connection with improved error handling."""
        self.connection_retry_count = 0
        
        while self.connection_retry_count < self.max_retry_attempts:
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                connection_methods = self._get_connection_methods()
                
                for method_name, method_config in connection_methods:
                    try:
                        logger.info(f"Trying connection method: {method_name}")
                        self.cap = cv2.VideoCapture(method_config['url'], method_config.get('backend', cv2.CAP_ANY))
                        
                        if self.cap is None or not self.cap.isOpened():
                            continue
                        
                        for prop, value in method_config.get('props', {}).items():
                            self.cap.set(prop, value)
                        
                        if self._test_connection_stability():
                            self.camera_connected = True
                            logger.info(f"Camera connected successfully using {method_name}")
                            return True
                        else:
                            self.cap.release()
                            self.cap = None
                            
                    except Exception as e:
                        logger.warning(f"Connection method {method_name} failed: {e}")
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                
                self.connection_retry_count += 1
                if self.connection_retry_count < self.max_retry_attempts:
                    logger.info(f"Retrying camera connection in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay = min(self.retry_delay * 1.5, 5)
                    
            except Exception as e:
                logger.error(f"Camera initialization error: {e}")
                self.connection_retry_count += 1
        
        logger.error("Failed to initialize camera after all attempts")
        return False

    def _get_connection_methods(self) -> List[Tuple[str, Dict]]:
        """Get list of connection methods to try."""
        methods = []
        
        if self.rtsp_link:
            methods.extend([
                ("RTSP_FFMPEG_OPTIMIZED", {
                    'url': self.rtsp_link,
                    'backend': cv2.CAP_FFMPEG,
                    'props': {
                        cv2.CAP_PROP_BUFFERSIZE: 1,
                        cv2.CAP_PROP_FRAME_WIDTH: self.target_width,
                        cv2.CAP_PROP_FRAME_HEIGHT: self.target_height,
                        cv2.CAP_PROP_FPS: 15
                    }
                }),
                ("RTSP_FFMPEG_SIMPLE", {
                    'url': self.rtsp_link,
                    'backend': cv2.CAP_FFMPEG,
                    'props': {
                        cv2.CAP_PROP_BUFFERSIZE: 2
                    }
                }),
                ("RTSP_ANY", {
                    'url': self.rtsp_link,
                    'backend': cv2.CAP_ANY,
                    'props': {}
                })
            ])
        else:
            methods.append(("LOCAL_VIDEO", {
                'url': self.video_path,
                'backend': cv2.CAP_ANY,
                'props': {}
            }))
        
        return methods

    def _test_connection_stability(self) -> bool:
        """Test if camera connection is stable."""
        if not self.cap or not self.cap.isOpened():
            return False
        
        successful_reads = 0
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                if np.mean(frame) > 5:
                    successful_reads += 1
            time.sleep(0.1)
        
        return successful_reads >= 3

    def read(self):
        return self.cap.read()

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()

    def get_frame(self):
        if not self.is_opened():
            logger.warning("Camera not open, attempting to re-initialize...")
            if not self.initialize():
                time.sleep(2)
                return None
        
        ret, frame = self.read()
        if ret:
            return frame
        else:
            time.sleep(0.05)
            return None