import os

import cv2
import numpy as np
import threading
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import psutil
import gc
import json
from datetime import datetime

from sort.sort import Sort
from config import settings
from core.optimized_detector import OptimizedVehicleDetector

logger = logging.getLogger(__name__)

class OptimizedParkingProcessor:
    
    def __init__(self):
        self.manual_counts = {"cars": None, "bikes": None} 
        self.detector = OptimizedVehicleDetector(
            model_path=settings.get('vehicle_model_path'),
            device='cpu'
        )
        
        self.tracker = Sort(
            max_age=settings.get('sort_max_age', 5),
            min_hits=settings.get('sort_min_hits', 2),
            iou_threshold=settings.get('sort_iou_threshold', 0.3)
        )
        
        self.car_free_count = 0
        self.bike_free_count = 0
        self.car_total_count = 0
        self.bike_total_count = 0
        self.last_updated = None
        self.counter_file_path = settings.get('counter_file_path', 'counter.json')
        
        self._initialize_counter_file()
        self._load_counts_from_file()

        self.vehicle_directions = {}
        self.track_class_labels = {}
        self.track_history = defaultdict(list)
        self.crossed_vehicles = set()
        
        self.midline = settings.get('midline', 360)
        self.rtsp_link = settings.get('rtsp_link')
        self.video_path = settings.get('video_path')
        
        self.target_width = settings.get('target_width', 1280)
        self.target_height = settings.get('target_height', 720)
        self.process_every_n_frames = settings.get('process_every_n_frames', 3)
        self.max_detections = settings.get('max_detections', 30)
        self.confidence_threshold = settings.get('confidence_threshold', 0.4)
        self.iou_threshold = settings.get('iou_threshold', 0.5)
        
        self.cap = None
        self.camera_connected = False
        self.connection_retry_count = 0
        self.max_retry_attempts = settings.get('max_retry_attempts', 3)
        self.retry_delay = settings.get('base_retry_delay', 1)
        
        self.latest_frame_bytes = None
        self.frame_skip_counter = 0
        self.last_successful_frame_time = time.time()
        self.processing_lock = threading.RLock()
        
        self.performance_stats = {
            'fps': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'total_frames_processed': 0,
            'detection_fps': 0
        }
        self.last_stats_update = time.time()
        
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        logger.info("OptimizedParkingProcessor initialized")

    def _initialize_counter_file(self):
        """
        Initializes the counter.json file with default values if it doesn't exist.
        """
        if not os.path.exists(self.counter_file_path):
            default_counts = {
                "bikes": {"free": 0, "total": 0},
                "cars": {"free": 0, "total": 0},
                "last_updated": datetime.now().isoformat()
            }
            with open(self.counter_file_path, 'w') as f:
                json.dump(default_counts, f, indent=2)
            logger.info(f"Created default counter file at {self.counter_file_path}")

    def _load_counts_from_file(self):
        """
        Loads vehicle counts and total capacities from the counter.json file.
        """
        try:
            with open(self.counter_file_path, 'r') as f:
                data = json.load(f)
                self.bike_free_count = data.get('bikes', {}).get('free', 0)
                self.bike_total_count = data.get('bikes', {}).get('total', 0)
                self.car_free_count = data.get('cars', {}).get('free', 0)
                self.car_total_count = data.get('cars', {}).get('total', 0)
                self.last_updated = data.get('last_updated', None)
            logger.info(f"Loaded counts from {self.counter_file_path}: Cars Free={self.car_free_count}, Bikes Free={self.bike_free_count}")
        except FileNotFoundError:
            logger.warning(f"Counter file not found at {self.counter_file_path}. Initializing with defaults.")
            self._initialize_counter_file()
            self._load_counts_from_file() # Try loading again after initialization
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.counter_file_path}: {e}. Resetting to defaults.")
            self._initialize_counter_file()
            self._load_counts_from_file() # Try loading again after reset
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading counts: {e}")
            self._initialize_counter_file()
            self._load_counts_from_file() # Try loading again after reset

    def _save_counts_to_file(self):
        """
        Saves the current vehicle counts and total capacities to the counter.json file.
        """
        with self.processing_lock:
            data = {
                "bikes": {"free": self.bike_free_count, "total": self.bike_total_count},
                "cars": {"free": self.car_free_count, "total": self.car_total_count},
                "last_updated": datetime.now().isoformat()
            }
            try:
                with open(self.counter_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                self.last_updated = data["last_updated"]
                logger.debug(f"Saved counts to {self.counter_file_path}")
            except Exception as e:
                logger.error(f"Error saving counts to {self.counter_file_path}: {e}")
    
    def initialize_camera(self) -> bool:
        """Initialize camera connection with improved error handling."""
        self.connection_retry_count = 0
        
        while self.connection_retry_count < self.max_retry_attempts:
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                # Try different connection methods
                connection_methods = self._get_connection_methods()
                
                for method_name, method_config in connection_methods:
                    try:
                        logger.info(f"Trying connection method: {method_name}")
                        self.cap = cv2.VideoCapture(method_config['url'], method_config.get('backend', cv2.CAP_ANY))
                        
                        if self.cap is None or not self.cap.isOpened():
                            continue
                        
                        # Set camera properties
                        for prop, value in method_config.get('props', {}).items():
                            self.cap.set(prop, value)
                        
                        # Test connection stability
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
            # RTSP connection methods
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
            # Local video file
            methods.append(("LOCAL_VIDEO", {
                'url': self.video_path,
                'backend': cv2.CAP_ANY,
                'props': {}
            }))
        
        return methods
    # def set_manual_counts(self, counts):
    #     """Thread-safe method to update manual override counts."""
    #     with self.lock:
    #         self.manual_counts = counts
    
    def _test_connection_stability(self) -> bool:
        """Test if camera connection is stable."""
        if not self.cap or not self.cap.isOpened():
            return False
        
        successful_reads = 0
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                if np.mean(frame) > 5:  # Check if frame is not too dark
                    successful_reads += 1
            time.sleep(0.1)
        
        return successful_reads >= 3
    
    def start_processing_thread(self):
        """Start the background frame grabbing and processing threads."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.shutdown_event.clear()

            # 1. Initialize shared resources for the threads
            self.latest_raw_frame = None
            self.frame_lock = threading.Lock() # A lock to safely share the frame

            # 2. Create and start the FRAME GRABBER thread (The Reporter)
            self.frame_grabber_thread = threading.Thread(
                target=self._frame_grabber_loop,
                daemon=True,
                name="FrameGrabber"
            )
            self.frame_grabber_thread.start()
            logger.info("Frame grabber thread started")

            # 3. Create and start the PROCESSING thread (The Anchor)
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="VideoProcessor"
            )
            self.processing_thread.start()
            logger.info("Processing thread started")
            
    def _frame_grabber_loop(self):
        """
        (This is the NEW 'Reporter' function)
        Continuously grabs frames from the camera and stores only the latest one.
        """
        # First, make sure the camera is initialized in this thread
        if not self.initialize_camera():
            logger.error("Failed to initialize camera for frame grabber")
            return

        while not self.shutdown_event.is_set():
            if not self.cap.isOpened():
                logger.warning("Camera not open, attempting to re-initialize...")
                if not self.initialize_camera():
                    time.sleep(2) # Wait before retrying
                    continue
                
            ret, frame = self.cap.read()
            if ret:
                # Use the lock to safely update the shared frame
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
            else:
                # If reading fails, sleep a bit to avoid a tight loop
                time.sleep(0.05)

    
    def _processing_loop(self):
        """
        (This is the MODIFIED 'Anchor' function)
        Main processing loop that works on the latest available frame.
        """
        # Camera initialization is now done in the grabber thread,
        # so we just wait for the first frame to be ready.
        logger.info("Processing loop waiting for first frame...")
        while self.latest_raw_frame is None and not self.shutdown_event.is_set():
            time.sleep(0.1)
        
        if self.shutdown_event.is_set():
            return # Exit if shutdown was called while waiting

        logger.info("First frame received, starting processing.")

        consecutive_failures = 0
        max_consecutive_failures = 10
        frame_times = deque(maxlen=30)
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Get the latest frame from the grabber thread
                current_frame = None
                with self.frame_lock:
                    if self.latest_raw_frame is not None:
                        current_frame = self.latest_raw_frame.copy()

                if current_frame is None:
                    # If no frame is available, wait a moment
                    time.sleep(0.02)
                    continue
                
                consecutive_failures = 0
                self.frame_skip_counter += 1
                
                # Process frame
                processed_frame = self._process_frame(current_frame)
                
                # Update frame buffer
                if processed_frame is not None:
                    with self.processing_lock:
                        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        self.latest_frame_bytes = buffer.tobytes()
                
                # Update performance stats
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                self._update_performance_stats(frame_times)
                
                # Adaptive frame rate control
                target_fps = 30
                target_frame_time = 1.0 / target_fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
    
    def _is_camera_healthy(self) -> bool:
        """Check if camera connection is healthy."""
        return (self.cap is not None and 
                self.cap.isOpened() and 
                self.camera_connected and
                time.time() - self.last_successful_frame_time < 10)
    
    def _handle_camera_failure(self) -> bool:
        """Handle camera connection failure."""
        logger.warning("Camera connection failure detected")
        self.camera_connected = False
        
        if self.rtsp_link:
            # Try to reconnect for RTSP
            return self.initialize_camera()
        else:
            # For local video, try to restart
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True
    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame for vehicle detection and counting."""
        try:
            # Resize frame for processing
            frame = self._resize_frame(frame)
            if frame is None:
                return None
            
            # Use full frame instead of cropping
            frame_height, frame_width = frame.shape[:2]
            
            # Run detection only every N frames
            if self.frame_skip_counter % self.process_every_n_frames == 0:
                detections = self.detector.detect_vehicles(
                    frame,
                    conf_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections
                )
                
                if detections:
                    self._update_tracking(frame, detections, frame_width, frame_height)
            
            # Draw overlays
            display_frame = self._draw_overlays(frame, frame_width, frame_height)
            
            self.last_successful_frame_time = time.time()
            self.performance_stats['total_frames_processed'] += 1
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def _resize_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Resize frame to target resolution."""
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        if width != self.target_width or height != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height), 
                             interpolation=cv2.INTER_LINEAR)
        return frame
    
    def set_manual_counts(self, counts: Dict[str, int]):
        """
        Thread-safe method to manually override the current vehicle counts.
        This is called from the Flask endpoint when a user edits the count.
        """
        # Use the existing re-entrant lock for thread safety
        with self.processing_lock:
            try:
                logger.info(f"Received manual count data: {counts}")
                
                # Validate and update car count
                if 'cars' in counts and isinstance(counts['cars'], int):
                    self.car_free_count = max(0, counts['cars'])
                    logger.info(f"Manual override: Car free count set to {self.car_free_count}")

                # Validate and update bike count
                if 'bikes' in counts and isinstance(counts['bikes'], int):
                    self.bike_free_count = max(0, counts['bikes'])
                    logger.info(f"Manual override: Bike free count set to {self.bike_free_count}")
                
                self._save_counts_to_file()

            except (TypeError, ValueError) as e:
                logger.error(f"Invalid data format for manual count update: {e}")
                
    
    def _update_tracking(self, frame: np.ndarray, detections: List[List[float]], 
                        frame_width: int, frame_height: int):
        """Update vehicle tracking and counting."""
        try:
            # Convert detections to numpy array
            dets = np.array(detections)
            
            # Update tracker
            tracks = self.tracker.update(dets[:, :5])
            
            # Process each track
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure bounding box is within frame bounds
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(x1 + 1, min(x2, frame_width))
                y2 = max(y1 + 1, min(y2, frame_height))
                
                # Get vehicle class
                cls = self._get_vehicle_class(detections, bbox)
                if cls is not None:
                    self.track_class_labels[track_id] = cls
                
                # Update vehicle counting using original logic
                self._update_vehicle_count_original(track_id, x1, y1, x2, y2)
                
                # Update track history
                center_y = (y1 + y2) // 2
                self.track_history[track_id].append(center_y)
                if len(self.track_history[track_id]) > 10:
                    self.track_history[track_id].pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating tracking: {e}")
    
    def _update_vehicle_count_original(self, track_id: int, x1: int, y1: int, x2: int, y2: int):
        """Update vehicle count using the original logic from video_processor.py"""
        center_y = (y1 + y2) // 2
        
        # Initialize direction if not set
        if track_id not in self.vehicle_directions:
            self.vehicle_directions[track_id] = None
        
        # Set initial direction
        if self.vehicle_directions[track_id] is None:
            if center_y < self.midline:
                self.vehicle_directions[track_id] = 'up'
            else:
                self.vehicle_directions[track_id] = 'down'
        else:
            # Check for midline crossing
            if self.vehicle_directions[track_id] == 'up' and center_y > self.midline:
                # Vehicle crossed from top to bottom (entering)
                vehicle_class = self.track_class_labels.get(track_id)
                with self.processing_lock:
                    if vehicle_class == 2:  # Car
                        self.car_free_count = max(0, self.car_free_count - 1)
                        logger.info(f"🚗 Car {track_id} ENTERED - Free Cars: {self.car_free_count}")
                    elif vehicle_class == 3:  # Motorcycle
                        self.bike_free_count = max(0, self.bike_free_count - 1)
                        logger.info(f"🏍️ Bike {track_id} ENTERED - Free Bikes: {self.bike_free_count}")
                    self._save_counts_to_file()
                self.vehicle_directions[track_id] = 'crossed_down'
                
            elif self.vehicle_directions[track_id] == 'down' and center_y < self.midline:
                # Vehicle crossed from bottom to top (exiting)
                vehicle_class = self.track_class_labels.get(track_id)
                with self.processing_lock:
                    if vehicle_class == 2:  # Car
                        self.car_free_count = min(self.car_total_count, self.car_free_count + 1)
                        logger.info(f"🚗 Car {track_id} EXITED - Free Cars: {self.car_free_count}")
                    elif vehicle_class == 3:  # Motorcycle
                        self.bike_free_count = min(self.bike_total_count, self.bike_free_count + 1)
                        logger.info(f"🏍️ Bike {track_id} EXITED - Free Bikes: {self.bike_free_count}")
                    self._save_counts_to_file()
                self.vehicle_directions[track_id] = 'crossed_up'
    
    def _get_vehicle_class(self, detections: List[List[float]], bbox: np.ndarray) -> Optional[int]:
        """Get vehicle class for a bounding box."""
        x1, y1, x2, y2 = bbox
        
        for det in detections:
            det_x1, det_y1, det_x2, det_y2, _, det_cls = det
            # Check if bounding boxes overlap
            if not (x2 < det_x1 or x1 > det_x2 or y2 < det_y1 or y1 > det_y2):
                return int(det_cls)
        return None
    
    def _draw_overlays(self, frame: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        """Draw overlays on frame."""
        # Use the original fixed midline from settings
        midline = self.midline
        
        # Draw midline
        cv2.line(frame, (0, midline), (frame_width, midline), (255, 0, 0), 2)
        
        # Draw tracking boxes and vehicle information
        for track_id, track_info in self.track_history.items():
            if len(track_info) > 0:
                # Get the latest position
                latest_y = track_info[-1]
                latest_x = frame_width // 2  # Approximate x position
                
                # Draw tracking point
                cv2.circle(frame, (latest_x, latest_y), 5, (0, 255, 255), -1)
                
                # Draw vehicle ID and class
                vehicle_class = self.track_class_labels.get(track_id, "Unknown")
                direction = self.vehicle_directions.get(track_id, "Unknown")
                text = f"ID:{track_id} C:{vehicle_class} D:{direction}"
                cv2.putText(frame, text, (latest_x + 10, latest_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw counts
        cv2.putText(frame, f"Cars: {self.car_total_count - self.car_free_count}/{self.car_total_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"Bikes: {self.bike_total_count - self.bike_free_count}/{self.bike_total_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Draw total vehicles (occupied)
        occupied_cars = self.car_total_count - self.car_free_count
        occupied_bikes = self.bike_total_count - self.bike_free_count
        total_occupied = occupied_cars + occupied_bikes
        cv2.putText(frame, f"Occupied: {total_occupied}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Draw performance stats
        fps_text = f"FPS: {self.performance_stats['fps']:.1f}"
        cv2.putText(frame, fps_text, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw connection status
        status_color = (0, 255, 0) if self.camera_connected else (0, 0, 255)
        status_text = "Connected" if self.camera_connected else "Disconnected"
        cv2.putText(frame, status_text, (frame_width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw midline position info
        midline_text = f"Midline: {midline}"
        cv2.putText(frame, midline_text, (frame_width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _update_performance_stats(self, frame_times: deque):
        """Update performance statistics."""
        current_time = time.time()
        if current_time - self.last_stats_update > 1.0:  # Update every second
            # Calculate FPS
            if len(frame_times) > 0:
                avg_frame_time = sum(frame_times) / len(frame_times)
                self.performance_stats['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Get system stats
            process = psutil.Process()
            self.performance_stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            self.performance_stats['cpu_usage_percent'] = process.cpu_percent()
            
            # Get detection stats
            detector_stats = self.detector.get_performance_stats()
            self.performance_stats['detection_fps'] = detector_stats.get('fps', 0)
            
            self.last_stats_update = current_time
    
    def get_current_frame_bytes(self) -> bytes:
        """Get current frame as bytes."""
        with self.processing_lock:
            if self.latest_frame_bytes is None:
                # Create placeholder frame
                placeholder = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Initializing...", (50, self.target_height // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                return buffer.tobytes() if buffer is not None else b''
            return self.latest_frame_bytes
    
    def get_counts(self) -> Dict[str, int]:
        """Get current vehicle counts."""
        with self.processing_lock:
            return {
                'cars': {
                    'free': max(0, self.car_free_count),
                    'total': self.car_total_count
                },
                'bikes': {
                    'free': max(0, self.bike_free_count),
                    'total': self.bike_total_count
                },
                'last_updated': self.last_updated
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_counts(self):
        """Reset vehicle counts."""
        with self.processing_lock:
            self.car_free_count = self.car_total_count
            self.bike_free_count = self.bike_total_count
            self.crossed_vehicles.clear()
            self._save_counts_to_file()
        logger.info("Vehicle counts reset")
    
    def cleanup(self):
        """Clean up resources."""
        self.shutdown_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.detector.cleanup()
        
        # Clear memory
        self.track_history.clear()
        self.vehicle_directions.clear()
        self.track_class_labels.clear()
        self.crossed_vehicles.clear()
        
        gc.collect()
        
        logger.info("OptimizedParkingProcessor cleanup completed")
