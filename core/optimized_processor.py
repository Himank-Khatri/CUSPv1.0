import cv2
import numpy as np
import threading
import time
import logging
import json
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import gc

from sort.sort import Sort
from config import settings
from core.optimized_detector import OptimizedVehicleDetector
from core.video_capture import VideoCapture
from core.drawing_utils import draw_overlays
from core.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class OptimizedParkingProcessor:
    def __init__(self):
        self.manual_counts = {"cars": None, "bikes": None}
        self.counter_file = 'data/counter.json'
        self._load_counts()

        self.detector = OptimizedVehicleDetector(
            model_path=settings.get('vehicle_model_path'),
            device='cpu'
        )
        
        self.tracker = Sort(
            max_age=settings.get('sort_max_age', 5),
            min_hits=settings.get('sort_min_hits', 2),
            iou_threshold=settings.get('sort_iou_threshold', 0.3)
        )
        
        self.vehicle_directions = {}
        self.track_class_labels = {}
        self.track_history = defaultdict(list)
        
        self.midline = settings.get('midline', 360)
        
        self.target_width = settings.get('target_width', 1280)
        self.target_height = settings.get('target_height', 720)
        self.process_every_n_frames = settings.get('process_every_n_frames', 3)
        self.max_detections = settings.get('max_detections', 30)
        self.confidence_threshold = settings.get('confidence_threshold', 0.4)
        self.iou_threshold = settings.get('iou_threshold', 0.5)
        
        self.video_capture = VideoCapture(settings.parking_system)
        self.performance_tracker = PerformanceTracker(self.detector)
        
        self.latest_frame_bytes = None
        self.frame_skip_counter = 0
        self.processing_lock = threading.RLock()
        
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        logger.info("OptimizedParkingProcessor initialized")
    
    def _load_counts(self):
        """Load counts from the JSON file."""
        try:
            with open(self.counter_file, 'r') as f:
                data = json.load(f)
                self.car_count = data.get('cars', {}).get('occupied', 0)
                self.bike_count = data.get('bikes', {}).get('occupied', 0)
                self.total_car_slots = data.get('cars', {}).get('total', 0)
                self.total_bike_slots = data.get('bikes', {}).get('total', 0)
                logger.info(f"Counts loaded from {self.counter_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading counter file: {e}. Initializing counts to 0.")
            self.car_count = 0
            self.bike_count = 0
            self.total_car_slots = 32  # Default
            self.total_bike_slots = 500 # Default
            self._update_counter_file()

    def _update_counter_file(self):
        """Update the JSON counter file with the current counts."""
        try:
            with self.processing_lock:
                counts = {
                    "cars": {
                        "occupied": self.car_count,
                        "total": self.total_car_slots
                    },
                    "bikes": {
                        "occupied": self.bike_count,
                        "total": self.total_bike_slots
                    },
                    "last_updated": datetime.utcnow().isoformat() + "Z"
                }
                with open(self.counter_file, 'w') as f:
                    json.dump(counts, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to update counter file: {e}")

    def start_processing_thread(self):
        """Start the background frame grabbing and processing threads."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.shutdown_event.clear()

            self.latest_raw_frame = None
            self.frame_lock = threading.Lock()

            self.frame_grabber_thread = threading.Thread(
                target=self._frame_grabber_loop,
                daemon=True,
                name="FrameGrabber"
            )
            self.frame_grabber_thread.start()
            logger.info("Frame grabber thread started")

            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="VideoProcessor"
            )
            self.processing_thread.start()
            logger.info("Processing thread started")
            
    def _frame_grabber_loop(self):
        """Continuously grabs frames from the camera and stores only the latest one."""
        if not self.video_capture.initialize():
            logger.error("Failed to initialize camera for frame grabber")
            return

        while not self.shutdown_event.is_set():
            frame = self.video_capture.get_frame()
            if frame is not None:
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
    
    def _processing_loop(self):
        """Main processing loop that works on the latest available frame."""
        logger.info("Processing loop waiting for first frame...")
        while self.latest_raw_frame is None and not self.shutdown_event.is_set():
            time.sleep(0.1)
        
        if self.shutdown_event.is_set():
            return

        logger.info("First frame received, starting processing.")

        frame_times = deque(maxlen=30)
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                current_frame = None
                with self.frame_lock:
                    if self.latest_raw_frame is not None:
                        current_frame = self.latest_raw_frame.copy()

                if current_frame is None:
                    time.sleep(0.02)
                    continue
                
                self.frame_skip_counter += 1
                
                processed_frame = self._process_frame(current_frame)
                
                if processed_frame is not None:
                    with self.processing_lock:
                        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        self.latest_frame_bytes = buffer.tobytes()
                
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                self.performance_tracker.update(frame_times)
                
                target_fps = 30
                target_frame_time = 1.0 / target_fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame for vehicle detection and counting."""
        try:
            frame = self._resize_frame(frame)
            if frame is None:
                return None
            
            frame_height, frame_width = frame.shape[:2]
            
            if self.frame_skip_counter % self.process_every_n_frames == 0:
                detections = self.detector.detect_vehicles(
                    frame,
                    conf_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections
                )
                
                if detections:
                    self._update_tracking(frame, detections, frame_width, frame_height)
            
            display_frame = draw_overlays(
                frame, 
                self.get_counts(), 
                self.performance_tracker.get_stats(), 
                self.video_capture.camera_connected, 
                self.midline
            )
            
            self.performance_tracker.increment_frame_count()
            
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
        """Thread-safe method to manually override the current vehicle counts."""
        with self.processing_lock:
            try:
                logger.info(f"Received manual count data: {counts}")
                
                if 'cars' in counts and isinstance(counts['cars'], int):
                    self.car_count = max(0, counts['cars'])
                    logger.info(f"Manual override: Car count set to {self.car_count}")

                if 'bikes' in counts and isinstance(counts['bikes'], int):
                    self.bike_count = max(0, counts['bikes'])
                    logger.info(f"Manual override: Bike count set to {self.bike_count}")

            except (TypeError, ValueError) as e:
                logger.error(f"Invalid data format for manual count update: {e}")
                
    def _update_tracking(self, frame: np.ndarray, detections: List[List[float]], 
                        frame_width: int, frame_height: int):
        """Update vehicle tracking and counting."""
        try:
            dets = np.array(detections)
            
            tracks = self.tracker.update(dets[:, :5])
            
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                x1, y1, x2, y2 = map(int, bbox)
                
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(x1 + 1, min(x2, frame_width))
                y2 = max(y1 + 1, min(y2, frame_height))
                
                cls = self._get_vehicle_class(detections, bbox)
                if cls is not None:
                    self.track_class_labels[track_id] = cls
                
                self._update_vehicle_count_original(track_id, x1, y1, x2, y2)
                
                center_y = (y1 + y2) // 2
                self.track_history[track_id].append(center_y)
                if len(self.track_history[track_id]) > 10:
                    self.track_history[track_id].pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating tracking: {e}")
    
    def _update_vehicle_count_original(self, track_id: int, x1: int, y1: int, x2: int, y2: int):
        """Update vehicle count using the original logic from video_processor.py"""
        center_y = (y1 + y2) // 2
        
        if track_id not in self.vehicle_directions:
            self.vehicle_directions[track_id] = None
        
        if self.vehicle_directions[track_id] is None:
            if center_y < self.midline:
                self.vehicle_directions[track_id] = 'up'
            else:
                self.vehicle_directions[track_id] = 'down'
        else:
            if self.vehicle_directions[track_id] == 'up' and center_y > self.midline:
                vehicle_class = self.track_class_labels.get(track_id)
                with self.processing_lock:
                    if vehicle_class == 2:
                        self.car_count += 1
                        logger.info(f"🚗 Car {track_id} ENTERED - Total Cars: {self.car_count}")
                    elif vehicle_class == 3:
                        self.bike_count += 1
                        logger.info(f"🏍️ Bike {track_id} ENTERED - Total Bikes: {self.bike_count}")
                self._update_counter_file()
                self.vehicle_directions[track_id] = 'crossed_down'
                
            elif self.vehicle_directions[track_id] == 'down' and center_y < self.midline:
                vehicle_class = self.track_class_labels.get(track_id)
                with self.processing_lock:
                    if vehicle_class == 2:
                        self.car_count = max(0, self.car_count - 1)
                        logger.info(f"🚗 Car {track_id} EXITED - Total Cars: {self.car_count}")
                    elif vehicle_class == 3:
                        self.bike_count = max(0, self.bike_count - 1)
                        logger.info(f"🏍️ Bike {track_id} EXITED - Total Bikes: {self.bike_count}")
                self._update_counter_file()
                self.vehicle_directions[track_id] = 'crossed_up'
    
    def _get_vehicle_class(self, detections: List[List[float]], bbox: np.ndarray) -> Optional[int]:
        """Get vehicle class for a bounding box."""
        x1, y1, x2, y2 = bbox
        
        for det in detections:
            det_x1, det_y1, det_x2, det_y2, _, det_cls = det
            if not (x2 < det_x1 or x1 > det_x2 or y2 < det_y1 or y1 > det_y2):
                return int(det_cls)
        return None
    
    def get_current_frame_bytes(self) -> bytes:
        """Get current frame as bytes."""
        with self.processing_lock:
            if self.latest_frame_bytes is None:
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
                    'occupied': max(0, self.car_count),
                    'total': self.total_car_slots
                },
                'bikes': {
                    'occupied': max(0, self.bike_count),
                    'total': self.total_bike_slots
                },
                'total': max(0, self.car_count + self.bike_count)
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_tracker.get_stats()
    
    def reset_counts(self):
        """Reset vehicle counts."""
        with self.processing_lock:
            self.car_count = 0
            self.bike_count = 0
            self._update_counter_file()
        logger.info("Vehicle counts reset")
    
    def cleanup(self):
        """Clean up resources."""
        self.shutdown_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        self.video_capture.release()
        
        self.detector.cleanup()
        
        self.track_history.clear()
        self.vehicle_directions.clear()
        self.track_class_labels.clear()
        
        gc.collect()
        
        logger.info("OptimizedParkingProcessor cleanup completed")
