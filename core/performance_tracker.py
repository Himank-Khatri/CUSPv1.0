import time
import psutil
from collections import deque
from typing import Dict

class PerformanceTracker:
    def __init__(self, detector):
        self.detector = detector
        self.performance_stats = {
            'fps': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'total_frames_processed': 0,
            'detection_fps': 0
        }
        self.last_stats_update = time.time()

    def update(self, frame_times: deque):
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

    def increment_frame_count(self):
        self.performance_stats['total_frames_processed'] += 1

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_stats.copy()