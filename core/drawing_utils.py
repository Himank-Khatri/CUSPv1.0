import cv2
import numpy as np
from typing import Dict

def draw_overlays(frame: np.ndarray, counts: Dict[str, int], performance_stats: Dict, camera_connected: bool, midline: int) -> np.ndarray:
    """Draw overlays on frame."""
    frame_height, frame_width = frame.shape[:2]
    
    # Draw midline
    cv2.line(frame, (0, midline), (frame_width, midline), (255, 0, 0), 2)
    
    # Draw counts
    cv2.putText(frame, f"Cars: {counts.get('cars', 0)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Bikes: {counts.get('bikes', 0)}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw total vehicles
    total_vehicles = counts.get('total', 0)
    cv2.putText(frame, f"Total: {total_vehicles}", (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Draw performance stats
    fps_text = f"FPS: {performance_stats.get('fps', 0):.1f}"
    cv2.putText(frame, fps_text, (10, frame_height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw connection status
    status_color = (0, 255, 0) if camera_connected else (0, 0, 255)
    status_text = "Connected" if camera_connected else "Disconnected"
    cv2.putText(frame, status_text, (frame_width - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Draw midline position info
    midline_text = f"Midline: {midline}"
    cv2.putText(frame, midline_text, (frame_width - 200, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame