from flask import Flask, render_template, Response, jsonify, request
import cv2
import logging
import threading
import os
import time

from config import settings
from core.optimized_processor import OptimizedParkingProcessor

# Ensure logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

processor = OptimizedParkingProcessor()
processor.start_processing_thread() # Start the background processing thread

# Global variable to control video streaming
video_streaming_enabled = True

@app.route("/")
def home():
    """Serves the dashboard template"""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Serves the live video stream"""
    def generate_frames():
        global video_streaming_enabled
        while True:
            if video_streaming_enabled:
                frame_data = processor.get_current_frame_bytes()
                if frame_data:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                else:
                    logger.info("Can't display frames because there is no frame data")
            else:
                # If streaming is disabled, yield an empty frame or simply pass
                # To avoid breaking the multipart stream, we can yield a minimal frame or just sleep
                # For now, we'll just sleep to reduce CPU usage and not send frames.
                # The client-side JS will handle hiding the img.
                time.sleep(0.1) # Sleep a bit to prevent busy-waiting

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_counts")
def get_counts():
    """Returns live vehicle counts"""
    counts = processor.get_counts()
    response = jsonify(counts)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/camera_status")
def camera_status():
    """Returns detailed camera connection status"""
    # The optimized processor doesn't have get_camera_status, so we'll create a simple status
    status = {
        "connected": processor.camera_connected,
        "retry_count": processor.connection_retry_count,
        "max_retry_attempts": processor.max_retry_attempts
    }
    return jsonify(status)

@app.route("/reset_counts")
def reset_counts():
    """Reset vehicle counts"""
    processor.reset_counts()
    counts = processor.get_counts()
    return jsonify({"status": "Counts reset", "cars": counts['cars'], "bikes": counts['bikes']})

@app.route("/reconnect_camera")
def reconnect_camera_endpoint():
    """Force camera reconnection"""
    success = processor.initialize_camera()
    return jsonify({
        "status": "success" if success else "failed",
        "message": "Camera reconnection " + ("successful" if success else "failed")
    })

@app.route("/debug_rtsp")
def debug_rtsp():
    """Debug RTSP connection issues"""
    debug_info = {
        "camera_connected": processor.camera_connected,
        "connection_retry_count": processor.connection_retry_count,
        "max_retry_attempts": processor.max_retry_attempts,
        "rtsp_link": processor.rtsp_link,
        "video_path": processor.video_path,
        "performance_stats": processor.get_performance_stats()
    }
    return jsonify(debug_info)

# This is the new endpoint that your JavaScript is trying to call
@app.route("/update_manual_counts", methods=['POST'])
def update_manual_counts():
    """Receives manually updated counts from the frontend."""
    try:
        data = request.get_json()
        logger.info(f"Received manual count update: {data}")
        
        # Call the method on the processor to update the counts
        processor.set_manual_counts(data)
        
        return jsonify({"status": "success", "message": "Counts updated manually."})
    except Exception as e:
        logger.error(f"Error updating manual counts: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/test_simple_rtsp")
def test_simple_rtsp():
    """Simple RTSP connection test"""
    test_result = {
        "success": processor.initialize_camera(),
        "message": "RTSP connection test completed"
    }
    return jsonify(test_result)

@app.route("/toggle_video_feed", methods=['POST'])
def toggle_video_feed():
    global video_streaming_enabled
    data = request.get_json()
    enable = data.get('enable')
    if isinstance(enable, bool):
        video_streaming_enabled = enable
        logger.info(f"Video streaming enabled: {video_streaming_enabled}")
        return jsonify(success=True, video_streaming_enabled=video_streaming_enabled)
    return jsonify(success=False, message="Invalid 'enable' value"), 400

app.config['DEBUG'] = True

if __name__ == "__main__":
    try:
        logger.info("Starting Optimized Smart Parking System...")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        processor.cleanup()
        cv2.destroyAllWindows()
