import os
import cv2
import base64
import numpy as np
import importlib
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, flash, session
from werkzeug.utils import secure_filename
import time
import traceback

# Import application modules
from utils.config import Config
from utils.logger import setup_logger
# Reload the database module to ensure the latest changes are loaded
import database
importlib.reload(database)
from database import Database
from detr_utils import DetrFaceUtils
from camera import CameraManager, CameraSource, Camera

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
logger = setup_logger(name="web_app")

# Initialize configuration
config = Config()

# Initialize database
database = Database(config.get("database", "path", "data/database.db"))

# Helper function to ensure database has the latest methods
def ensure_database_ready():
    """Ensure the database instance has all required methods"""
    global database
    
    # Check if the required methods exist
    required_methods = ['get_distinct_departments', 'get_distinct_positions']
    missing_methods = [method for method in required_methods 
                       if not hasattr(database, method)]
    
    # If any methods are missing, reinitialize the database
    if missing_methods:
        logger.warning(f"Missing database methods: {missing_methods}. Reinitializing database.")
        # Reload the database module
        import importlib
        import database as db_module
        importlib.reload(db_module)
        # Recreate the database instance
        from database import Database
        database = Database(config.get("database", "path", "data/database.db"))
    
    return database

# Initialize face utils with DETR
tolerance = config.get("face_recognition", "tolerance", 0.7)
device = config.get("face_recognition", "device", None)
face_utils = DetrFaceUtils(tolerance=tolerance, device=device)

# Load known faces from database
def load_known_faces():
    try:
        # Ensure database is ready with all methods
        db = ensure_database_ready()
        
        known_faces = db.get_all_face_encodings()
        face_utils.load_known_faces(known_faces)
        logger.info(f"Loaded {len(known_faces)} known faces")
    except Exception as e:
        logger.error(f"Error loading known faces: {str(e)}")

load_known_faces()

# Initialize camera manager
camera_manager = CameraManager()

# Add default webcam if no cameras are configured
saved_cameras = config.get("camera", "saved_cameras", [])
if not saved_cameras:
    camera_manager.add_camera(
        "Default Webcam",
        CameraSource.WEBCAM,
        0,
        config.get("camera", "default_resolution", [640, 480])
    )
else:
    for camera in saved_cameras:
        camera_manager.add_camera(
            camera["name"],
            CameraSource[camera["type"]],
            camera["source"],
            camera["resolution"]
        )

# Set active camera
camera_names = camera_manager.get_available_cameras()
if camera_names:
    try:
        first_camera_name = next(iter(camera_names))
        logger.info(f"Setting active camera to: {first_camera_name}")
        camera_manager.set_active_camera(first_camera_name)
        if not camera_manager.start_camera(first_camera_name):
            logger.error(f"Failed to start camera: {first_camera_name}")
        else:
            logger.info(f"Camera started successfully: {first_camera_name}")
    except Exception as e:
        logger.error(f"Error initializing camera: {str(e)}")
else:
    logger.warning("No cameras available. Video streaming will be disabled.")

# Generate camera frames for streaming
def generate_frames(with_detection=True):
    # Create a fallback frame (black image with text)
    fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        fallback_frame,
        "Camera not available",
        (100, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    error_count = 0
    max_errors = 10
    
    while True:
        try:
            success, frame = camera_manager.read_active_camera()
            
            # Reset error count on successful frame
            if success:
                error_count = 0
            else:
                error_count += 1
                logger.warning(f"Failed to read camera: {error_count}/{max_errors}")
                
                # If too many consecutive errors, use fallback frame
                if error_count >= max_errors:
                    logger.error("Too many camera read errors, using fallback frame")
                    frame = fallback_frame.copy()
                    # Add timestamp to fallback frame
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(
                        frame,
                        timestamp,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                else:
                    # Skip this frame and try again
                    time.sleep(0.1)
                    continue
            
            if with_detection and success:
                try:
                    # Detect and recognize faces using DETR
                    results = face_utils.detect_and_recognize(frame)
                    frame = face_utils.draw_recognition_results(frame, results)
                    
                    # Check if any face is recognized for attendance
                    for result in results:
                        if result['id'] is not None and result['confidence'] >= face_utils.tolerance:
                            # Trigger attendance recording in a non-blocking way
                            success, status, _ = database.record_attendance(result['id'])
                            if success:
                                cv2.putText(
                                    frame,
                                    f"{result['name']} - {status.upper()} RECORDED",
                                    (10, frame.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2
                                )
                except Exception as e:
                    logger.error(f"Error in face detection: {str(e)}")
                    # Add error message to frame
                    cv2.putText(
                        frame,
                        f"Detection error: {str(e)[:30]}...",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame in the HTTP response
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
        except Exception as e:
            logger.error(f"Error in generate_frames: {str(e)}")
            
            # Create an error frame
            error_frame = fallback_frame.copy()
            cv2.putText(
                error_frame,
                f"Error: {str(e)[:30]}...",
                (50, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            
            # Yield the error frame
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Wait before retry
            time.sleep(0.5)

@app.route('/')
def index():
    """Render the main page with attendance functionality"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for the attendance page"""
    return Response(generate_frames(with_detection=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/enrollment')
def enrollment():
    """Render the enrollment page"""
    # Ensure database is ready with all methods
    db = ensure_database_ready()
    
    # Get departments from database for dropdown
    try:
        departments = db.get_distinct_departments()
        positions = db.get_distinct_positions()
    except AttributeError as e:
        # Log the error
        logger.error(f"AttributeError in enrollment: {str(e)}")
        # Fallback if methods aren't loaded yet
        departments = []
        positions = []
    
    return render_template('enrollment.html', 
                          departments=departments, 
                          positions=positions)

@app.route('/enrollment_feed')
def enrollment_feed():
    """Video streaming route for the enrollment page"""
    return Response(generate_frames(with_detection=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    """Capture a face for enrollment"""
    success, frame = camera_manager.read_active_camera()
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture image'})
    
    # Detect face in the frame using DETR
    face_locations = face_utils.detect_faces(frame)
    if not face_locations:
        return jsonify({'success': False, 'message': 'No face detected'})
    
    # Use the first face detected
    face_location = face_locations[0]
    
    # Encode the face using DETR
    face_encoding = face_utils.encode_face(frame, face_location)
    if face_encoding is None:
        return jsonify({'success': False, 'message': 'Failed to encode face'})
    
    # Draw rectangle around the face and convert to base64 for display
    top, right, bottom, left = face_location
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Convert to JPEG and then to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Store in session for later use during enrollment submission
    session['face_encoding'] = face_encoding.tolist()
    
    return jsonify({
        'success': True, 
        'image': frame_base64,
        'message': 'Face captured successfully'
    })

@app.route('/submit_enrollment', methods=['POST'])
def submit_enrollment():
    """Save a new employee to the database"""
    try:
        # Ensure database is ready with all methods
        db = ensure_database_ready()
        
        employee_id = request.form.get('employee_id')
        name = request.form.get('name')
        department = request.form.get('department')
        position = request.form.get('position')
        
        # Get face encoding from session
        face_encoding = session.get('face_encoding')
        if not face_encoding:
            flash('No face capture found. Please capture a face image first.', 'danger')
            return redirect(url_for('enrollment'))
        
        # Convert back to numpy array
        face_encoding = np.array(face_encoding)
        
        # Add employee to database
        success, result = db.add_employee(
            employee_id, 
            name, 
            department, 
            position, 
            face_encoding
        )
        
        if success:
            # Reload known faces after adding new employee
            load_known_faces()
            flash(f'Employee {name} added successfully!', 'success')
        else:
            flash(f'Failed to add employee: {result}', 'danger')
        
        # Clear session data
        if 'face_encoding' in session:
            del session['face_encoding']
            
        return redirect(url_for('enrollment'))
    
    except Exception as e:
        flash(f'Error submitting enrollment: {str(e)}', 'danger')
        logger.error(f"Error submitting enrollment: {str(e)}")
        return redirect(url_for('enrollment'))

@app.route('/reports')
def reports():
    """Render the reports page"""
    # Ensure database is ready with all methods
    db = ensure_database_ready()
    
    # Get all employees for the filters
    try:
        employees = db.get_all_employees()
    except Exception as e:
        logger.error(f"Error getting employees: {str(e)}")
        employees = []
    
    return render_template('reports.html', employees=employees)

@app.route('/get_attendance_data', methods=['POST'])
def get_attendance_data():
    """Get attendance data for reports"""
    try:
        # Ensure database is ready with all methods
        db = ensure_database_ready()
        
        employee_id = request.form.get('employee_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        
        attendance_records = db.get_attendance_records(
            employee_id=employee_id if employee_id else None,
            start_date=start_date if start_date else None,
            end_date=end_date if end_date else None
        )
        
        return jsonify({
            'success': True,
            'data': attendance_records
        })
    
    except Exception as e:
        logger.error(f"Error getting attendance data: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/settings')
def settings():
    """Render the settings page"""
    camera_sources = [{'id': source.value, 'name': source.name} 
                      for source in CameraSource]
    
    cameras = []
    for name in camera_manager.get_available_cameras():
        try:
            camera_info = camera_manager.get_camera_info(name)
            if camera_info:
                cameras.append({
                    'name': name,
                    'type': camera_info['type'],
                    'source': camera_info['source'],
                    'resolution': camera_info['resolution']
                })
        except KeyError as e:
            # Log the error and continue
            logger.error(f"KeyError accessing camera info: {str(e)}")
            # Add with basic info
            cameras.append({
                'name': name,
                'type': "Unknown",
                'source': "Unknown",
                'resolution': [640, 480]
            })
    
    recognition_settings = {
        'tolerance': face_utils.tolerance,
        'device': face_utils.device,
    }
    
    return render_template('settings.html', 
                          camera_sources=camera_sources,
                          cameras=cameras,
                          recognition_settings=recognition_settings)

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """Save settings"""
    try:
        # Save face recognition settings
        tolerance = float(request.form.get('tolerance', 0.7))
        device = request.form.get('device', face_utils.device)
        
        face_utils.tolerance = tolerance
        
        # Device setting requires restart
        if device != face_utils.device:
            flash('Device setting change requires application restart to take effect', 'warning')
        
        # Save to config
        config.set("face_recognition", "tolerance", tolerance)
        config.set("face_recognition", "device", device)
        
        flash('Settings saved successfully!', 'success')
        return redirect(url_for('settings'))
    
    except Exception as e:
        flash(f'Error saving settings: {str(e)}', 'danger')
        logger.error(f"Error saving settings: {str(e)}")
        return redirect(url_for('settings'))

@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Add a new camera"""
    try:
        name = request.form.get('name')
        source_type = int(request.form.get('source_type', 0))
        source_path = request.form.get('source_path')
        
        # Convert source_path to int if it's a number (for webcam index)
        if source_type == CameraSource.WEBCAM.value:
            try:
                source_path = int(source_path)
            except ValueError:
                pass
        
        width = int(request.form.get('width', 640))
        height = int(request.form.get('height', 480))
        
        # Add camera
        camera_manager.add_camera(
            name,
            CameraSource(source_type),
            source_path,
            (width, height)
        )
        
        # Save to config
        saved_cameras = config.get("camera", "saved_cameras", [])
        saved_cameras.append({
            "name": name,
            "type": CameraSource(source_type).name,
            "source": source_path,
            "resolution": [width, height]
        })
        config.set("camera", "saved_cameras", saved_cameras)
        
        flash(f'Camera {name} added successfully!', 'success')
        return redirect(url_for('settings'))
    
    except Exception as e:
        flash(f'Error adding camera: {str(e)}', 'danger')
        logger.error(f"Error adding camera: {str(e)}")
        return redirect(url_for('settings'))

@app.route('/set_active_camera', methods=['POST'])
def set_active_camera():
    """Set the active camera"""
    try:
        name = request.form.get('name')
        
        # Stop current camera
        current_name = camera_manager.get_active_camera_name()
        if current_name:
            camera_manager.stop_camera(current_name)
        
        # Set and start new camera
        camera_manager.set_active_camera(name)
        camera_manager.start_camera(name)
        
        flash(f'Camera switched to {name}', 'success')
        return redirect(url_for('settings'))
    
    except Exception as e:
        flash(f'Error setting active camera: {str(e)}', 'danger')
        logger.error(f"Error setting active camera: {str(e)}")
        return redirect(url_for('settings'))

@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    """Remove a camera"""
    try:
        name = request.form.get('name')
        
        # Remove from camera manager
        camera_manager.remove_camera(name)
        
        # Remove from config
        saved_cameras = config.get("camera", "saved_cameras", [])
        saved_cameras = [cam for cam in saved_cameras if cam["name"] != name]
        config.set("camera", "saved_cameras", saved_cameras)
        
        flash(f'Camera {name} removed', 'success')
        return redirect(url_for('settings'))
    
    except Exception as e:
        flash(f'Error removing camera: {str(e)}', 'danger')
        logger.error(f"Error removing camera: {str(e)}")
        return redirect(url_for('settings'))

@app.route('/get_available_webcams', methods=['GET'])
def get_available_webcams():
    """Get list of available webcams connected to the device"""
    try:
        webcams = []
        # Try to check the first 10 indices (0-9) to find available cameras
        for i in range(10):
            try:
                # Try multiple backends in order of preference
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
                webcam_opened = False
                
                for backend in backends:
                    try:
                        logger.info(f"Trying to open webcam {i} with backend {backend}")
                        cap = cv2.VideoCapture(i, backend)
                        
                        if cap.isOpened():
                            # Get camera information
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            if width > 0 and height > 0:  # Valid dimensions indicate a working camera
                                # Try to get one frame to make sure camera works
                                ret, _ = cap.read()
                                if ret:
                                    webcams.append({
                                        'id': i,
                                        'name': f"Webcam {i}",
                                        'resolution': f"{width}x{height}",
                                        'backend': backend
                                    })
                                    webcam_opened = True
                                    logger.info(f"Successfully opened webcam {i} with backend {backend}")
                                    break  # Use the first working backend
                    except Exception as be:
                        logger.warning(f"Error trying backend {backend} for webcam {i}: {str(be)}")
                    finally:
                        if 'cap' in locals() and cap is not None:
                            cap.release()
                
                if webcam_opened:
                    logger.info(f"Webcam {i} is available and working")
                else:
                    logger.info(f"Webcam {i} is not available or not working")
                    
            except Exception as ce:
                logger.warning(f"Error checking webcam {i}: {str(ce)}")
        
        logger.info(f"Found {len(webcams)} available webcams")
        return jsonify({
            'success': True,
            'webcams': webcams
        })
    except Exception as e:
        logger.error(f"Error getting available webcams: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e),
            'trace': traceback.format_exc()
        })

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Stop all cameras when the application is shutting down"""
    camera_manager.stop_all_cameras()

if __name__ == '__main__':
    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/faces", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Verify camera is working
    camera_working = False
    if camera_manager.get_active_camera_name():
        success, frame = camera_manager.read_active_camera()
        camera_working = success
        if success:
            logger.info("Camera test successful - video streaming is available")
        else:
            logger.warning("Camera test failed - video streaming may be limited")
    else:
        logger.warning("No active camera set - video streaming will be unavailable")
    
    # Run the Flask application
    app.run(debug=True) 