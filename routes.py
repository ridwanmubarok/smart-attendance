import os
import torch
import base64
import json
import numpy as np
import face_recognition
import cv2
from flask import render_template, request, jsonify, redirect, url_for, current_app, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
from datetime import datetime, timedelta
from transformers import DetrImageProcessor, DetrForObjectDetection
from models import Employee, FaceData, Attendance, db, Configuration
from app import app
from flask_sock import Sock
import threading
import queue
import time

# Initialize model and processor globally for better performance
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Initialize Flask-Sock
sock = Sock(app)

# Store active streams
active_streams = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    # Get quick stats
    total_employees = Employee.query.count()
    today = datetime.now().date()
    today_attendance = Attendance.query.filter(
        db.func.date(Attendance.check_in) == today
    ).count()
    late_today = Attendance.query.filter(
        db.func.date(Attendance.check_in) == today,
        Attendance.status == 'late'
    ).count()
    
    return render_template('index.html',
                         total_employees=total_employees,
                         today_attendance=today_attendance,
                         late_today=late_today)

@app.route('/employees')
def employees():
    # Get page number from request, default to 1
    page = request.args.get('page', 1, type=int)
    
    # Get records per page, default to 10
    per_page = request.args.get('per_page', 10, type=int)
    
    # Limit per_page to reasonable values
    if per_page > 100:
        per_page = 100
    # Ensure per_page is at least 1
    if per_page < 1:
        per_page = 1
    
    # Get paginated results
    pagination = Employee.query.order_by(Employee.name).paginate(page=page, per_page=per_page, error_out=False)
    employees_list = pagination.items
    
    # Get total records
    total_records = Employee.query.count()
    
    # Calculate page ranges for larger datasets
    page_range = 5  # Show 5 pages before and after current page
    start_page = max(1, page - page_range)
    end_page = min(pagination.pages, page + page_range)
    
    return render_template('employees.html', 
                          employees=employees_list,
                          pagination=pagination,
                          total_records=total_records,
                          page=page,
                          per_page=per_page,
                          start_page=start_page,
                          end_page=end_page)

@app.route('/employees/add', methods=['POST'])
def add_employee():
    try:
        employee = Employee(
            employee_id=request.form['employee_id'],
            name=request.form['name'],
            position=request.form['position'],
            department=request.form['department']
        )
        db.session.add(employee)
        db.session.commit()
        return redirect(url_for('employees'))
    except Exception as e:
        return str(e), 400

@app.route('/employees/<int:employee_id>/delete', methods=['POST'])
def delete_employee(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    db.session.delete(employee)
    db.session.commit()
    return '', 204

@app.route('/employees/<int:employee_id>')
def view_employee(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    attendance = Attendance.query.filter_by(employee_id=employee_id).order_by(Attendance.check_in.desc()).limit(10).all()
    return render_template('employee_detail.html', employee=employee, attendance=attendance)

@app.route('/employees/<int:employee_id>/face', methods=['GET', 'POST'])
def upload_employee_face(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload_face.html', error='No file part', employee=employee)
        
        files = request.files.getlist('image')
        success_count = 0
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process face encoding
                face_image = face_recognition.load_image_file(filepath)
                face_encodings = face_recognition.face_encodings(face_image)
                
                if not face_encodings:
                    os.remove(filepath)
                    continue
                
                # Create and save face data
                face_data = FaceData(
                    employee_id=employee.id,
                    image=filename
                )
                face_data.set_encoding(face_encodings[0])
                
                db.session.add(face_data)
                success_count += 1
        
        if success_count > 0:
            db.session.commit()
            # Return with success message for toast
            return redirect(url_for('view_employee', employee_id=employee.id, 
                                   upload_status='success', 
                                   count=success_count))
        else:
            return render_template('upload_face.html', error='No valid faces detected in the uploaded images', employee=employee)
    
    return render_template('upload_face.html', employee=employee)

@app.route('/employees/<int:employee_id>/face/<int:face_id>/delete', methods=['POST'])
def delete_employee_face(employee_id, face_id):
    try:
        # Find the face data by ID
        face_data = FaceData.query.get_or_404(face_id)
        
        # Verify that the face belongs to the correct employee
        if face_data.employee_id != employee_id:
            return jsonify({'error': 'Face data does not belong to this employee'}), 403
        
        # Get the image filename before deleting the record
        image_filename = face_data.image
        
        # Delete the face data record
        db.session.delete(face_data)
        db.session.commit()
        
        # Delete the actual image file
        try:
            image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image_filename)
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            # Log the error but don't fail if file deletion fails
            print(f"Error removing file: {str(e)}")
        
        return jsonify({
            'status': 'success',
            'message': 'Face data deleted successfully'
        }), 200
    except Exception as e:
        # Rollback in case of error
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Failed to delete face data: {str(e)}'
        }), 500

@app.route('/attendance/report')
def attendance_report():
    start_date = request.args.get('start_date', 
                                 datetime.now().replace(day=1).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', 
                               datetime.now().strftime('%Y-%m-%d'))
    department = request.args.get('department', '')
    
    # Get page number from request, default to 1
    page = request.args.get('page', 1, type=int)
    
    # Get records per page, default to 10 but allow users to choose
    per_page = request.args.get('per_page', 10, type=int)
    # Limit per_page to reasonable max value only
    if per_page > 100:
        per_page = 100
    # Ensure per_page is at least 1
    if per_page < 1:
        per_page = 1
    
    query = Attendance.query.join(Employee)
    
    if start_date:
        query = query.filter(Attendance.check_in >= start_date)
    if end_date:
        query = query.filter(Attendance.check_in <= end_date + ' 23:59:59')
    if department:
        query = query.filter(Employee.department == department)
    
    # Order by check_in date descending
    query = query.order_by(Attendance.check_in.desc())
    
    # Get paginated results
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    attendance_records = pagination.items
    
    # Get total records for summary
    total_records = query.count()
    
    # Get unique departments for filter
    departments = db.session.query(Employee.department).distinct().all()
    
    # Calculate page ranges for larger datasets
    page_range = 5  # Show 5 pages before and after current page
    start_page = max(1, page - page_range)
    end_page = min(pagination.pages, page + page_range)
    
    return render_template('attendance_report.html',
                         attendance_records=attendance_records,
                         pagination=pagination,
                         total_records=total_records,
                         departments=[d[0] for d in departments if d[0]],
                         page=page,
                         per_page=per_page,
                         start_page=start_page,
                         end_page=end_page)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get confidence threshold from configuration
        confidence_threshold = float(Configuration.get_value('confidence_threshold', '80')) / 100
        
        # Remove the data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to numpy array
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            return jsonify({"detected_employees": []})
            
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Get all known faces from database
        known_faces = FaceData.query.all()
        detected_employees = []
        
        current_time = datetime.now()
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = current_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            if known_faces:
                known_encodings = [face.get_encoding() for face in known_faces]
                known_employees = [face.employee for face in known_faces]
                
                # Compare face with known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        employee = known_employees[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        
                        # Only process if confidence is higher than configured threshold
                        if confidence > confidence_threshold:
                            # Get check-in time from configuration
                            check_in_time = Configuration.get_value('check_in_time', '08:00')
                            check_in_hour = int(check_in_time.split(':')[0])
                            
                            # Get all attendance records for today for this employee
                            today_attendance = Attendance.query.filter(
                                Attendance.employee_id == employee.id,
                                Attendance.check_in >= today_start,
                                Attendance.check_in <= today_end
                            ).order_by(Attendance.check_in.desc()).all()
                            
                            status = None
                            attendance_action = None

                            if not today_attendance:
                                # No attendance today - create check-in
                                status = "Checked In"
                                attendance = Attendance(
                                    employee_id=employee.id,
                                    check_in=current_time,
                                    status='present' if current_time.hour < check_in_hour else 'late'
                                )
                                attendance_action = attendance
                            else:
                                latest_attendance = today_attendance[0]
                                
                                # If the latest record has no check_out
                                if latest_attendance.check_out is None:
                                    # Check if enough time has passed for check-out (minimum 1 hour)
                                    time_diff = current_time - latest_attendance.check_in
                                    if time_diff.total_seconds() >= 3600:  # 1 hour in seconds
                                        status = "Checked Out"
                                        latest_attendance.check_out = current_time
                                        attendance_action = latest_attendance
                                else:
                                    # If last record has check_out and it's been at least 1 hour since last check-out
                                    time_since_last_checkout = current_time - latest_attendance.check_out
                                    if time_since_last_checkout.total_seconds() >= 3600:
                                        status = "Checked In"
                                        attendance = Attendance(
                                            employee_id=employee.id,
                                            check_in=current_time,
                                            status='present' if current_time.hour < check_in_hour else 'late'
                                        )
                                        attendance_action = attendance

                            # Only process if we have an action to take
                            if attendance_action and status:
                                if status == "Checked In":
                                    db.session.add(attendance_action)
                                db.session.commit()
                                
                                detected_employees.append({
                                    "name": employee.name,
                                    "employee_id": employee.employee_id,
                                    "status": status,
                                    "confidence": float(confidence),
                                    "face_location": face_location,
                                    "time": current_time.strftime("%H:%M:%S")
                                })
                            else:
                                # If no action taken, still show detection but with "Already Processed" status
                                detected_employees.append({
                                    "name": employee.name,
                                    "employee_id": employee.employee_id,
                                    "status": "Already Processed",
                                    "confidence": float(confidence),
                                    "face_location": face_location,
                                    "time": current_time.strftime("%H:%M:%S")
                                })
        
        return jsonify({
            "detected_employees": detected_employees
        })
    
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', error='No file part')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('upload.html', uploaded_file_url=url_for('uploaded_file', filename=filename))
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect', methods=['GET', 'POST'])
def detect_faces():
    if request.method == 'POST':
        try:
            image_path = request.form.get('image_path')
            if not image_path:
                raise ValueError("No image path provided")
            
            # Remove the URL prefix and get the filename
            filename = image_path.split('/')[-1]
            full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Image file not found")
            
            # Load and process the image
            image = Image.open(full_path)
            detections = process_image(image)
            
            return render_template('result.html', 
                                detections=detections,
                                image_path=url_for('uploaded_file', filename=filename))
        
        except Exception as e:
            return render_template('upload.html', error=str(e))
    
    return render_template('upload.html')

@app.route('/live')
def live_detection():
    return render_template('live.html')

def process_image(image):
    # Prepare the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Get predictions
    outputs = model(**inputs)
    
    # Convert outputs to probabilities
    probas = outputs.logits.softmax(-1)[0, :, :]
    
    # Convert boxes from center format to corner format
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    # Get detections
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.9:
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                'box': box,
                'score': round(score.item(), 3),
                'label': model.config.id2label[label.item()]
            })
    return detections

@app.route('/config')
def config_page():
    # Get all configurations
    config = {
        'check_in_time': Configuration.get_value('check_in_time', '08:00'),
        'check_out_time': Configuration.get_value('check_out_time', '17:00'),
        'confidence_threshold': Configuration.get_value('confidence_threshold', '80'),
        'detection_interval': Configuration.get_value('detection_interval', '1000')
    }
    return render_template('config.html', config=config)

@app.route('/api/config/time', methods=['POST'])
def save_time_config():
    try:
        data = request.get_json()
        check_in_time = data.get('check_in_time')
        check_out_time = data.get('check_out_time')

        if not check_in_time or not check_out_time:
            return jsonify({'error': 'Missing required fields'}), 400

        Configuration.set_value('check_in_time', check_in_time)
        Configuration.set_value('check_out_time', check_out_time)

        return jsonify({'message': 'Time configuration saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/face', methods=['POST'])
def save_face_config():
    try:
        data = request.get_json()
        confidence_threshold = data.get('confidence_threshold')
        detection_interval = data.get('detection_interval')

        if confidence_threshold is None or detection_interval is None:
            return jsonify({'error': 'Missing required fields'}), 400

        Configuration.set_value('confidence_threshold', str(confidence_threshold))
        Configuration.set_value('detection_interval', str(detection_interval))

        return jsonify({'message': 'Face recognition settings saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_all_attendance', methods=['POST'])
def delete_all_attendance():
    try:
        # Delete all records from the attendance table
        Attendance.query.delete()
        # Commit the changes
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'All attendance records have been deleted successfully'
        }), 200
    except Exception as e:
        # Rollback in case of error
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Failed to delete records: {str(e)}'
        }), 500

@app.route('/api/today-attendance')
def get_today_attendance():
    try:
        # Get today's date range
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())
        
        # Get all attendance records for today
        attendance_records = (Attendance.query
            .join(Employee)
            .filter(
                Attendance.check_in >= today_start,
                Attendance.check_in <= today_end
            )
            .order_by(Attendance.check_in.desc())
            .all())
        
        # Count statistics
        present_count = sum(1 for record in attendance_records if record.status == 'present')
        late_count = sum(1 for record in attendance_records if record.status == 'late')
        
        # Format attendance records
        attendance_data = []
        for record in attendance_records:
            attendance_data.append({
                'employee_name': record.employee.name,
                'employee_id': record.employee.employee_id,
                'department': record.employee.department,
                'check_in': record.check_in.strftime('%H:%M:%S'),
                'check_out': record.check_out.strftime('%H:%M:%S') if record.check_out else None,
                'status': record.status
            })
        
        return jsonify({
            'status': 'success',
            'total_count': len(attendance_records),
            'present_count': present_count,
            'late_count': late_count,
            'attendance': attendance_data
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/config/camera', methods=['GET', 'POST'])
def camera_config():
    try:
        if request.method == 'POST':
            data = request.get_json()
            
            # Validate required fields
            if not data or 'camera_source' not in data:
                return jsonify({'error': 'Missing required fields'}), 400

            # Save camera configuration
            Configuration.set_value('camera_source', data['camera_source'])
            
            if data['camera_source'] == 'webcam':
                Configuration.set_value('active_camera_id', data['active_camera_id'])
            else:
                # For CCTV, construct and save the complete RTSP URL
                rtsp_url = data['rtsp_url'].strip()
                username = data.get('cctv_username', '').strip()
                password = data.get('cctv_password', '').strip()

                # If URL doesn't contain credentials and credentials are provided, add them
                if '@' not in rtsp_url and username and password:
                    from urllib.parse import urlparse, urlunparse
                    parsed = urlparse(rtsp_url)
                    netloc = f"{username}:{password}@{parsed.netloc}"
                    rtsp_url = urlunparse(parsed._replace(netloc=netloc))

                Configuration.set_value('rtsp_url', rtsp_url)
                Configuration.set_value('cctv_username', username)
                Configuration.set_value('cctv_password', password)

            return jsonify({'message': 'Camera configuration saved successfully'}), 200
        else:
            # GET method - return current configuration
            camera_source = Configuration.get_value('camera_source', 'webcam')
            config = {
                'camera_source': camera_source,
                'active_camera_id': Configuration.get_value('active_camera_id', ''),
                'rtsp_url': Configuration.get_value('rtsp_url', ''),
                'cctv_username': Configuration.get_value('cctv_username', ''),
                'cctv_password': Configuration.get_value('cctv_password', '')
            }
            return jsonify(config), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/camera/preview', methods=['POST'])
def camera_preview():
    try:
        data = request.get_json()
        print(f"Received preview request with data: {data}")
        
        # Validate required fields for CCTV preview
        if not data or 'rtsp_url' not in data:
            return jsonify({'error': 'Missing RTSP URL'}), 400

        # Get credentials and URL
        rtsp_url = data['rtsp_url'].strip()
        username = data.get('cctv_username', '').strip()
        password = data.get('cctv_password', '').strip()

        print(f"Original RTSP URL: {rtsp_url}")
        print(f"Username: {username}")
        print(f"Password: {'*' * len(password) if password else 'None'}")

        # Validate RTSP URL format for Tapo TC60
        if not rtsp_url.startswith('rtsp://'):
            return jsonify({'error': 'Invalid RTSP URL format. Must start with rtsp://'}), 400

        # If URL already contains credentials, use it as is
        if '@' not in rtsp_url and username and password:
            try:
                # Parse the URL to insert credentials
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(rtsp_url)
                netloc = f"{username}:{password}@{parsed.netloc}"
                rtsp_url = urlunparse(parsed._replace(netloc=netloc))
                print(f"Modified RTSP URL with credentials: {rtsp_url}")
            except Exception as e:
                print(f"Error parsing RTSP URL: {str(e)}")
                return jsonify({'error': f'Invalid RTSP URL format: {str(e)}'}), 400

        # Test network connectivity first
        import socket
        try:
            ip = rtsp_url.split('@')[-1].split(':')[0]
            port = 554
            print(f"Testing connection to {ip}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 seconds timeout
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result != 0:
                return jsonify({
                    'error': 'Cannot connect to camera. Please check:'
                    '\n1. Camera is powered on'
                    '\n2. Camera is connected to the same network'
                    '\n3. IP address is correct'
                    '\n4. Port 554 is not blocked by firewall'
                }), 400
        except Exception as e:
            print(f"Network connectivity test failed: {str(e)}")
            return jsonify({
                'error': f'Network connectivity test failed: {str(e)}'
            }), 400

        # Test the RTSP connection with timeout
        print("Attempting to open RTSP connection...")
        
        # Set OpenCV RTSP preferences
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 seconds timeout
        
        if not cap.isOpened():
            print("Failed to open RTSP connection")
            return jsonify({
                'error': 'Failed to connect to CCTV camera. Please check:'
                '\n1. RTSP service is enabled on the camera'
                '\n2. Username and password are correct'
                '\n3. RTSP URL format is correct'
                f'\n4. Debug info: URL={rtsp_url}'
            }), 400

        print("RTSP connection opened successfully")
        
        # Read a frame to verify the stream
        print("Attempting to read frame...")
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from RTSP stream")
            cap.release()
            return jsonify({
                'error': 'Failed to read from CCTV camera stream. Please check:'
                '\n1. Camera is powered on'
                '\n2. Network bandwidth is sufficient'
                '\n3. RTSP stream is not being accessed by other applications'
                f'\n4. Debug info: URL={rtsp_url}'
            }), 400

        print("Successfully read frame from RTSP stream")
        cap.release()
        return jsonify({'message': 'Successfully connected to CCTV camera'}), 200

    except Exception as e:
        print(f"Error in camera_preview: {str(e)}")
        return jsonify({'error': f'Error testing CCTV connection: {str(e)}'}), 500

@app.route('/api/config')
def get_config():
    try:
        config = Configuration.query.first()
        if not config:
            return jsonify({
                'detection_interval': 1000,
                'confidence_threshold': 80,
                'check_in_time': '08:00',
                'check_out_time': '17:00'
            })
        
        return jsonify({
            'detection_interval': config.detection_interval,
            'confidence_threshold': config.confidence_threshold,
            'check_in_time': config.check_in_time.strftime('%H:%M'),
            'check_out_time': config.check_out_time.strftime('%H:%M')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_attendance/<int:attendance_id>', methods=['POST'])
def delete_attendance(attendance_id):
    try:
        # Find the attendance record by ID
        attendance = Attendance.query.get_or_404(attendance_id)
        
        # Delete the record
        db.session.delete(attendance)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Attendance record deleted successfully'
        }), 200
    except Exception as e:
        # Rollback in case of error
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Failed to delete record: {str(e)}'
        }), 500

def process_rtsp_stream(rtsp_url, ws):
    try:
        print(f"Starting RTSP stream from: {rtsp_url}")
        
        # Set OpenCV RTSP preferences
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|reorder_queue_size;0|buffer_size;0'
        
        # Open RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        
        # Configure capture properties for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print(f"Failed to open RTSP stream: {rtsp_url}")
            ws.send(json.dumps({'error': 'Failed to open RTSP stream'}))
            return
        
        print(f"Successfully opened RTSP stream: {rtsp_url}")
        
        # Get actual stream properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Stream properties - FPS: {fps}, Resolution: {width}x{height}")
        
        frame_count = 0
        start_time = time.time()
        last_frame_time = 0
        frame_interval = 1.0 / 30  # Target 30 FPS
        
        # JPEG encoding parameters - increased quality for better color
        jpeg_encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        
        while True:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                continue  # Skip if too soon for next frame
                
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from RTSP stream")
                break
            
            frame_count += 1
            elapsed_time = current_time - start_time
            
            if elapsed_time >= 1:  # Log FPS every second
                actual_fps = frame_count / elapsed_time
                print(f"Actual FPS: {actual_fps:.2f}")
                frame_count = 0
                start_time = current_time
            
            try:
                # No need to convert BGR to RGB since we're sending directly to browser
                # Browser expects BGR format when decoding JPEG
                
                # Apply light color correction if needed
                # frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)  # Optional: adjust brightness/contrast
                
                # Encode frame to JPEG directly from BGR
                ret, buffer = cv2.imencode('.jpg', frame, jpeg_encode_param)
                if not ret:
                    print("Failed to encode frame to JPEG")
                    continue
                
                # Send raw bytes through WebSocket
                ws.send(buffer.tobytes())
                last_frame_time = current_time
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                break
            
    except Exception as e:
        print(f"Error processing RTSP stream: {str(e)}")
        try:
            ws.send(json.dumps({'error': str(e)}))
        except:
            pass
    finally:
        if cap:
            cap.release()
        if ws in active_streams:
            del active_streams[ws]
        print("RTSP stream processing stopped")

@sock.route('/ws/stream')
def stream_socket(ws):
    try:
        # Get RTSP URL from initial message
        message = ws.receive()
        data = json.loads(message)
        rtsp_url = data.get('rtsp_url')
        
        if not rtsp_url:
            ws.send(json.dumps({'error': 'No RTSP URL provided'}))
            return

        # Set OpenCV RTSP preferences
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|reorder_queue_size;0|buffer_size;0'
        
        # Initialize video capture
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            ws.send(json.dumps({'error': 'Failed to open RTSP stream'}))
            return

        last_frame_time = 0
        frame_interval = 1.0 / 30  # Target 30 FPS

        while True:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent CPU overload
                continue

            ret, frame = cap.read()
            if not ret:
                ws.send(json.dumps({'error': 'Failed to read frame'}))
                break

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Send frame as binary data
            try:
                ws.send(buffer.tobytes())
            except Exception as e:
                print(f"Error sending frame: {e}")
                break

            last_frame_time = current_time

    except Exception as e:
        print(f"Stream error: {e}")
        try:
            ws.send(json.dumps({'error': str(e)}))
        except:
            pass
    finally:
        if 'cap' in locals():
            cap.release()