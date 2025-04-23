# Smart Attendance System (AIoT-Based)

A modern face detection and recognition system that combines Artificial Intelligence (AI) and Internet of Things (IoT) technologies. Built with Flask, DETR (DEtection TRansformer), OpenCV, and deep learning models for accurate and efficient attendance management.

## Developed By

This project was developed as part of the Internet of Things course at Asia Cyber University by:

**Team 3**

- Akmal Fauzy
- Faisal Dzulfikar
- Heru Saputra
- Ridwan Mubarok

## Key Features

### AI Capabilities

- **DETR-based Detection**: Utilizing Facebook's Detection Transformer (DETR) for state-of-the-art object detection
- **Face Recognition**: Deep learning-based face recognition with high accuracy
- **Real-time Processing**: Live video stream processing with optimized performance
- **Confidence Scoring**: Advanced scoring system for detection reliability

### IoT Integration

- **Camera Integration**: Support for multiple camera types (webcam, IP cameras, CCTV)
- **Real-time Data Collection**: Continuous monitoring and data gathering
- **Edge Computing Ready**: Optimized for edge device deployment
- **Device Management**: Camera and sensor management interface

### System Features

- **Live Detection**: Real-time face detection and recognition
- **Attendance Tracking**: Automated attendance recording and management
- **Modern UI/UX**: Responsive and intuitive user interface
- **Analytics Dashboard**: Attendance patterns and insights
- **Multi-device Support**: Works across various devices and cameras

## Technical Architecture

### AI Components

- **DETR (Detection Transformer)**
  - End-to-end object detection model
  - Transformer-based architecture for improved accuracy
  - Direct set prediction approach
  - Parallel decoding for faster processing

### Face Recognition Pipeline

1. Face Detection using DETR
2. Face Alignment and Normalization
3. Feature Extraction using Deep Neural Networks
4. Face Matching and Recognition
5. Confidence Score Calculation

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Webcam or IP camera
- pip (Python package manager)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/smart-attendance.git
cd smart-attendance
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env file with your configuration
```

## System Configuration

### Camera Setup

1. Local Webcam:

   ```python
   CAMERA_SOURCE = 'webcam'
   CAMERA_ID = 0  # Default webcam
   ```

2. IP Camera:
   ```python
   CAMERA_SOURCE = 'ip_camera'
   CAMERA_URL = 'rtsp://your_camera_ip'
   ```

### AI Model Configuration

```python
DETR_CONFIG = {
    'model_type': 'detr_resnet50',
    'confidence_threshold': 0.85,
    'device': 'cuda'  # or 'cpu'
}
```

## Running the Application

1. Initialize the database:

```bash
flask db init
flask db migrate
flask db upgrade
```

2. Start the Flask development server:

```bash
python app.py
```

3. Access the web interface:

```
http://localhost:5000
```

## Usage Guide

### 1. Face Registration

- Navigate to "Upload Face"
- Enter employee/student details
- Upload a clear front-facing photo
- System will process and store face embeddings

### 2. Live Detection

- Access "Live Detection" page
- Grant camera permissions
- System will automatically:
  - Detect faces using DETR
  - Match against registered faces
  - Record attendance with timestamp
  - Display confidence scores

### 3. Attendance Management

- View real-time attendance status
- Access historical attendance data
- Generate attendance reports
- Export data in various formats

## Project Structure

```
smart-attendance/
├── app/
│   ├── ai/
│   │   ├── detr/           # DETR model implementation
│   │   │   └── utils/         # AI utilities
│   │   └── sensors/       # Additional sensor support
│   ├── models/           # Database models
│   ├── templates/        # HTML templates
│   └── static/          # Static assets
├── config/             # Configuration files
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Performance Optimization

- GPU acceleration support
- Batch processing for multiple faces
- Edge computing capabilities
- Optimized video stream processing

## Security Features

- Encrypted face data storage
- Secure API endpoints
- Access control and authentication
- Data privacy compliance

## Contributing

We welcome contributions! Please check our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Development Team

This project was developed as an academic project for the Internet of Things course at Asia Cyber University.

### Team 3 Members

- **Akmal Fauzy**
- **Faisal Dzulfikar**
- **Heru Saputra**
- **Ridwan Mubarok**

## Acknowledgments

- DETR: Facebook AI Research
- OpenCV Community
- Flask Framework
- PyTorch Team
- Asia Cyber University Faculty and Staff
