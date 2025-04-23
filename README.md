# Smart Attendance System

A modern face detection and recognition system built with Flask, OpenCV, and deep learning models.

## Features

- Upload and process images for face detection
- Real-time face detection using webcam
- Face recognition and matching against stored faces
- Modern and responsive UI
- Easy to use interface

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (for live detection)

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

4. Set up the environment variables (optional):

```bash
cp .env.example .env
# Edit .env file with your configuration
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

3. Open your web browser and navigate to:

```
http://localhost:5000
```

## Usage

1. **Upload Image**

   - Click on "Upload Image" in the navigation
   - Select an image file
   - Click "Upload" to process the image
   - View detection results

2. **Live Detection**

   - Click on "Live Detection" in the navigation
   - Grant camera permissions when prompted
   - Click "Start Camera" to begin detection
   - View real-time detection results

3. **Upload Face**
   - Click on "Upload Face" in the navigation
   - Enter the person's name
   - Upload a clear front-facing photo
   - The face will be added to the recognition database

## Project Structure

```
smart-attendance/
├── app.py              # Main Flask application
├── models.py           # Database models
├── routes.py           # Application routes
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── result.html
│   ├── live.html
│   └── upload_face.html
├── media/             # Uploaded images
│   └── faces/        # Stored face images
└── instance/         # SQLite database
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
