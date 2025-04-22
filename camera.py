import os
import cv2
import time
import threading
import logging
import platform
from enum import Enum
from datetime import datetime
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)

class CameraSource(Enum):
    """Enum untuk tipe sumber kamera"""
    WEBCAM = 0
    IP_CAMERA = 1
    VIDEO_FILE = 2
    SMARTPHONE = 3
    RTSP = 4

class Camera:
    """Kelas untuk mengelola kamera tunggal"""
    
    def __init__(self, name, source_type, source, resolution=(640, 480)):
        """
        Initialize a camera instance
        
        Args:
            name (str): Name of the camera
            source_type (CameraSource): Type of camera source (webcam, video file, RTSP)
            source: Camera index for webcam, file path for video file, URL for RTSP
            resolution (tuple): Desired resolution (width, height)
        """
        self.name = name
        self.source_type = source_type
        self.source = source
        self.resolution = resolution
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.latest_frame = None
        self.frame_time = 0
        self.frame_count = 0
        self.start_time = 0
        self.last_read_success = False
        
        # Track FPS
        self.fps = 0
        self.fps_frames = 0
        self.fps_start = 0
        
        # Track consecutive failures
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        logger.info(f"Camera '{name}' initialized: {source_type.name}, Source: {source}, Resolution: {resolution}")
    
    def open(self):
        """
        Open the camera with appropriate backend
        
        Returns:
            bool: True if camera was opened successfully, False otherwise
        """
        # Close any existing capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Define preferred backends based on platform
        os_name = platform.system()
        
        if os_name == "Windows":
            backends = [
                cv2.CAP_DSHOW,    # DirectShow (preferred for Windows)
                cv2.CAP_MSMF,     # Microsoft Media Foundation
                cv2.CAP_ANY       # Auto-detect (fallback)
            ]
        else:  # Linux, macOS, etc.
            backends = [
                cv2.CAP_V4L2,     # Video4Linux2 (Linux)
                cv2.CAP_AVFOUNDATION,  # AVFoundation framework (macOS)
                cv2.CAP_ANY       # Auto-detect (fallback)
            ]
            
        # Video file or RTSP stream doesn't need multiple backends
        if self.source_type != CameraSource.WEBCAM:
            backends = [cv2.CAP_FFMPEG]  # Use FFMPEG for files and streams
        
        # Try each backend with retries
        max_retries = 3
        delay_between_retries = 1.0  # seconds
        
        for backend in backends:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Trying to open camera '{self.name}' with backend {backend}, attempt {attempt+1}/{max_retries}")
                    
                    if self.source_type == CameraSource.WEBCAM:
                        self.cap = cv2.VideoCapture(self.source, backend)
                    else:
                        self.cap = cv2.VideoCapture(self.source)
                    
                    # Set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    
                    # Check if camera is opened successfully
                    if self.cap.isOpened():
                        # Try to read a test frame to validate
                        ret, test_frame = self.cap.read()
                        
                        if ret and test_frame is not None and test_frame.shape[0] > 0 and test_frame.shape[1] > 0:
                            # Get and log actual resolution (may differ from requested)
                            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            logger.info(f"Camera '{self.name}' opened successfully with backend {backend}")
                            logger.info(f"Actual resolution: {actual_width}x{actual_height}")
                            
                            # Set buffer size properties to minimize latency
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            return True
                        else:
                            logger.warning(f"Camera '{self.name}' opened but failed to read test frame, retrying...")
                            self.cap.release()
                            self.cap = None
                    else:
                        logger.warning(f"Failed to open camera '{self.name}' with backend {backend}, attempt {attempt+1}")
                
                except Exception as e:
                    logger.error(f"Error opening camera '{self.name}' with backend {backend}: {str(e)}")
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                
                # Wait before retrying
                time.sleep(delay_between_retries)
        
        # All backends and retry attempts failed
        logger.error(f"Failed to open camera '{self.name}' after trying all backends and {max_retries} attempts per backend")
        return False
    
    def start(self):
        """
        Start the camera thread
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            logger.warning(f"Camera '{self.name}' is already running")
            return True
        
        # Open the camera if not already open
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                logger.error(f"Failed to open camera '{self.name}', cannot start")
                return False
        
        # Start thread
        self.running = True
        self.thread = threading.Thread(target=self._update, name=f"Camera-{self.name}")
        self.thread.daemon = True
        self.thread.start()
        
        self.start_time = time.time()
        self.fps_start = time.time()
        logger.info(f"Camera '{self.name}' started")
        
        return True
    
    def stop(self):
        """Stop the camera thread"""
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=3.0)  # Wait for thread to finish with timeout
            self.thread = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger.info(f"Camera '{self.name}' stopped")
    
    def _update(self):
        """Thread function to continuously update frames"""
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning(f"Camera '{self.name}' lost connection, attempting to reopen")
                    if not self.open():
                        time.sleep(1.0)  # Wait before retrying
                        continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    # Reset failure counter on success
                    self.consecutive_failures = 0
                    
                    # Update frame with thread safety
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.last_read_success = True
                        self.frame_time = time.time()
                        self.frame_count += 1
                    
                    # Calculate FPS
                    self.fps_frames += 1
                    elapsed = time.time() - self.fps_start
                    if elapsed >= 1.0:
                        self.fps = self.fps_frames / elapsed
                        self.fps_frames = 0
                        self.fps_start = time.time()
                else:
                    self.consecutive_failures += 1
                    logger.warning(f"Failed to read frame from camera '{self.name}', failure #{self.consecutive_failures}")
                    
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error(f"Too many consecutive failures ({self.consecutive_failures}) for camera '{self.name}', reopening")
                        self.cap.release()
                        self.cap = None
                        self.consecutive_failures = 0
                
                # Small delay to prevent CPU overuse
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in camera thread for '{self.name}': {str(e)}")
                self.consecutive_failures += 1
                time.sleep(0.5)  # Longer delay after error
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"Too many errors in camera thread for '{self.name}', attempting to reopen")
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    self.consecutive_failures = 0
    
    def read(self):
        """
        Read the latest frame from the camera
        
        Returns:
            tuple: (success, frame) where success is True if a frame was available
        """
        with self.lock:
            if self.latest_frame is not None and self.last_read_success:
                # Return a copy to avoid reference issues
                return True, self.latest_frame.copy()
            else:
                # Create a black frame with error message if no valid frame is available
                if self.resolution:
                    black_frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
                    cv2.putText(
                        black_frame,
                        f"No frame available from {self.name}",
                        (20, self.resolution[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8,
                        (255, 255, 255),
                        2
                    )
                    return False, black_frame
                return False, None
    
    def is_running(self):
        """Check if camera is currently running"""
        return self.running and self.thread is not None and self.thread.is_alive()
    
    def get_fps(self):
        """Get current FPS (frames per second)"""
        return self.fps
    
    def get_camera_info(self):
        """Get detailed camera information including status"""
        info = {
            'type': self.source_type.name,
            'source': self.source,
            'resolution': self.resolution,
            'running': self.is_running(),
            'fps': self.get_fps(),
            'consecutive_failures': self.consecutive_failures,
            'max_consecutive_failures': self.max_consecutive_failures
        }
        
        # Add more detailed status
        if self.cap is not None and self.cap.isOpened():
            info['status'] = 'connected'
        elif self.running:
            info['status'] = 'reconnecting'
        else:
            info['status'] = 'disconnected'
            
        return info

class CameraManager:
    """Kelas untuk mengelola multiple kamera"""
    
    def __init__(self):
        """Inisialisasi camera manager"""
        self.cameras = {}  # Dict dari nama kamera ke objek Camera
        self.active_camera = None  # Nama kamera aktif
    
    def add_camera(self, name, source_type=CameraSource.WEBCAM, source_path=0, resolution=(640, 480)):
        """
        Menambahkan kamera baru
        
        Args:
            name (str): Nama kamera untuk identifikasi
            source_type (CameraSource): Tipe sumber kamera
            source_path (str/int): Path ke sumber
            resolution (tuple): Resolusi kamera (lebar, tinggi)
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Cek apakah kamera dengan nama yang sama sudah ada
            if name in self.cameras:
                print(f"Camera with name '{name}' already exists")
                return False
            
            # Buat objek Camera baru
            camera = Camera(name, source_type, source_path, resolution)
            
            # Simpan ke dictionary
            self.cameras[name] = camera
            
            # Jika belum ada kamera aktif, set ini sebagai aktif
            if self.active_camera is None:
                self.active_camera = name
            
            return True
        except Exception as e:
            print(f"Error adding camera: {str(e)}")
            return False
    
    def remove_camera(self, name):
        """
        Menghapus kamera
        
        Args:
            name (str): Nama kamera yang akan dihapus
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if name not in self.cameras:
                print(f"Camera '{name}' not found")
                return False
            
            # Stop kamera jika sedang berjalan
            camera = self.cameras[name]
            if camera.is_running():
                camera.stop()
            
            # Hapus dari dictionary
            del self.cameras[name]
            
            # Reset active camera jika yang dihapus adalah active camera
            if self.active_camera == name:
                if self.cameras:
                    # Set kamera aktif ke kamera pertama yang tersisa
                    self.active_camera = list(self.cameras.keys())[0]
                else:
                    self.active_camera = None
            
            return True
        except Exception as e:
            print(f"Error removing camera: {str(e)}")
            return False
    
    def start_camera(self, name):
        """
        Memulai kamera
        
        Args:
            name (str): Nama kamera
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Cek apakah kamera ada
            if name not in self.cameras:
                print(f"Camera '{name}' not found")
                return False
            
            camera = self.cameras[name]
            
            # Jika kamera sudah berjalan, tidak perlu start ulang
            if camera.is_running():
                # Set sebagai kamera aktif
                self.active_camera = name
                return True
            
            # Start kamera
            for attempt in range(3):  # Try up to 3 times
                print(f"Attempting to start camera {name} (attempt {attempt+1}/3)")
                if camera.start():
                    print(f"Camera {name} started successfully")
                    # Set sebagai kamera aktif
                    self.active_camera = name
                    return True
                else:
                    print(f"Failed to start camera {name} on attempt {attempt+1}")
                    time.sleep(1)  # Wait before retry
            
            print(f"All attempts to start camera {name} failed")
            return False
            
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False
    
    def stop_camera(self, name):
        """
        Menghentikan kamera
        
        Args:
            name (str): Nama kamera
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if name not in self.cameras:
                print(f"Camera '{name}' not found")
                return False
            
            camera = self.cameras[name]
            camera.stop()
            
            # Reset active camera jika yang dihentikan adalah active camera
            if self.active_camera == name:
                self.active_camera = None
            
            return True
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")
            return False
    
    def stop_all_cameras(self):
        """Menghentikan semua kamera"""
        for name, camera in self.cameras.items():
            camera.stop()
        
        self.active_camera = None
    
    def set_active_camera(self, name):
        """
        Mengatur kamera aktif
        
        Args:
            name (str): Nama kamera
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if name not in self.cameras:
                print(f"Camera '{name}' not found")
                return False
            
            # Stop kamera aktif sebelumnya jika ada
            if self.active_camera and self.active_camera != name:
                if self.cameras[self.active_camera].is_running():
                    self.cameras[self.active_camera].stop()
            
            # Set kamera aktif baru
            self.active_camera = name
            
            # Start kamera aktif jika belum berjalan
            if not self.cameras[name].is_running():
                success = self.cameras[name].start()
                if not success:
                    print(f"Failed to start camera '{name}'")
                    return False
            
            return True
        except Exception as e:
            print(f"Error setting active camera: {str(e)}")
            return False
    
    def read_active_camera(self):
        """
        Membaca frame dari kamera aktif
        
        Returns:
            tuple: (ret, frame) seperti cv2.VideoCapture.read()
        """
        if not self.active_camera or self.active_camera not in self.cameras:
            return False, None
        
        camera = self.cameras[self.active_camera]
        return camera.read()
    
    def get_active_camera_name(self):
        """Mendapatkan nama kamera aktif"""
        return self.active_camera
    
    def get_available_cameras(self):
        """
        Mendapatkan daftar kamera yang tersedia
        
        Returns:
            dict: Dictionary dari nama kamera ke objek Camera
        """
        return self.cameras
    
    def get_camera_info(self, name):
        """
        Mendapatkan informasi kamera
        
        Args:
            name (str): Nama kamera
            
        Returns:
            dict: Informasi kamera atau None jika tidak ditemukan
        """
        if name not in self.cameras:
            return None
        
        camera = self.cameras[name]
        
        return {
            'name': name,
            'type': camera.source_type.name,
            'source': camera.source,
            'resolution': camera.resolution,
            'running': camera.is_running(),
            'fps': camera.get_fps(),
            'consecutive_failures': camera.consecutive_failures,
            'max_consecutive_failures': camera.max_consecutive_failures
        } 