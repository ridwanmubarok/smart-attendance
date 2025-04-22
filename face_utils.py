import cv2
import face_recognition
import numpy as np
from datetime import datetime

class FaceUtils:
    def __init__(self, tolerance=0.6, model="hog"):
        """
        Inisialisasi utilitas pengenalan wajah
        
        Args:
            tolerance (float): Toleransi jarak untuk pengenalan wajah (0.0-1.0), semakin kecil semakin ketat
            model (str): Model deteksi wajah ('hog' cepat atau 'cnn' akurat)
        """
        self.tolerance = tolerance
        self.model = model
        # Kamus untuk memetakan id karyawan ke encoding wajah
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
    
    def load_known_faces(self, known_faces):
        """
        Memuat encoding wajah yang sudah dikenal dari database
        
        Args:
            known_faces (list): List objek dengan id, name, dan face_encoding
        """
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        
        for face in known_faces:
            self.known_face_encodings.append(face['face_encoding'])
            self.known_face_ids.append(face['id'])
            self.known_face_names.append(face['name'])
    
    def detect_faces(self, frame):
        """
        Mendeteksi lokasi wajah dalam frame
        
        Args:
            frame (numpy.ndarray): Frame gambar untuk deteksi wajah
            
        Returns:
            list: List koordinat (top, right, bottom, left) untuk setiap wajah
        """
        # Convert BGR (OpenCV format) ke RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteksi semua wajah dalam frame
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        
        return face_locations
    
    def encode_face(self, frame, face_location=None):
        """
        Mengubah wajah menjadi encoding numerik
        
        Args:
            frame (numpy.ndarray): Frame gambar yang berisi wajah
            face_location (tuple, optional): Lokasi wajah (top, right, bottom, left)
            
        Returns:
            numpy.ndarray: Encoding wajah
        """
        # Convert BGR (OpenCV format) ke RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if face_location:
            # Jika lokasi wajah diberikan, encode hanya wajah itu
            encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
        else:
            # Jika tidak, encode semua wajah yang ditemukan
            encoding = face_recognition.face_encodings(rgb_frame)
            if len(encoding) > 0:
                encoding = encoding[0]  # Ambil yang pertama saja
            else:
                return None
        
        return encoding
    
    def recognize_face(self, face_encoding):
        """
        Mengenali wajah dengan membandingkan dengan wajah yang sudah dikenal
        
        Args:
            face_encoding (numpy.ndarray): Encoding wajah yang akan dikenali
            
        Returns:
            tuple: (id, name, confidence) jika wajah dikenali, atau (None, None, 0.0)
        """
        if len(self.known_face_encodings) == 0:
            return None, None, 0.0
        
        # Bandingkan dengan semua wajah yang dikenal
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        # Dapatkan index jarak terkecil
        best_match_index = np.argmin(face_distances)
        
        # Hitung nilai confidence (0-1), 1 adalah confidence tertinggi
        confidence = 1 - face_distances[best_match_index]
        
        # Jika confidence melebihi threshold, kembalikan id dan nama
        if confidence >= self.tolerance:
            return self.known_face_ids[best_match_index], self.known_face_names[best_match_index], confidence
        
        return None, None, confidence
    
    def detect_and_recognize(self, frame):
        """
        Deteksi dan kenali semua wajah dalam frame
        
        Args:
            frame (numpy.ndarray): Frame gambar untuk diproses
            
        Returns:
            list: List objek [{'id', 'name', 'confidence', 'location', 'encoding'}]
        """
        # Convert BGR (OpenCV format) ke RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteksi lokasi semua wajah
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        
        # Encode setiap wajah yang terdeteksi
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        results = []
        
        # Loop melalui setiap wajah yang terdeteksi
        for i, face_encoding in enumerate(face_encodings):
            # Kenali wajah
            id, name, confidence = self.recognize_face(face_encoding)
            
            # Tambahkan hasil ke list
            results.append({
                'id': id,
                'name': name if name else "Unknown",
                'confidence': confidence,
                'location': face_locations[i],
                'encoding': face_encoding
            })
        
        return results
    
    def draw_face_locations(self, frame, face_locations, labels=None):
        """
        Menggambar kotak dan label di sekitar wajah yang terdeteksi
        
        Args:
            frame (numpy.ndarray): Frame gambar untuk digambar
            face_locations (list): List koordinat (top, right, bottom, left)
            labels (list, optional): List label untuk wajah yang terdeteksi
            
        Returns:
            numpy.ndarray: Frame dengan kotak dan label
        """
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Gambar kotak
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Gambar label jika ada
            if labels and i < len(labels):
                label = labels[i]
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        return frame
    
    def draw_recognition_results(self, frame, results):
        """
        Menggambar hasil pengenalan wajah pada frame
        
        Args:
            frame (numpy.ndarray): Frame gambar untuk digambar
            results (list): List hasil pengenalan wajah
            
        Returns:
            numpy.ndarray: Frame dengan hasil pengenalan wajah
        """
        for result in results:
            top, right, bottom, left = result['location']
            
            # Tentukan warna berdasarkan confidence
            if result['id'] is not None:
                # Dikenali: hijau dengan opacity berdasarkan confidence
                color = (0, int(255 * result['confidence']), 0)
                label = f"{result['name']} ({result['confidence']:.2f})"
            else:
                # Tidak dikenali: merah
                color = (0, 0, 255)
                label = f"Unknown ({result['confidence']:.2f})"
            
            # Gambar kotak
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Gambar label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            # Tambahkan timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
        return frame 