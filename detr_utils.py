import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torch.nn import functional as F

class DetrFaceUtils:
    def __init__(self, tolerance=0.7, device=None):
        """
        Inisialisasi utilitas deteksi dan pengenalan wajah dengan DETR
        
        Args:
            tolerance (float): Threshold untuk similarity (0.0-1.0), semakin besar semakin ketat
            device (str): Device untuk inferensi ('cuda' atau 'cpu')
        """
        self.tolerance = tolerance
        
        # Pilih device berdasarkan ketersediaan GPU
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load feature extractor dan model DETR pre-trained
        self.feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50').to(self.device)
        
        # Definisikan label untuk deteksi (COCO dataset)
        # DETR dilatih pada COCO, dimana label 1 adalah 'person'
        self.person_label = 1
        
        # Untuk pengenalan wajah, gunakan resnet sebagai feature extractor
        self.face_feature_extractor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Kamus untuk memetakan id karyawan ke encoding wajah
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        
        # Path untuk menyimpan face encodings
        self.faces_dir = "data/faces"
        os.makedirs(self.faces_dir, exist_ok=True)

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
        # Convert BGR (OpenCV format) ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Process image untuk DETR
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Lakukan inferensi
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process hasil
        width, height = pil_image.size
        results = self.feature_extractor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=0.7
        )[0]
        
        # Filter hanya untuk 'person'
        face_boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label == self.person_label:  # 'person' class in COCO
                # Convert ke format (top, right, bottom, left)
                # DETR mengembalikan (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = box.tolist()
                top, right, bottom, left = int(y_min), int(x_max), int(y_max), int(x_min)
                
                # Fokus hanya pada bagian wajah (estimasi dari proporsi tubuh)
                # Ini adalah pendekatan sederhana, dalam kasus nyata mungkin perlu face detector khusus
                face_height = (bottom - top) // 3
                face_top = top
                face_bottom = top + face_height
                face_width = int(face_height * 0.8)
                face_left = left + (right - left) // 2 - face_width // 2
                face_right = face_left + face_width
                
                # Pastikan koordinat dalam batas frame
                face_top = max(0, face_top)
                face_left = max(0, face_left)
                face_bottom = min(height, face_bottom)
                face_right = min(width, face_right)
                
                face_boxes.append((face_top, face_right, face_bottom, face_left))
        
        return face_boxes
    
    def encode_face(self, frame, face_location=None):
        """
        Mengubah wajah menjadi encoding numerik
        
        Args:
            frame (numpy.ndarray): Frame gambar yang berisi wajah
            face_location (tuple, optional): Lokasi wajah (top, right, bottom, left)
            
        Returns:
            numpy.ndarray: Encoding wajah
        """
        # Convert BGR (OpenCV format) ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if face_location:
            # Jika lokasi wajah diberikan, crop bagian wajah
            top, right, bottom, left = face_location
            face_image = rgb_frame[top:bottom, left:right]
        else:
            # Jika tidak, gunakan seluruh frame
            face_image = rgb_frame
        
        # Convert ke PIL Image
        pil_image = Image.fromarray(face_image)
        
        # Extract features
        tensor = self.face_feature_extractor(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.backbone(tensor)[0]
            # Flatten dan normalisasi
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten().cpu().numpy()
            features = features / np.linalg.norm(features)
        
        return features
    
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
        
        # Hitung similarity dengan semua wajah yang dikenal
        similarities = []
        for encoding in self.known_face_encodings:
            similarity = cosine_similarity([face_encoding], [encoding])[0][0]
            similarities.append(similarity)
        
        # Dapatkan index similarity tertinggi
        best_match_index = np.argmax(similarities)
        confidence = similarities[best_match_index]
        
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
        # Deteksi lokasi semua wajah
        face_locations = self.detect_faces(frame)
        
        results = []
        
        # Loop melalui setiap wajah yang terdeteksi
        for face_location in face_locations:
            # Encode wajah
            face_encoding = self.encode_face(frame, face_location)
            
            # Kenali wajah
            id, name, confidence = self.recognize_face(face_encoding)
            
            # Tambahkan hasil ke list
            results.append({
                'id': id,
                'name': name if name else "Unknown",
                'confidence': confidence,
                'location': face_location,
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
        
        return frame 