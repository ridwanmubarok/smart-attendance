import os
import json
import logging
from typing import Dict, Any

class Config:
    """Kelas untuk mengelola konfigurasi aplikasi"""
    
    DEFAULT_CONFIG = {
        "app": {
            "name": "Sistem Absensi dengan Pendeteksi Wajah",
            "window_size": [1024, 768]
        },
        "database": {
            "path": "data/database.db"
        },
        "face_recognition": {
            "tolerance": 0.6,
            "model": "hog",  # 'hog' (CPU) atau 'cnn' (GPU)
            "detection_frequency": 3  # Deteksi setiap n frame
        },
        "camera": {
            "default_source": "webcam",
            "default_resolution": [640, 480],
            "saved_cameras": []
        },
        "attendance": {
            "auto_check_out": True,  # Otomatis checkout saat pengenalan berikutnya
            "notification_enabled": True
        }
    }
    
    def __init__(self, config_path="data/config.json"):
        """
        Inisialisasi konfigurasi
        
        Args:
            config_path (str): Path ke file konfigurasi
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Memuat konfigurasi dari file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                    # Update default config dengan nilai yang dimuat
                    self._update_dict(self.config, loaded_config)
                    
                    logging.info(f"Konfigurasi dimuat dari {self.config_path}")
            else:
                logging.warning(f"File konfigurasi tidak ditemukan di {self.config_path}, menggunakan default")
                self.save_config()
        except Exception as e:
            logging.error(f"Error saat memuat konfigurasi: {str(e)}")
    
    def save_config(self):
        """Menyimpan konfigurasi ke file"""
        try:
            # Memastikan direktori ada
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logging.info(f"Konfigurasi disimpan ke {self.config_path}")
            return True
        except Exception as e:
            logging.error(f"Error saat menyimpan konfigurasi: {str(e)}")
            return False
    
    def get(self, section, key=None, default=None):
        """
        Mendapatkan nilai konfigurasi
        
        Args:
            section (str): Bagian konfigurasi
            key (str, optional): Kunci konfigurasi. Jika None, kembalikan seluruh bagian
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default
        """
        try:
            if key is None:
                return self.config.get(section, {})
            
            return self.config.get(section, {}).get(key, default)
        except Exception:
            return default
    
    def set(self, section, key, value):
        """
        Mengubah nilai konfigurasi
        
        Args:
            section (str): Bagian konfigurasi
            key (str): Kunci konfigurasi
            value: Nilai baru
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if section not in self.config:
                self.config[section] = {}
                
            self.config[section][key] = value
            return True
        except Exception as e:
            logging.error(f"Error saat mengubah konfigurasi: {str(e)}")
            return False
    
    def _update_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Update nested dictionary secara rekursif
        
        Args:
            target (dict): Dictionary target yang akan diupdate
            source (dict): Dictionary sumber
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value