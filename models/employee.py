import os
import numpy as np
from datetime import datetime

class Employee:
    """Model untuk data karyawan"""
    
    def __init__(self, id=None, employee_id="", name="", department="", position="", face_encoding=None,
                 created_at=None, updated_at=None):
        """
        Inisialisasi object Employee
        
        Args:
            id (int, optional): ID database
            employee_id (str): ID karyawan (kode unik)
            name (str): Nama karyawan
            department (str): Departemen
            position (str): Posisi/jabatan
            face_encoding (numpy.ndarray, optional): Encoding wajah karyawan
            created_at (datetime, optional): Waktu pembuatan
            updated_at (datetime, optional): Waktu update terakhir
        """
        self.id = id
        self.employee_id = employee_id
        self.name = name
        self.department = department
        self.position = position
        self.face_encoding = face_encoding
        self.created_at = created_at if created_at else datetime.now()
        self.updated_at = updated_at if updated_at else datetime.now()
    
    @classmethod
    def from_dict(cls, data):
        """
        Membuat Employee dari dictionary
        
        Args:
            data (dict): Dictionary dengan data karyawan
            
        Returns:
            Employee: Objek Employee
        """
        return cls(
            id=data.get('id'),
            employee_id=data.get('employee_id', ''),
            name=data.get('name', ''),
            department=data.get('department', ''),
            position=data.get('position', ''),
            face_encoding=data.get('face_encoding'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
    
    def to_dict(self):
        """
        Mengkonversi Employee ke dictionary
        
        Returns:
            dict: Dictionary dengan data karyawan
        """
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'name': self.name,
            'department': self.department,
            'position': self.position,
            'face_encoding': self.face_encoding,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    def __str__(self):
        """String representation"""
        return f"Employee(id={self.id}, employee_id={self.employee_id}, name={self.name})" 