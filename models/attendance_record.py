from datetime import datetime

class AttendanceRecord:
    """Model untuk rekaman absensi"""
    
    def __init__(self, id=None, employee_id=None, check_in=None, check_out=None, date=None,
                 employee_name=None, employee_code=None):
        """
        Inisialisasi object AttendanceRecord
        
        Args:
            id (int, optional): ID database
            employee_id (int): ID database karyawan
            check_in (datetime, optional): Waktu check-in
            check_out (datetime, optional): Waktu check-out
            date (str, optional): Tanggal absensi (YYYY-MM-DD)
            employee_name (str, optional): Nama karyawan (untuk tampilan)
            employee_code (str, optional): Kode karyawan (untuk tampilan)
        """
        self.id = id
        self.employee_id = employee_id
        
        # Set check-in time
        if isinstance(check_in, str):
            try:
                self.check_in = datetime.strptime(check_in, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                self.check_in = None
        else:
            self.check_in = check_in
            
        # Set check-out time
        if isinstance(check_out, str):
            try:
                self.check_out = datetime.strptime(check_out, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                self.check_out = None
        else:
            self.check_out = check_out
        
        # Set date
        if date is None and self.check_in is not None:
            self.date = self.check_in.strftime('%Y-%m-%d')
        else:
            self.date = date
            
        # Tambahan informasi tentang karyawan
        self.employee_name = employee_name
        self.employee_code = employee_code
    
    @classmethod
    def from_dict(cls, data):
        """
        Membuat AttendanceRecord dari dictionary
        
        Args:
            data (dict): Dictionary dengan data absensi
            
        Returns:
            AttendanceRecord: Objek AttendanceRecord
        """
        return cls(
            id=data.get('id'),
            employee_id=data.get('employee_id'),
            check_in=data.get('check_in'),
            check_out=data.get('check_out'),
            date=data.get('date'),
            employee_name=data.get('name'),
            employee_code=data.get('employee_code')
        )
    
    def to_dict(self):
        """
        Mengkonversi AttendanceRecord ke dictionary
        
        Returns:
            dict: Dictionary dengan data absensi
        """
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'check_in': self.check_in,
            'check_out': self.check_out,
            'date': self.date,
            'employee_name': self.employee_name,
            'employee_code': self.employee_code
        }
    
    def get_duration(self):
        """
        Menghitung durasi antara check-in dan check-out
        
        Returns:
            tuple: (hours, minutes) atau None jika tidak bisa dihitung
        """
        if self.check_in and self.check_out:
            duration = self.check_out - self.check_in
            total_seconds = duration.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            return hours, minutes
        
        return None
    
    def get_duration_str(self):
        """
        Mendapatkan string durasi antara check-in dan check-out
        
        Returns:
            str: String durasi atau "-" jika tidak bisa dihitung
        """
        duration = self.get_duration()
        if duration:
            hours, minutes = duration
            return f"{hours}h {minutes}m"
        
        return "-"
    
    def get_check_in_str(self):
        """String waktu check-in"""
        if self.check_in:
            return self.check_in.strftime('%H:%M:%S')
        return "-"
    
    def get_check_out_str(self):
        """String waktu check-out"""
        if self.check_out:
            return self.check_out.strftime('%H:%M:%S')
        return "-"
    
    def __str__(self):
        """String representation"""
        return f"AttendanceRecord(id={self.id}, employee={self.employee_name}, date={self.date}, in={self.get_check_in_str()}, out={self.get_check_out_str()})"