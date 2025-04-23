import numpy as np
from app import db
from datetime import datetime

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100))
    department = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    face_data = db.relationship('FaceData', backref='employee', lazy=True)
    attendance = db.relationship('Attendance', backref='employee', lazy=True)

    def __repr__(self):
        return f'<Employee {self.name}>'

class FaceData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    image = db.Column(db.String(255), nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_encoding(self, encoding_array):
        """Store face encoding as binary data"""
        if isinstance(encoding_array, np.ndarray):
            self.face_encoding = encoding_array.tobytes()
        else:
            self.face_encoding = np.array(encoding_array, dtype=np.float64).tobytes()

    def get_encoding(self):
        """Retrieve face encoding as numpy array"""
        if self.face_encoding:
            return np.frombuffer(self.face_encoding, dtype=np.float64)
        return None

    def __repr__(self):
        return f'<FaceData {self.employee.name}>'

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    check_in = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    check_out = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='present')  # present, late, absent
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Attendance {self.employee.name} - {self.check_in.date()}>'

class Configuration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(255), nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    @staticmethod
    def get_value(key, default=None):
        config = Configuration.query.filter_by(key=key).first()
        return config.value if config else default

    @staticmethod
    def set_value(key, value):
        config = Configuration.query.filter_by(key=key).first()
        if config:
            config.value = value
            config.updated_at = datetime.utcnow()
        else:
            config = Configuration(key=key, value=value)
            db.session.add(config)
        db.session.commit() 