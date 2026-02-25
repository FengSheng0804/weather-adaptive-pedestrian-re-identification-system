from datetime import datetime
from app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    image_records = db.relationship('ImageRecord', backref='author', lazy='dynamic')
    video_records = db.relationship('VideoRecord', backref='author', lazy='dynamic')
    task_records = db.relationship('TaskRecord', backref='author', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ImageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_path = db.Column(db.String(256), nullable=False)
    restored_path = db.Column(db.String(256))
    weather_type = db.Column(db.String(64))
    model_used = db.Column(db.String(64))
    psnr = db.Column(db.Float)
    ssim = db.Column(db.Float)
    processing_time = db.Column(db.Float)
    media_type = db.Column(db.String(16), default='image')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class VideoRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query_video_path = db.Column(db.String(256), nullable=False)
    gallery_video_path = db.Column(db.String(256), nullable=False)
    result_video_path = db.Column(db.String(256))
    target_person_image = db.Column(db.String(256))
    processing_time = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class TaskRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_type = db.Column(db.String(64)) # 'weather_restoration' or 'pedestrian_reid'
    status = db.Column(db.String(64), default='pending') # 'pending', 'processing', 'completed', 'failed'
    result_data = db.Column(db.Text) # JSON string for storing extra results
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_pinned = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
