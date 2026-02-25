from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import ImageRecord, VideoRecord, TaskRecord

user_bp = Blueprint('user', __name__)

@user_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for('admin.dashboard'))
    recent_tasks = TaskRecord.query.filter_by(user_id=current_user.id).order_by(TaskRecord.created_at.desc()).limit(5).all()
    return render_template('user/dashboard.html', recent_tasks=recent_tasks)

@user_bp.route('/gallery')
@login_required
def gallery():
    page = request.args.get('page', 1, type=int)
    images = ImageRecord.query.filter_by(user_id=current_user.id, media_type='image').order_by(ImageRecord.created_at.desc()).paginate(page=page, per_page=12)
    return render_template('user/gallery.html', images=images)

@user_bp.route('/videos')
@login_required
def videos():
    page = request.args.get('page', 1, type=int)
    reid_videos = VideoRecord.query.filter_by(user_id=current_user.id).order_by(VideoRecord.created_at.desc()).paginate(page=page, per_page=12)
    weather_videos = ImageRecord.query.filter_by(user_id=current_user.id, media_type='video').order_by(ImageRecord.created_at.desc()).paginate(page=page, per_page=12)
    return render_template('user/videos.html', videos=reid_videos, weather_videos=weather_videos)

@user_bp.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    tasks = TaskRecord.query.filter_by(user_id=current_user.id).order_by(TaskRecord.created_at.desc()).paginate(page=page, per_page=20)
    return render_template('user/history.html', tasks=tasks)

@user_bp.route('/delete_image/<int:id>', methods=['POST'])
@login_required
def delete_image(id):
    record = ImageRecord.query.get_or_404(id)
    if record.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    db.session.delete(record)
    db.session.commit()
    return jsonify({'success': True})

@user_bp.route('/delete_video/<int:id>', methods=['POST'])
@login_required
def delete_video(id):
    record = VideoRecord.query.get_or_404(id)
    if record.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    db.session.delete(record)
    db.session.commit()
    return jsonify({'success': True})

@user_bp.route('/delete_task/<int:id>', methods=['POST'])
@login_required
def delete_task(id):
    record = TaskRecord.query.get_or_404(id)
    if record.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    db.session.delete(record)
    db.session.commit()
    return jsonify({'success': True})
