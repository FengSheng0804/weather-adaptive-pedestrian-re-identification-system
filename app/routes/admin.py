from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.models import User, TaskRecord, Announcement
from app import db

admin_bp = Blueprint('admin', __name__)

@admin_bp.before_request
@login_required
def require_admin():
    if not current_user.is_admin:
        flash('您没有权限访问此页面', 'danger')
        return redirect(url_for('main.index'))

@admin_bp.route('/dashboard')
def dashboard():
    total_users = User.query.count()
    total_tasks = TaskRecord.query.count()
    pending_tasks = TaskRecord.query.filter_by(status='pending').count()
    completed_tasks = TaskRecord.query.filter_by(status='completed').count()
    return render_template('admin/dashboard.html', 
                           total_users=total_users, 
                           total_tasks=total_tasks,
                           pending_tasks=pending_tasks,
                           completed_tasks=completed_tasks)

@admin_bp.route('/users')
def users():
    page = request.args.get('page', 1, type=int)
    users = User.query.paginate(page=page, per_page=20)
    return render_template('admin/users.html', users=users)

@admin_bp.route('/tasks')
def tasks():
    page = request.args.get('page', 1, type=int)
    tasks = TaskRecord.query.order_by(TaskRecord.created_at.desc()).paginate(page=page, per_page=20)
    return render_template('admin/tasks.html', tasks=tasks)

@admin_bp.route('/system')
def system():
    # Mock system metrics
    metrics = {
        'cpu_usage': 45,
        'memory_usage': 60,
        'disk_usage': 30,
        'gpu_usage': 80
    }
    return render_template('admin/system.html', metrics=metrics)

@admin_bp.route('/announcements', methods=['GET', 'POST'])
def announcements():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        is_pinned = request.form.get('is_pinned') == 'on'
        announcement = Announcement(title=title, content=content, is_pinned=is_pinned)
        db.session.add(announcement)
        db.session.commit()
        flash('公告发布成功', 'success')
        return redirect(url_for('admin.announcements'))
    
    page = request.args.get('page', 1, type=int)
    announcements = Announcement.query.order_by(Announcement.is_pinned.desc(), Announcement.created_at.desc()).paginate(page=page, per_page=10)
    return render_template('admin/announcements.html', announcements=announcements)
