from flask import Blueprint, render_template, request, jsonify, current_app, url_for, Response
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import uuid
import time
import hashlib
import cv2
from app import db
from app.models import VideoRecord, TaskRecord

reid_bp = Blueprint('reid', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
progress_dict = {}
frame_buffers = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@reid_bp.route('/workbench', methods=['GET'])
@login_required
def workbench():
    return render_template('reid/workbench.html')

@reid_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        
        # Calculate file hash to check if it already exists
        file_content = file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        file.seek(0) # Reset file pointer after reading
        
        unique_filename = f"{file_hash}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_VIDEOS_FOLDER'], unique_filename)
        
        # Only save and convert if the file doesn't already exist
        if not os.path.exists(filepath):
            file.save(filepath)
            
            # Convert avi/mov to mp4 for browser compatibility
            if ext in ['avi', 'mov']:
                import subprocess
                mp4_filename = f"{unique_filename.rsplit('.', 1)[0]}.mp4"
                mp4_filepath = os.path.join(current_app.config['UPLOAD_VIDEOS_FOLDER'], mp4_filename)
                try:
                    subprocess.run(['ffmpeg', '-i', filepath, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', mp4_filepath], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.remove(filepath)
                    unique_filename = mp4_filename
                except Exception as e:
                    print(f"Error converting video: {e}")
        else:
            # If it was an avi/mov that was previously converted, we need to use the mp4 filename
            if ext in ['avi', 'mov']:
                mp4_filename = f"{unique_filename.rsplit('.', 1)[0]}.mp4"
                mp4_filepath = os.path.join(current_app.config['UPLOAD_VIDEOS_FOLDER'], mp4_filename)
                if os.path.exists(mp4_filepath):
                    unique_filename = mp4_filename
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': unique_filename,
            'url': url_for('static', filename=f'uploads/videos/{unique_filename}')
        })
    return jsonify({'error': 'Invalid file type'}), 400

@reid_bp.route('/video_feed/<int:task_id>')
@login_required
def video_feed(task_id):
    def generate():
        while True:
            if task_id in frame_buffers:
                frame_bytes = frame_buffers[task_id]
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04) # Limit to approx 25 fps
            
            # Check if task is done or cancelled to stop the stream
            if task_id not in progress_dict:
                break
                
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@reid_bp.route('/progress/<int:task_id>')
@login_required
def get_progress(task_id):
    task = TaskRecord.query.get(task_id)
    if not task or task.user_id != current_user.id:
        return jsonify({'error': 'Task not found'}), 404
        
    if task.status == 'processing' and task_id not in progress_dict:
        # Stale task from previous server run
        task.status = 'failed'
        db.session.commit()
        return jsonify({'error': 'Task failed due to server restart', 'status': 'failed'})
        
    if task.status == 'cancelled':
        return jsonify({'error': 'Task was cancelled', 'status': 'cancelled'})
        
    if task.status == 'completed':
        return jsonify({'progress': 100, 'status': 'completed'})
        
    progress_info = progress_dict.get(task_id, {'progress': 0})
    return jsonify({'progress': progress_info.get('progress', 0), 'status': task.status})

@reid_bp.route('/cancel/<int:task_id>', methods=['POST'])
@login_required
def cancel_task(task_id):
    task = TaskRecord.query.get(task_id)
    if not task or task.user_id != current_user.id:
        return jsonify({'error': 'Task not found'}), 404
        
    if task.status == 'processing':
        task.status = 'cancelled'
        db.session.commit()
        if task_id in progress_dict:
            progress_dict[task_id]['cancel'] = True
        return jsonify({'message': 'Task cancelled successfully'})
        
    return jsonify({'error': 'Task is not processing'}), 400

@reid_bp.route('/create_task', methods=['POST'])
@login_required
def create_task():
    task = TaskRecord(task_type='pedestrian_reid', status='pending', user_id=current_user.id)
    db.session.add(task)
    db.session.commit()
    return jsonify({'task_id': task.id})

@reid_bp.route('/process', methods=['POST'])
@login_required
def process():
    data = request.json
    query_video = data.get('query_video')
    gallery_video = data.get('gallery_video')
    target_bbox = data.get('target_bbox') # [x, y, w, h]
    task_id = data.get('task_id')
    
    if not query_video or not gallery_video or not target_bbox or not task_id:
        return jsonify({'error': 'Missing parameters'}), 400
        
    task = TaskRecord.query.get(task_id)
    if not task or task.user_id != current_user.id:
        return jsonify({'error': 'Task not found'}), 404
    
    task.status = 'processing'
    db.session.commit()
    
    progress_dict[task_id] = {'progress': 0, 'cancel': False}
    
    start_time = time.time()
    
    def progress_callback(current, total, frame=None):
        if progress_dict.get(task_id, {}).get('cancel', False):
            return False # Signal to abort
            
        if total > 0:
            progress_dict[task_id]['progress'] = int((current / total) * 100)
            
        if frame is not None:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_buffers[task_id] = buffer.tobytes()
        
        return True # Signal to continue
    
    query_video_path = os.path.join(current_app.config['UPLOAD_VIDEOS_FOLDER'], query_video)
    gallery_video_path = os.path.join(current_app.config['UPLOAD_VIDEOS_FOLDER'], gallery_video)
    
    # Use task_id to ensure unique filenames for each run
    gallery_base = os.path.splitext(gallery_video)[0]
    query_base = os.path.splitext(query_video)[0]
    
    result_video = f"reid_{task_id}_{gallery_base}.mp4"
    result_video_path = os.path.join(current_app.config['REID_VIDEOS_FOLDER'], result_video)
    
    target_image = f"target_{task_id}_{query_base}.jpg"
    target_image_path = os.path.join(current_app.config['REID_IMAGES_FOLDER'], target_image)
    
    try:
        from app.utils.reid_inference import process_reid_video
        top_k_results = process_reid_video(
            query_video_path,
            gallery_video_path,
            target_bbox,
            result_video_path,
            target_image_path,
            progress_callback
        )
        
        if top_k_results is False:
            # Task was cancelled
            if os.path.exists(result_video_path):
                os.remove(result_video_path)
            if os.path.exists(target_image_path):
                os.remove(target_image_path)
            if task_id in progress_dict:
                del progress_dict[task_id]
            if task_id in frame_buffers:
                del frame_buffers[task_id]
            return jsonify({'error': 'Task cancelled'}), 400
            
        # Convert output video to H.264 for browser compatibility
        import subprocess
        temp_filepath = result_video_path + ".temp.mp4"
        os.rename(result_video_path, temp_filepath)
        try:
            subprocess.run(['ffmpeg', '-i', temp_filepath, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', result_video_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_filepath)
        except Exception as e:
            print(f"Error converting output video: {e}")
            # If conversion fails, restore the original output
            if os.path.exists(temp_filepath):
                os.rename(temp_filepath, result_video_path)
                
    except Exception as e:
        print(f"Error running ReID inference: {e}")
        task.status = 'failed'
        db.session.commit()
        if task_id in progress_dict:
            del progress_dict[task_id]
        if task_id in frame_buffers:
            del frame_buffers[task_id]
        return jsonify({'error': str(e)}), 500
    
    processing_time = time.time() - start_time
    
    task.status = 'completed'
    task.completed_at = db.func.now()
    
    # Clean up progress
    if task_id in progress_dict:
        del progress_dict[task_id]
        
    if task_id in frame_buffers:
        del frame_buffers[task_id]
    
    # Create video record
    record = VideoRecord(
        query_video_path=query_video,
        gallery_video_path=gallery_video,
        result_video_path=result_video,
        target_person_image=target_image,
        processing_time=processing_time,
        user_id=current_user.id
    )
    db.session.add(record)
    db.session.commit()
    
    return jsonify({
        'message': 'Processing completed',
        'result_video_url': url_for('static', filename=f'output/reid/videos/{result_video}'),
        'target_image_url': url_for('static', filename=f'output/reid/images/{target_image}'),
        'record_id': record.id,
        'task_id': task_id,
        'top_k': top_k_results
    })
