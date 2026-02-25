import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys
import os

# Add person_reID folder to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '../../person_reID'))
from model import PCB, ft_net_dense, PCB_test

_reid_model = None
_yolo_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_reid_model(model_name='DenseNet121', class_num=751):
    global _reid_model
    if _reid_model is None:
        if model_name == 'DenseNet121':
            _reid_model = ft_net_dense(class_num)
            weights_path = os.path.join(os.path.dirname(__file__), '../../person_reID/weights/DenseNet121/net_last.pth')
        elif model_name == 'PCB':
            _reid_model = PCB(class_num)
            weights_path = os.path.join(os.path.dirname(__file__), '../../person_reID/weights/PCB/net_last.pth')
        else:
            raise ValueError("Unsupported model name")
        
        try:
            print(f"Loading ReID weights from: {weights_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            _reid_model.load_state_dict(torch.load(weights_path, map_location=_device))
            print(f"ReID model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading ReID weights: {e}")
            raise e
            
        if model_name == 'PCB':
            _reid_model = PCB_test(_reid_model)
        else:
            _reid_model.classifier.classifier = nn.Sequential()
            
        _reid_model.to(_device)
        _reid_model.eval()
    return _reid_model

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        weights_path = os.path.join(os.path.dirname(__file__), '../../yolov11_person_detection/weights/yolo11s.pt')
        _yolo_model = YOLO(weights_path)
    return _yolo_model

def extract_feature(model, img, model_name='DenseNet121'):
    h, w = 256, 128
    if model_name == 'PCB':
        h, w = 384, 192


    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = data_transforms(img)
    img = img.unsqueeze(0).to(_device)
        
    with torch.no_grad():
        if model_name == 'PCB':
            feature = model(img)
            # norm feature
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(feature, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            feature = feature.div(fnorm.expand_as(feature))
            feature = feature.view(feature.size(0), -1)
        else:
            feature = model(img)
            fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
            feature = feature.div(fnorm.expand_as(feature))
        
    return feature

def process_reid_video(query_video_path, gallery_video_path, target_bbox, output_video_path, target_image_path, progress_callback=None):
    model_name = 'PCB' # 'PCB' or 'DenseNet121'
    reid_model = get_reid_model(model_name)
    yolo_model = get_yolo_model()
    
    if not os.path.isabs(query_video_path) or not os.path.exists(query_video_path):
        # try relative to static/uploads/videos
        query_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static/uploads/videos', os.path.basename(query_video_path)))
        
    if not os.path.isabs(gallery_video_path) or not os.path.exists(gallery_video_path):
        gallery_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static/uploads/videos', os.path.basename(gallery_video_path)))
        
    # 1. Extract query image from query video using bbox
    cap_query = cv2.VideoCapture(query_video_path)
    
    # Seek to the specific time if provided
    if 'time' in target_bbox:
        cap_query.set(cv2.CAP_PROP_POS_MSEC, target_bbox['time'] * 1000)
        
    ret, frame = cap_query.read()
    cap_query.release()
    
    if not ret:
        # Fallback to first frame if seeking fails
        cap_query = cv2.VideoCapture(query_video_path)
        ret, frame = cap_query.read()
        cap_query.release()
        if not ret:
            raise ValueError("Could not read query video")
        
    h, w = frame.shape[:2]
    # target_bbox is [x, y, width, height] in relative coordinates (0-1)
    x1 = int(target_bbox['x'] * w)
    y1 = int(target_bbox['y'] * h)
    bw = int(target_bbox['w'] * w)
    bh = int(target_bbox['h'] * h)
    x2 = x1 + bw
    y2 = y1 + bh
    
    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box coordinates")
    
    query_img = frame[y1:y2, x1:x2]
    cv2.imwrite(target_image_path, query_img)
    
    query_feature = extract_feature(reid_model, query_img, model_name)
    
    # 2. Process gallery video
    cap = cv2.VideoCapture(gallery_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    threshold = 0.65
    frame_count = 0
    all_matches = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = yolo_model(frame, classes=[0], verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                
                bx1 = max(0, bx1); by1 = max(0, by1)
                bx2 = min(width, bx2); by2 = min(height, by2)
                
                if bx2 <= bx1 or by2 <= by1:
                    continue
                
                person_img = frame[by1:by2, bx1:bx2]
                if person_img.size == 0:
                    continue

                gallery_feature = extract_feature(reid_model, person_img, model_name)
                score = torch.mm(query_feature, gallery_feature.t()).item()
                
                color = (0, 0, 255)
                label = f"Person {score:.2f}"
                if score > threshold:
                    color = (0, 255, 0)
                    label = f"Match {score:.2f}"
                    all_matches.append({
                        'score': score,
                        'time': frame_count / fps if fps > 0 else 0,
                        'frame_index': frame_count,
                        'image': person_img.copy()
                    })
                
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
                cv2.putText(frame, label, (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        out.write(frame)
        frame_count += 1
        
        if progress_callback:
            if total_frames > 0:
                should_continue = progress_callback(frame_count, total_frames, frame)
            else:
                should_continue = progress_callback(frame_count, -1, frame)
            
            if should_continue is False:
                cap.release()
                out.release()
                return False
                
    cap.release()
    out.release()
    
    # Sort matches by score descending and get top 3
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter out matches that are too close in time (e.g., within 2 seconds)
    filtered_matches = []
    for match in all_matches:
        if not filtered_matches:
            filtered_matches.append(match)
        else:
            # Check if this match is at least 2 seconds away from all existing filtered matches
            is_distinct = True
            for existing in filtered_matches:
                if abs(match['time'] - existing['time']) < 2.0:
                    is_distinct = False
                    break
            if is_distinct:
                filtered_matches.append(match)
                
        if len(filtered_matches) >= 3:
            break
            
    # Format time string
    top_k = []
    base_dir = os.path.dirname(target_image_path)
    base_name = os.path.splitext(os.path.basename(target_image_path))[0]
    
    for i, match in enumerate(filtered_matches):
        total_seconds = int(match['time'])
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Save matched image
        result_img_name = f"result_{base_name}_rank_{i+1}.jpg"
        result_img_path = os.path.join(base_dir, result_img_name)
        cv2.imwrite(result_img_path, match['image'])
        
        top_k.append({
            'id': i + 1,
            'score': float(match['score']),
            'time': time_str,
            'seconds': float(match['time']),
            'frame_index': match.get('frame_index'),
            'image_url': f"output/reid/images/{result_img_name}" # Relative Web Path
        })
        
    return top_k
