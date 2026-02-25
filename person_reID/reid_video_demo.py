import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys
import os

# Add person_reID folder to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), 'person_reID'))
from model import PCB, ft_net_dense, PCB_test

def load_reid_model(model_name='DenseNet121', class_num=751, model_path=None):
    if model_name == 'DenseNet121':
        model = ft_net_dense(class_num)
    elif model_name == 'PCB':
        model = PCB(class_num)
    else:
        raise ValueError("Unsupported model name")
    
    # Load trained weights
    # We need to be careful with loading state dict if the model was trained with DataParallel or similar
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        # Try loading with CPU mapping if CUDA is not available or other issues
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    print(f"Model {model_name} loaded successfully")

    # Remove the final classifier layer for feature extraction
    if model_name == 'PCB':
        model = PCB_test(model)
    else:
        model.classifier.classifier = nn.Sequential()
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def extract_feature(model, img, model_name='DenseNet121'):
    # Transforms
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
    img = img.unsqueeze(0)
    
    if torch.cuda.is_available():
        img = img.cuda()
        
    with torch.no_grad():
        if model_name == 'PCB':
            feature = model(img) # [1, 2048, 6]
            # norm feature
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(feature, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            feature = feature.div(fnorm.expand_as(feature))
            feature = feature.view(feature.size(0), -1) # [1, 12288]
        else:
            feature = model(img)
            # Normalize feature
            fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
            feature = feature.div(fnorm.expand_as(feature))
        
    return feature

def select_query_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    scale_factor = 1.3  # 放大显示倍数
    
    window_name = "Select Query Person"
    cv2.namedWindow(window_name)

    # 读取第一帧获取尺寸
    ret, frame = cap.read()
    if not ret: return None
    orig_h, orig_w = frame.shape[:2]
    disp_w, disp_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    ctrl_h = 60 # 控制栏高度

    # 状态变量 (使用字典以便在闭包中修改)
    # selecting: 是否在画框模式
    # mouse_down: 鼠标左键是否按下
    # sel_start: 画框起点(显示坐标)
    # sel_curr: 画框终点(显示坐标)
    # confirmed_roi: 最终选定的区域(原始坐标x,y,w,h)
    # selection_frame: 进入选择模式时冻结的原始帧
    state = {
        'selecting': False,
        'mouse_down': False,
        'sel_start': None,
        'sel_curr': None,
        'confirmed_roi': None,
        'selection_frame': None,
        'playing': False,
        'need_refresh': True,
        'current_frame': frame.copy()
    }

    # 按钮定义函数
    def get_buttons():
        btns = []
        # y坐标基于 disp_h (画面下方)
        y1, y2 = disp_h + 10, disp_h + 50
        if not state['selecting']:
            btns.append({'rect': (10, y1, 110, y2), 'text': "Select", 'action': 'start_select'})
            # btns.append({'rect': (120, y1, 220, y2), 'text': "Play/Pause", 'action': 'toggle_play'})
            btns.append({'rect': (230, y1, 330, y2), 'text': "Quit", 'action': 'quit'})
        else:
            btns.append({'rect': (10, y1, 110, y2), 'text': "Confirm", 'action': 'confirm'})
            btns.append({'rect': (120, y1, 220, y2), 'text': "Cancel", 'action': 'cancel_select'})
        return btns

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击了按钮
            if y > disp_h:
                for btn in get_buttons():
                    bx1, by1, bx2, by2 = btn['rect']
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        handle_action(btn['action'])
                        return
            
            # 如果在选择模式且点击在画面内 -> 开始画框
            if state['selecting'] and y < disp_h:
                state['mouse_down'] = True
                state['sel_start'] = (x, y)
                state['sel_curr'] = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if state['selecting'] and state['mouse_down']:
                # 限制坐标在画面内
                curr_x = min(max(0, x), disp_w)
                curr_y = min(max(0, y), disp_h)
                state['sel_curr'] = (curr_x, curr_y)

        elif event == cv2.EVENT_LBUTTONUP:
            if state['selecting'] and state['mouse_down']:
                state['mouse_down'] = False
                curr_x = min(max(0, x), disp_w)
                curr_y = min(max(0, y), disp_h)
                state['sel_curr'] = (curr_x, curr_y)

    def handle_action(action):
        if action == 'quit':
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif action == 'toggle_play':
            state['playing'] = not state['playing']
        elif action == 'start_select':
            state['selecting'] = True
            state['playing'] = False
            # 冻结当前帧，用于画框
            state['selection_frame'] = state['current_frame'].copy()
            state['sel_start'] = None
            state['sel_curr'] = None
        elif action == 'cancel_select':
            state['selecting'] = False
            state['sel_start'] = None
        elif action == 'confirm':
            if state['sel_start'] and state['sel_curr']:
                # 转换坐标回原始图像
                x1, y1 = state['sel_start']
                x2, y2 = state['sel_curr']
                
                # 确保 x1<x2, y1<y2
                rx1, rx2 = sorted([x1, x2])
                ry1, ry2 = sorted([y1, y2])
                
                # 缩放回去
                ox1 = int(rx1 / scale_factor)
                oy1 = int(ry1 / scale_factor)
                owa = int((rx2 - rx1) / scale_factor)
                oha = int((ry2 - ry1) / scale_factor)
                
                if owa > 5 and oha > 5:
                    state['confirmed_roi'] = (ox1, oy1, owa, oha)
                else:
                    print("Selection too small!")

    def on_trackbar(val):
        state['playing'] = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        current_frame_idx = val
        state['need_refresh'] = True

    cv2.setMouseCallback(window_name, on_mouse)
    cv2.createTrackbar("Frame", window_name, 0, total_frames-1, on_trackbar)

    while True:
        if state['confirmed_roi']:
            break

        # 视频播放逻辑
        if state['playing']:
            ret, frame = cap.read()
            if not ret:
                state['playing'] = False
            else:
                state['current_frame'] = frame.copy()
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos("Frame", window_name, pos)
        elif state['need_refresh']:
            pos = cv2.getTrackbarPos("Frame", window_name)
            # 避免重复 seek
            # cap.set(cv2.CAP_PROP_POS_FRAMES, pos) 
            # 实际上 trackbar 回调已经做了 seek，这里只需 read
            ret, frame = cap.read()
            if ret:
                state['current_frame'] = frame.copy()
            state['need_refresh'] = False

        # 绘图逻辑
        # 1. 确定底图
        if state['selecting'] and state['selection_frame'] is not None:
            img_to_show = state['selection_frame']
        else:
            img_to_show = state['current_frame']
            
        # 放大
        disp_img = cv2.resize(img_to_show, (disp_w, disp_h))
        
        # 2. 如果正在选择，画绿框
        if state['selecting'] and state['sel_start'] and state['sel_curr']:
            cv2.rectangle(disp_img, state['sel_start'], state['sel_curr'], (0, 255, 0), 2)
            
        # 3. 创建包含控制栏的画布
        canvas = np.zeros((disp_h + ctrl_h, disp_w, 3), dtype=np.uint8)
        canvas[:disp_h, :disp_w] = disp_img
        
        # 绘制底部背景
        cv2.rectangle(canvas, (0, disp_h), (disp_w, disp_h+ctrl_h), (50, 50, 50), -1)
        
        # 4. 绘制按钮
        for btn in get_buttons():
            bx1, by1, bx2, by2 = btn['rect']
            # 按钮背景
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (100, 100, 100), -1)
            # 按钮边框
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
            # 按钮文字
            label = btn['text']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = bx1 + (bx2 - bx1 - tw) // 2
            ty = by1 + (by2 - by1 + th) // 2
            cv2.putText(canvas, label, (tx, ty), font, font_scale, (255, 255, 255), thickness)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF
        if key == 27: # ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if state['confirmed_roi'] and state['selection_frame'] is not None:
        x, y, w, h = state['confirmed_roi']
        query_img = state['selection_frame'][y:y+h, x:x+w]
        return query_img
    return None

def main(model_name, reid_weights_path, yolo_weights_path, video_path1, video_path2):
    # 1. Load Models
    print("Loading ReID model...")
    reid_model = load_reid_model(model_name=model_name, model_path=reid_weights_path)
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_weights_path)
    
    # 2. Select Video and Query Person
    # You can change these paths or use a file dialog
    if not os.path.exists(video_path1):
        print("Video 1 not found!")
        return

    print("Opening video 1 for query selection...")
    query_img = select_query_from_video(video_path1)
    
    if query_img is None:
        print("No query person selected.")
        return

    # Extract query feature
    query_feature = extract_feature(reid_model, query_img, model_name=model_name)
    
    # 3. Process Second Video
    if not os.path.exists(video_path2):
        print("Video 2 not found!")
        return

    cap = cv2.VideoCapture(video_path2)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video writer
    save_path = "output_reid_result.mp4"
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except:
            pass
            
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print("Processing video 2...")
    threshold = 0.6 # Similarity threshold, adjust as needed

    # 使用与视频1相同的放大倍数
    scale_factor = 1.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # YOLO Detection
        results = yolo_model(frame, classes=[0], verbose=False) # class 0 is person
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                # Check boundaries
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(width, x2); y2 = min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop person
                person_img = frame[y1:y2, x1:x2]
                
                if person_img.size == 0:
                    continue

                # ReID
                gallery_feature = extract_feature(reid_model, person_img, model_name=model_name)
                
                # Compare
                # query_feature: [1, 512], gallery_feature: [1, 512]
                # Matrix multiplication for cosine similarity (since features are normalized)
                score = torch.mm(query_feature, gallery_feature.t()).item()
                
                # Visualization
                color = (0, 0, 255) # Red for non-match
                label = f"Person {score:.2f}"
                if score > threshold:
                    color = (0, 255, 0) # Green for match
                    label = f"Match {score:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 放大显示
        display_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        cv2.imshow("ReID Result", display_frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Result saved to {save_path}")

if __name__ == "__main__":
    model_name = 'PCB' # 'DenseNet121' or 'PCB'
    reid_weights_path = f'./person_reID/weights/{model_name}/net_last.pth' # DenseNet121 or PCB weights path
    yolo_weights_path = './yolov11_person_detection/weights/yolo11s.pt'
    video_path1 = "F:\\desktop\\graduation_design\\adaptive-pedestrian-re-identification-system\\datasets\\TestDataset\\videos\\test3\\terrace2-c0.mp4"
    video_path2 = "F:\\desktop\\graduation_design\\adaptive-pedestrian-re-identification-system\\datasets\\TestDataset\\videos\\test3\\terrace2-c2.mp4"

    main(model_name, reid_weights_path, yolo_weights_path, video_path1, video_path2)
