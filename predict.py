import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import pickle
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import time
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import KalmanFilter
from PIL import Image, ImageDraw, ImageFont

MODEL_DIR = "model_advanced"
CONFIDENCE_THRESHOLD = 0.7 
SMOOTHING_FRAMES = 5 
SEQUENCE_LENGTH = 15
CAMERA_INDEX = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
    config = pickle.load(f)
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, num_layers=2, num_classes=27):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.softmax(x)

model = CNNLSTM(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    num_classes=config['num_classes']
).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "asl_cnn_lstm.pt"), map_location=device))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

def compute_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cos_val = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8)
    return np.arccos(np.clip(cos_val, -1.0, 1.0))

def compute_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_features_from_landmarks(lm):
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
    coords = []
    pts_2d = []
    for p in lm:
        coords.extend([p.x - wrist[0], p.y - wrist[1], p.z - wrist[2]])
        pts_2d.append([p.x, p.y])
    
    fingers = [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
        [13, 14, 15, 16], [17, 18, 19, 20],
    ]
    angles = []
    for f in fingers:
        angles.append(compute_angle(pts_2d[f[0]], pts_2d[f[1]], pts_2d[f[2]]))
        angles.append(compute_angle(pts_2d[f[1]], pts_2d[f[2]], pts_2d[f[3]]))
        angles.append(compute_angle(pts_2d[0], pts_2d[f[0]], pts_2d[f[1]]))
    
    tip_ids = [4, 8, 12, 16, 20]
    palm_center = np.mean([pts_2d[i] for i in [0, 5, 9, 13, 17]], axis=0)
    distances = []
    for t in tip_ids:
        distances.append(compute_distance(pts_2d[t], pts_2d[0]))
        distances.append(compute_distance(pts_2d[t], palm_center))
    
    return np.array(coords + angles + distances, dtype=np.float32)

class TrajectorySmoother:
    def __init__(self, n_features=21*3):
        self.kf = KalmanFilter(dim_x=n_features, dim_z=n_features)
        self.kf.F = np.eye(n_features)
        self.kf.H = np.eye(n_features)
        self.kf.P *= 10
        self.kf.R = 0.5
        self.kf.Q = 0.05
        self.initialized = False
    
    def smooth(self, landmarks_flat):
        if not self.initialized:
            self.kf.x = landmarks_flat
            self.initialized = True
            return landmarks_flat
        self.kf.predict()
        self.kf.update(landmarks_flat)
        return self.kf.x

class FeedbackGenerator:
    def __init__(self):
        self.tip_ids = [4, 8, 12, 16, 20]
        self.finger_names = ['Ngon cai', 'Ngon tro', 'Ngon giua', 'Ngon ap ut', 'Ngon ut']

    def analyze_hand_posture(self, landmarks):
        feedback = []
        tips = [landmarks[i] for i in self.tip_ids]
        mcp_joints = [landmarks[i] for i in [2, 6, 10, 14, 18]]
        
        for i, (tip, mcp) in enumerate(zip(tips, mcp_joints)):
            # tip = [x, y, z]
            if tip[1] < mcp[1] - 0.05:
                feedback.append(f"[dong y] {self.finger_names[i]} dang duoi")
            elif tip[1] > mcp[1] + 0.02:
                feedback.append(f"[canh bao] {self.finger_names[i]} nen duoi thang hon")
        
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        
        if abs(index_mcp[0] - wrist[0]) < 0.05:
            feedback.append("Hay xoay ban tay ve phia camera ro hon")
        
        return feedback

trajectory_smoother = TrajectorySmoother()
feedback_gen = FeedbackGenerator()
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)
confidence_history = deque(maxlen=10)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17),(0,17),
]
FINGERTIPS = {4, 8, 12, 16, 20}

def draw_hand_skeleton(frame, hand_landmarks):
    h, w = frame.shape[:2]
    lm = hand_landmarks.landmark
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        color = (255, 255, 255) if i in FINGERTIPS else (200, 200, 200)
        cv2.circle(frame, (px, py), 6 if i in FINGERTIPS else 4, color, -1)
        if i in FINGERTIPS:
            cv2.circle(frame, (px, py), 8, (0, 0, 255), 2)
    return pts

def draw_feedback(frame, feedback_list, y_offset=80):
    for i, msg in enumerate(feedback_list[:4]):
        cv2.putText(frame, msg, (20, y_offset + i * 35),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def draw_confidence_meter(frame, confidence, x=20, y=300):
    h = 150
    filled = int(h * confidence)
    cv2.rectangle(frame, (x, y), (x+25, y+h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y + h - filled), (x+25, y+h), (0, 255, 0) if confidence > 0.6 else (0, 165, 255), -1)
    cv2.putText(frame, f"{confidence*100:.0f}%", (x-5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def draw_text_unicode(frame, text, position, color=(0,255,255), size=28):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", size)
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)

    return np.array(img_pil)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    print("ASL Advanced Predictor running - press Q to quit.")
    print("Tính năng mới: Kalman smoothing, Reject mechanism, Phản hồi tư thế")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now
        
        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            pts = draw_hand_skeleton(frame, hand_lm)
            
            raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]).flatten()
            smoothed_landmarks = trajectory_smoother.smooth(raw_landmarks) 
            smoothed_lm = smoothed_landmarks.reshape(21, 3)
            
            class FakeLM: pass
            fake_lm = [type('', (), {'x': p[0], 'y': p[1], 'z': p[2]})() for p in smoothed_lm]
            feat = extract_features_from_landmarks(fake_lm)
            
            frame_buffer.append(feat)
            
            feedback_msgs = feedback_gen.analyze_hand_posture(smoothed_lm)
            draw_feedback(frame, feedback_msgs)
            
            if len(frame_buffer) == SEQUENCE_LENGTH:
                sequence = np.array(frame_buffer)
                feature_dim = sequence.shape[1]
                sequence_scaled = scaler.transform(sequence.reshape(-1, feature_dim)).reshape(SEQUENCE_LENGTH, feature_dim).reshape(SEQUENCE_LENGTH, 88)
                seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    log_probs = model(seq_tensor)
                    probs = torch.exp(log_probs)
                    max_prob, idx = torch.max(probs, dim=1)
                    confidence = max_prob.item()
                
                label = le.classes_[idx.item()]
                confidence_history.append(confidence)
                avg_conf = np.mean(confidence_history)
                
                label = le.classes_[idx.item()]
                if label == 'U':
                    index_tip = smoothed_lm[8]
                    middle_tip = smoothed_lm[12]
                    thumb_tip = smoothed_lm[4]

                    gap = np.linalg.norm(index_tip[:2] - middle_tip[:2])

                    v1 = index_tip[:2] - middle_tip[:2]
                    v2 = thumb_tip[:2] - middle_tip[:2]

                    cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    angle = np.arccos(np.clip(cos_val, -1.0, 1.0))

                    if gap > 0.07 and angle > 0.6:
                        label = 'K'

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(label)
                
                smoothed_label = Counter(prediction_buffer).most_common(1)[0][0] if prediction_buffer else "?"
                
                x1, y1, x2, y2 = (min(p[0] for p in pts)-20, min(p[1] for p in pts)-20, max(p[0] for p in pts)+20, max(p[1] for p in pts)+20)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 100, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                result_text = f"{smoothed_label}" if confidence >= CONFIDENCE_THRESHOLD else "? (Low confidence)"
                cv2.putText(frame, result_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                
                draw_confidence_meter(frame, confidence)
                
                if confidence < CONFIDENCE_THRESHOLD:
                    cv2.putText(frame, "[canh bao] Khong ro, hay giu tay ro hon", (w-400, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            status = f"FPS: {fps:.1f} | Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            frame_buffer.clear()
            prediction_buffer.clear()
            text = "Khong thay ban tay"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            x = w - text_width - 20 
            y = 40             

            cv2.putText(frame, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("ASL Advanced Recognition", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")

if __name__ == "__main__":
    main()
