import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import pickle
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import time

MODEL_PATH = "model/asl_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
SCALER_PATH = "model/asl_scaler.pkl"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.45
SMOOTHING_FRAMES = 12

with open(MODEL_PATH, "rb") as f: clf = pickle.load(f)
with open(ENCODER_PATH, "rb") as f: le = pickle.load(f)  
with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17),
]
FINGERTIPS = {4, 8, 12, 16, 20}

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
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    angles = []
    for f in fingers:
        angles.append(compute_angle(pts_2d[f[0]], pts_2d[f[1]], pts_2d[f[2]]))
        angles.append(compute_angle(pts_2d[f[1]], pts_2d[f[2]], pts_2d[f[3]]))
        angles.append(compute_angle(pts_2d[0],    pts_2d[f[0]], pts_2d[f[1]]))
    tip_ids = [4, 8, 12, 16, 20]
    palm_center = np.mean([pts_2d[i] for i in [0, 5, 9, 13, 17]], axis=0)
    distances = []
    for t in tip_ids:
        distances.append(compute_distance(pts_2d[t], pts_2d[0]))
        distances.append(compute_distance(pts_2d[t], palm_center))
    return np.array(coords + angles + distances, dtype=np.float32)

def draw_hand_skeleton(frame, hand_landmarks):
    h, w = frame.shape[:2]
    lm   = hand_landmarks.landmark
    pts  = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        if i in FINGERTIPS:
            cv2.circle(frame, (px, py), 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 7, (0,   0,   230), -1, cv2.LINE_AA)
        elif i == 0:
            cv2.circle(frame, (px, py), 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 7, (0,   200, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (0,   0,   200), -1, cv2.LINE_AA)
    return pts

def get_bounding_box(pts, w, h, pad=20):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(0, min(xs)-pad), max(0, min(ys)-pad),
            min(w-1, max(xs)+pad), min(h-1, max(ys)+pad))

def draw_prediction_box(frame, label, confidence, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}  {confidence*100:.0f}%"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.6, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    tx, ty = x1, max(y1 - 10, th + 10)
    cv2.rectangle(frame, (tx-4, ty-th-8), (tx+tw+4, ty+bl), (0,200,0), -1)
    cv2.putText(frame, text, (tx, ty), font, scale, (0,0,0), thick, cv2.LINE_AA)

def draw_big_letter(frame, letter):
    h, w  = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 5.0, 8
    (tw, _), _ = cv2.getTextSize(letter, font, scale, thick)
    tx, ty = w - tw - 20, h - 60
    cv2.putText(frame, letter, (tx+3, ty+3), font, scale, (0,0,0),    thick+2, cv2.LINE_AA)
    cv2.putText(frame, letter, (tx,   ty),   font, scale, (0,255,80), thick,   cv2.LINE_AA)

def draw_hud(frame, fps, no_hand=False):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, h-40), (w, h), (30,30,30), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    status = "No hand detected" if no_hand else "Hand detected"
    cv2.putText(frame, f"FPS: {fps:.1f}   {status}   [Q] Quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)
    prev_time = time.time()
    print("ASL Predictor running - press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        now, prev_time = time.time(), time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now
        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            pts = draw_hand_skeleton(frame, hand_lm)
            feat = extract_features_from_landmarks(hand_lm.landmark)
            feat_sc = scaler.transform(feat.reshape(1, -1))
            proba = clf.predict_proba(feat_sc)[0]
            idx = np.argmax(proba)
            conf = proba[idx]
            label = le.classes_[idx]
            if conf >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(label)
            smoothed = Counter(prediction_buffer).most_common(1)[0][0] \
                       if prediction_buffer else "?"
            x1, y1, x2, y2 = get_bounding_box(pts, w, h)
            draw_prediction_box(frame, smoothed, conf, x1, y1, x2, y2)
            if conf >= CONFIDENCE_THRESHOLD:
                draw_big_letter(frame, smoothed)
            draw_hud(frame, fps, no_hand = False)
        else:
            prediction_buffer.clear()
            draw_hud(frame, fps, no_hand = True)
        cv2.imshow("ASL Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")

if __name__ == "__main__":
    main()
