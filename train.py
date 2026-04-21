import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from sklearn.ensemble import VotingClassifier

DATASET_DIR = "dataset/asl_alphabet_train/asl_alphabet_train"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "asl_scaler.pkl")
MAX_PER_CLASS = 1000  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
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
        rx = p.x - wrist[0]
        ry = p.y - wrist[1]
        rz = p.z - wrist[2]
        coords.extend([rx, ry, rz])
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
        p0 = pts_2d[f[0]]
        p1 = pts_2d[f[1]]
        p2 = pts_2d[f[2]]
        p3 = pts_2d[f[3]]
        angles.append(compute_angle(p0, p1, p2))
        angles.append(compute_angle(p1, p2, p3))
        angles.append(compute_angle(pts_2d[0], p0, p1))
    tip_ids = [4, 8, 12, 16, 20]
    palm_center = np.mean([pts_2d[i] for i in [0, 5, 9, 13, 17]], axis=0)
    distances = []
    for t in tip_ids:
        distances.append(compute_distance(pts_2d[t], pts_2d[0]))   # đến cổ tay
        distances.append(compute_distance(pts_2d[t], palm_center)) # đến tâm lòng bàn tay
    return np.array(coords + angles + distances, dtype=np.float32)

def extract_landmarks(image_path, flip=False):
    img = cv2.imread(image_path)
    if img is None:
        return None
    if flip:
        img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    return extract_features_from_landmarks(lm)

def add_noise(features, sigma=0.005):
    return features + np.random.normal(0, sigma, features.shape).astype(np.float32)

def build_dataset():
    X, y = [], []
    classes = sorted(os.listdir(DATASET_DIR))
    print(f"Found {len(classes)} classes: {classes}\n")
    for label in classes:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        images = [f for f in os.listdir(label_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if MAX_PER_CLASS:
            images = images[:MAX_PER_CLASS]
        ok = 0
        for fname in tqdm(images, desc=f"[{label}]", leave=False):
            fpath = os.path.join(label_dir, fname)
            feat = extract_landmarks(fpath, flip=False)
            if feat is not None:
                X.append(feat)
                y.append(label)
                X.append(add_noise(feat, 0.004))
                y.append(label)
                X.append(add_noise(feat, 0.007))
                y.append(label)
                ok += 1
            feat_f = extract_landmarks(fpath, flip=True)
            if feat_f is not None:
                X.append(feat_f)
                y.append(label)
                ok += 0  
        print(f"  {label}: {ok}/{len(images)} images processed  "
              f"(~{ok*3} samples với augmentation)")
    return np.array(X), np.array(y)

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("  ASL Sign Language - Training (Improved)")
    print("=" * 60)
    print("\n[1/4] Extracting features + augmentation ...")
    X, y = build_dataset()
    print(f"\n  Total samples: {len(X)}")
    if len(X) == 0:
        print("ERROR: No samples extracted.")
        return
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("\n[2/4] Scaling features ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\n[3/4] Training ensemble classifier ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    clf1 = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        random_state=42, n_jobs=-1
    )
    clf2 = RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_leaf=2,
        random_state=7, n_jobs=-1
    )
    clf = VotingClassifier(
        estimators=[('rf1', clf1), ('rf2', clf2)],
        voting='soft',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Validation accuracy: {acc * 100:.2f}%")
    print("\n  Classification report:")
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names  = le.inverse_transform(present_labels)
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_names))
    print("\n[4/4] Saving model ...")
    with open(MODEL_PATH, "wb") as f: pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"  Model   -> {MODEL_PATH}")
    print(f"  Encoder -> {ENCODER_PATH}")
    print(f"  Scaler  -> {SCALER_PATH}")
    print("\nDone!")

if __name__ == "__main__":
    train()
