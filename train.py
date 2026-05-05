import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = "dataset/asl_alphabet_train/asl_alphabet_train"
MODEL_DIR = "model_advanced"
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_PER_CLASS = 800
SEQUENCE_LENGTH = 15  
CONFIDENCE_REJECT_THRESHOLD = 0.6  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

def compute_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cos_val = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8)
    return np.arccos(np.clip(cos_val, -1.0, 1.0))

def compute_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_features_from_landmarks(lm):
    """Trích xuất 88 đặc trưng như cũ"""
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

def extract_landmarks_single(image_path, flip=False):
    img = cv2.imread(image_path)
    if img is None:
        return None
    if flip:
        img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    return extract_features_from_landmarks(result.multi_hand_landmarks[0].landmark)

def extract_sequence_from_images(image_paths):
    features = []
    for path in image_paths[:SEQUENCE_LENGTH]:
        feat = extract_landmarks_single(path)
        if feat is not None:
            features.append(feat)
        else:
            features.append(np.zeros(88, dtype=np.float32))  # padding
    if len(features) < SEQUENCE_LENGTH:
        padding = [np.zeros(88, dtype=np.float32)] * (SEQUENCE_LENGTH - len(features))
        features.extend(padding)
    return np.array(features[:SEQUENCE_LENGTH])

def add_temporal_noise(sequence, noise_level=0.003):
    return sequence + np.random.normal(0, noise_level, sequence.shape)

class ASLDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class CNNLSTM(nn.Module):
    """Mô hình CNN + LSTM cho nhận diện chuỗi thời gian"""
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

class RejectClassifier:
    def __init__(self, model, threshold=CONFIDENCE_REJECT_THRESHOLD):
        self.model = model
        self.threshold = threshold
    
    def predict_with_reject(self, x):
        with torch.no_grad():
            probs = torch.exp(self.model(x))
            max_prob, pred = torch.max(probs, dim=1)
            reject_mask = (max_prob < self.threshold).cpu().numpy()
            predictions = pred.cpu().numpy()
            predictions[reject_mask] = -1  # -1 = reject class
            return predictions, max_prob.cpu().numpy()

def build_sequence_dataset():
    X_sequences = []
    y_labels = []
    classes = sorted(os.listdir(DATASET_DIR))
    
    print(f"Xây dựng sequence dataset từ {len(classes)} classes...")
    
    for label in classes:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        
        images = [f for f in os.listdir(label_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images = images[:MAX_PER_CLASS]
        
        for i in range(0, len(images) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            seq_paths = [os.path.join(label_dir, images[i+j]) for j in range(SEQUENCE_LENGTH)]
            seq_feat = extract_sequence_from_images(seq_paths)
            
            non_zero_frames = np.sum(np.sum(np.abs(seq_feat), axis=1) > 0)
            if non_zero_frames >= SEQUENCE_LENGTH * 0.7:
                X_sequences.append(seq_feat)
                y_labels.append(label)
                X_sequences.append(add_temporal_noise(seq_feat, 0.002))
                y_labels.append(label)
        print(f"  {label}: {len([x for x,y in zip(X_sequences, y_labels) if y==label])} sequences")
    return np.array(X_sequences), np.array(y_labels)

def train():
    print("=" * 60)
    print("  ASL Advanced Training  ")
    print("=" * 60)
    print("\n[1/5] Building sequence dataset...")
    X, y = build_sequence_dataset()
    print(f"  Total sequences: {len(X)}")
    
    if len(X) == 0:
        print("ERROR: No valid sequences found!")
        return
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"  Number of classes: {num_classes}")
    
    print("\n[2/5] Normalizing features...")
    scaler = StandardScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    
    train_dataset = ASLDataset(X_train, y_train)
    test_dataset = ASLDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\n[3/5] Training CNN-LSTM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTM(input_dim=88, hidden_dim=128, num_layers=2, num_classes=num_classes).to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_acc = 0
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = correct / total
        scheduler.step(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "asl_cnn_lstm.pt"))
    
    print(f"\n  Best validation accuracy: {best_acc*100:.2f}%")
    
    print("\n[4/5] Evaluating with reject mechanism...")
    reject_classifier = RejectClassifier(model, threshold=0.6)
    model.eval()
    
    all_preds = []
    all_confidences = []
    all_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds, confs = reject_classifier.predict_with_reject(batch_X)
            all_preds.extend(preds)
            all_confidences.extend(confs)
            all_true.extend(batch_y.numpy())
    
    valid_mask = np.array(all_preds) != -1
    if np.sum(valid_mask) > 0:
        valid_acc = np.mean(np.array(all_preds)[valid_mask] == np.array(all_true)[valid_mask])
        reject_rate = 1 - np.mean(valid_mask)
        print(f"  Valid accuracy (after reject): {valid_acc*100:.2f}%")
        print(f"  Reject rate: {reject_rate*100:.2f}%")
    
    print("\n[5/5] Saving model and preprocessing objects...")
    
    with open(os.path.join(MODEL_DIR, "reject_classifier.pkl"), "wb") as f:
        pickle.dump(reject_classifier, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'input_dim': 88,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_classes': num_classes,
        'reject_threshold': CONFIDENCE_REJECT_THRESHOLD
    }
    with open(os.path.join(MODEL_DIR, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    print(f"  Model saved to {MODEL_DIR}/")
    print("\nDone! Advanced model ready for prediction.")

if __name__ == "__main__":
    train()
