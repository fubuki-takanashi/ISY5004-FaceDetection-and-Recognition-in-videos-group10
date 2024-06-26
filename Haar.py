import os
import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt

WIDER_FACE_PATH = r'D:\Document\5004\video_store\Wider_face'
IMAGE_PATH = os.path.join(WIDER_FACE_PATH, 'WIDER_val', 'images')
ANNOTATION_PATH = os.path.join(WIDER_FACE_PATH, 'wider_face_split', 'wider_face_val_bbx_gt.txt')

def load_annotations(anno_path):
    annotations = {}
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            filename = lines[i].strip()
            num_faces = int(lines[i + 1].strip())
            boxes = []
            for j in range(num_faces):
                box = list(map(int, lines[i + 2 + j].strip().split()[:4]))
                box[2] += box[0]
                box[3] += box[1]
                boxes.append(box)
            annotations[filename] = boxes
            i += 2 + num_faces
    return annotations

annotations = load_annotations(ANNOTATION_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(filename):
    image_path = os.path.join(IMAGE_PATH, filename)
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} does not exist!")
        return [], []

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    pred_boxes = []
    for (x, y, w, h) in faces:
        pred_boxes.append([x, y, x + w, y + h])
    
    return annotations[filename], pred_boxes

results = []
for filename in tqdm(annotations.keys()):
    result = process_image(filename)
    results.append(result)

all_true_boxes = [result[0] for result in results if result]
all_pred_boxes = [result[1] for result in results if result]

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    
    xi1 = max(x1, xx1)
    yi1 = max(y1, yy1)
    xi2 = min(x2, xx2)
    yi2 = min(y2, yy2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def evaluate(all_true_boxes, all_pred_boxes, iou_threshold=0.5):
    y_true = []
    y_scores = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    ious = []

    for true_boxes, pred_boxes in zip(all_true_boxes, all_pred_boxes):
        matched = [False] * len(true_boxes)

        for pred_box in pred_boxes:
            best_iou = 0
            best_idx = -1
            for i, true_box in enumerate(true_boxes):
                iou = compute_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            y_scores.append(best_iou)
            
            if best_iou >= iou_threshold:
                y_true.append(1)
                true_positive += 1
                ious.append(best_iou)
                if best_idx >= 0:
                    matched[best_idx] = True
            else:
                y_true.append(0)
                false_positive += 1

        for matched_flag in matched:
            if not matched_flag:
                y_true.append(1)
                y_scores.append(0)
                false_negative += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    mean_iou = np.mean(ious) if len(ious) > 0 else 0
    ap = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return ap, fpr, tpr, roc_auc, mean_iou, precision, recall

ap, fpr, tpr, roc_auc, mean_iou, precision, recall = evaluate(all_true_boxes, all_pred_boxes)
print(f'Average Precision (AP): {ap:.4f}')
print(f'Area Under Curve (AUC): {roc_auc:.4f}')
print(f'Mean IoU: {mean_iou:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

np.savez('roc_data.npz', fpr=fpr, tpr=tpr, roc_auc=roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
