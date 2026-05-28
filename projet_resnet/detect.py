import cv2
import torch
import numpy as np
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Charge les classes officielles directement depuis le modèle (evite les IndexError)
print("[>] Loading Faster R-CNN model (ResNet-50 backbone)...")
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
COCO_CLASSES = weights.meta["categories"]

model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"[OK] Model loaded on: {device}")

# Chemins relatifs au dossier du script (garanti de fonctionner)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_INPUT = os.path.join(SCRIPT_DIR, "m.mp4")
VIDEO_OUTPUT = os.path.join(SCRIPT_DIR, "m_annotated.mp4")

cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_INPUT}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

SKIP_FRAMES = 3  # Traite 1 frame sur 3 pour aller 3x plus vite
THRESHOLD = 0.5
all_objects = set()
frame_count = 0

print(f"[>] Processing {total_frames} frames... (skip={SKIP_FRAMES})")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Sauvegarde les frames non traitees pour garder la video fluide
    if frame_count % SKIP_FRAMES != 0:
        out.write(frame)
        continue

    if frame_count % (SKIP_FRAMES * 10) == 0:
        print(f"  Frame {frame_count}/{total_frames} processed...")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    mask = preds["scores"] > THRESHOLD
    boxes = preds["boxes"][mask].cpu().numpy()
    labels = preds["labels"][mask].cpu().numpy()
    scores = preds["scores"][mask].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        # Securite anti-crash si le label sort des limites
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"obj_{int(label)}"
        all_objects.add(class_name)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{class_name} {score:.2f}"
        cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print(f"[OK] Annotated video saved: {VIDEO_OUTPUT}")
print("[-] Unique objects detected in the video:")
for obj in sorted(all_objects):
    print(f"  - {obj}")
print("="*50)