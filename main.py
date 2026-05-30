import cv2
import torch
import json
import logging
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_faster_rcnn_detection(
    video_path: Path,
    output_dir: Path,
    skip_frames: int = 3,
    conf_thresh: float = 0.5,
    save_annotated: bool = False
) -> dict | None:
    if not video_path.exists():
        logging.error(f"📁 Fichier vidéo introuvable : {video_path}")
        return None

    logging.info("🤖 Chargement de Faster R-CNN (ResNet-50)...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    COCO_CLASSES = weights.meta["categories"]
    logging.info(f"✅ Modèle chargé sur : {device} | Classes COCO : {len(COCO_CLASSES)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"❌ Impossible d'ouvrir la vidéo : {video_path.name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = output_dir / f"{video_path.stem}_annotated_resnet.mp4"
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    detections_data = {
        "source_video": video_path.name,
        "model_used": "fasterrcnn_resnet50_fpn",
        "confidence_threshold": conf_thresh,
        "fps": fps,
        "skip_frames": skip_frames,
        "frame_detections": []
    }

    frame_idx = 0
    processed_count = 0
    logging.info(f"🎬 Traitement de {total_frames} frames (1 sur {skip_frames})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ✅ CORRECTION ICI : pas de .unsqueeze(0)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).to(device)

        with torch.no_grad():
            preds = model([img_tensor])[0]  # [img_tensor] est une liste de tenseurs 3D [C,H,W]

        mask = preds["scores"] > conf_thresh
        boxes = preds["boxes"][mask].cpu().numpy()
        labels = preds["labels"][mask].cpu().numpy()
        scores = preds["scores"][mask].cpu().numpy()

        if len(boxes) > 0:
            objects_in_frame = []
            for box, label, score in zip(boxes, labels, scores):
                cls_id = int(label)
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                objects_in_frame.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(score), 3),
                    "bbox": [round(float(x), 1) for x in box]
                })

            timestamp = round(frame_idx / fps, 2)
            detections_data["frame_detections"].append({
                "frame_id": frame_idx,
                "timestamp_seconds": timestamp,
                "objects": objects_in_frame,
                "object_count": len(objects_in_frame)
            })

        if out:
            for obj in objects_in_frame:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj['class_name']} {obj['confidence']:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            out.write(frame)

        processed_count += 1
        if processed_count % 10 == 0:
            logging.info(f"📊 Progression : {processed_count} frames traitées ({frame_idx}/{total_frames})")
        frame_idx += 1

    cap.release()
    if out:
        out.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "resnet_fasterRCNN_detections.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2, ensure_ascii=False)
        
    logging.info(f"💾 Résultats sauvegardés : {json_path}")
    logging.info(f"✅ Détection terminée. {len(detections_data['frame_detections'])} frames avec objets.")
    return detections_data

if __name__ == "__main__":
    VIDEO_PATH = Path(r"C:\Users\Exenia\Desktop\videoss\video2.mp4")
    OUT_DIR = VIDEO_PATH.parent / "resnet_pz6"
    run_faster_rcnn_detection(VIDEO_PATH, OUT_DIR, skip_frames=3, conf_thresh=0.5, save_annotated=False)
