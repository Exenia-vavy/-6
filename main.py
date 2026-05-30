import cv2
import json
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_yolo_detection(
    video_path: Path, 
    output_dir: Path, 
    model_name: str = "yolov8n.pt", 
    conf_thresh: float = 0.5,
    save_annotated: bool = False
) -> dict | None:
    """
    Détecte des objets dans une vidéo avec YOLOv8 et génère un JSON structuré.
    """
    if not video_path.exists():
        logging.error(f"📁 Fichier vidéo introuvable : {video_path}")
        return None

    # Récupération du FPS pour le calcul des timestamps
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        logging.warning("⚠️ FPS non détecté, valeur par défaut : 25.0")
        fps = 25.0

    logging.info(f"🤖 Chargement du modèle YOLO : {model_name}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        logging.error(f"❌ Échec du chargement YOLO : {e}")
        return None

    logging.info("🔍 Lancement de la détection (mode stream pour optimiser la RAM)...")
    # stream=True évite de charger toute la vidéo en mémoire
    results = model.predict(
        source=str(video_path), 
        conf=conf_thresh, 
        save=save_annotated, 
        verbose=False, 
        stream=True
    )

    detections_data = {
        "source_video": video_path.name,
        "model_used": model_name,
        "confidence_threshold": conf_thresh,
        "fps": fps,
        "frame_detections": []
    }

    frame_idx = 0
    frames_with_objects = 0

    for res in results:
        # Extraction des boîtes si des objets sont détectés
        if res.boxes is not None and len(res.boxes) > 0:
            objects_in_frame = []
            for box in res.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                bbox = [round(float(x), 1) for x in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]

                objects_in_frame.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": bbox
                })

            timestamp = round(frame_idx / fps, 2)
            detections_data["frame_detections"].append({
                "frame_id": frame_idx,
                "timestamp_seconds": timestamp,
                "objects": objects_in_frame,
                "object_count": len(objects_in_frame)
            })
            frames_with_objects += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            logging.info(f"📊 Progression : {frame_idx}/{total_frames} frames")

    logging.info(f"✅ Détection terminée. {frames_with_objects} frames contiennent au moins 1 objet.")

    # Sauvegarde JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "yolo_detections.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Résultats sauvegardés : {json_path}")

    if save_annotated:
        logging.info("🎬 Vidéo annotée disponible dans : runs/detect/predict/")

    return detections_data


# ─────────────────────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH = Path(r"C:\Users\Exenia\Desktop\videoss\video2.mp4")
    OUT_DIR = VIDEO_PATH.parent / "yolo_pz5"

    # save_annotated=False par défaut pour gagner du temps/disk. 
    # Mettez True si vous avez besoin de la vidéo marquée pour le rapport.
    run_yolo_detection(VIDEO_PATH, OUT_DIR, model_name="yolov8n.pt", conf_thresh=0.5, save_annotated=False)
