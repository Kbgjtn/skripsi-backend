import subprocess
import os
import cv2

from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from .config import get_settings
from .diseases import DISEASES_ENUM

settings = get_settings()
models: Dict[str, Any] = {}

def predict_with_classifier(image_path: Path, imgsz: int) -> Dict[str, Any]:
    print(f"running yolo classification on {image_path}...")
    model = models.get("classifier")
    if not model:
        # This check is redundant if called from the endpoint, but good practice.
        raise RuntimeError("Image classification model not loaded!")

    results = model(image_path, imgsz=imgsz)
    result = results[0]
    names_dict = result.names
    probs = result.probs
    
    # Extract the top 3 prediction details
    # probs.top5 and probs.top5conf give the indices and confidences for the top 5 predictions
    top5_indices = probs.top5
    top5_confidences = probs.top5conf.tolist() # Convert tensor to list

    # Ensure we don't try to access more results than available
    num_results = min(3, len(top5_indices))

    top_predictions = []
    for i in range(num_results):
        class_index = top5_indices[i]
        confidence = top5_confidences[i]
        class_name = names_dict[class_index]

        prediction_details = {
            "class_name": class_name,
            "confidence": confidence
        }

        disease_info = DISEASES_ENUM.get(class_name.replace("-", "_"))
        if disease_info:
            prediction_details.update(disease_info)

        top_predictions.append(prediction_details)

    
    labeled_image_path = None
    if top_predictions:
        # Get top-1 prediction for labeling
        top1 = top_predictions[0]
        label_text = f"{top1['class_name']}: {top1['confidence']:.2f}"
        
        # Load image with OpenCV
        image = cv2.imread(str(image_path))
        
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)  # White
        bg_color = (0, 0, 255)  # Red background
        thickness = 2
        
        # Calculate text size to draw a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_origin = (10, 30)
        
        # Draw the background rectangle
        cv2.rectangle(image, (text_origin[0] - 5, text_origin[1] - text_height - 5), (text_origin[0] + text_width + 5, text_origin[1] + baseline), bg_color, cv2.FILLED)
        
        # Draw the text label
        cv2.putText(image, label_text, text_origin, font, font_scale, font_color, thickness)
        
        # Save the labeled image
        output_filename = f"{image_path.stem}_predicted.jpg"
        labeled_image_path = settings.PREDICTIONS_DIRECTORY / output_filename
        cv2.imwrite(str(labeled_image_path), image)
        print(f"Saved labeled image to: {labeled_image_path}")



    return {
        "path": f"assets/{output_filename}",
        "predictions": top_predictions,
        "model": "YOLO Classifier"
    }

def predict_with_detection(video_path: Path, conf: float, imgsz: int, speed_factor: float) -> Dict[str, Any]:
    """
    Performs object detection on a video. It first saves a temporary video
    using OpenCV and then uses FFmpeg to re-encode it to a web-compatible format.
    """
    print(f"Running YOLO detection on {video_path} with conf={conf}, imgsz={imgsz}, speed_factor={speed_factor}...")
    yolo_model = models.get("detection")
    if not yolo_model:
        raise RuntimeError("YOLO detection model not loaded!")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define a temporary path for the initial OpenCV output
    temp_output_path = settings.PREDICTIONS_DIRECTORY / f"{video_path.stem}_temp.mp4"
    final_output_filename = f"{video_path.stem}_predicted.mp4"
    final_output_path = settings.PREDICTIONS_DIRECTORY / final_output_filename

    # Use the reliable 'mp4v' codec for the initial write.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise RuntimeError("Could not initialize temporary VideoWriter.")

    all_detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, conf=conf, imgsz=imgsz)
        frame_detections = []
        
        for result in results:
            for box in result.boxes.data:
                box_data = box.tolist()
                x1, y1, x2, y2, pred_conf, class_id = box_data
                class_name = yolo_model.names[int(class_id)]

                prediction_details = {
                        "box_coordinates": [x1, y1, x2, y2], 
                        "confidence": pred_conf, 
                        "class_name": class_name,
                }

                disease_info = DISEASES_ENUM.get(class_name.replace("-", "_"))
                if disease_info:
                    prediction_details.update(disease_info)

                frame_detections.append(prediction_details)
                color, label = (0, 255, 0), f"{class_name}: {pred_conf:.2f}"
                font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - text_height - baseline), (int(x1) + text_width, int(y1)), color, cv2.FILLED)
                cv2.putText(frame, label, (int(x1), int(y1) - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        if frame_detections:
            all_detections.append(frame_detections)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Temporary video saved to: {temp_output_path}")

    # --- Re-encode with FFmpeg using a subprocess ---
    print("Re-encoding video with FFmpeg for web compatibility...")
    ffmpeg_command = [
        "ffmpeg",
        "-i", str(temp_output_path), # Input file
        "-y",                      # Overwrite output file if it exists
        "-filter:v", f"setpts={speed_factor}*PTS", # Change video speed
        "-c:v", "libx264",         # Use the H.264 video codec
        "-pix_fmt", "yuv420p",     # Pixel format for maximum compatibility
        "-preset", "fast",         # Encoding speed preset
        "-crf", "23",              # Constant Rate Factor for quality (lower is better)
        str(final_output_path)     # Final output file
    ]
    
    try:
        # We use capture_output=True to get stdout/stderr for debugging
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print("FFmpeg re-encoding successful.")
        print("FFmpeg stdout:", result.stdout)
        # Clean up the temporary file
        os.remove(temp_output_path)
        print(f"Removed temporary file: {temp_output_path}")

    except subprocess.CalledProcessError as e:
        # If FFmpeg fails, log the error and raise an exception
        print("FFmpeg re-encoding failed.")
        print("FFmpeg stderr:", e.stderr)
        # Clean up the temporary file even on failure
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise HTTPException(status_code=500, detail=f"Failed to process video with FFmpeg: {e.stderr}")

    print(f"Saved labeled video to: {final_output_filename}")

    return {
        "path": f"assets/{final_output_filename}",
        "detections": all_detections,
        "model": "YOLOv8 Detection"
    }
