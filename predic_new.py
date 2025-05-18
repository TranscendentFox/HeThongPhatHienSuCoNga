import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
from ultralytics import YOLO
# sys.path.append('FallDetection/ViTPose/')
from ViTPose.mmpose.apis.inference import init_pose_model, inference_top_down_pose_model
from rlstm_model import KAMTFENet
import time
import uuid
import os

def calculate_fps(prev_time, frame_count):
    """
    Tính FPS dựa trên thời gian xử lý và số khung hình.
    """
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0
    return fps, current_time

def normalize_keypoints_sequence(keypoints):
    """
    Chuẩn hóa keypoints theo từng frame.
    - Gốc là hông phải (index 7)
    - Scale theo khoảng cách giữa 2 vai (index 1 và 4)
    """
    keypoints_norm = np.zeros_like(keypoints)
    for t in range(keypoints.shape[0]):
        frame = keypoints[t]
        
        # Lấy hông phải làm gốc
        origin = frame[7]
        centered = frame - origin

        # Tính khoảng cách giữa hai vai
        right_shoulder = frame[1]
        left_shoulder = frame[4]
        shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Tránh chia cho 0
        if shoulder_dist < 1e-5:
            shoulder_dist = 1.0

        # Chuẩn hóa bằng cách chia cho khoảng cách vai
        normalized = centered / shoulder_dist
        keypoints_norm[t] = normalized
    return keypoints_norm

def extract_keypoints_from_frame(frame, yolo_model, pose_model, expected_keypoints=17):
    """
    Trích xuất điểm mấu chốt từ một khung hình bằng YOLOv8 và ViTPose.
    """
    KEYPOINT_MAPPING_COCO = {
        1: (5, 6),  # Neck: trung bình giữa left shoulder (5) và right shoulder (6)
        2: 6,       # Right shoulder
        3: 8,       # Right elbow
        4: 10,      # Right wrist
        5: 5,       # Left shoulder
        6: 7,       # Left elbow
        7: 9,       # Left wrist
        8: 12,      # Right hip
        9: 14,      # Right knee
        10: 16,     # Right foot
        11: 11,     # Left hip
        12: 13,     # Left knee
        13: 15,     # Left foot
        14: 2,      # Right eye
        15: 1,      # Left eye
        16: 4,      # Right ear
        17: 3       # Left ear
    }

    # Resize khung hình
    frame = cv2.resize(frame, (640, 480))
    frame_with_bbox = frame.copy()

    # Phát hiện người bằng YOLO
    results = yolo_model(frame, device=0)
    boxes = results[0].boxes
    boxes = boxes.xyxy[boxes.cls == 0].cpu().numpy()  # Lọc chỉ lấy người (class 0)
    if len(boxes) == 0:
        return None, None, frame_with_bbox

    # Lấy bounding box đầu tiên (giả định chỉ có một người)
    x_min, y_min, x_max, y_max = map(int, boxes[0][:4])
    bbox = (x_min, y_min, x_max, y_max)
    
    # Vẽ hình chữ nhật quanh người
    cv2.rectangle(frame_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    
    person_img = frame[y_min:y_max, x_min:x_max]

    # Chuẩn bị ảnh cho ViTPose
    person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    
    # Tạo person_results với bounding box
    person_results = [{'bbox': np.array([0, 0, person_img.shape[1], person_img.shape[0]])}]
    
    # Dự đoán keypoints bằng ViTPose
    pose_results = inference_top_down_pose_model(
        pose_model,
        person_img,
        person_results=person_results,
        format='xywh'
    )
    
    if pose_results:
        keypoints = np.zeros((17, 2), dtype=np.float32)
        keypoints_pred = pose_results[0][0]['keypoints'][:, :2]  # [17, 2], chỉ lấy x, y
        
        # Ánh xạ từ COCO keypoints sang hệ thống
        for i in range(1, 18):
            if i == 1:  # Neck
                left_shoulder = keypoints_pred[KEYPOINT_MAPPING_COCO[i][0]]
                right_shoulder = keypoints_pred[KEYPOINT_MAPPING_COCO[i][1]]
                neck = [(left_shoulder[0] + right_shoulder[0]) / 2,
                        (left_shoulder[1] + right_shoulder[1]) / 2]
                keypoints[i-1] = neck
            else:
                idx = KEYPOINT_MAPPING_COCO[i]
                keypoints[i-1] = keypoints_pred[idx]
        return keypoints, bbox, frame_with_bbox
    else:
        return None, bbox, frame_with_bbox

def save_frame_buffer(frame_buffer, output_dir, fps, width, height):
    """
    Lưu buffer khung hình thành một video clip.
    """
    if not frame_buffer:
        return None
    clip_name = f"fall_clip_{uuid.uuid4().hex}.mp4"
    clip_path = os.path.join(output_dir, clip_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    for frame in frame_buffer:
        out.write(frame)
    out.release()
    return clip_path

def predict_and_display_video(video_path, model_path, yolo_model, pose_model, window_size=30, 
                              device='cuda' if torch.cuda.is_available() else 'cpu', 
                              output_dir="fall_clips", output_video_path=None, 
                              extra_frames=30, max_buffer_size=60):
    """
    Predict falls from video, display video with live labels, FPS, bounding boxes,
    measure RLSTM inference time, and save output video if specified.
    """
    # Load KAMTFENet model
    try:
        model = KAMTFENet(
            num_keypoints=17, 
            seq_len=30, 
            input_size=34, 
            hidden_size=256, 
            num_layers=3
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return "error"

    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return "error"

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 640
    height = 480

    # Initialize video writer if output path provided
    out = None
    if output_video_path:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"Failed to initialize video writer for {output_video_path}")
                out = None
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            out = None

    # Initialize buffers and counters
    keypoints_buffer = []
    frame_count = 0
    predictions = []
    inference_times = []
    frame_buffer = []
    prev_time = time.time()
    fps_count = 0
    display_fps = 0
    frame_fall_count = 0
    cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_count += 1
        print(f"Processing frame {frame_count}...")

        # Resize frame
        frame = cv2.resize(frame, (width, height))

        # Extract keypoints and bounding box
        keypoints, bbox, frame_with_bbox = extract_keypoints_from_frame(frame, yolo_model, pose_model)
        if keypoints is None:
            keypoints = np.zeros((17, 2), dtype=np.float32)
        
        # Store frame in buffer
        frame_buffer.append(frame_with_bbox.copy())
        if len(frame_buffer) > max_buffer_size:
            frame_buffer.pop(0)

        # Normalize keypoints
        keypoints = keypoints[np.newaxis, ...]
        keypoints_normalized = normalize_keypoints_sequence(keypoints)[0]
        keypoints_buffer.append(keypoints_normalized)

        # Perform prediction if buffer is full
        label = "No Fall"
        if len(keypoints_buffer) >= window_size:
            window = np.stack(keypoints_buffer[-window_size:], axis=0)
            
            if not np.all(window == 0):
                window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Measure inference time
                if device == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with torch.no_grad():
                        output = model(window_tensor)
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_time = start_event.elapsed_time(end_event) / 1000.0
                else:
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(window_tensor)
                    inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
                
                _, predicted = torch.max(output, 1)
                if predicted.item() == 1:
                    label = "Fall"
                    predictions.append(1)
                    frame_fall_count += 1
                    if frame_fall_count >= 5:
                        save_frame_buffer(frame_buffer, output_dir, fps, width, height)
                        frame_fall_count = 0
                else:
                    predictions.append(0)
            else:
                predictions.append(0)

            if len(keypoints_buffer) > window_size:
                keypoints_buffer.pop(0)

        # Calculate FPS
        if fps_count >= 10:
            display_fps, prev_time = calculate_fps(prev_time, fps_count)
            fps_count = 0

        # Add labels and metrics to frame
        cv2.putText(frame_with_bbox, f"Frame {frame_count}: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)
        cv2.putText(frame_with_bbox, f"FPS: {display_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            cv2.putText(frame_with_bbox, f"RLSTM Inference: {avg_inference_time*1000:.2f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display frame
        cv2.imshow("Fall Detection", frame_with_bbox)

        # Write frame to output video
        if out:
            out.write(frame_with_bbox)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Print inference statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        print(f"\nAverage RLSTM inference time: {avg_inference_time*1000:.2f} ms")
        print(f"RLSTM inference FPS: {inference_fps:.2f}")
    else:
        print("\nNo RLSTM inferences performed.")

    # Calculate final prediction
    fall_count = np.sum(np.array(predictions) == 1)
    total_windows = len(predictions)
    fall_ratio = fall_count / total_windows if total_windows > 0 else 0
    threshold = 0.5
    final_prediction = "fall" if fall_ratio > threshold else "no_fall"

    print(f"\nPrediction results:")
    print(f"Frames predicted as 'fall': {fall_count}/{total_windows}")
    print(f"Fall ratio: {fall_ratio:.2f}")
    print(f"Final prediction: {final_prediction}")
    print(f"Average overall FPS: {display_fps:.2f}")

    return final_prediction

if __name__ == "__main__":
    video_path = "FallClipsTest/1.avi"
    model_path = "best_kamtfenet_new.pth"
    output_video_path = "output_labeled_video.mp4"

    yolo_model = YOLO("yolov8x.pt")
    config_path = 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
    checkpoint_path = 'models/vitpose/vitpose_weights/vitpose_base_coco_aic_mpii.pth'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vitpose_model = init_pose_model(config_path, checkpoint_path, device=device)

    prediction = predict_and_display_video(
        video_path=video_path,
        model_path=model_path,
        yolo_model=yolo_model,
        pose_model=vitpose_model,
        window_size=30,
        device=device,
        output_video_path=output_video_path
    )