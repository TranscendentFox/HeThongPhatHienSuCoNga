import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import uuid
import os
from collections import defaultdict, deque
from ultralytics import YOLO
from ViTPose.mmpose.apis.inference import init_pose_model, inference_top_down_pose_model
from rlstm_model import KAMTFENet

KEYPOINT_MAPPING_COCO = {
    1: (5, 6), 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12,
    9: 14, 10: 16, 11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3
}

def calculate_fps(prev_time, frame_count):
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return fps, current_time

def normalize_keypoints_sequence(keypoints):
    keypoints_norm = np.zeros_like(keypoints)
    for t in range(keypoints.shape[0]):
        frame = keypoints[t]
        origin = frame[7]
        centered = frame - origin
        shoulder_dist = np.linalg.norm(frame[1] - frame[4])
        shoulder_dist = shoulder_dist if shoulder_dist >= 1e-5 else 1.0
        normalized = centered / shoulder_dist
        keypoints_norm[t] = normalized
    return keypoints_norm

def save_frame_buffer(frame_buffer, output_dir, fps, width, height):
    if not frame_buffer:
        return None
    os.makedirs(output_dir, exist_ok=True)
    clip_name = f"fall_clip_{uuid.uuid4().hex}.mp4"
    clip_path = os.path.join(output_dir, clip_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    for frame in frame_buffer:
        out.write(frame)
    out.release()
    return clip_path

def predict_multi_person(video_path, model_path, yolo_model, pose_model,
                         window_size=30, device='cuda' if torch.cuda.is_available() else 'cpu',
                         output_dir="fall_clips", output_video_path=None,
                         max_buffer_size=60):

    model = KAMTFENet(num_keypoints=17, seq_len=30, input_size=34, hidden_size=256, num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = 640, 480

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Buffers
    id_to_keypoints_buffer = defaultdict(lambda: deque(maxlen=window_size))
    id_to_frame_buffer = defaultdict(lambda: deque(maxlen=max_buffer_size))
    id_to_fall_count = defaultdict(int)

    frame_count, fps_count = 0, 0
    prev_time = time.time()
    display_fps = 0

    cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame_count += 1
        fps_count += 1

        # Tracking người
        results = yolo_model.track(frame, persist=True, device=device, classes=[0])
        boxes = results[0].boxes
        ids = results[0].boxes.id

        if boxes is None or ids is None:
            continue

        for i, box in enumerate(boxes):
            track_id = int(ids[i].item())
            x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
            person_img = frame[y_min:y_max, x_min:x_max]
            person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            person_results = [{'bbox': np.array([0, 0, person_img.shape[1], person_img.shape[0]])}]
            pose_results = inference_top_down_pose_model(pose_model, person_img_rgb, person_results, format='xywh')
            if not pose_results:
                continue

            keypoints_pred = pose_results[0][0]['keypoints'][:, :2]
            keypoints = np.zeros((17, 2), dtype=np.float32)
            for j in range(1, 18):
                if j == 1:
                    left = keypoints_pred[5]
                    right = keypoints_pred[6]
                    keypoints[j - 1] = (left + right) / 2
                else:
                    keypoints[j - 1] = keypoints_pred[KEYPOINT_MAPPING_COCO[j]]

            keypoints_normalized = normalize_keypoints_sequence(keypoints[np.newaxis, ...])[0]
            id_to_keypoints_buffer[track_id].append(keypoints_normalized)
            id_to_frame_buffer[track_id].append(frame.copy())

            label = "No Fall"

            if len(id_to_keypoints_buffer[track_id]) == window_size:
                window = np.stack(id_to_keypoints_buffer[track_id], axis=0)
                if not np.all(window == 0):
                    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                    _, pred = torch.max(output, 1)

                    if pred.item() == 1:
                        label = "Fall"
                        id_to_fall_count[track_id] += 1
                        if id_to_fall_count[track_id] >= 5:
                            save_frame_buffer(list(id_to_frame_buffer[track_id]), output_dir, fps, width, height)
                            id_to_fall_count[track_id] = 0
                    else:
                        id_to_fall_count[track_id] = 0

            # Vẽ khung và label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(frame, f"ID {track_id}: {label}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)

        if fps_count >= 10:
            display_fps, prev_time = calculate_fps(prev_time, fps_count)
            fps_count = 0

        cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Fall Detection", frame)
        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "FallClipsTest/2.avi"  #video.mp4
    model_path = "best_kamtfenet_new.pth"
    output_video_path = "output_multi_person.mp4"

    yolo_model = YOLO("yolov8x.pt")
    config_path = 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
    checkpoint_path = 'models/vitpose/vitpose_weights/vitpose_base_coco_aic_mpii.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vitpose_model = init_pose_model(config_path, checkpoint_path, device=device)

    predict_multi_person(video_path, model_path, yolo_model, vitpose_model,
                         window_size=30, device=device,
                         output_video_path=output_video_path)


# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import time
# import uuid
# import os
# from collections import defaultdict, deque
# from ultralytics import YOLO
# from ViTPose.mmpose.apis.inference import init_pose_model, inference_top_down_pose_model
# from rlstm_model import KAMTFENet
#
# KEYPOINT_MAPPING_COCO = {
#     1: (5, 6), 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12,
#     9: 14, 10: 16, 11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3
# }
#
# # Hàm tính FPS
# def calculate_fps(prev_time, frame_count):
#     current_time = time.time()
#     elapsed_time = current_time - prev_time
#     fps = frame_count / elapsed_time if elapsed_time > 0 else 0
#     return fps, current_time
#
# # Hàm chuẩn hóa keypoints
# def normalize_keypoints_sequence(keypoints):
#     keypoints_norm = np.zeros_like(keypoints)
#     for t in range(keypoints.shape[0]):
#         frame = keypoints[t]
#         origin = frame[7]
#         centered = frame - origin
#         shoulder_dist = np.linalg.norm(frame[1] - frame[4])
#         shoulder_dist = shoulder_dist if shoulder_dist >= 1e-5 else 1.0
#         normalized = centered / shoulder_dist
#         keypoints_norm[t] = normalized
#     return keypoints_norm
#
# # Hàm lưu video clip té ngã vào thư mục theo tài khoản và ngày
# def save_frame_buffer(frame_buffer, output_root_dir, fps, width, height, user_id):
#     if not frame_buffer:
#         return None
#
#     # Lấy ngày và giờ hiện tại để tạo thư mục
#     now = time.localtime()
#     date_str = time.strftime("%Y-%m-%d", now)
#     time_str = time.strftime("%H-%M-%S", now)
#
#     # Tạo đường dẫn thư mục theo user_id và ngày
#     output_dir = os.path.join(output_root_dir, str(user_id), date_str)
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Tạo tên file clip với thời gian hiện tại
#     clip_name = f"fall_{time_str}.mp4"
#     clip_path = os.path.join(output_dir, clip_name)
#
#     # Ghi video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
#     for frame in frame_buffer:
#         out.write(frame)
#     out.release()
#
#     return clip_path
#
# # Hàm chính phát hiện té ngã và lưu video clip
# def predict_multi_person(video_path, model_path, yolo_model, pose_model,
#                          window_size=30, device='cuda' if torch.cuda.is_available() else 'cpu',
#                          output_root_dir="fall_clips", output_video_path=None,
#                          max_buffer_size=60, user_id="default_user"):
#
#     model = KAMTFENet(num_keypoints=17, seq_len=30, input_size=34, hidden_size=256, num_layers=3)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open video: {video_path}")
#         return
#
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width, height = 640, 480
#
#     out = None
#     if output_video_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     # Buffers
#     id_to_keypoints_buffer = defaultdict(lambda: deque(maxlen=window_size))
#     id_to_frame_buffer = defaultdict(lambda: deque(maxlen=max_buffer_size))
#     id_to_fall_count = defaultdict(int)
#
#     frame_count, fps_count = 0, 0
#     prev_time = time.time()
#     display_fps = 0
#
#     cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (width, height))
#         frame_count += 1
#         fps_count += 1
#
#         # Tracking người
#         results = yolo_model.track(frame, persist=True, device=device, classes=[0])
#         boxes = results[0].boxes
#         ids = results[0].boxes.id
#
#         if boxes is None or ids is None:
#             continue
#
#         for i, box in enumerate(boxes):
#             track_id = int(ids[i].item())
#             x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
#             person_img = frame[y_min:y_max, x_min:x_max]
#             person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
#
#             person_results = [{'bbox': np.array([0, 0, person_img.shape[1], person_img.shape[0]])}]
#             pose_results = inference_top_down_pose_model(pose_model, person_img_rgb, person_results, format='xywh')
#             if not pose_results:
#                 continue
#
#             keypoints_pred = pose_results[0][0]['keypoints'][:, :2]
#             keypoints = np.zeros((17, 2), dtype=np.float32)
#             for j in range(1, 18):
#                 if j == 1:
#                     left = keypoints_pred[5]
#                     right = keypoints_pred[6]
#                     keypoints[j - 1] = (left + right) / 2
#                 else:
#                     keypoints[j - 1] = keypoints_pred[KEYPOINT_MAPPING_COCO[j]]
#
#             keypoints_normalized = normalize_keypoints_sequence(keypoints[np.newaxis, ...])[0]
#             id_to_keypoints_buffer[track_id].append(keypoints_normalized)
#             id_to_frame_buffer[track_id].append(frame.copy())
#
#             label = "No Fall"
#
#             if len(id_to_keypoints_buffer[track_id]) == window_size:
#                 window = np.stack(id_to_keypoints_buffer[track_id], axis=0)
#                 if not np.all(window == 0):
#                     input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
#                     with torch.no_grad():
#                         output = model(input_tensor)
#                     _, pred = torch.max(output, 1)
#
#                     if pred.item() == 1:
#                         label = "Fall"
#                         id_to_fall_count[track_id] += 1
#                         if id_to_fall_count[track_id] >= 5:
#                             # Lưu video clip khi phát hiện té ngã
#                             save_frame_buffer(list(id_to_frame_buffer[track_id]), output_root_dir, fps, width, height, user_id)
#                             id_to_fall_count[track_id] = 0
#                     else:
#                         id_to_fall_count[track_id] = 0
#
#             # Vẽ khung và label
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#             cv2.putText(frame, f"ID {track_id}: {label}", (x_min, y_min - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)
#
#         if fps_count >= 10:
#             display_fps, prev_time = calculate_fps(prev_time, fps_count)
#             fps_count = 0
#
#         cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#
#         cv2.imshow("Fall Detection", frame)
#         if out:
#             out.write(frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     if out:
#         out.release()
#     cv2.destroyAllWindows()
#
#     # Print inference statistics
#     if inference_times:
#         avg_inference_time = np.mean(inference_times)
#         inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
#         print(f"\nAverage RLSTM inference time: {avg_inference_time * 1000:.2f} ms")
#         print(f"RLSTM inference FPS: {inference_fps:.2f}")
#     else:
#         print("\nNo RLSTM inferences performed.")
#
#     # Calculate final prediction
#     fall_count = np.sum(np.array(predictions) == 1)
#     total_windows = len(predictions)
#     fall_ratio = fall_count / total_windows if total_windows > 0 else 0
#     threshold = 0.5
#     final_prediction = "fall" if fall_ratio > threshold else "no_fall"
#
#     print(f"\nPrediction results:")
#     print(f"Frames predicted as 'fall': {fall_count}/{total_windows}")
#     print(f"Fall ratio: {fall_ratio:.2f}")
#     print(f"Final prediction: {final_prediction}")
#     print(f"Average overall FPS: {display_fps:.2f}")

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import uuid
import os
from collections import defaultdict, deque
from ultralytics import YOLO
from ViTPose.mmpose.apis.inference import init_pose_model, inference_top_down_pose_model
from rlstm_model import KAMTFENet

KEYPOINT_MAPPING_COCO = {
    1: (5, 6), 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12,
    9: 14, 10: 16, 11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3
}


# Hàm tính FPS
def calculate_fps(prev_time, frame_count):
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return fps, current_time


# Hàm chuẩn hóa keypoints
def normalize_keypoints_sequence(keypoints):
    keypoints_norm = np.zeros_like(keypoints)
    for t in range(keypoints.shape[0]):
        frame = keypoints[t]
        origin = frame[7]
        centered = frame - origin
        shoulder_dist = np.linalg.norm(frame[1] - frame[4])
        shoulder_dist = shoulder_dist if shoulder_dist >= 1e-5 else 1.0
        normalized = centered / shoulder_dist
        keypoints_norm[t] = normalized
    return keypoints_norm


# Hàm lưu video clip té ngã vào thư mục theo tài khoản và ngày
def save_frame_buffer(frame_buffer, output_root_dir, fps, width, height, user_id):
    if not frame_buffer:
        return None

    # Lấy ngày và giờ hiện tại để tạo thư mục
    now = time.localtime()
    date_str = time.strftime("%Y-%m-%d", now)
    time_str = time.strftime("%H-%M-%S", now)

    # Tạo đường dẫn thư mục theo user_id và ngày
    output_dir = os.path.join(output_root_dir, str(user_id), date_str)
    os.makedirs(output_dir, exist_ok=True)

    # Tạo tên file clip với thời gian hiện tại
    clip_name = f"fall_{time_str}.mp4"
    clip_path = os.path.join(output_dir, clip_name)

    # Ghi video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    for frame in frame_buffer:
        out.write(frame)
    out.release()

    return clip_path


# # Hàm chính phát hiện té ngã và lưu video clip
# def predict_multi_person(video_path, model_path, yolo_model, pose_model,
#                          window_size=30, device='cuda' if torch.cuda.is_available() else 'cpu',
#                          output_root_dir="fall_clips", output_video_path=None,
#                          max_buffer_size=60, user_id="default_user"):
#     model = KAMTFENet(num_keypoints=17, seq_len=30, input_size=34, hidden_size=256, num_layers=3)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open video: {video_path}")
#         return
#
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width, height = 640, 480
#
#     out = None
#     if output_video_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     # Buffers
#     id_to_keypoints_buffer = defaultdict(lambda: deque(maxlen=window_size))
#     id_to_frame_buffer = defaultdict(lambda: deque(maxlen=max_buffer_size))
#     id_to_fall_count = defaultdict(int)
#
#     frame_count, fps_count = 0, 0
#     prev_time = time.time()
#     display_fps = 0
#
#     # Lưu trữ thời gian suy luận
#     inference_times = []
#     predictions = []
#
#     cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (width, height))
#         frame_count += 1
#         fps_count += 1
#
#         # Tracking người
#         results = yolo_model.track(frame, persist=True, device=device, classes=[0])
#         boxes = results[0].boxes
#         ids = results[0].boxes.id
#
#         if boxes is None or ids is None:
#             continue
#
#         for i, box in enumerate(boxes):
#             track_id = int(ids[i].item())
#             x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
#             person_img = frame[y_min:y_max, x_min:x_max]
#             person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
#
#             person_results = [{'bbox': np.array([0, 0, person_img.shape[1], person_img.shape[0]])}]
#             pose_results = inference_top_down_pose_model(pose_model, person_img_rgb, person_results, format='xywh')
#             if not pose_results:
#                 continue
#
#             keypoints_pred = pose_results[0][0]['keypoints'][:, :2]
#             keypoints = np.zeros((17, 2), dtype=np.float32)
#             for j in range(1, 18):
#                 if j == 1:
#                     left = keypoints_pred[5]
#                     right = keypoints_pred[6]
#                     keypoints[j - 1] = (left + right) / 2
#                 else:
#                     keypoints[j - 1] = keypoints_pred[KEYPOINT_MAPPING_COCO[j]]
#
#             keypoints_normalized = normalize_keypoints_sequence(keypoints[np.newaxis, ...])[0]
#             id_to_keypoints_buffer[track_id].append(keypoints_normalized)
#             id_to_frame_buffer[track_id].append(frame.copy())
#
#             label = "No Fall"
#
#             if len(id_to_keypoints_buffer[track_id]) == window_size:
#                 window = np.stack(id_to_keypoints_buffer[track_id], axis=0)
#                 if not np.all(window == 0):
#                     input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
#
#                     # Tính thời gian suy luận RLSTM
#                     start_time = time.time()
#                     with torch.no_grad():
#                         output = model(input_tensor)
#                     inference_time = time.time() - start_time
#                     inference_times.append(inference_time)
#
#                     _, pred = torch.max(output, 1)
#                     predictions.append(pred.item())
#
#                     if pred.item() == 1:
#                         label = "Fall"
#                         id_to_fall_count[track_id] += 1
#                         if id_to_fall_count[track_id] >= 5:
#                             # Lưu video clip khi phát hiện té ngã
#                             save_frame_buffer(list(id_to_frame_buffer[track_id]), output_root_dir, fps, width, height,
#                                               user_id)
#                             id_to_fall_count[track_id] = 0
#                     else:
#                         id_to_fall_count[track_id] = 0
#
#             # Vẽ khung và label
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#             cv2.putText(frame, f"ID {track_id}: {label}", (x_min, y_min - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)
#
#         if fps_count >= 10:
#             display_fps, prev_time = calculate_fps(prev_time, fps_count)
#             fps_count = 0
#
#         cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#
#         cv2.imshow("Fall Detection", frame)
#         if out:
#             out.write(frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     if out:
#         out.release()
#     cv2.destroyAllWindows()
#
#     # Print inference statistics
#     if inference_times:
#         avg_inference_time = np.mean(inference_times)
#         inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
#         print(f"\nAverage RLSTM inference time: {avg_inference_time * 1000:.2f} ms")
#         print(f"RLSTM inference FPS: {inference_fps:.2f}")
#     else:
#         print("\nNo RLSTM inferences performed.")
#
#     # Calculate final prediction
#     fall_count = np.sum(np.array(predictions) == 1)
#     total_windows = len(predictions)
#     fall_ratio = fall_count / total_windows if total_windows > 0 else 0
#     threshold = 0.5
#     final_prediction = "fall" if fall_ratio > threshold else "no_fall"
#
#     print(f"\nPrediction results:")
#     print(f"Frames predicted as 'fall': {fall_count}/{total_windows}")
#     print(f"Fall ratio: {fall_ratio:.2f}")
#     print(f"Final prediction: {final_prediction}")
#     print(f"Average overall FPS: {display_fps:.2f}")

# def predict_multi_person(video_path, model_path, yolo_model, pose_model,
#                          window_size=30, device='cuda' if torch.cuda.is_available() else 'cpu',
#                          output_root_dir="fall_clips", output_video_path=None,
#                          max_buffer_size=60, user_id="default_user",
#                          headless=True):  # <-- thêm headless
#
#     model = KAMTFENet(num_keypoints=17, seq_len=30, input_size=34, hidden_size=256, num_layers=3)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open video: {video_path}")
#         return
#
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width, height = 640, 480
#
#     out = None
#     if output_video_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     id_to_keypoints_buffer = defaultdict(lambda: deque(maxlen=window_size))
#     id_to_frame_buffer = defaultdict(lambda: deque(maxlen=max_buffer_size))
#     id_to_fall_count = defaultdict(int)
#
#     frame_count, fps_count = 0, 0
#     prev_time = time.time()
#     display_fps = 0
#     inference_times = []
#     predictions = []
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (width, height))
#         frame_count += 1
#         fps_count += 1
#
#         results = yolo_model.track(frame, persist=True, device=device, classes=[0])
#         boxes = results[0].boxes
#         ids = results[0].boxes.id
#
#         if boxes is None or ids is None:
#             continue
#
#         for i, box in enumerate(boxes):
#             track_id = int(ids[i].item())
#             x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
#             person_img = frame[y_min:y_max, x_min:x_max]
#             person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
#
#             person_results = [{'bbox': np.array([0, 0, person_img.shape[1], person_img.shape[0]])}]
#             pose_results = inference_top_down_pose_model(pose_model, person_img_rgb, person_results, format='xywh')
#             if not pose_results:
#                 continue
#
#             keypoints_pred = pose_results[0][0]['keypoints'][:, :2]
#             keypoints = np.zeros((17, 2), dtype=np.float32)
#             for j in range(1, 18):
#                 if j == 1:
#                     left = keypoints_pred[5]
#                     right = keypoints_pred[6]
#                     keypoints[j - 1] = (left + right) / 2
#                 else:
#                     keypoints[j - 1] = keypoints_pred[KEYPOINT_MAPPING_COCO[j]]
#
#             keypoints_normalized = normalize_keypoints_sequence(keypoints[np.newaxis, ...])[0]
#             id_to_keypoints_buffer[track_id].append(keypoints_normalized)
#             id_to_frame_buffer[track_id].append(frame.copy())
#
#             label = "No Fall"
#             if len(id_to_keypoints_buffer[track_id]) == window_size:
#                 window = np.stack(id_to_keypoints_buffer[track_id], axis=0)
#                 if not np.all(window == 0):
#                     input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
#
#                     start_time = time.time()
#                     with torch.no_grad():
#                         output = model(input_tensor)
#                     inference_time = time.time() - start_time
#                     inference_times.append(inference_time)
#
#                     _, pred = torch.max(output, 1)
#                     predictions.append(pred.item())
#
#                     if pred.item() == 1:
#                         label = "Fall"
#                         id_to_fall_count[track_id] += 1
#                         if id_to_fall_count[track_id] >= 5:
#                             save_frame_buffer(list(id_to_frame_buffer[track_id]), output_root_dir, fps, width, height, user_id)
#                             id_to_fall_count[track_id] = 0
#                     else:
#                         id_to_fall_count[track_id] = 0
#
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#             cv2.putText(frame, f"ID {track_id}: {label}", (x_min, y_min - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)
#
#         if fps_count >= 10:
#             display_fps, prev_time = calculate_fps(prev_time, fps_count)
#             fps_count = 0
#
#         if not headless:
#             cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#             cv2.imshow("Fall Detection", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         if out:
#             out.write(frame)
#
#     cap.release()
#     if out:
#         out.release()
#     if not headless:
#         cv2.destroyAllWindows()
#
#     if inference_times:
#         avg_inference_time = np.mean(inference_times)
#         inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
#         print(f"\nAverage RLSTM inference time: {avg_inference_time * 1000:.2f} ms")
#         print(f"RLSTM inference FPS: {inference_fps:.2f}")
#     else:
#         print("\nNo RLSTM inferences performed.")
#
#     fall_count = np.sum(np.array(predictions) == 1)
#     total_windows = len(predictions)
#     fall_ratio = fall_count / total_windows if total_windows > 0 else 0
#     threshold = 0.5
#     final_prediction = "fall" if fall_ratio > threshold else "no_fall"
#
#     print(f"\nPrediction results:")
#     print(f"Frames predicted as 'fall': {fall_count}/{total_windows}")
#     print(f"Fall ratio: {fall_ratio:.2f}")
#     print(f"Final prediction: {final_prediction}")
#     print(f"Average overall FPS: {display_fps:.2f}")
#
#
# if __name__ == "__main__":
#     video_path = "rtsp://admin:QTCDRN@192.168.2.177:554/live1.264"  # hoặc "video.mp4"
#     # name = (video_path.split("/")[1].split("."))[:-1]
#
#     model_path = "best_kamtfenet_new.pth"
#     output_video_path = f"FallClipsOutput/output_multi_person_(1).mp4"
#
#     yolo_model = YOLO("best.pt")
#     config_path = 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
#     checkpoint_path = 'models/vitpose/vitpose_weights/vitpose_base_coco_aic_mpii.pth'
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     vitpose_model = init_pose_model(config_path, checkpoint_path, device=device)
#
#     # Gọi hàm dự đoán và phát hiện té ngã
#     predict_multi_person(video_path, model_path, yolo_model, vitpose_model,
#                          window_size=30, device=device,
#                          output_video_path=output_video_path, user_id="user_123", headless=False)

