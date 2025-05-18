import torch
import cv2
import time
import os
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from rlstm_model import KAMTFENet  # Đảm bảo bạn có model này

def calculate_fps(prev_time, frame_count):
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return fps, current_time

# def normalize_keypoints_sequence(keypoints):
#     keypoints_norm = np.zeros_like(keypoints)
#     for t in range(keypoints.shape[0]):
#         frame = keypoints[t]
#         origin = frame[7]
#         centered = frame - origin
#         shoulder_dist = np.linalg.norm(frame[5] - frame[6])
#         shoulder_dist = shoulder_dist if shoulder_dist >= 1e-5 else 1.0
#         normalized = centered / shoulder_dist
#         keypoints_norm[t] = normalized
#     return keypoints_norm


# def normalize_keypoints_sequence(keypoints):
#
#
#     """
#     Chuẩn hóa keypoints theo từng frame.
#     - Gốc là hông phải (index 12 trong chuẩn COCO).
#     - Scale theo khoảng cách giữa hai vai (index 5 và 6 trong chuẩn COCO).
#     """
#     if not isinstance(keypoints, np.ndarray):
#         print("✘ Lỗi: Keypoints không phải mảng numpy")
#         return np.zeros((1, 17, 2))
#
#     keypoints = np.squeeze(keypoints)
#     if len(keypoints.shape) == 2:
#         keypoints = keypoints[np.newaxis, ...]
#     if len(keypoints.shape) != 3 or keypoints.shape[1] != 17 or keypoints.shape[2] != 2:
#         print(f"✘ Lỗi: Shape keypoints không hợp lệ {keypoints.shape}. Mong đợi [N, 17, 2]")
#         return np.zeros((1, 17, 2))
#
#     keypoints_norm = np.zeros_like(keypoints)
#     shoulder_dists = []
#
#     for t in range(keypoints.shape[0]):
#         frame = keypoints[t]
#         right_shoulder = frame[6]  # Index 6: Right shoulder
#         left_shoulder = frame[5]   # Index 5: Left shoulder
#         if np.all(right_shoulder == 0) or np.all(left_shoulder == 0):
#             print(f"Khung {t}: Vai không hợp lệ (tọa độ bằng 0)")
#             continue
#
#         shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)
#         print(f"Khoảng cách vai khung {t}: {shoulder_dist:.2f}")
#         if shoulder_dist >= 0.01:
#             shoulder_dists.append(shoulder_dist)
#
#     mean_shoulder_dist = np.mean(shoulder_dists) if shoulder_dists else 100.0
#     print(f"Khoảng cách vai trung bình: {mean_shoulder_dist:.2f}")
#
#     for t in range(keypoints.shape[0]):
#         frame = keypoints[t]
#         origin = frame[12]  # Index 12: Right hip
#         if np.all(origin == 0):
#             print(f"Khung {t}: Hông phải bằng 0, không chuẩn hóa")
#             keypoints_norm[t] = frame
#             continue
#         centered = frame - origin
#         normalized = centered / mean_shoulder_dist if mean_shoulder_dist > 0 else centered
#         keypoints_norm[t] = normalized
#
#     return keypoints_norm

def normalize_keypoints_sequence(keypoints):
    """
    Chuẩn hóa keypoints theo từng frame.
    - Gốc là hông phải (index 12)
    - Scale theo khoảng cách giữa 2 vai (index 6 và 5)
    """
    keypoints_norm = np.zeros_like(keypoints)
    for t in range(keypoints.shape[0]):
        frame = keypoints[t]

        # Lấy hông phải làm gốc
        origin = frame[12]
        centered = frame - origin

        # Tính khoảng cách giữa hai vai
        right_shoulder = frame[6]
        left_shoulder = frame[5]
        shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)

        # Tránh chia cho 0
        if shoulder_dist < 1e-5:
            shoulder_dist = 1.0
        # Chuẩn hóa bằng cách chia cho khoảng cách vai
        normalized = centered / shoulder_dist
        keypoints_norm[t] = normalized
    return keypoints_norm

def save_frame_buffer(frame_buffer, output_root_dir, fps, width, height, user_id):
    if not frame_buffer:
        return None
    now = time.localtime()
    date_str = time.strftime("%Y-%m-%d", now)
    time_str = time.strftime("%H-%M-%S", now)
    output_dir = os.path.join(output_root_dir, str(user_id), date_str)
    os.makedirs(output_dir, exist_ok=True)
    clip_name = f"fall_{time_str}.mp4"
    clip_path = os.path.join(output_dir, clip_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    for frame in frame_buffer:
        out.write(frame)
    out.release()
    return clip_path

def predict_multi_person(video_path, model_path, yolo_model,
                         window_size=30, device='cuda',
                         output_root_dir="fall_clips", output_video_path=None,
                         max_buffer_size=60, user_id="default_user",
                         headless=True):

    model = KAMTFENet(num_keypoints=17, seq_len=window_size, input_size=34, hidden_size=256, num_layers=3)
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

    id_to_keypoints_buffer = defaultdict(lambda: deque(maxlen=window_size))
    id_to_frame_buffer = defaultdict(lambda: deque(maxlen=max_buffer_size))
    id_to_fall_count = defaultdict(int)

    frame_count, fps_count = 0, 0
    prev_time = time.time()
    display_fps = 0
    inference_times = []
    predictions = []

    # Định nghĩa skeleton (COCO format)
    skeleton = [
        (5, 7), (7, 9), (6, 8), (8, 10),  # tay
        (5, 6),                           # vai
        (11, 13), (13, 15), (12, 14), (14, 16),  # chân
        (11, 12), (5, 11), (6, 12)        # hông và thân
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame_count += 1
        fps_count += 1

        results = yolo_model.track(frame, persist=True, device=device, verbose=False, conf=0.6)

        if results and results[0].keypoints is not None:
            boxes = results[0].boxes
            keypoints_list = results[0].keypoints.xy  # (num_person, 17, 2)

            for i, (box, kps) in enumerate(zip(boxes, keypoints_list)):
                track_id = int(boxes.id[i].item()) if boxes.id is not None else i
                kps = kps.cpu().numpy().astype(np.float32)

                # Vẽ keypoints
                for x, y in kps:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

                # Vẽ skeleton
                for i1, i2 in skeleton:
                    x1, y1 = kps[i1]
                    x2, y2 = kps[i2]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                if kps.shape[0] != 17:
                    continue

                keypoints_normalized = normalize_keypoints_sequence(kps[np.newaxis, ...])[0]
                id_to_keypoints_buffer[track_id].append(keypoints_normalized)
                id_to_frame_buffer[track_id].append(frame.copy())

                label = "No Fall"

                if len(id_to_keypoints_buffer[track_id]) == window_size:
                    window = np.stack(id_to_keypoints_buffer[track_id], axis=0)

                    print(window)

                    if not np.all(window == 0):
                        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                        start_time = time.time()
                        with torch.no_grad():
                            output = model(input_tensor)
                        inference_time = time.time() - start_time
                        inference_times.append(inference_time)
                        _, pred = torch.max(output, 1)
                        predictions.append(pred.item())

                        if pred.item() == 1:
                            label = "Fall"
                            id_to_fall_count[track_id] += 1
                            if id_to_fall_count[track_id] >= 5:
                                save_frame_buffer(list(id_to_frame_buffer[track_id]), output_root_dir, fps, width, height, user_id)
                                id_to_fall_count[track_id] = 0
                        else:
                            id_to_fall_count[track_id] = 0

                x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.putText(frame, f"ID {track_id}: {label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)

        if fps_count >= 10:
            display_fps, prev_time = calculate_fps(prev_time, fps_count)
            fps_count = 0

        cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if not headless:
            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out:
            out.write(frame)

    cap.release()
    if out:
        out.release()
    if not headless:
        cv2.destroyAllWindows()

    if inference_times:
        avg_inference_time = np.mean(inference_times)
        print(f"Avg RLSTM inference time: {avg_inference_time * 1000:.2f} ms")
        print(f"Inference FPS: {1 / avg_inference_time:.2f}")

    fall_count = sum(np.array(predictions) == 1)
    total = len(predictions)
    ratio = fall_count / total if total else 0
    print(f"Fall ratio: {ratio:.2f}, Final: {'fall' if ratio > 0.5 else 'no_fall'}")


# def predict_multi_person(video_path, model_path, yolo_model,
#                          window_size=30, device='cuda',
#                          output_root_dir="fall_clips", output_video_path=None,
#                          max_buffer_size=60, user_id="default_user",
#                          headless=True):
#
#     model = KAMTFENet(num_keypoints=17, seq_len=window_size, input_size=34, hidden_size=256, num_layers=3)
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
#         results = yolo_model.track(frame, persist=True, device=device, verbose=False)
#
#         if results and results[0].keypoints is not None:
#             for i, (box, kps) in enumerate(zip(results[0].boxes, results[0].keypoints.xy)):
#                 track_id = int(results[0].boxes.id[i].item()) if results[0].boxes.id is not None else i
#                 kps = kps.cpu().numpy().astype(np.float32)
#
#                 if kps.shape[0] != 17:
#                     continue
#
#                 keypoints_normalized = normalize_keypoints_sequence(kps[np.newaxis, ...])[0]
#                 id_to_keypoints_buffer[track_id].append(keypoints_normalized)
#                 id_to_frame_buffer[track_id].append(frame.copy())
#
#                 label = "No Fall"
#
#                 if len(id_to_keypoints_buffer[track_id]) == window_size:
#                     window = np.stack(id_to_keypoints_buffer[track_id], axis=0)
#                     if not np.all(window == 0):
#                         input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
#                         start_time = time.time()
#                         with torch.no_grad():
#                             output = model(input_tensor)
#                         inference_time = time.time() - start_time
#                         inference_times.append(inference_time)
#                         _, pred = torch.max(output, 1)
#                         predictions.append(pred.item())
#
#                         if pred.item() == 1:
#                             label = "Fall"
#                             id_to_fall_count[track_id] += 1
#                             if id_to_fall_count[track_id] >= 5:
#                                 save_frame_buffer(list(id_to_frame_buffer[track_id]), output_root_dir, fps, width, height, user_id)
#                                 id_to_fall_count[track_id] = 0
#                         else:
#                             id_to_fall_count[track_id] = 0
#
#                 x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#                 cv2.putText(frame, f"ID {track_id}: {label}", (x_min, y_min - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)
#
#         if fps_count >= 10:
#             display_fps, prev_time = calculate_fps(prev_time, fps_count)
#             fps_count = 0
#
#         cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
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
#         print(f"Avg RLSTM inference time: {avg_inference_time * 1000:.2f} ms")
#         print(f"Inference FPS: {1 / avg_inference_time:.2f}")
#
#     fall_count = sum(np.array(predictions) == 1)
#     total = len(predictions)
#     ratio = fall_count / total if total else 0
#     print(f"Fall ratio: {ratio:.2f}, Final: {'fall' if ratio > 0.5 else 'no_fall'}")


# from mediapipe import solutions as mp
#  # Giả sử bạn đã có KAMTFENet định nghĩa sẵn
#
# # Mediapipe Pose Model
# mp_pose = mp.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# # Cập nhật hàm để tích hợp Mediapipe Pose
# def predict_multi_person1(video_path, model_path, yolo_model, pose_model,
#                          window_size=30, device='cuda' if torch.cuda.is_available() else 'cpu',
#                          output_root_dir="fall_clips", output_video_path=None,
#                          max_buffer_size=60, user_id="default_user",
#                          headless=True):  # <-- thêm headless
#
#     # Khởi tạo mô hình RLSTM của bạn
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
#         # Dự đoán từ YOLO (phát hiện người)
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
#             # Sử dụng Mediapipe Pose để lấy keypoints
#             results_pose = pose.process(person_img_rgb)
#
#             if results_pose.pose_landmarks:
#                 keypoints_pred = results_pose.pose_landmarks.landmark
#                 keypoints = np.zeros((17, 2), dtype=np.float32)
#
#                 for j in range(17):
#                     keypoints[j] = [keypoints_pred[j].x, keypoints_pred[j].y]
#
#                 # Normalize keypoints
#                 keypoints_normalized = normalize_keypoints_sequence(keypoints[np.newaxis, ...])[0]
#                 id_to_keypoints_buffer[track_id].append(keypoints_normalized)
#                 id_to_frame_buffer[track_id].append(frame.copy())
#
#                 label = "No Fall"
#                 if len(id_to_keypoints_buffer[track_id]) == window_size:
#                     window = np.stack(id_to_keypoints_buffer[track_id], axis=0)
#                     if not np.all(window == 0):
#                         input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
#
#                         # Tính toán thời gian suy luận
#                         start_time = time.time()
#                         with torch.no_grad():
#                             output = model(input_tensor)
#                         inference_time = time.time() - start_time
#                         inference_times.append(inference_time)
#
#                         _, pred = torch.max(output, 1)
#                         predictions.append(pred.item())
#
#                         # Kiểm tra kết quả dự đoán
#                         if pred.item() == 1:
#                             label = "Fall"
#                             id_to_fall_count[track_id] += 1
#                             if id_to_fall_count[track_id] >= 5:
#                                 save_frame_buffer(list(id_to_frame_buffer[track_id]), output_root_dir, fps, width, height, user_id)
#                                 id_to_fall_count[track_id] = 0
#                         else:
#                             id_to_fall_count[track_id] = 0
#
#                 # Vẽ hộp và label lên frame
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#                 cv2.putText(frame, f"ID {track_id}: {label}", (x_min, y_min - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label == "Fall" else (0, 255, 0), 2)
#
#         # Hiển thị FPS nếu không ở chế độ headless
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

if __name__ == "__main__":
    video_path = 0
    # name = (video_path.split("/")[1].split("."))[:-1]
    output_video_path = f"FallClipsOutput/output_multi_person_(1).mp4"

    model_path = r"best_kamtfenet2.pth"
    yolo_model = YOLO("blened2.pt")
    predict_multi_person(video_path, model_path, yolo_model, output_video_path=output_video_path, device='cuda', headless=False)