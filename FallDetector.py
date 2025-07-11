import cv2
import time
import os
from ultralytics import YOLO
from playsound import playsound
import threading
from datetime import datetime
from collections import deque

class FallDetector:
    def __init__(self, video_path, model_path, alert_sound="alert.mp3", output_path="output_pose_fall.mp4"):
        # === CẤU HÌNH ===
        self.VIDEO_PATH = video_path
        self.MODEL_PATH = model_path
        self.ALERT_SOUND = alert_sound
        self.RESIZE_WIDTH = 640
        self.RESIZE_HEIGHT = 480
        self.KEYPOINT_THRESHOLD = 0.5
        self.MIN_VALID_KEYPOINTS = 8
        self.FALL_CONFIRM_FRAMES = 5
        self.RELATIVE_DROP_THRESHOLD = 0.01
        self.ALERT_INTERVAL = 5
        self.FALL_RECORD_SECONDS = 5
        self.BUFFER_SECONDS = 1
        self.LABEL_HISTORY_FRAMES = 20
        self.FALL_PERSISTENCE_SECONDS = 3

        # === KHỞI TẠO MODEL, VIDEO ===
        self.model = YOLO(self.MODEL_PATH)
        self.cap = cv2.VideoCapture(self.VIDEO_PATH)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))

        self.frame_id = 0
        self.last_alert_time = 0
        self.alert_lock = threading.Lock()
        self.frame_buffer = deque(maxlen=int(self.fps * self.BUFFER_SECONDS))

        # Biến theo dõi trạng thái
        self.prev_centers = {}
        self.last_seen_frame = {}
        self.fall_signs = {}
        self.label_history = {}
        self.track_start_frame = {}
        self.fall_confirmed_ids = set()
        self.fall_persist_start_time = {}
        self.persistent_alerted_ids = set()
        self.fall_ids_logged = set()
        self.recording_fall = {}
        self.log_file = "fall_log.txt"

    def play_alert_sound(self):
        with self.alert_lock:
            playsound(self.ALERT_SOUND)

    def log_fall(self, track_id):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{now}] Fall Detect - ID: {track_id}\n")

    def create_fall_writer(self, track_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fall_clip_{track_id}_{timestamp}.mp4"
        return cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))

    def check_object_missing(self):
        for tid in list(self.last_seen_frame):
            if self.frame_id - self.last_seen_frame[tid] > 30:
                self.prev_centers.pop(tid, None)
                self.last_seen_frame.pop(tid, None)
                self.fall_signs.pop(tid, None)
                self.label_history.pop(tid, None)
                self.track_start_frame.pop(tid, None)
                self.fall_confirmed_ids.discard(tid)
                self.fall_persist_start_time.pop(tid, None)
                self.persistent_alerted_ids.discard(tid)

    def draw_keypoints(self, frame, kps_xy, kps_conf):
        for (x, y), conf in zip(kps_xy, kps_conf):
            if conf >= self.KEYPOINT_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
            self.frame_buffer.append(frame.copy())
            start_time = time.time()
            fall_detected_this_frame = False

            try:
                results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
                result = results[0] if results else None
                if not result or not result.boxes or result.boxes.xyxy is None:
                    raise ValueError("Không có kết quả từ model")

                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []
                cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
                kps_xy = result.keypoints.xy.cpu().numpy() if result.keypoints else None
                kps_conf = result.keypoints.conf.cpu().numpy() if result.keypoints and result.keypoints.conf is not None else None

                self.check_object_missing()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    track_id = ids[i]
                    cls_label = "fall" if cls[i] == 1 else "person"

                    if track_id not in self.track_start_frame:
                        self.track_start_frame[track_id] = self.frame_id

                    if self.frame_id - self.track_start_frame[track_id] < self.LABEL_HISTORY_FRAMES:
                        continue

                    self.label_history.setdefault(track_id, deque(maxlen=self.LABEL_HISTORY_FRAMES)).append(cls_label)
                    label_seq = list(self.label_history[track_id])

                    recent_transition = False
                    if len(label_seq) == self.LABEL_HISTORY_FRAMES and 5 <= label_seq.count("person") <= 15:
                        try:
                            fall_start = label_seq.index("fall")
                            if (all(i == "person" for i in label_seq[:fall_start]) and
                                    all(j == "fall" for j in label_seq[fall_start:])):
                                recent_transition = True
                        except ValueError:
                            pass

                    if kps_conf is not None:
                        valid_kps = sum(1 for c in kps_conf[i] if c >= self.KEYPOINT_THRESHOLD)
                        if valid_kps < self.MIN_VALID_KEYPOINTS:
                            continue

                    w = x2 - x1
                    h = y2 - y1
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    prev_center = self.prev_centers.get(track_id, (x_center, y_center))
                    rel_delta_y = (y_center - prev_center[1]) / (h + 1e-6)

                    self.prev_centers[track_id] = (x_center, y_center)
                    self.last_seen_frame[track_id] = self.frame_id

                    self.fall_signs.setdefault(track_id, []).append(rel_delta_y)
                    if len(self.fall_signs[track_id]) > self.FALL_CONFIRM_FRAMES:
                        self.fall_signs[track_id].pop(0)

                    fall_by_movement = all(v > self.RELATIVE_DROP_THRESHOLD for v in self.fall_signs[track_id])
                    is_fall = (recent_transition and fall_by_movement) and (track_id not in self.fall_confirmed_ids)

                    if is_fall:
                        fall_detected_this_frame = True
                        self.fall_confirmed_ids.add(track_id)

                        if track_id not in self.fall_persist_start_time:
                            self.fall_persist_start_time[track_id] = time.time()

                        if track_id not in self.fall_ids_logged:
                            self.fall_ids_logged.add(track_id)
                            threading.Thread(target=self.log_fall, args=(track_id,), daemon=True).start()

                        if track_id not in self.recording_fall:
                            writer = self.create_fall_writer(track_id)
                            for f in self.frame_buffer:
                                writer.write(f)
                            self.recording_fall[track_id] = (time.time(), writer)

                        color = (0, 0, 255)
                        label = "fall"
                    elif cls_label == "person":
                        label = "person"
                        color = (0, 255, 0)
                    else:
                        label = "not fall"
                        color = (0, 125, 255)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} ID:{track_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if kps_xy is not None:
                        self.draw_keypoints(frame, kps_xy[i], kps_conf[i])

                for tid in self.fall_confirmed_ids:
                    if tid in self.fall_persist_start_time and tid not in self.persistent_alerted_ids:
                        duration = time.time() - self.fall_persist_start_time[tid]
                        if duration >= self.FALL_PERSISTENCE_SECONDS:
                            print(f"[ALERT] Người ID {tid} đã ngã {duration:.1f}s mà chưa đứng dậy.")
                            self.persistent_alerted_ids.add(tid)
                            threading.Thread(target=self.play_alert_sound, daemon=True).start()

                ended_ids = []
                for tid, (start_t, writer) in self.recording_fall.items():
                    writer.write(frame)
                    if time.time() - start_t >= self.FALL_RECORD_SECONDS:
                        writer.release()
                        ended_ids.append(tid)
                for tid in ended_ids:
                    self.recording_fall.pop(tid)

            except Exception as e:
                print(f"Lỗi tại frame {self.frame_id}: {e}")

            fps_disp = 1 / (time.time() - start_time + 1e-8)
            cv2.putText(frame, f"FPS: {fps_disp:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Fall Detection", frame)
            self.out.write(frame)
            self.frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.out.release()
        for _, writer in self.recording_fall.values():
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FallDetector(
        video_path="FallClipsTest/thangbe.mp4",
        model_path="yolov8-pose-finetune.pt",
        alert_sound="alert.mp3"
    )
    detector.run()
