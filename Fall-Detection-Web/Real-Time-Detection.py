import cv2
import os
import json
import time
import datetime
import threading
import multiprocessing
from collections import deque
from flask import Flask, render_template, Response, request, jsonify
from Email import send_email_alert
from twilio_alert import send_sms_alert
import queue

app = Flask(__name__)

from ultralytics import YOLO

model = YOLO("E:/FallDetection/FallDetection/yolov8-pose-finetune.pt")

# Global config
camera_source = "0"
enable_detection = False
enable_email = False
enable_sms = False
confidence = 0.5
fall_detected = False
cap = None
frame_buffer = deque(maxlen=100)
ALERTS_FILE = "alerts.json"
CONTACTS_FILE = "contacts.json"

# === TÍCH HỢP LOGIC TỪ FallDetector ===
# Cấu hình từ FallDetector
KEYPOINT_THRESHOLD = 0.5
MIN_VALID_KEYPOINTS = 8
FALL_CONFIRM_FRAMES = 5
RELATIVE_DROP_THRESHOLD = 0.01
LABEL_HISTORY_FRAMES = 20
FALL_PERSISTENCE_SECONDS = 3
MAX_DURATION = 10

# Biến theo dõi trạng thái từ FallDetector
prev_centers = {}
last_seen_frame = {}
fall_signs = {}
label_history = {}
track_start_frame = {}
fall_confirmed_ids = set()
fall_persist_start_time = {}
persistent_alerted_ids = set()
fall_ids_logged = set()
recording_fall = {}
frame_id = 0

# === THÊM PHẦN XỬ LÝ VIDEO RECORDING TỐI ƯU ===
video_write_queue = queue.Queue()
video_writers = {}
recording_threads = {}


class VideoRecorderThread(threading.Thread):
    """Thread riêng để xử lý việc ghi video mà không chặn luồng chính"""

    def __init__(self, track_id, output_path, fps=20, frame_size=(640, 480)):
        super().__init__(daemon=True)
        self.track_id = track_id
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.frame_queue = queue.Queue(maxsize=100)  # Buffer cho frames
        self.recording = True
        self.start_time = time.time()
        self.max_duration = MAX_DURATION  # Ghi tối đa 10 giây

    def run(self):
        """Chạy thread ghi video"""
        try:
            # Khởi tạo VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

            if not self.writer.isOpened():
                print(f"Không thể tạo VideoWriter cho track_id {self.track_id}")
                return

            print(f"Bắt đầu ghi video cho track_id {self.track_id}: {self.output_path}")

            frames_written = 0
            while self.recording and (time.time() - self.start_time) < self.max_duration:
                try:
                    # Lấy frame từ queue với timeout
                    frame = self.frame_queue.get(timeout=1.0)
                    if frame is None:  # Signal để dừng
                        break

                    # Resize frame và ghi
                    resized_frame = cv2.resize(frame, self.frame_size)
                    self.writer.write(resized_frame)
                    frames_written += 1

                    self.frame_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Lỗi khi ghi frame cho track_id {self.track_id}: {e}")
                    break

            print(f"Kết thúc ghi video cho track_id {self.track_id}. Đã ghi {frames_written} frames")

        except Exception as e:
            print(f"Lỗi trong VideoRecorderThread cho track_id {self.track_id}: {e}")
        finally:
            if self.writer:
                self.writer.release()
            self.cleanup()

    def add_frame(self, frame):
        """Thêm frame vào queue để ghi"""
        if self.recording and not self.frame_queue.full():
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                # Nếu queue đầy, bỏ qua frame cũ nhất
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Empty:
                    pass

    def stop_recording(self):
        """Dừng ghi video"""
        self.recording = False
        try:
            self.frame_queue.put_nowait(None)  # Signal để dừng
        except queue.Full:
            pass

    def cleanup(self):
        """Dọn dẹp resources"""
        global recording_threads, video_writers
        if self.track_id in recording_threads:
            recording_threads.pop(self.track_id, None)
        if self.track_id in video_writers:
            video_writers.pop(self.track_id, None)


def start_fall_recording(track_id, frame_buffer_list):
    """Bắt đầu ghi video khi phát hiện ngã"""
    global recording_threads, video_writers

    # Tạo tên file với timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "static/backup"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fall_clip_{track_id}_{timestamp}.mp4")

    # Tạo thread ghi video
    recorder = VideoRecorderThread(track_id, output_path)
    recording_threads[track_id] = recorder
    video_writers[track_id] = output_path

    # Bắt đầu thread
    recorder.start()

    # Ghi các frame trong buffer trước đó
    for buffered_frame in frame_buffer_list:
        recorder.add_frame(buffered_frame)

    print(f"Đã bắt đầu ghi video cho track_id {track_id}")
    return output_path


def add_frame_to_recording(track_id, frame):
    """Thêm frame vào video đang ghi"""
    if track_id in recording_threads:
        recording_threads[track_id].add_frame(frame)


def stop_fall_recording(track_id):
    """Dừng ghi video cho track_id"""
    if track_id in recording_threads:
        recording_threads[track_id].stop_recording()
        print(f"Đã dừng ghi video cho track_id {track_id}")


def cleanup_old_recordings():
    """Dọn dẹp các recording cũ đã kết thúc"""
    global recording_threads
    finished_ids = []
    for track_id, recorder in recording_threads.items():
        if not recorder.is_alive():
            finished_ids.append(track_id)

    for track_id in finished_ids:
        recording_threads.pop(track_id, None)


def check_object_missing():
    """Kiểm tra và xóa các object đã mất khỏi tracking"""
    global frame_id
    for tid in list(last_seen_frame.keys()):
        if frame_id - last_seen_frame[tid] > 30:
            # Dừng recording nếu object biến mất
            stop_fall_recording(tid)

            prev_centers.pop(tid, None)
            last_seen_frame.pop(tid, None)
            fall_signs.pop(tid, None)
            label_history.pop(tid, None)
            track_start_frame.pop(tid, None)
            fall_confirmed_ids.discard(tid)
            fall_persist_start_time.pop(tid, None)
            persistent_alerted_ids.discard(tid)


def draw_keypoints(frame, kps_xy, kps_conf):
    """Vẽ keypoints lên frame"""
    for (x, y), conf in zip(kps_xy, kps_conf):
        if conf >= KEYPOINT_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


def log_fall(track_id):
    """Ghi log khi phát hiện ngã"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("fall_log.txt", "a") as f:
        f.write(f"[{now}] Fall Detect - ID: {track_id}\n")


def create_alert_video_async(detected_ids_to_save):
    """Tạo video cảnh báo trong thread riêng để không chặn luồng chính"""

    def create_video():
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "static/history"
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(output_dir, f"fall_alert_{timestamp}.mp4")

            # Tạo video từ buffer
            fps = 20
            frame_size = (640, 480)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

            if out.isOpened():
                # Ghi frames từ buffer
                buffer_list = list(frame_buffer)
                for f in buffer_list:
                    resized_frame = cv2.resize(f, frame_size)
                    out.write(resized_frame)

                out.release()

                # Lưu cảnh báo
                max_conf = max([c for _, c in detected_ids_to_save])
                save_alert(video_path, max_conf)
                send_alerts(video_path, max_conf)
                print("Đã gửi rồi")
                print(f"Đã tạo video cảnh báo: {video_path}")
            else:
                print("Không thể tạo VideoWriter cho video cảnh báo")

        except Exception as e:
            print(f"Lỗi khi tạo video cảnh báo: {e}")

    # Chạy trong thread riêng
    threading.Thread(target=create_video, daemon=True).start()


def process_predictions(frame):
    """
    Hàm xử lý dự đoán với logic phát hiện ngã tích hợp từ FallDetector
    """
    global fall_detected, cap, frame_id, prev_centers, last_seen_frame, \
        fall_signs, label_history, track_start_frame, fall_confirmed_ids, \
        fall_persist_start_time, persistent_alerted_ids, fall_ids_logged, \
        recording_fall, confidence, video_writers

    if not enable_detection:
        return frame

    try:
        # Tracking với YOLO
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640)
        result = results[0] if results else None

        if not result or not result.boxes or result.boxes.xyxy is None:
            frame_id += 1
            # Vẫn thêm frame vào các recording đang hoạt động
            for track_id in list(recording_threads.keys()):
                add_frame_to_recording(track_id, frame)
            return frame

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []
        cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        kps_xy = result.keypoints.xy.cpu().numpy() if result.keypoints else None
        kps_conf = result.keypoints.conf.cpu().numpy() if result.keypoints and result.keypoints.conf is not None else None
        class_names = result.names

        # Kiểm tra object bị mất
        check_object_missing()

        # Dọn dẹp recordings cũ
        cleanup_old_recordings()

        fall_detected_this_frame = False
        detected_ids_to_save = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            track_id = ids[i] if i < len(ids) else None
            cls_label = class_names[cls[i]] if cls[i] in class_names else "unknown"

            if track_id is None:
                continue

            # Khởi tạo tracking cho ID mới
            if track_id not in track_start_frame:
                track_start_frame[track_id] = frame_id

            # Bỏ qua frame đầu để ổn định tracking
            if frame_id - track_start_frame[track_id] < LABEL_HISTORY_FRAMES:
                continue

            # Lưu lịch sử nhãn
            if track_id not in label_history:
                label_history[track_id] = deque(maxlen=LABEL_HISTORY_FRAMES)
            label_history[track_id].append(cls_label.lower())
            label_seq = list(label_history[track_id])

            # Kiểm tra chuyển đổi từ "person" sang "fall"
            recent_transition = False
            if len(label_seq) == LABEL_HISTORY_FRAMES and 5 <= label_seq.count("person") <= 15:
                try:
                    fall_start = label_seq.index("fall")
                    if (all(label == "person" for label in label_seq[:fall_start]) and
                            all(label == "fall" for label in label_seq[fall_start:])):
                        recent_transition = True
                except ValueError:
                    pass

            # Kiểm tra keypoints hợp lệ
            valid_keypoints = True
            if kps_conf is not None and i < len(kps_conf):
                valid_kps = sum(1 for c in kps_conf[i] if c >= KEYPOINT_THRESHOLD)
                if valid_kps < MIN_VALID_KEYPOINTS:
                    valid_keypoints = False

            if not valid_keypoints:
                continue

            # Tính toán chuyển động
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            prev_center = prev_centers.get(track_id, (x_center, y_center))
            rel_delta_y = (y_center - prev_center[1]) / (h + 1e-6)

            prev_centers[track_id] = (x_center, y_center)
            last_seen_frame[track_id] = frame_id

            # Theo dõi dấu hiệu ngã
            if track_id not in fall_signs:
                fall_signs[track_id] = []
            fall_signs[track_id].append(rel_delta_y)
            if len(fall_signs[track_id]) > FALL_CONFIRM_FRAMES:
                fall_signs[track_id].pop(0)

            # Kiểm tra ngã dựa trên chuyển động
            fall_by_movement = (len(fall_signs[track_id]) == FALL_CONFIRM_FRAMES and
                                all(v > RELATIVE_DROP_THRESHOLD for v in fall_signs[track_id]))

            # Xác định ngã
            is_fall = (recent_transition and fall_by_movement and
                       track_id not in fall_confirmed_ids)

            if is_fall:
                fall_detected_this_frame = True
                fall_confirmed_ids.add(track_id)
                detected_ids_to_save.append((track_id, confidence))  # confidence giả định

                # Bắt đầu theo dõi thời gian ngã
                if track_id not in fall_persist_start_time:
                    fall_persist_start_time[track_id] = time.time()

                # Ghi log
                if track_id not in fall_ids_logged:
                    fall_ids_logged.add(track_id)
                    threading.Thread(target=log_fall, args=(track_id,), daemon=True).start()

                # Bắt đầu ghi video riêng cho track_id này
                if track_id not in recording_threads:
                    buffer_list = list(frame_buffer)
                    start_fall_recording(track_id, buffer_list)

                color = (0, 0, 255)  # Đỏ cho ngã
                label_text = "Fall"
            elif cls_label == "person":
                color = (0, 255, 0)  # Xanh cho bình thường
                label_text = "Person"
            else:
                color = (0, 125, 255)  # Vàng cho không ngã
                label_text = "Not fall"

            # Vẽ bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label_text} ID:{track_id}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Vẽ keypoints
            if kps_xy is not None and i < len(kps_xy):
                draw_keypoints(frame, kps_xy[i], kps_conf[i])

        # Thêm frame hiện tại vào tất cả recordings đang hoạt động
        for track_id in list(recording_threads.keys()):
            add_frame_to_recording(track_id, frame)

        # Kiểm tra cảnh báo liên tục
        current_time = time.time()
        for tid in fall_confirmed_ids.copy():
            if tid in fall_persist_start_time and tid not in persistent_alerted_ids:
                duration = current_time - fall_persist_start_time[tid]
                if duration >= FALL_PERSISTENCE_SECONDS:
                    print(f"[ALERT] Người ID {tid} đã ngã {duration:.1f}s mà chưa đứng dậy.")
                    persistent_alerted_ids.add(tid)

        # Dừng recording sau thời gian nhất định
        for tid in list(recording_threads.keys()):
            if tid in fall_persist_start_time:
                duration = current_time - fall_persist_start_time[tid]
                if duration >= MAX_DURATION:  # Dừng sau 10 giây
                    stop_fall_recording(tid)


        # Xử lý cảnh báo và lưu file (async)
        if detected_ids_to_save:
            fall_detected = True
            create_alert_video_async(detected_ids_to_save)

    except Exception as e:
        print(f"Lỗi trong process_predictions: {e}")

    frame_id += 1
    return frame


def load_contacts():
    if not os.path.exists(CONTACTS_FILE):
        return {"emails": [], "phones": []}
    with open(CONTACTS_FILE, "r") as f:
        return json.load(f)


def save_contacts(data):
    with open(CONTACTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_alert(video_path, conf):
    data = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video": video_path,
        "confidence": round(conf, 2)
    }
    alerts = []
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
    alerts.insert(0, data)
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2)


def send_alerts(video_path, conf):
    if enable_email:
        multiprocessing.Process(target=send_email_alert, kwargs={
            "label": "Fall Detected!",
            "confidence_score": conf,
            "attachment_paths": [video_path]
        }).start()
    if enable_sms:
        multiprocessing.Process(target=send_sms_alert).start()


def generate_frames():
    global cap
    while True:
        if cap is None or not cap.isOpened():
            update_camera_source(camera_source)
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            update_camera_source(camera_source)
            time.sleep(1)
            continue

        frame_buffer.append(frame.copy())

        if enable_detection:
            try:
                frame = process_predictions(frame)
            except Exception as e:
                print(f"Lỗi khi detect: {e}")

        try:

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Lỗi encode frame: {e}")
            break


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Đọc lịch sử từ file lịch sử
def load_history():
    try:
        with open('alerts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Lưu lịch sử vào file lịch sử
def save_history(history):
    with open('alerts.json', 'w') as f:
        json.dump(history, f, indent=4)


# Route để lấy lịch sử
@app.route("/get_history", methods=["GET"])
def get_history():
    history = load_history()
    return jsonify({"history": history})


# Route để xoá lịch sử ngã
@app.route("/delete_history", methods=["POST"])
def delete_history():
    data = request.get_json()
    timestamp = data.get("timestamp")

    if not timestamp:
        return jsonify({"error": "Không tìm thấy timestamp."}), 400

    history = load_history()

    # Lọc các sự kiện không phải là sự kiện cần xoá
    new_history = [event for event in history if event["time"] != timestamp]

    save_history(new_history)

    return jsonify({"message": "✅ Xoá lịch sử thành công."})


# Route để tìm kiếm lịch sử theo video
@app.route("/search_by_video", methods=["GET"])
def search_by_video():
    video_name = request.args.get("video_name")

    if not video_name:
        return jsonify({"error": "Không tìm thấy tên video."}), 400

    history = load_history()

    result = [event for event in history if video_name.lower() in event["video"].lower()]

    if not result:
        return jsonify({"message": "Không tìm thấy video trong lịch sử cảnh báo."})

    return jsonify({"history": result})


# Route để tìm kiếm lịch sử theo thời gian
@app.route("/search_by_time", methods=["GET"])
def search_by_time():
    start_time = request.args.get("start_time")
    end_time = request.args.get("end_time")

    if not start_time or not end_time:
        return jsonify({"error": "Thiếu thông tin về thời gian bắt đầu hoặc kết thúc."}), 400

    try:
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return jsonify({"error": "Định dạng thời gian không hợp lệ."}), 400

    history = load_history()

    result = []
    for event in history:
        event_time = datetime.datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S")
        if start_time <= event_time <= end_time:
            result.append(event)

    if not result:
        return jsonify({"message": "Không có cảnh báo trong khoảng thời gian này."})

    return jsonify({"history": result})


# Route để thêm sự kiện ngã (chỉ là ví dụ, thường sẽ tự động từ hệ thống)
@app.route("/add_history", methods=["POST"])
def add_history():
    data = request.get_json()
    event = {
        "time": data["time"],
        "video": data["video"],
        "confidence": data["confidence"],
        "message": data["message"]
    }

    history = load_history()
    history.append(event)
    save_history(history)

    return jsonify({"message": "✅ Lịch sử ngã đã được thêm."})


@app.route("/save_config", methods=["POST"])
def save_config():
    global camera_source, enable_detection, enable_email, enable_sms, confidence
    data = request.get_json()
    camera_source = data.get("camera_source", "0")
    enable_detection = data.get("enable_detection", True)
    enable_email = data.get("enable_email", True)
    enable_sms = data.get("enable_sms", True)
    confidence = float(data.get("conf", 0.8))

    print(enable_detection)
    print(enable_email)

    # Cập nhật camera
    update_camera_source(camera_source)

    return jsonify({"message": "Cấu hình đã được lưu."})


# Đọc danh sách
@app.route("/get_contacts")
def get_contacts():
    if not os.path.exists(CONTACTS_FILE):
        return {"emails": [], "phones": []}
    with open(CONTACTS_FILE, "r") as f:
        return json.load(f)


# Ghi danh sách
def save_contacts(data):
    with open(CONTACTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# -------------------- EMAIL --------------------

@app.route("/get_emails")
def get_emails():
    contacts = load_contacts()
    return jsonify(contacts.get("emails", []))


@app.route("/add_email", methods=["POST"])
def add_email():
    data = request.get_json()
    email = data.get("email", "").strip()
    if not email:
        return jsonify({"error": "Email rỗng"}), 400

    contacts = load_contacts()
    if email not in contacts["emails"]:
        contacts["emails"].append(email)
        save_contacts(contacts)
        return jsonify({"message": "Đã thêm email"}), 200
    return jsonify({"message": "Email đã tồn tại"}), 200


@app.route("/delete_email", methods=["POST"])
def delete_email():
    data = request.get_json()
    email = data.get("email", "").strip()
    contacts = load_contacts()
    if email in contacts["emails"]:
        contacts["emails"].remove(email)
        save_contacts(contacts)
        return jsonify({"message": f"Đã xoá email {email}"}), 200
    return jsonify({"message": "Email không tồn tại"}), 400


# -------------------- PHONE --------------------

@app.route("/get_phones")
def get_phones():
    contacts = load_contacts()
    return jsonify(contacts.get("phones", []))


@app.route("/add_phone", methods=["POST"])
def add_phone():
    data = request.get_json()
    phone = data.get("phone", "").strip()
    if not phone:
        return jsonify({"error": "Số điện thoại rỗng"}), 400

    contacts = load_contacts()
    if phone not in contacts["phones"]:
        contacts["phones"].append(phone)
        save_contacts(contacts)
        return jsonify({"message": "Đã thêm số điện thoại"}), 200
    return jsonify({"message": "Số điện thoại đã tồn tại"}), 200


@app.route("/delete_phone", methods=["POST"])
def delete_phone():
    data = request.get_json()
    phone = data.get("phone", "").strip()
    contacts = load_contacts()
    if phone in contacts["phones"]:
        contacts["phones"].remove(phone)
        save_contacts(contacts)
        return jsonify({"message": "Đã xoá số điện thoại"}), 200
    return jsonify({"message": "Số điện thoại không tồn tại"}), 400


@app.route("/delete_alert", methods=["POST"])
def delete_alert():
    data = request.get_json()
    time = data.get("time", None)

    if not time:
        return jsonify({"message": "Không có thời gian video để xoá."}), 400

    if not os.path.exists(ALERTS_FILE):
        return jsonify({"message": "Không có dữ liệu."})

    with open(ALERTS_FILE, "r") as f:
        alerts = json.load(f)

    print("Đang tìm")

    # Tìm video theo thời gian
    alert_to_delete = next((alert for alert in alerts if alert['time'] == time), None)

    if alert_to_delete:
        video_path = alert_to_delete.get("video")
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Đã xoá video tại: {video_path}")
            except Exception as e:
                return jsonify({"message": f"Không thể xoá video: {str(e)}"}), 500
        else:
            return jsonify({"message": "Video không tồn tại."}), 400

        alerts.remove(alert_to_delete)  # Xóa phần tử khỏi danh sách
        with open(ALERTS_FILE, "w") as f:
            json.dump(alerts, f, indent=2)

        return jsonify({"message": "Đã xoá cảnh báo."})
    else:
        return jsonify({"message": "Cảnh báo không hợp lệ."}), 400


@app.route("/history")
def view_history():
    return render_template("history.html")


@app.route("/get_alerts")
def get_alerts():
    min_conf = float(request.args.get("min_conf", 0.0))
    date_filter = request.args.get("date", None)
    if not os.path.exists(ALERTS_FILE):
        return jsonify([])
    with open(ALERTS_FILE, "r") as f:
        alerts = json.load(f)

    # 🔥 Sửa lỗi đường dẫn: chuyển "\" thành "/"
    for alert in alerts:
        alert["video"] = alert["video"].replace("\\", "/")

    result = []
    for alert in alerts:
        if alert["confidence"] < min_conf:
            continue
        if date_filter and not alert["time"].startswith(date_filter):
            continue
        result.append(alert)

    return jsonify(result)


CONFIG_PATH = 'config.json'


# Hàm load và save config
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


@app.route("/update_settings", methods=["POST"])
def update_settings():
    global enable_detection, enable_email, enable_sms, confidence
    data = request.get_json()
    print(request)
    key = data.get("key")
    value = data.get("value")

    config = load_config()

    if key not in config:
        return jsonify({"error": "Trường cấu hình không hợp lệ."}), 400

    # Kiểm tra kiểu và giá trị
    try:
        if key == "conf":
            value = float(value)
            if not 0 <= value <= 1:
                return jsonify({"error": "Ngưỡng confidence phải từ 0 đến 1."}), 400
            confidence = value
        elif key == "enable_detection":
            enable_detection = value
        elif key == "enable_email":
            enable_email = value
        elif key == "enable_sms":
            enable_sms = value
        elif key == "camera_source":
            camera_config = load_camera_config()
            if value in camera_config:
                value = value
            else:
                value = '0'

            check = update_camera_source(value)
            if check is False:
                return jsonify({"error": "Không khởi tạo được camera"}), 400
    except:
        return jsonify({"error": "Giá trị không hợp lệ."}), 400

    config[key] = value
    save_config(config)

    # Tạo phản hồi chi tiết
    readable = {
        "enable_detection": "phát hiện ngã",
        "enable_email": "gửi email",
        "enable_sms": "gửi SMS",
        "camera_source": "nguồn camera",
        "conf": "ngưỡng phát hiện"
    }

    print("📝 Config đã cập nhật:", config)
    if key in ["enable_detection", "enable_email", "enable_sms"]:
        status_text = "bật" if value else "tắt"
        message = f"Đã {status_text} {readable[key]}."
    elif key in ["conf"]:
        message = f"Chuyển thành công {readable[key]} thành {value}."
    elif key in ["camera_source"]:
        name_camera = camera_config[value]["name"]
        message = f"Chuyển thành công {readable[key]} thành {name_camera}."
    else:
        message = "Cập nhật thành công."

    return jsonify({"message": message}), 200


# Đọc cấu hình camera từ file JSON
def load_camera_config():
    with open('camera_config.json', 'r') as f:
        return json.load(f)


# Route để lấy thông tin cấu hình camera
@app.route("/get_camera_config", methods=["GET"])
def get_camera_config():
    camera_config = load_camera_config()
    return jsonify(camera_config)


# Route để lấy thông tin cấu hình camera (tên camera)
@app.route("/get_camera_name", methods=["GET"])
def get_camera_name():
    # Lấy thông tin camera từ config và camera_config
    config = load_config()
    camera_config = load_camera_config()

    camera_source = str(config["camera_source"])

    # Kiểm tra camera trong camera_config
    if camera_source in camera_config:
        camera_name = camera_config[camera_source]["name"]
        return jsonify({"camera_name": camera_name})

    return jsonify({"error": "Camera không hợp lệ."}), 400


def update_camera_source(camera_index):
    global cap
    if cap is not None:
        cap.release()  # Giải phóng camera hiện tại nếu có
    try:
        # Chuyển đổi camera_index thành kiểu phù hợp
        if camera_index.isdigit():
            cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print(f"Không thể mở camera với nguồn {camera_index}")
            cap = None
            return False
        else:
            print(f"Camera đã được khởi tạo với nguồn {camera_index}")
            config = load_config()
            config["camera_source"] = camera_index  # Sửa lỗi: lưu đúng giá trị camera_index
            save_config(config)
            return True
    except Exception as e:
        print(f"Lỗi khi khởi tạo camera {camera_index}: {str(e)}")
        cap = None
        return False


# Route để lấy trạng thái recording hiện tại
@app.route("/get_recording_status", methods=["GET"])
def get_recording_status():
    """Lấy thông tin về các recording đang hoạt động"""
    status = {}
    for track_id, recorder in recording_threads.items():
        if recorder.is_alive():
            duration = time.time() - recorder.start_time
            status[track_id] = {
                "recording": True,
                "duration": round(duration, 1),
                "output_path": video_writers.get(track_id, "")
            }
    return jsonify(status)


# Route để dừng recording thủ công
@app.route("/stop_recording", methods=["POST"])
def stop_recording_manual():
    """Dừng recording cho một track_id cụ thể"""
    data = request.get_json()
    track_id = data.get("track_id")

    if track_id is None:
        return jsonify({"error": "Thiếu track_id"}), 400

    track_id = int(track_id)
    if track_id in recording_threads:
        stop_fall_recording(track_id)
        return jsonify({"message": f"Đã dừng recording cho track_id {track_id}"}), 200
    else:
        return jsonify({"error": f"Không tìm thấy recording cho track_id {track_id}"}), 404


# Route để lấy thống kê hệ thống
@app.route("/get_system_stats", methods=["GET"])
def get_system_stats():
    """Lấy thống kê về hệ thống"""
    stats = {
        "active_recordings": len(recording_threads),
        "frame_buffer_size": len(frame_buffer),
        "fall_confirmed_ids": len(fall_confirmed_ids),
        "total_tracked_objects": len(prev_centers),
        "current_frame_id": frame_id
    }
    return jsonify(stats)


if __name__ == "__main__":
    # Khởi tạo camera ban đầu
    # update_camera_source(camera_source)
    app.run(debug=True)