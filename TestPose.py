import cv2
import time
import os
from ultralytics import YOLO

# Load model YOLO-Pose
model = YOLO("yolov8-pose-finetune.pt")
names = model.names  # names[0] = 'person', names[1] = 'fall' (nếu có)

# Đường dẫn video hoặc ảnh (có thể thay bằng webcam: 0)
video_path = "FallClipsTest/173.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông số ảnh đầu vào
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video output
out = cv2.VideoWriter("output_pose.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Folder để lưu file .txt annotation
output_txt_folder = "pose_txt"
os.makedirs(output_txt_folder, exist_ok=True)

frame_id = 0  # Đánh số frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    results = model(frame)

    height_img, width_img = frame.shape[:2]
    txt_lines = []

    for result in results:
        # Bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes else []

        kps_xy = result.keypoints.xy.cpu().numpy() if result.keypoints else None

        # Kiểm tra có attribute 'conf' không trước khi truy cập
        if result.keypoints and hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None:
            kps_conf = result.keypoints.conf.cpu().numpy()
        else:
            kps_conf = None

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cls_id = classes[i]

            # Bounding box center và size (chuẩn hóa)
            x_center = (x1 + x2) / 2 / width_img
            y_center = (y1 + y2) / 2 / height_img
            w = (x2 - x1) / width_img
            h = (y2 - y1) / height_img

            # Lấy keypoint + visibility
            keypoints = []
            if kps_xy is not None:
                if kps_conf is not None:
                    for (x, y), c in zip(kps_xy[i], kps_conf[i]):
                        x_n = x / width_img
                        y_n = y / height_img
                        v = 2 if c >= 0.5 else 0
                        keypoints.extend([x_n, y_n, v])
                else:
                    for (x, y) in kps_xy[i]:
                        x_n = x / width_img
                        y_n = y / height_img
                        v = 2  # hoặc 1
                        keypoints.extend([x_n, y_n, v])
            else:
                keypoints = [0, 0, 0] * 17

            # Tạo dòng annotation
            line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} " + \
                   " ".join([f"{kp:.6f}" if isinstance(kp, float) else str(kp) for kp in keypoints])
            txt_lines.append(line)

            # Vẽ bbox và label
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            label = names[cls_id] if cls_id < len(names) else f"id:{cls_id}"
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1i, y1i - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Vẽ keypoints
            if kps_xy is not None:
                for (x, y), c in zip(kps_xy[i], kps_conf[i]):
                    if c >= 0.5:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Lưu annotation ra file txt
    txt_path = os.path.join(output_txt_folder, f"{frame_id:06d}.txt")
    with open(txt_path, "w") as f:
        for line in txt_lines:
            f.write(line + "\n")

    # Hiển thị FPS
    end_time = time.time()
    fps_disp = 1 / (end_time - start_time + 1e-8)
    cv2.putText(frame, f"FPS: {fps_disp:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Hiển thị và lưu frame
    cv2.imshow("YOLOv8 Pose Detection", frame)
    out.write(frame)

    frame_id += 1

    # start_time = cv2.getTickCount()
    # while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 20000:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
