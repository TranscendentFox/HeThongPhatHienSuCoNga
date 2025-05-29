import cv2
import os

# Đường dẫn thư mục chứa video cũ
INPUT_FOLDER = "static/history"
# Tạo thư mục lưu video mới (nếu muốn riêng)
OUTPUT_FOLDER = INPUT_FOLDER  # có thể để riêng như "static/converted"

# Tạo thư mục nếu chưa có
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Duyệt tất cả file trong thư mục
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".mp4") and "_avc1" not in filename:
        input_path = os.path.join(INPUT_FOLDER, filename)

        # Mở video bằng OpenCV
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Không mở được: {filename}")
            continue

        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tên video mới
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Ghi video mới với codec avc1 (H.264)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"🔄 Đang chuyển: {filename} -> {output_filename}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        print(f"✅ Đã lưu: {output_filename}")
