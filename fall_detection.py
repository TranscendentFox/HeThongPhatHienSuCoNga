from multi_predict import predict_multi_person
from ultralytics import YOLO
from ViTPose.mmpose.apis.inference import init_pose_model

import torch

# Đọc đường dẫn camera hoặc video từ file
with open("camera_source.txt", "r") as f:
    video_path = f.read().strip()

model_path = "best_kamtfenet_new.pth"
output_video_path = None
output_dir = "fall_clips"
user_id = "user_123"

yolo_model = YOLO("yolo12x.pt")
config_path = 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
checkpoint_path = 'models/vitpose/vitpose_weights/vitpose_base_coco_aic_mpii.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vitpose_model = init_pose_model(config_path, checkpoint_path, device=device)

predict_multi_person(video_path, model_path, yolo_model, vitpose_model,
                     window_size=30,
                     device=device,
                     output_root_dir=output_dir,
                     output_video_path=output_video_path,
                     user_id=user_id)
