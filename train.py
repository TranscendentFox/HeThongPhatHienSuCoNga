import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR
from rlstm_model import KAMTFENet  # Giả sử bạn đã có đường dẫn đến model này

# Hàm tải dữ liệu gốc
def load_original_data(data_dir, expected_frames=30, expected_keypoints=17, expected_coords=2):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục {data_dir} không tồn tại")

    window_files = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
    if not window_files:
        raise FileNotFoundError(f"Không tìm thấy file .npz nào trong {data_dir}")

    tensors = []
    labels = []

    for window_path in window_files:
        window_file = os.path.basename(window_path)
        try:
            data = np.load(window_path)
            tensor = data['data']  # [30, 17, 2]
            label = data['label']  # "fall" hoặc "no_fall"

            if tensor.shape != (expected_frames, expected_keypoints, expected_coords):
                print(f"File {window_file} có kích thước không đúng: {tensor.shape}, bỏ qua")
                continue

            if label == "fall":
                label_num = 1
            elif label == "no_fall":
                label_num = 0
            else:
                print(f"File {window_file} có nhãn không hợp lệ: {label}, bỏ qua")
                continue

            tensor = tensor.transpose(1, 0, 2)  # [17, 30, 2]
            tensors.append(tensor)
            labels.append(label_num)

        except Exception as e:
            print(f"Lỗi khi xử lý {window_file}: {str(e)}")

    if not tensors:
        raise ValueError("Không có dữ liệu hợp lệ nào được tải")

    tensors = torch.tensor(np.stack(tensors), dtype=torch.float32)  # [num_samples, 17, 30, 2]
    labels = torch.tensor(labels, dtype=torch.long)  # [num_samples]

    return tensors, labels

# Hàm huấn luyện
def train(data_dir, model_save_path="best_model.pth", batch_size=64, num_epochs=100, 
          learning_rate=0.001, train_split=0.7, expected_frames=30, expected_keypoints=17, expected_coords=2):
    """
    Huấn luyện mô hình KAMTFENet trên dữ liệu keypoints.

    Args:
        data_dir (str): Đường dẫn đến thư mục chứa file .npz.
        model_save_path (str): Đường dẫn lưu mô hình tốt nhất.
        batch_size (int): Kích thước batch.
        num_epochs (int): Số epoch huấn luyện.
        learning_rate (float): Tốc độ học ban đầu.
        train_split (float): Tỷ lệ chia tập huấn luyện (0-1).
        expected_frames (int): Số khung hình mong đợi (mặc định 30).
        expected_keypoints (int): Số keypoints mong đợi (mặc định 17).
        expected_coords (int): Số tọa độ mỗi keypoint (mặc định 2).
    """
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tải dữ liệu
    try:
        tensors, labels = load_original_data(data_dir, expected_frames, expected_keypoints, expected_coords)
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return

    # Chuyển dữ liệu sang định dạng phù hợp với KAMTFENet: (num_samples, 30, 17, 2)
    tensors = tensors.permute(0, 2, 1, 3)  # Từ (num_samples, 17, 30, 2) sang (num_samples, 30, 17, 2)
    print(f"Loaded {tensors.size(0)} samples with shape {tensors.shape}")

    # Tạo dataset
    dataset = TensorDataset(tensors, labels)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Khởi tạo mô hình
    model = KAMTFENet(num_keypoints=expected_keypoints, seq_len=expected_frames, 
                      input_size=expected_keypoints * expected_coords, hidden_size=256, num_layers=3)
    model.to(device)

    # Hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # Giảm learning rate 0.5 sau mỗi 20 epoch

    # Lưu trữ mô hình tốt nhất
    best_acc = 0.0

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Xóa gradient
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)  # (batch_size, 2)
            loss = criterion(outputs, targets)

            # Backward
            loss.backward()
            optimizer.step()

            # Tính toán thống kê
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Đánh giá trên tập kiểm tra
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        test_loss /= test_total
        test_acc = test_correct / test_total

        # Cập nhật learning rate
        scheduler.step()

        # In kết quả
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Lưu mô hình tốt nhất
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with Test Acc: {best_acc:.4f}")

    print(f"Training completed. Best Test Acc: {best_acc:.4f}")

if __name__ == "__main__":
    data_dir = "data/processed/sliding_windows/ur_dataset"  # Thay bằng đường dẫn thực tế
    train(data_dir, model_save_path="best_kamtfenet.pth")

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import numpy as np
# import os
# from sklearn.metrics import accuracy_score
# import logging
# from src.modeling.rlstm_model import KAMTFENet  # Import từ file trước

# # Thiết lập logging
# logging.basicConfig(filename='output/logs/train.log', level=logging.INFO,
#                     format='%(asctime)s - %(message)s')

# class FallDataset(Dataset):
#     def __init__(self, data_dir):
#         """
#         Dataset cho dữ liệu tăng cường.
        
#         Args:
#             data_dir (str): Thư mục chứa các file .npz.
#         """
#         self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
#         self.data_dir = data_dir
#         self.label_map = {'no_fall': 0, 'fall': 1}
    
#     def __len__(self):
#         return len(self.data_files)
    
#     def __getitem__(self, idx):
#         file_path = os.path.join(self.data_dir, self.data_files[idx])
#         data = np.load(file_path)
#         tensor = data['data']  # [30, 17, 2]
#         label = self.label_map[data['label']]  # Chuyển thành số: 0 hoặc 1
        
#         return torch.tensor(tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# def train_model(data_dir, model_path, batch_size=32, num_epochs=300, lr=0.0005):
#     """
#     Huấn luyện mô hình KAMTFENet.
    
#     Args:
#         data_dir (str): Thư mục chứa dữ liệu tăng cường.
#         model_path (str): Đường dẫn lưu trọng số mô hình.
#         batch_size (int): Kích thước batch.
#         num_epochs (int): Số epoch huấn luyện.
#         lr (float): Learning rate.
#     """
#     # Thiết bị
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device: {device}")
    
#     # Tạo dataset và split train/test
#     dataset = FallDataset(data_dir)
#     train_size = int(0.7 * len(dataset))  # 70% train
#     test_size = len(dataset) - train_size  # 30% test
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Khởi tạo mô hình, loss và optimizer
#     model = KAMTFENet().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#     # Huấn luyện
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         for batch_tensors, batch_labels in train_loader:
#             batch_tensors, batch_labels = batch_tensors.to(device), batch_labels.to(device)
            
#             # Forward pass
#             outputs = model(batch_tensors)  # [batch_size, 2]
#             loss = criterion(outputs, batch_labels)
            
#             # Backward pass và optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         train_loss /= len(train_loader)
        
#         # Đánh giá trên tập test
#         model.eval()
#         all_preds = []
#         all_labels = []
#         with torch.no_grad():
#             for batch_tensors, batch_labels in test_loader:
#                 batch_tensors, batch_labels = batch_tensors.to(device), batch_labels.to(device)
#                 outputs = model(batch_tensors)
#                 _, preds = torch.max(outputs, 1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(batch_labels.cpu().numpy())
        
#         test_accuracy = accuracy_score(all_labels, all_preds)
        
#         # Ghi log
#         logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
#         print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
        
#         # Lưu mô hình sau mỗi 50 epoch
#         if (epoch + 1) % 50 == 0:
#             torch.save(model.state_dict(), f"{model_path}/model_epoch_{epoch+1}.pth")
#             logging.info(f"Saved model at epoch {epoch+1}")
    
#     # Lưu mô hình cuối cùng
#     torch.save(model.state_dict(), f"{model_path}/model_final.pth")
#     logging.info(f"Final model saved to {model_path}/model_final.pth")

# if __name__ == "__main__":
#     data_dir = "data/processed/enhanced_features/ur_dataset"
#     model_path = "models/rlstm_weights"
#     os.makedirs(model_path, exist_ok=True)
#     train_model(data_dir, model_path)