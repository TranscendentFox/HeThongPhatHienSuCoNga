import torch
import torch.nn as nn

# Custom RLSTM Cell
class CustomRLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # Thêm tầng linear cho residual connection
        self.residual_linear = nn.Linear(input_size, hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)
        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        # Biến đổi x trước khi cộng
        x_transformed = self.residual_linear(x)  # (64, 34) -> (64, 256)
        h_t_prime = h_t + x_transformed  # (64, 256) + (64, 256)
        return h_t_prime, c_t

# RLSTM Network
class RLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(RLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Tạo các ô RLSTM cho mỗi lớp
        self.rlstm_cells = nn.ModuleList([
            CustomRLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        # Khởi tạo trạng thái ẩn
        if hidden is None:
            hidden = [
                (torch.zeros(batch_size, self.hidden_size).to(x.device),
                 torch.zeros(batch_size, self.hidden_size).to(x.device))
                for _ in range(self.num_layers)
            ]

        outputs = []
        current_input = x

        # Xử lý từng bước thời gian
        for t in range(seq_len):
            x_t = current_input[:, t, :]
            layer_hidden = hidden

            # Duyet qua tung lop
            for layer_idx in range(self.num_layers):
                h_t, c_t = self.rlstm_cells[layer_idx](x_t, layer_hidden[layer_idx])
                layer_hidden[layer_idx] = (h_t, c_t)
                x_t = h_t

            outputs.append(h_t)
            hidden = layer_hidden

        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden

# Classification Layers
class ClassificationLayers(nn.Module):
    def __init__(self, input_size=256, dropout_rate=0.3):
        super(ClassificationLayers, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.classifier(x)

class AdaptiveKeypointAttention(nn.Module):
    def __init__(self, num_keypoints=17, reduction_ratio=1):
        super(AdaptiveKeypointAttention, self).__init__()
        self.num_keypoints = num_keypoints
        self.batch_norm = nn.BatchNorm2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, num_keypoints * 2))
        self.max_pool = nn.AdaptiveMaxPool2d((1, num_keypoints * 2))
        self.fc1 = nn.Linear(num_keypoints * 2 * 2, num_keypoints // reduction_ratio)
        self.fc2 = nn.Linear(num_keypoints // reduction_ratio, num_keypoints)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, seq_len, num_keypoints, coords) = (64, 30, 17, 2)
        batch_size, seq_len, num_keypoints, coords = x.size()
        x_reshaped = x.view(batch_size, 1, seq_len, num_keypoints * coords)  # (64, 1, 30, 34)

        # Batch normalization
        x_norm = self.batch_norm(x_reshaped)  # (64, 1, 30, 34)

        # Global average and max pooling
        x_avg = self.avg_pool(x_norm).view(batch_size, num_keypoints * coords)  # (64, 34)
        x_max = self.max_pool(x_norm).view(batch_size, num_keypoints * coords)  # (64, 34)

        # Ghép avg và max
        x_concat = torch.cat([x_avg, x_max], dim=1)  # (64, 68)

        # Tính trọng số chú ý
        x_fc = torch.relu(self.fc1(x_concat))  # (64, 17)
        weights = self.sigmoid(self.fc2(x_fc)).view(batch_size, 1, num_keypoints, 1)  # (64, 1, 17, 1)

        # Áp dụng trọng số
        x_att = x * weights  # (64, 30, 17, 2) * (64, 1, 17, 1) -> (64, 30, 17, 2)
        return x_att

# Complete KAMTFENet Model
class KAMTFENet(nn.Module):
    def __init__(self, num_keypoints=17, seq_len=30, input_size=34, hidden_size=256, num_layers=3):
        super(KAMTFENet, self).__init__()
        self.num_keypoints = num_keypoints
        self.seq_len = seq_len

        # Adaptive keypoint attention module
        self.attention = AdaptiveKeypointAttention(num_keypoints=num_keypoints)

        # RLSTM
        self.rlstm = RLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        # Classification layers
        self.classifier = ClassificationLayers(input_size=hidden_size)

    def forward(self, x):
        # x: (batch_size, seq_len, num_keypoints, 2)
        batch_size, seq_len, num_keypoints, coords = x.size()
        assert seq_len == self.seq_len, f"Expected seq_len {self.seq_len}, got {seq_len}"
        assert num_keypoints == self.num_keypoints, f"Expected num_keypoints {self.num_keypoints}, got {num_keypoints}"

        # Apply attention
        x_att = self.attention(x)  # (batch_size, seq_len, num_keypoints, 2)
        x_att = x_att.view(batch_size, seq_len, num_keypoints * coords)  # (batch_size, seq_len, 34)

        # Pass through RLSTM
        rlstm_out, _ = self.rlstm(x_att)  # (batch_size, seq_len, hidden_size)
        rlstm_out = rlstm_out[:, -1, :]  # Take last time step: (batch_size, hidden_size)

        # Classify
        out = self.classifier(rlstm_out)  # (batch_size, 2)
        return out
    
    
# class KeypointAttentionModule(nn.Module):
#     def __init__(self, num_keypoints=17, reduction_ratio=1):
#         """
#         Khởi tạo Keypoint Attention Module (KAM).
        
#         Args:
#             num_keypoints (int): Số lượng điểm mấu chốt (mặc định là 17 theo Table 1).
#             reduction_ratio (int): Tỷ lệ giảm chiều trong tầng fully connected (mặc định là 1 theo bài báo).
#         """
#         super(KeypointAttentionModule, self).__init__()
#         self.num_keypoints = num_keypoints
#         self.reduction_ratio = reduction_ratio

#         # Batch normalization để chuẩn hóa dữ liệu đầu vào
#         self.batch_norm = nn.BatchNorm2d(num_keypoints)

#         # Tầng gộp trung bình và gộp cực đại (Avg/Max Pool - Pool2D)
#         # Kích thước kernel là (30, 2) để giảm chiều từ (30, 2) xuống (1, 1)
#         self.avg_pool2d = nn.AvgPool2d(kernel_size=(30, 2))  # 64 × 17 × 30 × 2 -> 64 × 17 × 1 × 1
#         self.max_pool2d = nn.MaxPool2d(kernel_size=(30, 2))  # 64 × 17 × 30 × 2 -> 64 × 17 × 1 × 1

#         # Tầng fully connected để dự đoán trọng số chú ý
#         # Attention FC1: 64 × 17 -> 64 × 17
#         self.fc1 = nn.Linear(num_keypoints, num_keypoints // reduction_ratio)
#         # Attention FC2: 64 × 17 -> 64 × 17
#         self.fc2 = nn.Linear(num_keypoints // reduction_ratio, num_keypoints)

#     def forward(self, x):
#         """
#         Xử lý dữ liệu qua Keypoint Attention Module.
        
#         Args:
#             x (torch.Tensor): Dữ liệu đầu vào với kích thước [batch_size, num_keypoints, num_frames, 2].
#                               Ví dụ: [64, 17, 30, 2].
        
#         Returns:
#             torch.Tensor: Dữ liệu sau khi tăng cường với cùng kích thước [batch_size, num_keypoints, num_frames, 2].
#         """
#         # Lưu kích thước ban đầu
#         batch_size, num_keypoints, num_frames, coords = x.size()  # [64, 17, 30, 2]

#         # Bước 1: Chuẩn hóa dữ liệu bằng BatchNorm
#         x_normalized = self.batch_norm(x)  # [64, 17, 30, 2]

#         # Bước 2: Gộp trung bình và gộp cực đại (Avg/Max Pool2D)
#         T_avg = self.avg_pool2d(x_normalized)  # [64, 17, 1, 1]
#         T_max = self.max_pool2d(x_normalized)  # [64, 17, 1, 1]

#         # Bước 3: Nối (concatenate) T_avg và T_max theo chiều thứ 3
#         # Loại bỏ các chiều 1x1 để nối
#         T_avg = T_avg.squeeze(-1).squeeze(-1)  # [64, 17]
#         T_max = T_max.squeeze(-1).squeeze(-1)  # [64, 17]
#         # Nối T_avg và T_max để tạo tensor [64, 17, 2]
#         concat = torch.stack([T_avg, T_max], dim=-1)  # [64, 17, 2]

#         # Bước 4: Global average pooling để hợp nhất đặc trưng
#         # Thay AvgPool1d bằng torch.mean trên chiều cuối
#         fused = torch.mean(concat, dim=-1)  # [64, 17]

#         # Bước 5: Đưa qua các tầng fully connected để dự đoán trọng số
#         z = F.relu(self.fc1(fused))  # [64, 17]
#         z = self.fc2(z)  # [64, 17]
#         z = torch.sigmoid(z)  # [64, 17], trọng số chú ý

#         # Bước 6: Tăng cường dữ liệu ban đầu bằng trọng số chú ý
#         # z cần được mở rộng để nhân với tensor x
#         z = z.unsqueeze(-1).unsqueeze(-1)  # [64, 17, 1, 1]
#         T_hat = x * z  # [64, 17, 30, 2], nhân element-wise

#         return T_hat
    
# class RLSTM(nn.Module):
#     def __init__(self, input_size=34, hidden_size=128, num_layers=3, output_size=256):
#         """
#         Khởi tạo RLSTM.
        
#         Args:
#             input_size (int): Kích thước đầu vào (34 = 17 keypoints × 2 coords).
#             hidden_size (int): Số đơn vị trong mỗi tầng ẩn (128).
#             num_layers (int): Số tầng ẩn (3).
#             output_size (int): Kích thước đầu ra (256).
#         """
#         super(RLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_size = output_size

#         # Tầng LSTM
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         # Tầng linear cho cấu trúc dư
#         self.residual = nn.Linear(input_size, hidden_size)
#         # Tầng linear để ánh xạ đầu ra
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         """
#         Xử lý dữ liệu qua RLSTM.
        
#         Args:
#             x (torch.Tensor): Dữ liệu đầu vào với kích thước [batch_size, num_frames, input_size].
#                               Ví dụ: [64, 30, 34].
        
#         Returns:
#             torch.Tensor: Dữ liệu đầu ra với kích thước [batch_size, num_frames, output_size].
#                           Ví dụ: [64, 30, 256].
#         """
#         batch_size, num_frames, _ = x.size()

#         # Khởi tạo trạng thái ẩn
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

#         # Đưa qua LSTM
#         out, (h_n, c_n) = self.lstm(x, (h0, c0))

#         # Cấu trúc dư
#         residual = self.residual(x)
#         out = out + residual

#         # Ánh xạ đầu ra
#         out = self.fc(out)

#         return out

# class ClassificationLayers(nn.Module):
#     def __init__(self, input_size=256, dropout_rate=0.3):
#         """
#         Khởi tạo Classification Layers.
        
#         Args:
#             input_size (int): Kích thước đầu vào (256, từ RLSTM).
#             dropout_rate (float): Tỷ lệ dropout (0.3).
#         """
#         super(ClassificationLayers, self).__init__()
#         self.input_size = input_size
#         self.dropout_rate = dropout_rate

#         # Tầng Classification1: 256 -> 128
#         self.fc1 = nn.Linear(input_size, 128)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         # Tầng Classification2: 128 -> 64
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         # Tầng Classification3: 64 -> 32
#         self.fc3 = nn.Linear(64, 32)
#         # Dropout: Áp dụng sau Classification3
#         self.dropout = nn.Dropout(p=dropout_rate)
#         # Tầng Classification4: 32 -> 2
#         self.fc4 = nn.Linear(32, 2)

#     def forward(self, x):
#         """
#         Xử lý dữ liệu qua Classification Layers.
        
#         Args:
#             x (torch.Tensor): Dữ liệu đầu vào với kích thước [batch_size, input_size].
#                               Ví dụ: [64, 256].
        
#         Returns:
#             torch.Tensor: Xác suất đầu ra với kích thước [batch_size, 2].
#                           Ví dụ: [64, 2].
#         """
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = self.fc4(x)
#         x = torch.sigmoid(x)
#         return x
    
# class KAMTFENet(nn.Module):
#     def __init__(self, num_keypoints=17, num_frames=30, input_size=34, hidden_size=128, num_layers=3, output_size=256, dropout_rate=0.3):
#         """
#         Khởi tạo mô hình KAMTFENet hoàn chỉnh.
        
#         Args:
#             num_keypoints (int): Số lượng điểm mấu chốt (17).
#             num_frames (int): Số khung hình trong cửa sổ trượt (30).
#             input_size (int): Kích thước đầu vào của RLSTM (34).
#             hidden_size (int): Số đơn vị trong mỗi tầng ẩn của RLSTM (128).
#             num_layers (int): Số tầng ẩn của RLSTM (3).
#             output_size (int): Kích thước đầu ra của RLSTM (256).
#             dropout_rate (float): Tỷ lệ dropout (0.3).
#         """
#         super(KAMTFENet, self).__init__()
#         self.num_keypoints = num_keypoints
#         self.num_frames = num_frames

#         # # Keypoint Attention Module
#         self.kam = KeypointAttentionModule(num_keypoints=num_keypoints)

#         # RLSTM
#         self.rlstm = RLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

#         # Classification Layers
#         self.classifier = ClassificationLayers(input_size=output_size, dropout_rate=dropout_rate)

#     def forward(self, x):
#         """
#         Xử lý dữ liệu qua KAMTFENet.
        
#         Args:
#             x (torch.Tensor): Dữ liệu đầu vào với kích thước [batch_size, num_keypoints, num_frames, 2].
#                               Ví dụ: [64, 17, 30, 2].
        
#         Returns:
#             torch.Tensor: Xác suất đầu ra với kích thước [batch_size, 2].
#                           Ví dụ: [64, 2].
#         """
#         # Bước 1: Đưa qua Keypoint Attention Module
#         x = self.kam(x)  # [64, 17, 30, 2]

#         # Bước 2: Hoán vị và định dạng lại cho RLSTM
#         x = x.permute(0, 2, 1, 3)  # [64, 30, 17, 2]
#         batch_size, num_frames, num_keypoints, coords = x.size()
#         x = x.reshape(batch_size, num_frames, num_keypoints * coords)  # [64, 30, 34]

#         # Bước 3: Đưa qua RLSTM
#         x = self.rlstm(x)  # [64, 30, 256]

#         # Bước 4: Lấy vector của khung hình cuối cùng (many-to-one)
#         x = x[:, -1, :]  # [64, 256]

#         # Bước 5: Đưa qua Classification Layers
#         x = self.classifier(x)  # [64, 2]

#         return x
    
# class RLSTM(nn.Module):
#     def __init__(self, input_size=34, hidden_size=256, num_layers=3):
#         """
#         Mạng RLSTM để trích xuất đặc trưng thời gian.
        
#         Args:
#             input_size (int): Số đặc trưng đầu vào (17 keypoints * 2 coords = 34).
#             hidden_size (int): Kích thước lớp ẩn (256 như bài báo).
#             num_layers (int): Số lớp LSTM 3.
#         """
#         super(RLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM với residual
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
#     def forward(self, x):
#         """
#         Forward pass của RLSTM.
        
#         Args:
#             x (torch.Tensor): Tensor đầu vào [batch_size, 30, 34]
#         Returns:
#             torch.Tensor: Đặc trưng thời gian [batch_size, 30, hidden_size]
#         """
#         batch_size = x.size(0)
        
#         # Khởi tạo hidden state và cell state
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
#         # LSTM forward
#         out, _ = self.lstm(x, (h0, c0))  # [batch_size, 30, hidden_size]
        
#         # Thêm residual (Eq. 7: h_{t+1}' = h_{t+1} + N_{t+1})
#         # Chuyển x về kích thước phù hợp để cộng
#         residual = nn.Linear(34, self.hidden_size)(x)  # [batch_size, 30, hidden_size]
#         out = out + residual
        
#         return out

# class FallClassifier(nn.Module):
#     def __init__(self, hidden_size=256, dropout_rate=0.3):
#         """
#         Mô hình phân loại té ngã.
        
#         Args:
#             hidden_size (int): Kích thước đầu ra của RLSTM (256).
#             dropout_rate (float): Tỷ lệ dropout 0.3.
#         """
#         super(FallClassifier, self).__init__()
        
#         # Các lớp fully connected (dựa trên Table 2)
#         self.fc1 = nn.Linear(hidden_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 2)  # 2 lớp: fall/no_fall
        
#         self.dropout = nn.Dropout(dropout_rate)
    
#     def forward(self, x):
#         """
#         Forward pass của classifier.
        
#         Args:
#             x (torch.Tensor): Đặc trưng thời gian [batch_size, hidden_size]
#         Returns:
#             torch.Tensor: Xác suất phân loại [batch_size, 2]
#         """
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = self.fc4(x)  # Không dùng sigmoid vì CrossEntropyLoss tự xử lý
#         return x

# class KAMTFENet(nn.Module):
#     def __init__(self):
#         """
#         Kết hợp RLSTM và FallClassifier thành mạng hoàn chỉnh.
#         """
#         super(KAMTFENet, self).__init__()
#         self.rlstm = RLSTM()
#         self.classifier = FallClassifier()
    
#     def forward(self, x):
#         """
#         Forward pass của toàn bộ mạng.
        
#         Args:
#             x (torch.Tensor): Tensor tăng cường [batch_size, 30, 17, 2]
#         Returns:
#             torch.Tensor: Xác suất phân loại [batch_size, 2]
#         """
#         # Chuyển tensor [B, 30, 17, 2] thành [B, 30, 34]
#         batch_size = x.size(0)
#         x = x.view(batch_size, 30, -1)  # [B, 30, 34]
        
#         # Trích xuất đặc trưng thời gian
#         temporal_features = self.rlstm(x)  # [B, 30, 256]
        
#         # Lấy đặc trưng của khung cuối cùng
#         last_frame_features = temporal_features[:, -1, :]  # [B, 256]
        
#         # Phân loại
#         output = self.classifier(last_frame_features)  # [B, 2]
#         return output

# def predict_fall(input_dir, output_dir, model_path=None):
#     """
#     Dự đoán té ngã từ các tensor tăng cường.
    
#     Args:
#         input_dir (str): Thư mục chứa tensor tăng cường (.npz).
#         output_dir (str): Thư mục lưu dự đoán.
#         model_path (str): Đường dẫn đến trọng số mô hình (nếu có).
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Khởi tạo mô hình
#     model = KAMTFENet()
#     if model_path and os.path.exists(model_path):
#         model.load_state_dict(torch.load(model_path))
#         print(f"Loaded model from {model_path}")
#     model.eval()
    
#     window_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])
    
#     with torch.no_grad():
#         for window_file in window_files:
#             window_path = os.path.join(input_dir, window_file)
#             data = np.load(window_path)
#             tensor = data['data']  # [30, 17, 2]
#             true_label = data['label']
            
#             # Chuyển sang torch tensor
#             tensor_torch = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)  # [1, 30, 17, 2]
            
#             # Dự đoán
#             output = model(tensor_torch)  # [1, 2]
#             probabilities = F.softmax(output, dim=1)  # [1, 2]
#             predicted_label = 'fall' if probabilities[0, 1] > 0.5 else 'no_fall'
            
#             # Lưu kết quả
#             output_path = os.path.join(output_dir, f"{window_file.split('.')[0]}_pred.txt")
#             with open(output_path, 'w') as f:
#                 f.write(f"True Label: {true_label}\n")
#                 f.write(f"Predicted Label: {predicted_label}\n")
#                 f.write(f"Probabilities: Fall={probabilities[0, 1]:.4f}, No_Fall={probabilities[0, 0]:.4f}\n")
#             print(f"Predicted {window_file}: {predicted_label} (True: {true_label})")

# if __name__ == "__main__":
#     input_dir = "data/processed/enhanced_features/ur_dataset"
#     output_dir = "data/processed/predictions/ur_dataset"
#     model_path = None  # Thay bằng "models/rlstm_weights/model.pth" nếu đã huấn luyện
#     predict_fall(input_dir, output_dir, model_path)

# class RLSTMManual(nn.Module):
#     def __init__(self, input_size=34, hidden_size=256):
#         super(RLSTMManual, self).__init__()
#         self.hidden_size = hidden_size
        
#         # Trọng số và bias cho các cổng
#         self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
#         self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
#         self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
#         self.Wc = nn.Linear(input_size + hidden_size, hidden_size)
    
#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         h = torch.zeros(batch_size, self.hidden_size).to(x.device)
#         c = torch.zeros(batch_size, self.hidden_size).to(x.device)
#         outputs = []
        
#         for t in range(seq_len):
#             xt = x[:, t, :]  # [batch_size, 34]
#             combined = torch.cat((h, xt), dim=1)  # [h_t, N_{t+1}]
            
#             ft = torch.sigmoid(self.Wf(combined))  # Eq. (4)
#             it = torch.sigmoid(self.Wi(combined))
#             ot = torch.sigmoid(self.Wo(combined))
#             ct_candidate = torch.tanh(self.Wc(combined))
#             c = ft * c + it * ct_candidate  # Eq. (5)
#             h = ot * torch.tanh(c)  # Eq. (6)
#             h = h + nn.Linear(34, self.hidden_size)(xt)  # Eq. (7)
#             outputs.append(h)
        
#         return torch.stack(outputs, dim=1)  # [batch_size, 30, hidden_size]


# import torch
# from torch import nn
#
# # (1) --- Dán toàn bộ định nghĩa model bạn đã đưa ở trên ---
# # ... dán CustomRLSTMCell, RLSTM, ClassificationLayers, AdaptiveKeypointAttention, KAMTFENet ...
#
# # (2) --- Đường dẫn file .pth ---
# pth_path = 'best_kamtfenet_new.pth'
#
# # (3) --- Load dữ liệu từ file ---
# data = torch.load(pth_path, map_location='cpu')
#
# # (4) --- Xử lý file nếu là dict có nhiều keys ---
# if isinstance(data, dict):
#     if 'state_dict' in data:
#         state_dict = data['state_dict']
#         print("[INFO] Found 'state_dict' in file.")
#     elif all(isinstance(v, torch.Tensor) for v in data.values()):
#         state_dict = data
#         print("[INFO] Loaded raw state_dict (model weights only).")
#     else:
#         print("[WARNING] Unknown file format. Keys found:", list(data.keys()))
#         state_dict = None
# else:
#     print("[ERROR] Không thể load model từ định dạng không xác định.")
#     state_dict = None
#
# # (5) --- Khởi tạo lại mô hình ---
model = KAMTFENet(num_keypoints=17, seq_len=30, input_size=34, hidden_size=256, num_layers=3)
#
# # (6) --- Nạp trọng số vào model nếu có ---
# if state_dict:
#     model.load_state_dict(state_dict)
#     print("[INFO] Model weights loaded successfully.")
#     print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters())}")
#
# # (7) --- Kiểm tra thông tin thêm nếu có ---
# if isinstance(data, dict):
#     for key in ['accuracy', 'val_accuracy', 'epoch', 'loss']:
#         if key in data:
#             print(f"[INFO] {key}: {data[key]}")

def evaluate_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, classification_report
    print(f"[RESULT] Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(classification_report(all_labels, all_preds))


import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_npz_dataset_from_folders(parent_folder):
    X_list = []
    y_list = []

    # Duyệt qua tất cả các thư mục con
    for root, dirs, files in os.walk(parent_folder):
        for fname in files:
            if fname.endswith('.npz'):
                file_path = os.path.join(root, fname)
                npz_file = np.load(file_path)

                X_list.append(npz_file['data'])  # Dữ liệu input
                y_list.append(npz_file['label'])  # Nhãn

    X_array = np.stack(X_list)

    # Ánh xạ nhãn
    label_map = {'no_fall': 0, 'fall': 1}
    y_array = np.array([label_map[str(label)] for label in y_list])

    return X_array, y_array


# Load dữ liệu
folder = "data/processed/enhanced_features/ur_dataset"
X, y = load_npz_dataset_from_folders(folder)

# Chuyển sang tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dataset + DataLoader
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

evaluate_model(model, test_loader)
