import torch
from transformers import ViTFeatureExtractor, ViTModel
from transformers import CLIPProcessor, CLIPModel


class Embedder:
    def __init__(self, model_type='clip', representation='cls',
                 model_kind='base', patch=32, device='cpu'):
        # Khởi tạo Embedder với loại model, kiểu đặc trưng, kiểu mô hình, kích thước patch và thiết bị sử dụng
        self.feature_type = representation  # Loại đặc trưng (cls hoặc mean)
        # Kiểm tra nếu có GPU thì sử dụng CUDA, nếu không sử dụng CPU
        self.device = 'cuda' if (torch.cuda.is_available()
                                 and device == 'cuda') else 'cpu'
        self.model_type = model_type
        
        # Chọn mô hình DINO hoặc CLIP
        if model_type == 'dino':
            # Đường dẫn mô hình DINO
            self.model_path = f'facebook/dino-vit{model_kind}{patch}'
            # Khởi tạo feature extractor và mô hình DINO
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path)
            self.model = ViTModel.from_pretrained(self.model_path)
            self.emb_dim = 768  # Kích thước đặc trưng cho DINO
        elif model_type == 'clip':
            # Đường dẫn mô hình CLIP
            self.model_path = f"openai/clip-vit-{model_kind}-patch{patch}"
            # Khởi tạo feature extractor và mô hình CLIP
            self.feature_extractor = CLIPProcessor.from_pretrained(self.model_path)
            self.model = CLIPModel.from_pretrained(self.model_path)
            self.emb_dim = self.model.projection_dim  # Kích thước đặc trưng cho CLIP
        
        # Đặt mô hình ở chế độ evaluation và di chuyển mô hình sang thiết bị (GPU hoặc CPU)
        self.model.eval()
        self.model.to(self.device)
        print(f'Using {self.device} device')
        
    def set_params(self, feature_type, device):
        # Cập nhật các tham số như loại đặc trưng và thiết bị sử dụng
        self.feature_type = feature_type
        new_device = 'cuda' if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        
        if self.device != new_device:
            # Nếu thiết bị thay đổi, cập nhật và chuyển mô hình sang thiết bị mới
            self.device = new_device
            self.model.to(self.device)
            print(f'Using {self.device} device')

    def image_embedding(self, image):
        # Đặt mô hình ở chế độ evaluation và tắt gradient
        self.model.eval()
        
        with torch.no_grad():
            # Tiền xử lý hình ảnh trước khi đưa vào mô hình
            inputs = self.feature_extractor(images=image,
                                            return_tensors="pt")
            
            # Tính toán đặc trưng từ mô hình
            if self.model_type == 'dino':
                outputs = self.model(**inputs.to(self.device))
                features = outputs.last_hidden_state.detach().squeeze(0)
            elif self.model_type == 'clip':
                outputs = self.model.get_image_features(**inputs.to(self.device))
                features = outputs.detach().squeeze(0)
        
        # Chuẩn hóa đặc trưng L2
        features = features / features.norm(dim=-1, keepdim=True)
        # Áp dụng Softmax lên đặc trưng
        features = torch.nn.functional.softmax(features, dim=-1)
        
        # Nếu đang sử dụng GPU, chuyển đặc trưng về CPU
        if self.device == 'cuda':
            features = features.cpu()
            
        # Xử lý đặc trưng cho mô hình DINO
        if self.model_type == 'dino':
            if self.feature_type == 'cls':
                # Trả về đặc trưng từ CLS token
                feature = features[0]
            else:
                # Trả về đặc trưng trung bình của tất cả các token
                feature = torch.mean(features, dim=0)
            return feature
        elif self.model_type == 'clip':
            # Trả về đặc trưng đã qua Softmax (có thể dùng cho CLIP)
            return features
