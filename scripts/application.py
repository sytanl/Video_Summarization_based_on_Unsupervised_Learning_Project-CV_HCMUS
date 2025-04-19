import os
import numpy as np
from tqdm import tqdm
import cv2 as cv

from torchvision.transforms import ToTensor
from PIL import Image

from model.embedder import Embedder
from model.propogator import Clusterer
from model.selector import Selector
from model.generator import Summarizer

from model.utils import count_frames, calculate_num_clusters


class VidSum():
    """Lớp VidSum dùng để tóm tắt video
    Thực hiện toàn bộ quá trình tóm tắt video được cung cấp
    """
    def __init__(self):
        # Khởi tạo các mô-đun chính: Trích đặc trưng (Embedder)
        self.embedder = Embedder(model_type='clip',
                                 representation='cls',
                                 model_kind='base',
                                 patch=32,
                                 device='cuda')  # Sử dụng GPU để tăng tốc

        # Các tham số sẽ được thiết lập sau
        self.input_frame_rate = None  # Tốc độ khung hình đầu vào

        # Các tham số điều chỉnh thuật toán tóm tắt
        self.method = None
        self.distance = None
        self.max_length = None
        self.modulation = None
        self.intermediate_components = None
        self.window_size = None
        self.min_seg_length = None

        # Các thiết lập cho embedding giảm chiều
        self.reduced_emb = None
        self.scoring_mode = None
        self.kf_mode = None
        self.bias = None

        # Các thông số đầu ra
        self.output_frame_rate = None
        self.max_length = None
        self.sum_rate = None
        self.extension = None
        self.k = None

        # Bảng mã hóa video theo phần mở rộng
        self.codec_dict = {
            'mp4': 'mp4v',
            'avi': 'DIVX',
            'webm': 'VP09'
        }

    def set_params(self, input_frame_rate, method, distance, max_length,
                   modulation, intermediate_components, window_size,
                   min_seg_length, reduced_emb, scoring_mode, kf_mode,
                   bias, output_frame_rate, sum_rate, extension, k):
        # Thiết lập tất cả các tham số từ bên ngoài
        self.input_frame_rate = int(input_frame_rate)
        self.method = method
        self.distance = distance
        self.max_length = int(max_length)
        self.modulation = float(10.0 ** float(modulation))
        self.intermediate_components = int(intermediate_components)
        self.window_size = int(window_size)
        self.min_seg_length = int(min_seg_length)

        self.reduced_emb = bool(reduced_emb)
        self.scoring_mode = scoring_mode
        self.kf_mode = kf_mode
        self.bias = float(bias)

        self.output_frame_rate = int(output_frame_rate)
        self.sum_rate = sum_rate
        self.extension = extension.lower()
        self.k = int(k)

    def generate_context(self, video_path):
        # Chuyển đổi ảnh từ numpy sang tensor
        transform = ToTensor()

        print(f'Đang trích xuất đặc trưng từ video: {video_path}')

        # Mở video
        cap = cv.VideoCapture(video_path)

        # Lấy thông tin video: fps và tổng số frame
        fps, total_frames = count_frames(video_path)

        if self.input_frame_rate is None:
            self.input_frame_rate = fps

        # Xác định bước nhảy giữa các khung hình
        frame_step = fps // self.input_frame_rate
        total_samples = (total_frames + frame_step - 1) // frame_step  # Làm tròn lên

        embeddings = []  # Danh sách chứa vector đặc trưng
        samples = []     # Danh sách chứa chỉ số khung hình

        # Tạo thanh tiến trình
        pbar = tqdm(total=total_samples)
        frame_idx = 0

        while True:
            ret, frame = cap.read()  # Đọc 1 khung hình
            if not ret:
                break
            if frame_idx % frame_step:
                frame_idx += 1
                continue

            # Chuyển ảnh sang kiểu PIL để xử lý
            img = Image.fromarray(frame, mode="RGB")
            img = transform(img).unsqueeze(0)  # Thêm chiều batch
            embedding = self.embedder.image_embedding(img)  # Trích đặc trưng

            embeddings.append(embedding)
            samples.append(frame_idx)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()  # Đóng video

        # Trả về ma trận embedding và danh sách khung hình tương ứng
        return np.vstack(embeddings), np.asarray(samples, dtype=np.int32)
    
    def localize_video(self, embeddings, flag):  
        # Tính toán số lượng cụm dựa trên số lượng khung hình  
        num_clusters = calculate_num_clusters(num_frames=embeddings.shape[0],  
                                            max_length=self.max_length,  
                                            modulation=self.modulation)  

        # Tạo đối tượng phân cụm  
        clusterer = Clusterer(method=self.method, distance=self.distance,  
                            num_clusters=num_clusters,  
                            embedding_dim=embeddings.shape[1],  
                            intermediate_components=self.intermediate_components)  
        
        # Tạo đối tượng chọn lựa  
        selector = Selector(window_size=self.window_size,  
                            min_seg_length=self.min_seg_length)  
        
        # Phân cụm và giảm chiều  
        labels, reduced_embeddings = clusterer.cluster(embeddings, flag)  
        
        # Chọn các phân đoạn  
        segments = selector.select(labels)  
        
        # In ra số lượng phân đoạn và cụm  
        print(f'Số lượng phân đoạn: {len(segments)}')  
        print(f'Số lượng cụm: {clusterer.num_clusters}')  
        
        return segments, labels, reduced_embeddings  

    def generate_summary(self, embeddings, samples, segments):  
        # Tạo đối tượng tổng hợp  
        summarizer = Summarizer(scoring_mode=self.scoring_mode,  
                                kf_mode=self.kf_mode, k=self.k)  
        
        # Tính toán điểm cho từng phân đoạn  
        scores = summarizer.score_segments(embeddings=embeddings,  
                                        segments=segments,  
                                        bias=self.bias)  
        
        # Kết hợp điểm số với thông tin mẫu  
        sampled_scores = [[sample, score]  
                        for sample, score in zip(samples, scores)]  
        
        # Sắp xếp theo chỉ số mẫu  
        sorted_scores = np.asarray(sorted(sampled_scores,  
                                        key=lambda x: x[0]))  
        
        # Chọn các khung hình chính  
        keyframe_indices = summarizer.select_keyframes(segments,  
                                                    scores,  
                                                    0)  
        
        print(f'Đã chọn {len(keyframe_indices)} khung hình chính')  
        
        # Lấy chỉ số khung hình chính từ mẫu  
        keyframe_idxs = np.asarray([samples[idx]  
                                    for idx in keyframe_indices])  
            
        return sorted_scores, keyframe_idxs  

    def generate_video(self, output_video_path, input_video_path,  
                    frame_indices):  
        # Đọc video thô  
        raw_video = cv.VideoCapture(input_video_path)  
        width = int(raw_video.get(cv.CAP_PROP_FRAME_WIDTH))  
        height = int(raw_video.get(cv.CAP_PROP_FRAME_HEIGHT))  
        
        # Tính FPS và chiều dài video  
        computed_fps, video_length = count_frames(input_video_path)  

        # Xác định FPS cho video đầu ra  
        if self.output_frame_rate is None:  
            fps = computed_fps  
        else:  
            fps = self.output_frame_rate  
        
        print(f'FPS: {fps}')  
        
        # Số lượng khung hình tối đa trong tóm tắt  
        frames_length = int(self.max_length * fps)  
        
        # Số lượng ước tính khung hình trong tóm tắt  
        estimated_length = int(video_length * self.sum_rate)  
        
        # Số lượng khung hình cuối cùng cho tóm tắt  
        summary_length = max(min(frames_length, estimated_length),  
                            len(frame_indices))  
        print(f'Số khung hình trong tóm tắt: {summary_length}')  
        
        # Độ dài của đoạn xung quanh mỗi khung hình chính  
        fragment_length = summary_length // len(frame_indices)  
        print(f'Độ dài đoạn nhúng: {fragment_length}')  
        
        # Độ rộng của đoạn tính toán  
        fragment_width = max(0, (fragment_length - 1) // 2)  
        print(f'Độ rộng của các đoạn: {fragment_width}')  
        
        # Thiết lập codec cho video đầu ra  
        output_codec = self.codec_dict[self.extension]  
        fourcc = cv.VideoWriter_fourcc(*output_codec)  
        video = cv.VideoWriter(output_video_path, fourcc,  
                            float(fps), (width, height))  
        
        cur_idx = 0  
        pbar = tqdm(total=len(frame_indices))  # Thanh tiến trình  
        kf_idx = 0  
        
        while True:  
            ret, frame = raw_video.read()  
            if not ret:  
                break  
            
            # Tìm khung hình chính để ghi  
            while kf_idx < len(frame_indices) and frame_indices[kf_idx] < cur_idx - fragment_width:  
                kf_idx += 1  
            if kf_idx < len(frame_indices) and abs(frame_indices[kf_idx] - cur_idx) <= fragment_width:  
                video.write(frame)  
            
            if cur_idx in frame_indices:  
                pbar.update(1)  # Cập nhật thanh tiến trình  
            
            cur_idx += 1  
        
        # Giải phóng tài nguyên video  
        raw_video.release()  
        video.release()  
        pbar.close()  

    def store_result(self, video_path, output_folder, data):  
        # Lấy tên video mà không có phần mở rộng  
        video_name = os.path.basename(video_path).split('.')[0]  
        video_folder = os.path.join(output_folder, video_name)  
        
        # Tạo thư mục nếu chưa tồn tại  
        if not os.path.exists(video_folder):  
            os.makedirs(video_folder)  
            
        # Tạo video  
        output_video_name = f'{video_name}_summary.{self.extension}'  
        output_video_path = os.path.join(video_folder,  
                                        output_video_name)  
        
        # Gọi hàm để tạo video tóm tắt  
        self.generate_video(output_video_path, video_path,  
                            data['keyframe_idxs'])  
        
        # Lưu điểm số  
        scores_path = os.path.join(video_folder,  
                                video_name + '_scores.npy')  
        np.save(scores_path, data['scores'])  
        
        # Lưu các phân đoạn  
        segments_path = os.path.join(video_folder,  
                                    video_name + '_segments.npy')  
        np.save(segments_path, data['segments'])  
        
        # Lưu nhãn  
        labels_path = os.path.join(video_folder,  
                                video_name + '_labels.npy')  
        np.save(labels_path, data['labels'])  
        
        # Lưu các nhúng đã giảm  
        reduced_embeddings_path = os.path.join(video_folder,  
                                            video_name + '_reduced_embeddings.npy')  
        np.save(reduced_embeddings_path, data['reduced_embeddings'])  
        
        return output_video_path  

    def summarize(self, video_path, output_folder, flag=False):  
        data = {}  
        
        # Tạo ngữ cảnh  
        data['embeddings'], data['samples'] = self.generate_context(video_path)  
        
        # Xác định vị trí video  
        local_information = self.localize_video(data['embeddings'], flag)  
        data['segments'], data['labels'], data['reduced_embeddings'] = local_information  
        
        # Tạo tóm tắt  
        embedding = data['reduced_embeddings'] if self.reduced_emb else data['embeddings']  
        summary = self.generate_summary(embedding,  
                                        data['samples'],  
                                        data['segments'])  
        data['scores'], data['keyframe_idxs'] = summary  
        
        # Tạo video  
        output_video_path = self.store_result(video_path, output_folder, data)  
        
        return output_video_path  