import numpy as np
from model.utils import mean_embeddings, similarity_score


def nomalize(arr):
    # Hàm chuẩn hóa: đưa các giá trị trong mảng về khoảng [0, 1]
    min_val = min(arr)
    max_val = max(arr)

    normalized = [(x - min_val) / (max_val - min_val) for x in arr]

    return normalized

class Summarizer:
    def __init__(self, scoring_mode, kf_mode, k):
        # Khởi tạo đối tượng Summarizer với các chế độ điểm số và chế độ keyframe
        print(f"Summarizer's scoring mode is {scoring_mode}")
        print(f"Input KF mode is {kf_mode}")
        
        self.scoring_mode = scoring_mode
        self.k = k  # Hệ số k, ảnh hưởng đến mức độ quan trọng của đặc trưng nội dung
        
        self.kf_mode = []  # Lưu trữ các chế độ keyframe
        if scoring_mode == 'mean':
            if 'mean' in kf_mode:
                self.kf_mode.append('mean')
        
        if 'middle' in kf_mode:
            self.kf_mode.append('middle')
        
        if 'ends' in kf_mode:
            self.kf_mode.append('ends')
            
        print(f"Summarizer's KF mode is {self.kf_mode}")

    def score_segments(self, embeddings, segments, bias):
        # Tính toán điểm số cho từng phân đoạn (segment) dựa trên các đặc trưng
        segment_scores = []
        
        for _, start, end in segments:
            # Lấy các đặc trưng liên quan đến phân đoạn hiện tại
            segment_features = embeddings[start:end]
            
            # Tính toán điểm số cho các khung hình trong phân đoạn
            if self.scoring_mode == "uniform":
                # Cho điểm đồng đều với trọng số cho các khung hình gần keyframes
                filling = len(segment_features)
                max_score = filling * (1 + bias)
                min_score = filling
                
                if bias < 0:
                    max_score, min_score = min_score, max_score
                
                # Tính toán điểm số theo một hàm cosine đối với vị trí của keyframes
                period = None
                if "middle" in self.kf_mode and "ends" in self.kf_mode:
                    period = 4 * np.pi
                elif "middle" in self.kf_mode or "ends" in self.kf_mode:
                    period = 2 * np.pi
                
                if period is not None:
                    start_phase = 0 if 'ends' in self.kf_mode else np.pi
                    end_phase = start_phase + period
                    magnitude = (max_score - min_score) / 2
                    domain = np.linspace(start_phase, end_phase,
                                         end - start)
                    
                    position_score = magnitude * np.cos(domain) + magnitude + min_score
                else:
                    position_score = [min_score] * len(segment_features)
                
                # Tính toán điểm số nội dung
                representative = mean_embeddings(segment_features)
                
                content_score = similarity_score(segment_features,
                                         representative).tolist()
                
                # Chuẩn hóa điểm số nội dung
                nomalized_content_score = nomalize(content_score)
                step_value = max(position_score) - min(position_score)

                # Tính toán điểm cuối cùng kết hợp giữa điểm vị trí và điểm nội dung
                score = np.asarray(position_score) +  self.k * np.asarray(nomalized_content_score) * step_value
            
            else:
                if self.scoring_mode == "mean":
                    # Nếu là scoring mode "mean", sử dụng trung bình của phân đoạn
                    representative = mean_embeddings(segment_features)
                else:
                    # Nếu không phải "mean", sử dụng khung hình giữa phân đoạn
                    representative = segment_features[len(segment_features)
                                                      // 2]
                
                # Tính toán điểm số nội dung với đại diện của phân đoạn
                score = similarity_score(segment_features,
                                         representative).tolist()
            
            segment_scores.extend(score)
        
        return np.asarray(segment_scores)

    def select_keyframes(self, segments, scores, length):
        # Chọn các keyframe từ các phân đoạn và điểm số của chúng
        keyframe_indices = []
        
        for _, start, end in segments:
            # Chọn keyframe theo các chế độ (mean, middle, ends)
            if 'mean' in self.kf_mode:
                segment_scores = scores[start:end]
                keyframe_indices.append(np.argmax(segment_scores) + start)
            
            if 'middle' in self.kf_mode:
                keyframe_indices.append((start + end) // 2)
                
            if 'ends' in self.kf_mode:
                keyframe_indices.append(start)
                keyframe_indices.append(end - 1)

        if length > 0:
            # Chọn thêm keyframe từ các khung hình không được chọn để đạt đủ độ dài
            unselected_indicies = np.setdiff1d(np.arange(len(scores)),
                                               keyframe_indices)
            
            unselected_scores = scores[unselected_indicies]
            
            # Lựa chọn thêm các keyframe từ các khung hình chưa được chọn
            remained_length = length - len(keyframe_indices)
            unselected_keyframes = np.argpartition(unselected_scores,
                                                   -remained_length)[-remained_length:]
            
            keyframe_indices.extend(unselected_indicies[unselected_keyframes])
        
        return np.sort(keyframe_indices)
