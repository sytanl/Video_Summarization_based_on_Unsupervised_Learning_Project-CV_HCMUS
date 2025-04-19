import cv2 as cv
import numpy as np
from scipy import sparse
from scipy.spatial.distance import jensenshannon

# Đếm số frame trong một video và lấy frame rate (số frame/giây)
def count_frames(video_path):
    video = cv.VideoCapture(video_path)
    fps = int(video.get(cv.CAP_PROP_FPS))  # Lấy frame rate

    count = 0
    while True:
        ret, _ = video.read()  # Đọc từng frame
        if not ret:
            break
        count += 1  # Đếm số frame

    video.release()
    return fps, count  # Trả về fps và tổng số frame


# Tính vector trung bình từ một tập các embedding
def mean_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


# Trả về hàm đo khoảng cách tương ứng với tên (string) đã cho
def distance_metric(distance):
    if distance == 'jensenshannon':
        return jensenshannon
    elif distance == 'euclidean':
        return distance 
    elif distance == 'cosine':
        return distance  
    else:
        raise ValueError(f'Unknown distance metric: {distance}')


# Tính độ tương đồng cosine giữa từng embedding và vector trung bình của chúng
def similarity_score(embeddings, mean=None):
    if mean is None:
        mean = mean_embeddings(embeddings)

    # Dot product giữa từng vector và vector trung bình chia cho tích độ dài (chuẩn hoá)
    return np.dot(embeddings, mean) / (np.linalg.norm(embeddings) * np.linalg.norm(mean))


# Tạo ma trận kết nối (connectivity matrix) từ nhãn cluster của từng vector
def construct_connectivity(data, labels):
    row = []
    col = []
    subcluster_dict = {}

    # Gom index của từng điểm theo cluster label
    for i in range(data.shape[0]):
        if labels[i] not in subcluster_dict:
            subcluster_dict[labels[i]] = []
        subcluster_dict[labels[i]].append(i)

    # Tạo liên kết giữa tất cả các cặp điểm trong cùng một cluster
    for subcluster in subcluster_dict.values():
        for i, element in enumerate(subcluster):
            for j in range(i + 1, len(subcluster)):
                row.append(element)
                col.append(subcluster[j])
                row.append(subcluster[j])
                col.append(element)

    # Tạo sparse matrix từ các chỉ số hàng và cột
    connectivity = sparse.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(data.shape[0], data.shape[0]))
    return connectivity


# Tính số lượng cluster phù hợp với độ dài video và frame rate
def calculate_num_clusters(num_frames, max_length, frame_rate=4, modulation=1e-3):
    """
    Tính số lượng cluster dựa trên số frame, độ dài tối đa và frame_rate.
    Sử dụng hàm sigmoid để đảm bảo số cluster tăng mượt theo số frame.
    """
    max_clusters = max_length * frame_rate
    num_clusters = max_clusters * 2.0 / (1 + np.exp((-modulation) * num_frames)) - max_clusters
    return int(num_clusters)
