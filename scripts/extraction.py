import os
import time
import argparse
import numpy as np

from model.propogator import Clusterer
from model.selector import Selector
from model.utils import calculate_num_clusters


# Hàm này thực hiện phân cụm và chọn các đoạn context từ các đặc trưng đã cho.
def localize_context(embeddings, method, n_clusters, window_size,
                     min_seg_length, distance, embedding_dim, intermediate_components=50, final_reducer='tsne'):
    clusterer = Clusterer(method, distance, n_clusters, embedding_dim, intermediate_components, final_reducer=final_reducer)
    selector = Selector(window_size, min_seg_length)
    labels, reduced_embeddings = clusterer.cluster(embeddings, flag=False)  # Thực hiện phân cụm
    return (labels, selector.select(labels),  # Chọn các đoạn (segments)
            clusterer.num_clusters, reduced_embeddings)  # Trả về số lượng cụm và các embedding giảm chiều


# Hàm này xử lý tất cả các video trong thư mục embeddings, thực hiện phân cụm và lưu kết quả.
def localize_videos(embedding_folder, context_folder, method,
                    max_len, window_size, min_seg_length,
                    distance, embedding_dim, modulation, intermediate_components, final_reducer='tsne'):
    # Duyệt qua tất cả các file trong thư mục embeddings
    for embedding_name in os.listdir(embedding_folder):
        if embedding_name.endswith('_embeddings.npy'):
            filename = embedding_name[:-len('_embeddings.npy')]
            print(f"Processing the context of video {filename}")
            
            embedding_file = os.path.join(embedding_folder, embedding_name)
            embeddings = np.load(embedding_file)
            print(f"The extracted context has {embeddings.shape[0]} embeddings")
            
            segment_file = filename + '_segments.npy'
            labels_file = filename + '_labels.npy'
            reduced_file = filename + '_reduced.npy'
            
            segment_path = os.path.join(context_folder, segment_file)
            labels_path = os.path.join(context_folder, labels_file)
            
            # Reduced embeddings
            reduced_path = os.path.join(embedding_folder, reduced_file)
            
            print(f'Clustering frames of {filename}')
            if os.path.exists(segment_path):  # Nếu đã có kết quả phân cụm thì bỏ qua
                continue
            
            # Tính toán số lượng cụm ban đầu
            num_clusters = calculate_num_clusters(num_frames=embeddings.shape[0],
                                                  max_length=max_len,
                                                  modulation=modulation)
            print(f"Initial number of clusters is {num_clusters} with modulation {modulation}")
            local_context = localize_context(embeddings=embeddings,
                                             method=method,
                                             n_clusters=num_clusters,
                                             window_size=window_size,
                                             min_seg_length=min_seg_length,
                                             distance=distance,
                                             embedding_dim=embedding_dim,
                                             intermediate_components=intermediate_components,
                                             final_reducer=final_reducer)
            
            labels, segments, n_clusters, reduced_embs = local_context
            print(f'Number of clusters: {n_clusters}')
            print(f'Number of segments: {len(segments)}')
            
            # Lưu kết quả phân cụm
            np.save(segment_path, segments)
            np.save(labels_path, labels)
            np.save(reduced_path, reduced_embs)


# Hàm main, nơi bắt đầu quá trình xử lý
def main():
    parser = argparse.ArgumentParser(description='Convert Global Context of Videos into Local Semantics.')
    
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')  # Đường dẫn đến thư mục chứa các file embedding
    parser.add_argument('--context-folder', type=str, required=True,
                        help='path to output folder for clustering')  # Đường dẫn đến thư mục lưu kết quả phân cụm
    
    # Các tham số để lựa chọn phương pháp phân cụm và các thông số liên quan
    parser.add_argument('--method', type=str, default='ours',
                        choices=['kmeans', 'dbscan', 'gaussian',
                                 'agglo', 'ours'],
                        help='clustering method')  # Phương pháp phân cụm
    parser.add_argument('--max-len', type=int, default=60,
                        help='Maximum length of output summarization in seconds')  # Độ dài tối đa của video tóm tắt
    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['jensenshannon', 'euclidean', 'cosine'],
                        help='distance metric for clustering')  # Metric khoảng cách cho phân cụm
    parser.add_argument('--embedding-dim', type=int, default=-1,
                        help='dimension of embeddings')  # Kích thước của vector embedding
    
    parser.add_argument('--window-size', type=int, default=10,
                        help='window size for smoothing')  # Kích thước cửa sổ cho phép làm mượt
    parser.add_argument('--min-seg-length', type=int, default=10,
                        help='minimum segment length')  # Độ dài tối thiểu của một đoạn
    parser.add_argument('--modulation', type=float, default=1e-3,
                        help='modulation factor for number of clusters')  # Hệ số điều chỉnh số lượng cụm
    parser.add_argument('--intermediate-components', type=int, default=-1,
                        help='intermediate components')  # Số lượng thành phần trung gian
    parser.add_argument('--final-reducer', type=str, default='tsne',
                        choices=['pca', 'tsne'],
                        help='final dimensionality reduction technique')  # Kỹ thuật giảm chiều cuối cùng
    
    args = parser.parse_args()

    # Gọi hàm xử lý phân cụm video
    localize_videos(embedding_folder=args.embedding_folder,
                    context_folder=args.context_folder,
                    method=args.method,
                    max_len=args.max_len,
                    window_size=args.window_size,
                    min_seg_length=args.min_seg_length,
                    distance=args.distance,
                    embedding_dim=args.embedding_dim,
                    modulation=args.modulation,
                    intermediate_components=args.intermediate_components,
                    final_reducer=args.final_reducer,
                    )


if __name__ == '__main__':
    start_time = time.time()  # Bắt đầu tính thời gian thực thi
    main()  # Thực thi hàm main
    print("--- %s seconds ---" % (time.time() - start_time))  # In ra thời gian thực thi
