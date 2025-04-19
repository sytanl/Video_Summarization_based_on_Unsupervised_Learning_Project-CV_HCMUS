import os
import time
import argparse
import numpy as np

from model.generator import Summarizer


def summarize_videos(embedding_folder, context_folder, summary_folder,
                     reduced_emb, scoring_mode, kf_mode, bias, key_length, k):
    # Tạo một đối tượng Summarizer để tiến hành tóm tắt video
    summarizer = Summarizer(scoring_mode, kf_mode, k)
    
    # Duyệt qua các file trong thư mục embedding_folder
    for embedding_name in os.listdir(embedding_folder):
        # Xác định tên file kết quả dựa trên việc có giảm chiều hay không
        file_end = '_reduced.npy' if reduced_emb else '_embeddings.npy'
        
        # Chỉ xử lý các file kết thúc với '_reduced.npy' hoặc '_embeddings.npy'
        if embedding_name.endswith(file_end):
            filename = embedding_name[:-len(file_end)]
            print(f"Processing the context of video {filename}")
            
            # Đọc các dữ liệu embeddings từ file
            embedding_name = os.path.join(embedding_folder, embedding_name)
            embeddings = np.load(embedding_name)
            print(f"The extracted context has {embeddings.shape[0]} embeddings")
            
            # Đọc các mẫu từ file
            samples_file = filename + '_samples.npy'
            samples_path = os.path.join(embedding_folder, samples_file)
            samples = np.load(samples_path)
            
            # Đọc các segments từ file context
            segments_path = os.path.join(context_folder, filename + '_segments.npy')
            segments = np.load(segments_path)
            print(f"The extracted context has {segments.shape[0]} segments")
            
            # Định nghĩa các file để lưu scores và keyframes
            scores_file = filename + '_scores.npy'
            keyframes_file = filename + '_keyframes.npy'
            
            scores_path = os.path.join(summary_folder, scores_file)
            keyframes_path = os.path.join(summary_folder, keyframes_file)
            
            print(f'Summarizing video {filename}')
            
            # Nếu file scores đã tồn tại, bỏ qua video này
            if os.path.exists(scores_path):
                continue
            
            # Tính toán điểm số cho các segments
            scores = summarizer.score_segments(embeddings=embeddings,
                                               segments=segments,
                                               bias=bias)
            
            # Kết hợp điểm số và samples lại với nhau
            sampled_scores = [[sample, score]
                              for sample, score in zip(samples, scores)
                              ]
            
            # Sắp xếp theo index của samples
            sorted_scores = np.asarray(sorted(sampled_scores,
                                              key=lambda x: x[0]))
            # Lưu kết quả scores vào file
            np.save(scores_path, sorted_scores)
            
            # Nếu key_length >= 0, thực hiện lựa chọn keyframes
            if key_length >= 0:
                keyframe_indices = summarizer.select_keyframes(segments,
                                                               scores,
                                                               key_length)
                
                print(f'Selected {len(keyframe_indices)} keyframes')
                
                # Lưu các chỉ số keyframe vào file
                keyframe_idxs = np.asarray([samples[idx]
                                            for idx in keyframe_indices])
                np.save(keyframes_path, np.sort(keyframe_idxs))


def main():
    # Tạo đối tượng parser để nhận các tham số đầu vào từ dòng lệnh
    parser = argparse.ArgumentParser(description='Generate Summaries from Partitioned Selections of Videos.')
    
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--context-folder', type=str, required=True,
                        help='path to output folder for clustering')
    parser.add_argument('--summary-folder', type=str, required=True,
                        help='path to output folder for summaries')
    
    # Các tham số cấu hình cho quá trình tính toán và tóm tắt
    parser.add_argument('--scoring-mode', type=str, default='mean',
                        choices=['mean', 'middle', 'uniform'],
                        help='Method of representing segments')
    parser.add_argument('--kf-mode', type=str, default='mean',
                        help='Method of representing segments')
    parser.add_argument('--reduced-emb', action='store_true',
                        help='Use reduced embeddings or not')
    parser.add_argument('--bias', type=float, default=0.5,
                        help='Bias for frames near the keyframes')
    
    # Số lượng keyframes cần chọn
    parser.add_argument('--key-length', type=int, default=-1,
                        help="Maximum number of keyframes to select, "
                        + "-1 to not select, 0 to auto select")
    parser.add_argument('--k', type=int, default=0,
                        help="parameter k for scoring phase")
    
    # Lấy các tham số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm summarize_videos với các tham số đã truyền vào
    summarize_videos(embedding_folder=args.embedding_folder,
                     context_folder=args.context_folder,
                     summary_folder=args.summary_folder,
                     reduced_emb=args.reduced_emb,
                     scoring_mode=args.scoring_mode,
                     kf_mode=args.kf_mode,
                     bias=args.bias,
                     key_length=args.key_length,
                     k=args.k
                     )


if __name__ == '__main__':
    # Đo thời gian thực thi của chương trình
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
