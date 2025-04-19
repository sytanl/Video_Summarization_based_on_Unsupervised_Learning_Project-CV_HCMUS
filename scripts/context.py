import argparse
import os
import time
import numpy as np
import cv2 as cv
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from model.embedder import Embedder
from model.utils import count_frames

# Hàm generate_context: trích xuất đặc trưng cho từng khung hình trong video và lưu chúng vào file
def generate_context(video_folder, filename, embedding_folder,
                     embedder, frame_rate=None):
    # Định nghĩa các phép biến đổi hình ảnh
    transform = ToTensor()
    
    # Lấy đường dẫn file video và file lưu embedding
    print(f'Extracting features for {filename}')
    video_name = os.path.splitext(filename)[0]
    video_file = os.path.join(video_folder, filename)
    embedding_file = os.path.join(embedding_folder, f'{video_name}_embeddings.npy')
    sample_file = os.path.join(embedding_folder, f'{video_name}_samples.npy')
    
    # Nếu đã có embedding và sample, không cần tính lại
    if os.path.exists(embedding_file) and os.path.exists(sample_file):
        return
    
    # Mở video và lấy thông tin frame rate, tổng số frame
    cap = cv.VideoCapture(video_file)
    fps, total_frames = count_frames(video_file)
    
    # Nếu không có frame_rate, sử dụng mặc định là fps
    if frame_rate is None:
        frame_rate = fps
    
    # Tính toán số lượng mẫu
    frame_step = fps // frame_rate
    total_samples = (total_frames + frame_step - 1) // frame_step
    
    # Tạo các biến để lưu trữ embeddings và samples
    embeddings = np.zeros((total_samples, embedder.emb_dim))
    samples = np.zeros((total_samples), dtype=np.int64)

    pbar = tqdm(total=total_samples)
    
    frame_idx = 0
    result_idx = 0
    
    # Đọc từng frame và tính toán embedding
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step:
            frame_idx += 1
            continue
        
        # Chuyển frame thành tensor PyTorch và trích xuất đặc trưng
        img = Image.fromarray(frame, mode="RGB")
        img = transform(img).unsqueeze(0)
        embedding = embedder.image_embedding(img)
        
        embeddings[result_idx] = embedding
        samples[result_idx] = frame_idx
        
        result_idx += 1
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Lưu embedding và sample vào file
    np.save(embedding_file, embeddings)
    np.save(sample_file, samples)
    

# Hàm videos_context: xử lý tất cả video trong thư mục và trích xuất đặc trưng cho mỗi video
def videos_context(video_folder, embedding_folder, frame_rate=None,
                   model_type='dino', model_kind='base', patch=16,
                   representation='cls', device='cuda'):
    # Khởi tạo embedder
    embedder = Embedder(model_type=model_type,
                        model_kind=model_kind,
                        patch=patch,
                        representation=representation,
                        device=device)

    # Duyệt qua tất cả video trong thư mục và trích xuất đặc trưng
    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):
            generate_context(video_folder, filename, embedding_folder,
                             embedder, frame_rate)


if __name__ == '__main__':
    start_time = time.time()
    
    # Cấu hình các tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Generating Contextual Features of Videos using DINO Embeddings')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='Path to folder to store embeddings')
    
    parser.add_argument('--representation', type=str, default='cls',
                        choices=['cls', 'mean'],
                        help='visual type')
    parser.add_argument('--frame-rate', type=int, 
                        help='Number of frames per second to sample from videos')
    
    # parser.add_argument('--model-name', type=str, default='b16',
    #                     choices=['b16', 'b8', 's16', 's8'],
    #                     help='Name of the DINO model')
    parser.add_argument('--model-type', type=str, default='dino',
                        choices=['dino', 'clip'])
    parser.add_argument('--model-kind', type=str, default='b',
                        choices=['b', 's', 'base'])
    parser.add_argument('--patch', type=int, default=16,
                        choices= [8, 16])
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run the model on')

    args = parser.parse_args()

    # Gọi hàm xử lý các video trong thư mục
    videos_context(video_folder=args.video_folder,
                   embedding_folder=args.embedding_folder,
                   frame_rate=args.frame_rate,
                   model_type=args.model_type,
                   model_kind=args.model_kind,
                   patch=args.patch,
                   representation=args.representation,
                   device=args.device,
                   )
    
    # In thời gian hoàn thành
    print("--- %s seconds ---" % (time.time() - start_time))
