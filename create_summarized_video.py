import os
import sys
import cv2
from scripts.application import VidSum

# Hàm lấy thời lượng video tính bằng giây
def get_video_duration_seconds(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video tại: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Tổng số khung hình
    fps = cap.get(cv2.CAP_PROP_FPS)  # Số khung hình/giây
    cap.release()
    return int(total_frames / fps) if fps > 0 else 0  # Thời lượng = tổng khung / fps


# Hàm tóm tắt một video cụ thể
def summarize_video(video_filename, input_dir, output_dir):
    video_path = os.path.join(input_dir, video_filename)
    duration = get_video_duration_seconds(video_path)

    vidsum = VidSum()  # Khởi tạo đối tượng tóm tắt video
    vidsum.set_params(
        input_frame_rate=4,               # Tốc độ khung hình đầu vào (4 FPS)
        method="ours",                    # Phương pháp phân đoạn: dùng thuật toán được đề cập trong báo cáo
        distance="cosine",                # Khoảng cách dùng trong clustering (cosine)
        max_length=duration // 3,         # Độ dài tối đa video tóm tắt (1/3 thời lượng gốc)
        modulation=1e-4,                  # Hệ số điều chỉnh độ nhạy trong segmentation
        intermediate_components=50,       # Số chiều trung gian sau giảm chiều PCA
        window_size=5,                    # Kích thước cửa sổ để xét đoạn liền kề
        min_seg_length=5,                 # Độ dài tối thiểu của một đoạn
        reduced_emb=True,                 # Dùng embedding đã giảm chiều
        scoring_mode="uniform",           # Chấm điểm đoạn bằng cách phân phối đều
        kf_mode=["middle", "ends"],       # Lựa chọn keyframe: giữa, đầu/cuối
        bias=0.5,                         # Tham số điều chỉnh điểm ưu tiên giữa các frame
        output_frame_rate=4,              # FPS đầu ra cho video tóm tắt
        sum_rate=0.4,                     # Tỷ lệ độ dài video tóm tắt / video gốc
        extension="mp4",                  # Định dạng xuất ra
        k=8                               # Tham số điều chỉnh công thức tính điểm cho frame
    )

    summary_path = vidsum.summarize(video_path, output_dir, flag=True)
    print(f"Video tóm tắt đã lưu tại: {summary_path}")


# Hàm chính: xử lý danh sách các video truyền từ dòng lệnh
def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python summarize_video.py video1.mp4 [video2.mp4 ...]")
        return

    input_dir = "videos"       # Thư mục chứa video gốc
    output_dir = "output"      # Thư mục để lưu video tóm tắt
    os.makedirs(output_dir, exist_ok=True)

    video_list = sys.argv[1:]  # Lấy danh sách tên file video từ dòng lệnh

    for video_file in video_list:
        try:
            summarize_video(video_file, input_dir, output_dir)
        except Exception as e:
            print(f"Lỗi khi xử lý {video_file}: {e}")


if __name__ == "__main__":
    main()
