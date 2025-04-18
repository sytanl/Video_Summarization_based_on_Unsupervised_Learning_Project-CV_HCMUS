import os
import sys
import cv2
from scripts.application import VidSum


def get_video_duration_seconds(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video tại: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(total_frames / fps) if fps > 0 else 0


def summarize_video(video_filename, input_dir, output_dir):
    video_path = os.path.join(input_dir, video_filename)
    duration = get_video_duration_seconds(video_path)

    vidsum = VidSum()
    vidsum.set_params(
        input_frame_rate=4,
        method="ours",
        distance="cosine",
        max_length=duration // 3,
        modulation=1e-4,
        intermediate_components=50,
        window_size=5,
        min_seg_length=5,
        reduced_emb=True,
        scoring_mode="uniform",
        kf_mode=["middle", "ends"],
        bias=0.5,
        output_frame_rate=4,
        sum_rate=0.4,
        extension="mp4"
    )

    summary_path = vidsum.summarize(video_path, output_dir, flag=True)
    print(f"Video tóm tắt đã lưu tại: {summary_path}")


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python summarize_video.py video1.mp4 [video2.mp4 ...]")
        return

    input_dir = "data/videos"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    video_list = sys.argv[1:]  # Lấy danh sách các video từ command line

    for video_file in video_list:
        try:
            summarize_video(video_file, input_dir, output_dir)
        except Exception as e:
            print(f"Lỗi khi xử lý {video_file}: {e}")


if __name__ == "__main__":
    main()
