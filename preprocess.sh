#!/bin/bash

# Kiểm tra xem đã truyền đủ 2 tham số: đường dẫn thư mục video và thư mục đầu ra chưa
if [ $# -ne 2 ]; then
    echo "Cách dùng: $0 đường_dẫn_thư_mục_video đường_dẫn_thư_mục_output"
    exit 1
fi

# Tạo thư mục embeddings và contexts bên trong thư mục đầu ra nếu chưa tồn tại
mkdir -p "$2/embeddings" "$2/contexts"

# Chạy script context.py để trích xuất embedding từ video
python scripts/context.py \
--video-folder "$1" \                         # Thư mục chứa các video gốc
--embedding-folder "$2/embeddings" \         # Thư mục để lưu vector đặc trưng (embedding) của mỗi frame
--frame-rate 4 \                              # Tốc độ lấy mẫu frame từ video: 4 fps (mỗi giây lấy 4 frame)
--representation cls                          # Kiểu trích xuất đặc trưng: dùng token CLS (class token)

# Chạy script extraction.py để thực hiện phân đoạn video theo ý nghĩa nội dung
python scripts/extraction.py \
--embedding-folder "$2/embeddings" \         # Thư mục chứa các embedding đã trích xuất từ video
--context-folder "$2/contexts" \             # Thư mục để lưu kết quả phân đoạn video theo ngữ nghĩa
--method ours \                              # Phương pháp phân đoạn: dùng thuật toán được đề cập trong báo cáo
--distance cosine \                          # Khoảng cách dùng để tính độ tương đồng: cosine similarity
--embedding-dim 3 \                          # Giảm số chiều embedding xuống 3 chiều trước khi phân tích
--window-size 5 \                            # Kích thước cửa sổ khi phân tích sự thay đổi nội dung: 5 frame
--min-seg-length 3 \                         # Độ dài tối thiểu của mỗi đoạn video sau khi phân đoạn là 3 frame
--modulation 1e-4                            # Hệ số điều chỉnh độ nhạy khi tách đoạn video (tác động đến ngưỡng phân đoạn)
