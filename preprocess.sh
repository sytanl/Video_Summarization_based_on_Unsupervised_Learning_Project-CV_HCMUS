#!/bin/bash

# Kiểm tra xem đã truyền đủ 2 tham số: đường dẫn thư mục video và thư mục đầu ra chưa
if [ $# -ne 2 ]; then
    echo "Cách dùng: $0 đường_dẫn_thư_mục_video đường_dẫn_thư_mục_output"
    exit 1
fi

# Tạo thư mục embeddings và contexts bên trong thư mục đầu ra nếu chưa tồn tại
mkdir -p "$2/embeddings" "$2/contexts"

# Chạy script context.py để trích xuất embedding từ video, trong đó:
# video-folder là thư mục chứa các video gốc
# embedding-folder là thư mục để lưu vector đặc trưng (embedding) của mỗi frame
# frame-rate là tốc độ lấy mẫu frame từ video: 4 fps (mỗi giây lấy 4 frame)
# representation là kiểu trích xuất đặc trưng: dùng token CLS (class token)
python scripts/context.py \
    --video-folder "$1" \
    --embedding-folder "$2/embeddings" \
    --frame-rate 4 \
    --representation cls

# Chạy script extraction.py để thực hiện phân đoạn video theo ý nghĩa nội dung, trong đó:
# embedding-folder là thư mục chứa các embedding đã trích xuất từ video
# context-folder là thư mục để lưu kết quả phân đoạn video theo ngữ nghĩa
# method ours là phương pháp phân đoạn: dùng thuật toán được đề cập trong báo cáo
# distance cosine là khoảng cách dùng để tính độ tương đồng: cosine similarity
# embedding-dim là giảm số chiều embedding xuống trước khi phân tích
# window-size là kích thước cửa sổ khi phân tích sự thay đổi nội dung: 5 frame
# min-seg-length là độ dài tối thiểu của mỗi đoạn video sau khi phân đoạn là 3 frame
# modulation là hệ số điều chỉnh độ nhạy khi tách đoạn video (tác động đến ngưỡng phân đoạn)
python scripts/extraction.py \
    --embedding-folder "$2/embeddings" \
    --context-folder "$2/contexts" \
    --method ours \
    --distance cosine \
    --embedding-dim 3 \
    --window-size 5 \
    --min-seg-length 3 \
    --modulation 1e-4
