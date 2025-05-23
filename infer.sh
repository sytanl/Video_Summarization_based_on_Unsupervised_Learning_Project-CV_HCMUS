#!/bin/bash

# Kiểm tra xem đã truyền đúng 2 đối số chưa: đường dẫn thư mục video và thư mục output
if [ $# -ne 2 ]; then
    echo "Cách dùng: $0 đường_dẫn_thư_mục_video đường_dẫn_thư_mục_output"
    exit 1
fi

# Tạo thư mục lưu kết quả tóm tắt bên trong thư mục output
mkdir -p "$2/summaries"

# Chạy script summarization.py để tạo video tóm tắt dựa trên kết quả embedding và phân đoạn ngữ nghĩa trước đó, trong đó:
# embedding-folder là thư mục chứa các embedding đã trích xuất
# context-folder là thư mục chứa thông tin ngữ cảnh đã phân đoạn
# summary-folder là thư mục để lưu điểm của các frame của video
# reduced-emb là biến boolean có sử dụng version rút gọn của embedding (đã giảm chiều trước đó)
# scoring-mode là cách chấm điểm quan trọng cho các đoạn video: phân phối đều (uniform)
# kf-mode là cách chọn keyframe: chọn frame giữa và 2 frame đầu/cuối của mỗi đoạn
# k là tham số điều chỉnh công thức tính điểm cho frame
python scripts/summarization.py \
    --embedding-folder "$2/embeddings" \
    --context-folder "$2/contexts" \
    --summary-folder "$2/summaries" \
    --reduced-emb \
    --scoring-mode "uniform" \
    --kf-mode "middle ends" \
    --k 8
