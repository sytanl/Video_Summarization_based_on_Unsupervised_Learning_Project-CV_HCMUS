#!/bin/bash

# Kiểm tra xem đã truyền đúng 2 đối số chưa: đường dẫn thư mục video và thư mục output
if [ $# -ne 2 ]; then
    echo "Cách dùng: $0 đường_dẫn_thư_mục_video đường_dẫn_thư_mục_output"
    exit 1
fi

# Tạo thư mục lưu kết quả tóm tắt bên trong thư mục output
mkdir -p "$2/summaries"

# Chạy script summarization.py để tạo video tóm tắt dựa trên kết quả embedding và phân đoạn ngữ nghĩa trước đó
python scripts/summarization.py \
--embedding-folder "$2/embeddings" \         # Thư mục chứa các embedding đã trích xuất
--context-folder "$2/contexts" \             # Thư mục chứa thông tin ngữ cảnh đã phân đoạn
--summary-folder "$2/summaries" \            # Thư mục để lưu điểm của các frame của video
--reduced-emb \                              # Sử dụng version rút gọn của embedding (đã giảm chiều trước đó)
--scoring-mode "uniform" \                   # Cách chấm điểm quan trọng cho các đoạn video: phân phối đều (uniform)
--kf-mode "middle ends" \                    # Cách chọn keyframe: chọn frame giữa và 2 frame đầu/cuối của mỗi đoạn
--k 8                                        # Tham số điều chỉnh công thức tính điểm cho frame
