# Video Summarization based on Unsupervised Learning

Dự án này là phiên bản đã sửa đổi thực hiện tóm tắt video dựa trên phương pháp học không giám sát dựa trên phương pháp gốc [TAC-SUM](https://github.com/hcmus-thesis-gulu/TAC-SUM/tree/main) (Temporal-Aware Cluster-based SUMmarization) và thể hiện kết quả F-measure avg có được cao hơn phương pháp gốc sau khi cải tiến cách tính điểm khung hình quan trọng.

## Cài đặt môi trường
1. **Tạo và kích hoạt môi trường ảo Conda**:
    ```bash
    conda create -n vidsum python=3.10 -y
    conda activate vidsum
2. **Clone repository và chuyển vào thư mục dự án**:
   ```bash
   git clone https://github.com/sytanl/Video_Summarization_based_on_Unsupervised_Learning_Project-CV_HCMUS.git
   cd Video_Summarization_based_on_Unsupervised_Learning_Project-CV_HCMUS
3. **Cài đặt các thư viện cần thiết:**:
    ```bash
    pip install -r requirements.txt
4. **Cài đặt tiện ích dos2unix để chuyển đổi định dạng dòng nếu chưa có**:
    ```bash 
    sudo apt install dos2unix
5. **Chuyển đổi script download_summe.sh sang định dạng Unix**:
    ```bash
    dos2unix data/download_summe.sh
6. **Tải dữ liệu video mẫu (SumMe), sau tải xong dữ liệu lưu ở thư mục videos**:
    ```bash
    bash data/download_summe.sh
## Đánh giá F-measure avg giữa Phương pháp TAC-SUM cũ và Phương pháp TAC-SUM cải tiến (kèm minh chứng sau khi cải tiến có được)
1. **Thiết lập `PYTHONPATH` để dự án có thể truy cập đúng các thư viện**:
    ```bash
    export PYTHONPATH=$(pwd)   
2. **Chuyển đổi script preprocess.sh sang định dạng Unix**:
    ```bash
    dos2unix preprocess.sh
3. **Tiền xử lý dữ liệu video đầu vào (thư mục output sẽ là *Test*)**:
    ```bash
    bash preprocess.sh videos Test
- Thư mục `Test` sẽ chứa:
     - `Test/embeddings/`: chứa các đặc trưng đã trích xuất từ 25 video đầu vào, bao gồm:
         - `embeddings.npy`: chứa toàn bộ embedding vectors của các khung hình của video.
         - `samples.npy`: chứa thông tin về các khung hình ở video tương ứng với từng embedding (ví dụ: chỉ số thời gian hoặc đoạn clip).
         - `reduced.npy`: chứa embedding sau khi đã được giảm chiều (dimensionality reduction), dùng để tăng hiệu suất tính toán.
    - `Test/contexts/`: chứa các thông tin về các đoạn video và nhãn phân loại, bao gồm:
         - `labels.npy`: chứa nhãn phân loại cho từng khung hình (ví dụ: hành động, cảnh).
         - `segments.npy`: chứa các khung hình được phân loại là quan trọng, có ý nghĩa.
4. **Chuyển đổi script infer.sh sang định dạng Unix**:
    ```bash
    dos2unix infer.sh
5. **Chạy mô hình suy luận để tóm tắt video (thư mục output sẽ là *Test*)**: Tinh chỉnh giá trị của tham số k tùy ý (tham số được nhắc tới trong mục *Tính điểm khung hình* của phần *Phương pháp tiếp cận* của báo cáo)
    ```bash
    bash infer.sh videos Test
 - Thư mục `Test` sẽ chứa:
     - `Test/summaries/`: chứa thông tin tóm tắt video, bao gồm:
         - `scores.npy`: chứa điểm số thể hiện mức độ quan trọng của từng frame của video, dùng để chọn ra các phần nổi bật cho bản tóm tắt.


> Ngoài ra thư mục Test chứa sẵn 4 thư mục:
> - `Test/embeddings_base`: Thư mục embeddings đã được tạo sẵn
> - `Test/contexts_base`: Thư mục contexts đã được tạo sẵn
> - `Test/summaries_0`: Thư mục summaries đã được tạo sẵn khi chạy file `infer.sh` với tham số k = 0.
> - `Test/summaries_8`: Thư mục summaries đã được tạo sẵn khi chạy file `infer.sh` với tham số k =8.
6. **Thực hiện chạy Notebook `run_evaluation.ipynb` để tìm điểm F-measure avg cao nhất thông qua hypersearch**.
* **Chạy ô notebook đầu tiên**: Sau khi đã chạy xong file `preprocess.sh` và `infer.sh` để tạo folder summaries trong thư mục Test, chạy ô notebook đầu tiên để thấy kết quả F-measure avg.
* **Ô notebook thứ hai**: ô này chạy như ô đầu tiên nhưng chỉ khác ở chỗ sử dụng folder summaries_0 đã được tạo từ trước khi chạy file `infer.sh` với tham số k = 0 (tính điểm frame theo công thức phương pháp TAC-SUM ban đầu), ô này đã được chạy sẵn và kết quả chạy được hiện ở dưới ô.
* **Ô notebook thứ ba**: ô này chạy như ô đầu tiên nhưng chỉ khác ở chỗ sử dụng folder summaries_8 đã được tạo từ trước khi chạy file `infer.sh` với tham số k = 8 (cho kết quả F-measure avg cao nhất), ô này đã được chạy sẵn và kết quả chạy được hiện ở dưới ô.

## Tạo video tóm tắt
> Đảm bảo đã hoàn tất tải video mẫu từ bước *Cài đặt môi trường* ở trên 

Chạy file `create_summarized_video.py` để tạo video tóm tắt từ tên video trong thư mục videos (dataset SUMME):

    python create_summarized_video.py video_name
* Ví dụ tóm tắt một video:
    ```bash
    python create_summarized_video.py "Air_Force_One.mp4"
* Ví dụ tóm tắt nhiều video:
    ```bash
    python create_summarized_video.py "Air_Force_One.mp4" "Cooking.mp4" "Scuba.mp4"
> Tên 25 Video trong Thư mục `videos`  
    1. Air_Force_One.mp4  
    2. Base jumping.mp4  
    3. Bearpark_climbing.mp4  
    4. Bike Polo.mp4  
    5. Bus_in_Rock_Tunnel.mp4  
    6. car_over_camera.mp4  
    7. Car_railcrossing.mp4  
    8. Cockpit_Landing.mp4  
    9. Cooking.mp4  
    10. Eiffel Tower.mp4  
    11. Excavators river crossing.mp4  
    12. Fire Domino.mp4  
    13. Jumps.mp4  
    14. Kids_playing_in_leaves.mp4  
    15. Notre_Dame.mp4  
    16. Paintball.mp4  
    17. paluma_jump.mp4  
    18. playing_ball.mp4  
    19. Playing_on_water_slide.mp4  
    20. Saving dolphines.mp4  
    21. Scuba.mp4  
    22. St Maarten Landing.mp4  
    23. Statue of Liberty.mp4  
    24. Uncut_Evening_Flight.mp4  
    25. Valparaiso_Downhill.mp4  

Video sau khi tóm tắt sẽ được lưu vào thư mục **output/video_name** (video_name là tên file) dưới dạng file.mp4.

