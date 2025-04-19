import numpy as np

class Selector:
    def __init__(self, window_size=5, min_seg_length=10):
        # window_size: số lượng khung hình dùng để làm mượt nhãn (label smoothing)
        # min_seg_length: độ dài tối thiểu của một đoạn (segment)
        self.window_size = window_size
        self.min_seg_length = min_seg_length

    # Hàm phân đoạn video dựa trên các nhãn đã được làm mượt
    def select(self, labels):
        segments = []   # Danh sách các đoạn có dạng (label, start, end)
        start = 0
        current_label = None
        window_size = min(self.window_size, len(labels))  # Đảm bảo không vượt quá số nhãn

        for i in range(len(labels)):
            # Làm mượt nhãn bằng cách lấy giá trị phổ biến nhất trong cửa sổ trượt
            if i < (window_size // 2):
                left = 0
                right = window_size
            elif i >= len(labels) - (window_size // 2):
                left = len(labels) - window_size
                right = len(labels)
            else:
                left = i - (window_size // 2)
                right = i + (window_size // 2) + 1

            window = labels[left:right]
            label = np.bincount(window).argmax()  # Nhãn phổ biến nhất trong cửa sổ

            # Phân đoạn video dựa trên sự thay đổi nhãn
            if i == 0:
                current_label = label
            elif i == len(labels) - 1:
                # Thêm đoạn cuối cùng
                segments.append((current_label, start, i+1))
            elif label != current_label:
                # Nếu đoạn hiện tại quá ngắn, xử lý để gộp lại hoặc chia lại
                if len(segments) > 0 and i - start < self.min_seg_length:
                    current_label = label

                    if segments[-1][0] == label:
                        # Nếu nhãn mới trùng với đoạn trước, gộp lại
                        segments[-1] = (segments[-1][0], segments[-1][1], i)
                        start = i
                    else:
                        # Nếu khác, chia đoạn làm hai
                        middle = (start + i) // 2
                        segments[-1] = (segments[-1][0], segments[-1][1], middle)
                        start = middle
                else:
                    # Nhãn thay đổi đủ dài để tạo đoạn mới
                    segments.append((current_label, start, i))
                    start = i
                    current_label = label

        # Hậu xử lý: gộp các đoạn liên tiếp có cùng nhãn
        post_segments = []
        current_label = None
        for label, start, end in segments:
            if current_label is None:
                current_label = label
                post_segments.append((current_label, start, end))
            elif label == current_label:
                post_segments[-1] = (current_label, post_segments[-1][1], end)
            else:
                current_label = label
                post_segments.append((current_label, start, end))

        return np.asarray(post_segments)  # Trả về mảng các đoạn có dạng (label, start, end)
