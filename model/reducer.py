from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

class Reducer:
    def __init__(self, intermediate_components=-1,
                 perplexity=30, num_components=-1, final_reducer='tsne'):
        # Khởi tạo các tham số cho quá trình giảm chiều dữ liệu
        self.intermediate_components = intermediate_components
        self.perplexity = perplexity
        self.num_components = num_components
        self.intermediate = intermediate_components != -1  # Kiểm tra có cần giảm chiều ban đầu không
        self.final = num_components != -1  # Kiểm tra có cần giảm chiều cuối cùng không
        self.final_reducer = final_reducer  # Phương pháp giảm chiều cuối cùng ('tsne' hoặc 'pca')
        
        # Nếu có cần giảm chiều ban đầu, sử dụng PCA
        if self.intermediate:
            self.pre_reducer = PCA(n_components=self.intermediate_components)
        
        # Chọn phương pháp giảm chiều cuối cùng (t-SNE hoặc PCA)
        if final_reducer == 'tsne':
            self.reducer = TSNE(n_components=self.num_components,
                                perplexity=self.perplexity,
                                metric='cosine'
                                )
        else:
            self.reducer = PCA(n_components=self.num_components)

    def reduce(self, embeddings, flag):
        # Nếu không có yêu cầu giảm chiều cuối cùng, trả lại dữ liệu ban đầu
        if not self.final:
            return embeddings, embeddings
        
        if not flag:
            # Nếu flag là False, chỉ thực hiện giảm chiều cho dữ liệu, không có thêm bước nào
            if self.intermediate:
                embeddings = self.pre_reducer.fit_transform(embeddings)
            return embeddings, self.reducer.fit_transform(embeddings)
        else:
            # Lấy số lượng mẫu và số đặc trưng của embeddings
            n_samples, n_features = embeddings.shape
            print(f"Embedding shape: {n_samples} samples, {n_features} features")

            # Nếu có yêu cầu giảm chiều ban đầu (PCA)
            if self.intermediate:
                max_components = min(n_samples, n_features)
                n_components = min(self.intermediate_components, max_components)
                print(f"Using PCA as pre-reducer with {n_components} components")
                self.pre_reducer = PCA(n_components=n_components)
                embeddings = self.pre_reducer.fit_transform(embeddings)

            # Tiến hành giảm chiều cuối cùng (t-SNE hoặc PCA)
            if self.final_reducer == 'tsne':
                # Nếu số lượng đặc trưng lớn hơn 50, cần giảm xuống 50 trước khi áp dụng t-SNE
                if embeddings.shape[1] > 50:
                    print(f"Reducing to 50 dims before t-SNE (current: {embeddings.shape[1]})")
                    self.pre_reducer = PCA(n_components=50)
                    embeddings = self.pre_reducer.fit_transform(embeddings)

                print(f"Using t-SNE as final reducer with {self.num_components} components and perplexity {self.perplexity}")
                self.reducer = TSNE(
                    n_components=min(self.num_components, embeddings.shape[1]),
                    perplexity=self.perplexity,
                    metric='cosine',
                    method='exact',
                    init='random',
                    learning_rate='auto'
                )
            else:
                max_components = min(embeddings.shape[0], embeddings.shape[1])
                n_components = min(self.num_components, max_components)
                print(f"Using PCA as final reducer with {n_components} components")
                self.reducer = PCA(n_components=n_components)
            
            return embeddings, self.reducer.fit_transform(embeddings)

class Scaler:
    def __init__(self):
        # Khởi tạo đối tượng StandardScaler để chuẩn hóa dữ liệu
        self.scaler = preprocessing.StandardScaler()
        
    def predict(self, embeddings):
        # Thực hiện chuẩn hóa dữ liệu
        return self.scaler.fit_transform(embeddings)
