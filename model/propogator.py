from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from model.reducer import Reducer, Scaler
from model.utils import distance_metric, construct_connectivity

class Clusterer:
    def __init__(self, method, distance, num_clusters, embedding_dim, intermediate_components=50, final_reducer='tsne'):
        # Lưu trữ các tham số nhập vào cho các thuật toán phân cụm
        self.method = method
        self.num_clusters = num_clusters
        self.distance = distance
        self.metric = distance_metric(distance)  # Chọn chỉ số khoảng cách dựa trên tham số nhập

        # Khởi tạo các đối tượng Reducer (giảm chiều dữ liệu) và Scaler (chuẩn hóa dữ liệu)
        self.reducer = Reducer(num_components=embedding_dim, intermediate_components=intermediate_components, final_reducer=final_reducer)
        self.scaler = Scaler()

        # Thông báo về phương pháp phân cụm được sử dụng
        if self.method == 'kmeans':
            print(f"Using K-Means with {num_clusters} clusters")
        elif self.method == 'dbscan':
            print(f"Using {self.distance} distance metric for DBSCAN")
        elif self.method == 'agglo':
            # Đặt chế độ linkage cho Agglomerative Clustering tùy theo khoảng cách
            self.linkage = 'ward' if distance == 'euclidean' else 'average'
            print(f"Using {distance} distance metric and {self.linkage} linkage for Agglomerative Clustering")
        elif self.method == 'gaussian':
            print("Using Bayesian inference for Gaussian Mixture Model")
        elif self.method == 'ours':
            print("Using our method")
        else:
            # Kiểm tra xem phương pháp phân cụm có hợp lệ không
            raise ValueError('Invalid clustering method')

    def cluster(self, embeddings, flag):
        # Giảm chiều dữ liệu và chuẩn hóa các embedding
        pre_embeddings, reduced_embeddings = self.reducer.reduce(embeddings, flag)
        print("HIHIHIHIHIHIHIHIHIHIHIHIHIHI: ", pre_embeddings.shape, reduced_embeddings.shape)
        scaled_embeddings = self.scaler.predict(reduced_embeddings)
        
        # Phương pháp phân cụm riêng (ours)
        if self.method == 'ours':
            print(f"Using our method with {self.distance} distance metric and {self.num_clusters} clusters")
            birch_model = Birch(threshold=0.5, n_clusters=None)
            subcluster_labels = birch_model.fit_predict(scaled_embeddings)
            
            # Xây dựng connectivity từ kết quả phân cụm con
            connectivity = construct_connectivity(scaled_embeddings, subcluster_labels)
            
            # Áp dụng AgglomerativeClustering với kết nối
            agglo_model = AgglomerativeClustering(n_clusters=self.num_clusters,
                                                  metric=self.distance,
                                                  linkage="single",
                                                  connectivity=connectivity
                                                  )
            
            labels = agglo_model.fit_predict(scaled_embeddings)
        else:
            # Áp dụng các phương pháp phân cụm khác
            if self.method == 'kmeans':
                model = KMeans(n_clusters=self.num_clusters, n_init='auto')
            elif self.method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10, metric=self.metric)
            elif self.method == 'agglo':
                model = AgglomerativeClustering(n_clusters=self.num_clusters,
                                                metric=self.distance,
                                                linkage=self.linkage
                                                )
            elif self.method == 'gaussian':
                model = BayesianGaussianMixture(n_components=self.num_clusters)
            else:
                raise ValueError('Invalid clustering method')
        
            # Áp dụng mô hình phân cụm đã chọn
            labels = model.fit_predict(scaled_embeddings)
        
        # Cập nhật số cụm khi sử dụng DBSCAN (tự động phát hiện số cụm)
        if self.method == 'dbscan':
            self.num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return labels, pre_embeddings
