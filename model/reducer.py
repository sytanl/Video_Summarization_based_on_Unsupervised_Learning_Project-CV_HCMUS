from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing


class Reducer:
    def __init__(self, intermediate_components=-1,
                 perplexity=30, num_components=-1, final_reducer='tsne'):
        self.intermediate_components = intermediate_components
        self.perplexity = perplexity
        self.num_components = num_components
        self.intermediate = intermediate_components != -1
        self.final = num_components != -1
        self.final_reducer = final_reducer
        # exit()
        # print("HIHIHIHIHIHIHIHIHIHIHIHIHIHI: ", self.intermediate, self.final)
        
        if self.intermediate:
            self.pre_reducer = PCA(n_components=self.intermediate_components)
        
        if final_reducer == 'tsne':
            self.reducer = TSNE(n_components=self.num_components,
                                perplexity=self.perplexity,
                                metric='cosine'
                                )
        else:
            self.reducer = PCA(n_components=self.num_components)

    def reduce(self, embeddings, flag):
        if not self.final:
            return embeddings, embeddings
        
        if not flag:
            if self.intermediate:
                embeddings = self.pre_reducer.fit_transform(embeddings)
                
            return embeddings, self.reducer.fit_transform(embeddings)
        else:
            n_samples, n_features = embeddings.shape
            print(f"Embedding shape: {n_samples} samples, {n_features} features")

            # Intermediate PCA (pre-reduction)
            if self.intermediate:
                max_components = min(n_samples, n_features)
                n_components = min(self.intermediate_components, max_components)
                print(f"Using PCA as pre-reducer with {n_components} components")
                self.pre_reducer = PCA(n_components=n_components)
                embeddings = self.pre_reducer.fit_transform(embeddings)


            # Final reducer
            if self.final_reducer == 'tsne':
                # t-SNE max input features should be <= 50
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
        self.scaler = preprocessing.StandardScaler()
        
    def predict(self, embeddings):
        return self.scaler.fit_transform(embeddings)
