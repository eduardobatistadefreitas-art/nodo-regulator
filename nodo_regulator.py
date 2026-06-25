"""
Módulo Proprietário NODORegulator.
Finalidade: Redução de dimensionalidade linear O(N) e filtragem de ruído global em grafos.
Autor: Engenharia de Dados
"""

import numpy as np
from scipy.sparse import diags, issparse
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class NODORegulator(BaseEstimator, TransformerMixin):
    """
    Filtro espectral baseado em Laplaciano normalizado simetricamente.
    Utiliza aproximação estocástica por projeções de âncoras para garantir 
    complexidade linear O(N) em cenários de alta escala.
    Aceita matrizes densas numpy e esparsas scipy (csr, csc).
    """
    def __init__(self, n_components=5, k_neighbors=20, batch_size=20000):
        self.n_components = n_components
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """
        Calcula o Laplaciano Normalizado Simetricamente e as projeções espectrais.
        """
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True)
        else:
            X = check_array(X, accept_sparse=['csr', 'csc'], ensure_2d=True)

        if self.n_components <= 0 or self.k_neighbors <= 0:
            raise ValueError("Os parâmetros n_components e k_neighbors devem ser inteiros positivos.")

        n_samples, n_features = X.shape

        if n_samples > self.batch_size:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.neighbors import NearestNeighbors

            n_anchors = min(self.batch_size // 4, 2000)
            
            X_dense = X.toarray() if issparse(X) else X
            kmeans = MiniBatchKMeans(n_clusters=n_anchors, batch_size=10000, random_state=42, n_init="auto")
            kmeans.fit(X_dense)
            anchors = kmeans.cluster_centers_

            nn_anchor = NearestNeighbors(n_neighbors=self.k_neighbors, metric="euclidean").fit(anchors)
            A_bridge = nn_anchor.kneighbors_graph(X, mode='connectivity')

            W = A_bridge.T.dot(A_bridge)
            degrees_w = np.array(W.sum(axis=1)).flatten()
            degrees_w[degrees_w == 0] = 1e-12
            D_w_inv = diags(1.0 / np.sqrt(degrees_w))
            L_anchor = diags(np.ones(n_anchors)) - D_w_inv.dot(W).dot(D_w_inv)

            _, vecs_anchor = eigsh(L_anchor, k=self.n_components + 1, which='SM')

            row_sums = np.array(A_bridge.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1e-12
            D_bridge_inv = diags(1.0 / row_sums)

            self.embedding_ = D_bridge_inv.dot(A_bridge).dot(vecs_anchor[:, 1:self.n_components + 1])
        
        else:
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(X, n_neighbors=self.k_neighbors, mode='connectivity', include_self=False)
            A = 0.5 * (A + A.T)

            degrees = np.array(A.sum(axis=1)).flatten()
            degrees[degrees == 0] = 1e-12
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            D_inv_sqrt = diags(d_inv_sqrt)

            I = diags(np.ones(n_samples))
            L_norm = I - D_inv_sqrt.dot(A).dot(D_inv_sqrt)

            vals, vecs = eigsh(L_norm, k=self.n_components + 1, which='SM')
            self.embedding_ = vecs[:, 1:self.n_components + 1]

        self.X_train_ = X
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """
        Projeta novos dados no espaço espectral reduzido (Extensão Out-of-Sample).
        """
        check_is_fitted(self, attributes=['embedding_', 'X_train_'])
        X = check_array(X, accept_sparse=['csr', 'csc'], ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Dimensão de features incompatível. Esperado {self.n_features_in_}, recebido {X.shape}")

        if not issparse(X) and not issparse(self.X_train_) and np.array_equal(X, self.X_train_):
            return self.embedding_

        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=self.k_neighbors, metric="euclidean").fit(self.X_train_)
        A_test = nn.kneighbors_graph(X, mode='connectivity')

        row_sums = np.array(A_test.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1e-12
        D_test_inv = diags(1.0 / row_sums)

        return D_test_inv.dot(A_test).dot(self.embedding_)
