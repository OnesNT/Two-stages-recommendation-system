import torch
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
import numpy as np
import time

class ItemCF:
    def __init__(self, top_k=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.top_k = top_k
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.num_users = 0
        self.num_items = 0
        print(f"🚀 Using device: {self.device}")

    def _compute_item_norms(self, matrix):
        """Compute L2 norm for each item (column) in sparse matrix"""
        return np.sqrt(matrix.power(2).sum(axis=0)).A1 + 1e-8  # avoid division by zero

    def fit(self, user_item_matrix: csr_matrix):
        start_time = time.time()
        self.num_users, self.num_items = user_item_matrix.shape
        print(f"\n📊 Training ItemCF model:")
        print(f"   - Users: {self.num_users}")
        print(f"   - Items: {self.num_items}")
        print(f"   - Density: {(user_item_matrix.nnz / (self.num_users * self.num_items)) * 100:.6f}%")

        self.user_item_matrix = user_item_matrix
        item_norms = self._compute_item_norms(user_item_matrix)

        # Initialize top-K sparse similarity matrix
        similarity = lil_matrix((self.num_items, self.num_items))

        print("\n⚙️ Computing item-item similarity (sparse top-K)...")
        for i in tqdm(range(self.num_items), desc="Items"):
            item_vec = user_item_matrix[:, i].toarray().flatten()
            dot_product = user_item_matrix.T.dot(item_vec).flatten()
            similarities = dot_product / (item_norms[i] * item_norms)

            # Remove self-similarity
            similarities[i] = -np.inf

            # Get top-K indices
            top_k_indices = np.argpartition(-similarities, self.top_k)[:self.top_k]
            top_k_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]

            for j in top_k_sorted:
                similarity[i, j] = similarities[j]

        # Convert to CSR for efficient indexing
        self.similarity_matrix = similarity.tocsr()

        duration = time.time() - start_time
        print(f"\n✅ Training finished in {duration:.2f} seconds")
        print(f"   - Sparse matrix size: {self.similarity_matrix.shape}")
        print(f"   - Non-zero similarities: {self.similarity_matrix.nnz}")

    def recommend(self, item_id: int, top_n=10):
        """
        Recommend top-N items similar to the given item ID.
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call `fit()` first.")
        # print(f"🔍 Recommendation for item {item_id}")
        # print(f"🔍 Similarity matrix shape: {self.similarity_matrix.shape}")
        sim_scores = self.similarity_matrix[item_id].toarray().flatten()
        top_n_indices = np.argsort(-sim_scores)[:top_n]
        return top_n_indices.tolist()
