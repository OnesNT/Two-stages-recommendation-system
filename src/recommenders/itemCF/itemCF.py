import torch
from scipy.sparse import csr_matrix

class ItemCF:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.num_users = 0
        self.num_items = 0

    def fit(self, user_item_matrix: csr_matrix):
        """
        Fit the model using a user-item sparse matrix (scipy.sparse).
        """
        self.num_users, self.num_items = user_item_matrix.shape

        # Convert to dense tensor (if matrix is small enough)
        print("🔁 Converting to torch tensor...")
        dense_matrix = torch.tensor(user_item_matrix.toarray(), dtype=torch.float32, device=self.device)

        # Normalize items (columns) - optional
        item_norms = torch.norm(dense_matrix, dim=0, keepdim=True) + 1e-8
        norm_matrix = dense_matrix / item_norms

        print("⚙️ Computing item-item similarity (dot product)...")
        # Compute item-item cosine similarity
        similarity = torch.matmul(norm_matrix.T, norm_matrix)

        self.similarity_matrix = similarity
        self.user_item_matrix = dense_matrix
        print("✅ Training complete.")

    def recommend(self, item_id: int, top_n=10):
        """
        Recommend top-N items similar to the given item ID.
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Get similarity scores
        sim_scores = self.similarity_matrix[item_id].clone()
        sim_scores[item_id] = -1e9  # exclude itself

        topk = torch.topk(sim_scores, top_n).indices.tolist()
        return topk
