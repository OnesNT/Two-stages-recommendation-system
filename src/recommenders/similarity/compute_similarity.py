import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMetric:
    def __init__(self, metric='cosine'):
        """
        Initialize the similarity metric.
        
        Parameters:
        metric (str): The similarity metric to use. Supported metrics: 'cosine', 'pearson', 'jaccard'.
        """
        self.metric = metric

    def compute(self, user_item_matrix):
        """
        Compute the similarity matrix based on the chosen metric.
        
        Parameters:
        user_item_matrix (numpy.ndarray): A matrix where rows represent users and columns represent items.
                                          The values represent the interaction strength (e.g., ratings, clicks).
        
        Returns:
        numpy.ndarray: A user-user similarity matrix.
        """
        if self.metric == 'cosine':
            return cosine_similarity(user_item_matrix)
        elif self.metric == 'pearson':
            return self._pearson_similarity(user_item_matrix)
        elif self.metric == 'jaccard':
            return self._jaccard_similarity(user_item_matrix)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")

    def _pearson_similarity(self, user_item_matrix):
        """
        Compute Pearson correlation similarity between users.
        """
        # Normalize the user-item matrix by subtracting the mean
        user_mean = np.nanmean(user_item_matrix, axis=1, keepdims=True)
        normalized_matrix = user_item_matrix - user_mean
        
        # Compute cosine similarity on the normalized matrix
        return cosine_similarity(normalized_matrix)

    def _jaccard_similarity(self, user_item_matrix):
        """
        Compute Jaccard similarity between users.
        Jaccard similarity is defined as the size of the intersection divided by the size of the union of the sets of items.
        """
        num_users = user_item_matrix.shape[0]
        similarity_matrix = np.zeros((num_users, num_users))
        
        for i in range(num_users):
            for j in range(num_users):
                # Get the sets of items interacted with by users i and j
                set_i = set(np.where(user_item_matrix[i] > 0)[0])
                set_j = set(np.where(user_item_matrix[j] > 0)[0])
                
                # Compute Jaccard similarity
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                
                if union == 0:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = intersection / union
        
        return similarity_matrix