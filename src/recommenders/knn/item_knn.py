import numpy as np
from recommenders.similarity.compute_similarity import SimilarityMetric

class ItemKNN:
    def __init__(self, k=5, similarity_metric='cosine'):
        """
        Initialize the ItemKNN model.
        
        Parameters:
        k (int): The number of nearest neighbours to consider.
        similarity_metric (str): The similarity metric to use. Supported metrics: 'cosine', 'pearson', 'jaccard'.
        """
        self.k = k
        self.similarity_metric = SimilarityMetric(metric=similarity_metric)
        self.item_similarity = None
        self.user_item_matrix = None
        self.num_users = None
        self.num_items = None

    def fit(self, user_item_matrix):
        """
        Fit the model to the user-item matrix.
        
        Parameters:
        user_item_matrix (numpy.ndarray or scipy.sparse.csr_matrix): A matrix where rows represent users and columns represent items.
                                                                    The values represent the interaction strength (e.g., ratings, clicks).
        """
        self.user_item_matrix = user_item_matrix
        self.num_users, self.num_items = user_item_matrix.shape
        
        # Compute item-item similarity matrix using the SimilarityMetric class
        self.item_similarity = self.similarity_metric.compute(user_item_matrix.T)
        print(f"Item similarity matrix shape: {self.item_similarity.shape}")

    def predict(self, user_id):
        """
        Predict items for a given user based on item-item similarity.
        
        Parameters:
        user_id (int): The ID of the user for whom to generate recommendations.
        
        Returns:
        numpy.ndarray: A list of item scores for the user.
        """
        # Get the items interacted with by the user
        user_interactions = self.user_item_matrix[user_id].toarray().flatten()
        interacted_items = np.where(user_interactions > 0)[0]
        # print(f"User {user_id} interacted with items: {interacted_items}")

        # Initialize the item scores
        item_scores = np.zeros(self.num_items)
        # print(f"Initial item scores shape: {item_scores.shape}")

        # Compute scores for each item
        for i in range(self.num_items):
            for j in interacted_items:
                if i in self._get_nearest_neighbours(j):
                    item_scores[i] += user_interactions[j] * self.item_similarity[i, j]
            # print(f"Item {i} score: {item_scores[i]}")

        # Exclude items already interacted with by the user
        print(f"User interactions shape: {user_interactions.shape}")
        item_scores[interacted_items] = 0


        return item_scores

    def _get_nearest_neighbours(self, item_id):
        """
        Get the k nearest neighbours of an item.
        
        Parameters:
        item_id (int): The ID of the item.
        
        Returns:
        numpy.ndarray: The IDs of the k nearest neighbours.
        """
        # Get the similarity scores for the item
        item_sim_scores = self.item_similarity[item_id]
        # print(f"Item {item_id} similarity scores: {item_sim_scores}")

        # Find the k nearest neighbours (excluding the item itself)
        nearest_neighbours = np.argsort(item_sim_scores)[-self.k-1:-1][::-1]
        # print(f"Item {item_id} nearest neighbours: {nearest_neighbours}")

        return nearest_neighbours

    def recommend(self, user_id, top_n=10):
        """
        Recommend top N items for a given user.
        
        Parameters:
        user_id (int): The ID of the user for whom to generate recommendations.
        top_n (int): The number of top items to recommend.
        
        Returns:
        list: A list of recommended item IDs.
        """
        item_scores = self.predict(user_id)
        recommended_items = np.argsort(item_scores)[-top_n:][::-1]
        print(f"Top {top_n} recommended items for user {user_id}: {recommended_items}")

        return recommended_items