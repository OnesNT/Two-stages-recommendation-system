import numpy as np
from recommenders.similarity.compute_similarity import SimilarityMetric

class UserKNN:
    def __init__(self, k=5, similarity_metric='cosine'):
        """
        Initialize the UserKNN model.
        
        Parameters:
        k (int): The number of nearest neighbours to consider.
        similarity_metric (str): The similarity metric to use. Supported metrics: 'cosine', 'pearson', 'jaccard'.
        """
        self.k = k
        self.similarity_metric = SimilarityMetric(metric=similarity_metric)
        self.user_similarity = None
        self.user_item_matrix = None
        self.num_users = None
        self.num_items = None

    def fit(self, user_item_matrix):
        """
        Fit the model to the user-item matrix.
        
        Parameters:
        user_item_matrix (scipy.sparse.csr_matrix): A sparse matrix where rows represent users and columns represent items.
                                                    The values represent the interaction strength (e.g., ratings, clicks).
        """
        self.user_item_matrix = user_item_matrix
        self.num_users, self.num_items = user_item_matrix.shape
        
        # Compute user-user similarity matrix
        self.user_similarity = self.similarity_metric.compute(user_item_matrix)
        print(f"User similarity matrix shape: {self.user_similarity.shape}")

    def predict(self, user_id):
        """
        Predict items for a given user based on their nearest neighbours.
        
        Parameters:
        user_id (int): The ID of the user for whom to generate recommendations.
        
        Returns:
        numpy.ndarray: A list of item scores for the user.
        """
        # Get the similarity scores for the user
        user_sim_scores = self.user_similarity[user_id]
        print(f"User similarity scores shape: {user_sim_scores.shape}")
        
        # Find the k nearest neighbours
        nearest_neighbours = np.argsort(user_sim_scores)[-self.k-1:-1][::-1]
        print(f"Nearest neighbours: {nearest_neighbours}")
        
        # Initialize the item scores
        item_scores = np.zeros(self.num_items)
        print(f"Item scores shape: {item_scores.shape}")
        
        # Compute the weighted sum of interactions from nearest neighbours
        for neighbour in nearest_neighbours:
            neighbour_interactions = self.user_item_matrix[neighbour].toarray().flatten()
            print(f"Neighbour {neighbour} interactions shape: {neighbour_interactions.shape}")
            item_scores += user_sim_scores[neighbour] * neighbour_interactions
        
        # Exclude items already interacted with by the user
        user_interactions = self.user_item_matrix[user_id].toarray().flatten()
        print(f"User interactions shape: {user_interactions.shape}")
        item_scores[user_interactions > 0] = 0
        
        return item_scores

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