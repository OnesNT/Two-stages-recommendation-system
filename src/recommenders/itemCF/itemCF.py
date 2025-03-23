import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class ItemCF:
    def __init__(self):
        """
        Initialize the ItemCF model.
        """
        self.item_weights = None
        self.item_similarity = None

    def fit(self, user_item_matrix):
        """
        Fit the model to the user-item matrix.
        
        Parameters:
        user_item_matrix (numpy.ndarray or scipy.sparse.csr_matrix): A matrix where rows represent users and columns represent items.
                                                                    The values represent the interaction strength (e.g., ratings, clicks).
        """
        if isinstance(user_item_matrix, np.ndarray):
            user_item_matrix = csr_matrix(user_item_matrix)
        
        self.num_users, self.num_items = user_item_matrix.shape
        
        # Compute item weights w_i
        self.item_weights = self._compute_item_weights(user_item_matrix)
        
        # Compute item-item similarity matrix
        self.item_similarity = self._compute_item_similarity(user_item_matrix)

    def _compute_item_weights(self, user_item_matrix):
        """
        Compute the weight w_i for each item.
        
        Parameters:
        user_item_matrix (scipy.sparse.csr_matrix): The user-item interaction matrix.
        
        Returns:
        numpy.ndarray: A vector of item weights.
        """
        # Number of users who interacted with each item
        item_interaction_counts = np.array(user_item_matrix.astype(bool).sum(axis=0)).flatten()
        
        # Compute w_i = sum_{u in U_i} 1 / |I_u|
        item_weights = np.zeros(self.num_items)
        for u in range(self.num_users):
            user_interactions = user_item_matrix[u].nonzero()[1]  # Items interacted with by user u
            num_interactions = len(user_interactions)
            if num_interactions > 0:
                item_weights[user_interactions] += 1 / num_interactions
        
        return item_weights

    def _compute_item_similarity(self, user_item_matrix):
        """
        Compute the item-item similarity matrix.
        
        Parameters:
        user_item_matrix (scipy.sparse.csr_matrix): The user-item interaction matrix.
        
        Returns:
        numpy.ndarray: A matrix of item-item similarity scores.
        """
        # Initialize the similarity matrix
        item_similarity = np.zeros((self.num_items, self.num_items))
        
        # Compute co-occurrence and similarity scores
        for u in range(self.num_users):
            user_interactions = user_item_matrix[u].nonzero()[1]  # Items interacted with by user u
            num_interactions = len(user_interactions)
            
            if num_interactions > 0:
                # Update similarity scores for all pairs of items interacted with by user u
                for i in user_interactions:
                    for j in user_interactions:
                        if i != j:
                            item_similarity[i, j] += 1 / (num_interactions * np.sqrt(self.item_weights[i] * self.item_weights[j]))
        
        return item_similarity

    def recommend(self, item_id, top_n=10):
        """
        Recommend top N related items for a given item.
        
        Parameters:
        item_id (int): The ID of the item for which to generate recommendations.
        top_n (int): The number of top items to recommend.
        
        Returns:
        list: A list of recommended item IDs.
        """
        # Get the similarity scores for the item
        item_sim_scores = self.item_similarity[item_id]
        
        # Sort and get the top N items (excluding the item itself)
        recommended_items = np.argsort(item_sim_scores)[-top_n-1:-1][::-1]
        
        return recommended_items