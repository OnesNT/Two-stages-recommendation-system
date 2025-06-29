import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .base import BaseRecommender

class UserBasedCF(BaseRecommender):
    """User-Based Collaborative Filtering recommender."""
    
    def __init__(self, config: Dict):
        """
        Initialize the User-CF recommender.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - n_neighbors: Number of similar users to consider
                - min_similarity: Minimum similarity threshold
        """
        super().__init__(config)
        self.n_neighbors = config.get('n_neighbors', 20)
        self.min_similarity = config.get('min_similarity', 0.1)
        self.user_item_matrix = None
        self.user_similarity = None
        
    def fit(self, user_item_matrix: np.ndarray) -> None:
        """
        Train the User-CF model.
        
        Args:
            user_item_matrix (np.ndarray): User-item interaction matrix
        """
        try:
            self.user_item_matrix = user_item_matrix
            
            # Compute user similarity matrix
            self.user_similarity = cosine_similarity(user_item_matrix)
            
            # Set self-similarity to 0 to avoid recommending items the user already interacted with
            np.fill_diagonal(self.user_similarity, 0)
            
        except Exception as e:
            logging.error(f"Error in User-CF training: {str(e)}")
            raise
            
    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): Target user ID
            n_items (int): Number of items to recommend
            exclude_items (Optional[List[int]]): Items to exclude from recommendations
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, score) tuples
        """
        try:
            if self.user_item_matrix is None or self.user_similarity is None:
                raise ValueError("Model not trained")
                
            # Get similar users
            user_sim = self.user_similarity[user_id]
            similar_users = np.argsort(user_sim)[::-1][:self.n_neighbors]
            
            # Filter by minimum similarity
            similar_users = [
                u for u in similar_users
                if user_sim[u] >= self.min_similarity
            ]
            
            if not similar_users:
                return []
                
            # Get items from similar users
            similar_users_matrix = self.user_item_matrix[similar_users]
            similar_users_weights = user_sim[similar_users].reshape(-1, 1)
            
            # Compute weighted average of similar users' ratings
            scores = np.sum(similar_users_matrix * similar_users_weights, axis=0)
            scores /= np.sum(similar_users_weights)
            
            # Set scores of items the user has already interacted with to -inf
            user_items = self.user_item_matrix[user_id] > 0
            scores[user_items] = -np.inf
            
            # Exclude additional items if necessary
            scores = self._exclude_items(scores, exclude_items)
            
            # Get top items
            top_item_indices = np.argsort(scores)[::-1][:n_items]
            recommendations = [
                (int(idx), float(scores[idx]))
                for idx in top_item_indices
                if scores[idx] > -np.inf
            ]
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error in User-CF recommendation: {str(e)}")
            return [] 