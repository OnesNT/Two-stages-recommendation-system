import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import logging

class EmotionAwareRetriever:
    def __init__(self, config: Dict):
        """
        Initialize the emotion-aware retriever.
        
        Args:
            config (Dict): Configuration dictionary containing parameters
        """
        self.config = config
        self.n_factors = config.get('n_factors', 50)
        self.n_candidates = config.get('n_candidates', 100)
        
    def fit(self, user_item_matrix: np.ndarray, emotion_matrix: np.ndarray) -> None:
        """
        Fit the retriever model using user-item interactions and emotion data.
        
        Args:
            user_item_matrix (np.ndarray): User-item interaction matrix
            emotion_matrix (np.ndarray): Matrix of user emotion features
        """
        self.user_item_matrix = user_item_matrix
        self.emotion_matrix = emotion_matrix
        
        # Perform matrix factorization
        U, sigma, Vt = svds(user_item_matrix, k=self.n_factors)
        self.user_factors = U
        self.item_factors = Vt.T
        
    def get_user_emotion_similarity(self, user_idx: int, emotion_state: Dict[str, float]) -> np.ndarray:
        """
        Calculate similarity between user's current emotional state and other users.
        
        Args:
            user_idx (int): Index of the target user
            emotion_state (Dict[str, float]): Current emotional state of the user
            
        Returns:
            np.ndarray: Array of similarity scores
        """
        current_emotion = np.array(list(emotion_state.values())).reshape(1, -1)
        return cosine_similarity(current_emotion, self.emotion_matrix)[0]
        
    def get_similar_items(self, item_idx: int) -> np.ndarray:
        """
        Find similar items based on latent factors.
        
        Args:
            item_idx (int): Index of the target item
            
        Returns:
            np.ndarray: Array of similarity scores
        """
        item_vec = self.item_factors[item_idx].reshape(1, -1)
        return cosine_similarity(item_vec, self.item_factors)[0]
        
    def retrieve_candidates(
        self,
        user_idx: int,
        emotion_state: Dict[str, float],
        item_history: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidate items for the user considering their emotional state.
        
        Args:
            user_idx (int): Index of the target user
            emotion_state (Dict[str, float]): Current emotional state of the user
            item_history (Optional[List[int]]): List of items already interacted with
            
        Returns:
            List[Tuple[int, float]]: List of (item_idx, score) tuples
        """
        try:
            # Get emotion-based user similarities
            user_emotion_sim = self.get_user_emotion_similarity(user_idx, emotion_state)
            
            # Get collaborative filtering scores
            cf_scores = self.user_factors[user_idx].dot(self.item_factors.T)
            
            # Combine scores from similar users
            similar_users = np.argsort(user_emotion_sim)[::-1][:10]  # Top 10 similar users
            similar_users_scores = np.mean(
                [self.user_factors[u].dot(self.item_factors.T) for u in similar_users],
                axis=0
            )
            
            # Combine different scores
            final_scores = 0.7 * cf_scores + 0.3 * similar_users_scores
            
            # Filter out items in history
            if item_history:
                final_scores[item_history] = -np.inf
                
            # Get top candidates
            top_items = np.argsort(final_scores)[::-1][:self.n_candidates]
            candidates = [(int(idx), float(final_scores[idx])) for idx in top_items]
            
            return candidates
            
        except Exception as e:
            logging.error(f"Error in candidate retrieval: {str(e)}")
            return []
            
    def update_model(self, user_idx: int, item_idx: int, rating: float, emotion: Dict[str, float]) -> None:
        """
        Update the model with new user interaction and emotion data.
        
        Args:
            user_idx (int): Index of the user
            item_idx (int): Index of the item
            rating (float): Rating given by the user
            emotion (Dict[str, float]): Emotional state during the interaction
        """
        try:
            # Update user-item matrix
            self.user_item_matrix[user_idx, item_idx] = rating
            
            # Update emotion matrix
            self.emotion_matrix[user_idx] = list(emotion.values())
            
            # Recompute matrix factorization
            U, sigma, Vt = svds(self.user_item_matrix, k=self.n_factors)
            self.user_factors = U
            self.item_factors = Vt.T
            
        except Exception as e:
            logging.error(f"Error in model update: {str(e)}") 