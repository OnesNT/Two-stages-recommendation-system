from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np

class BaseRecommender(ABC):
    """Base class for all recommender models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the recommender.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def fit(self, user_item_matrix: np.ndarray) -> None:
        """
        Train the recommender model.
        
        Args:
            user_item_matrix (np.ndarray): User-item interaction matrix
        """
        pass
        
    @abstractmethod
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
        pass
        
    def _exclude_items(
        self,
        scores: np.ndarray,
        exclude_items: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Set scores of excluded items to negative infinity.
        
        Args:
            scores (np.ndarray): Item scores
            exclude_items (Optional[List[int]]): Items to exclude
            
        Returns:
            np.ndarray: Modified scores
        """
        if exclude_items:
            scores[exclude_items] = -np.inf
        return scores 