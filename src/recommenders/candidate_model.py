import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

from .base import BaseRecommender
from .matrix_factorization import MatrixFactorizationRecommender
from .user_cf import UserBasedCF
from .item_cf import ItemBasedCF

class CandidateGenerator:
    """Combines multiple recommenders for candidate generation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the candidate generator.
        
        Args:
            config (Dict): Configuration dictionary containing settings for each recommender
        """
        self.config = config
        self.recommenders: Dict[str, BaseRecommender] = {}
        self.weights: Dict[str, float] = config.get('weights', {
            'mf': 0.4,
            'user_cf': 0.3,
            'item_cf': 0.3
        })
        
        # Initialize recommenders
        self._init_recommenders()
        
    def _init_recommenders(self) -> None:
        """Initialize all recommender models."""
        try:
            # Matrix Factorization
            if 'mf' in self.weights:
                self.recommenders['mf'] = MatrixFactorizationRecommender(
                    self.config.get('mf', {})
                )
                
            # User-Based CF
            if 'user_cf' in self.weights:
                self.recommenders['user_cf'] = UserBasedCF(
                    self.config.get('user_cf', {})
                )
                
            # Item-Based CF
            if 'item_cf' in self.weights:
                self.recommenders['item_cf'] = ItemBasedCF(
                    self.config.get('item_cf', {})
                )
                
        except Exception as e:
            logging.error(f"Error initializing recommenders: {str(e)}")
            raise
            
    def fit(self, user_item_matrix: np.ndarray) -> None:
        """
        Train all recommender models.
        
        Args:
            user_item_matrix (np.ndarray): User-item interaction matrix
        """
        for name, recommender in self.recommenders.items():
            try:
                logging.info(f"Training {name} recommender...")
                recommender.fit(user_item_matrix)
            except Exception as e:
                logging.error(f"Error training {name} recommender: {str(e)}")
                raise
                
    def generate_candidates(
        self,
        user_id: int,
        n_items: int = 100,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate candidate items using multiple recommenders.
        
        Args:
            user_id (int): Target user ID
            n_items (int): Number of candidates to generate
            exclude_items (Optional[List[int]]): Items to exclude
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, score) tuples
        """
        try:
            # Collect recommendations from each model
            all_recommendations = defaultdict(float)
            
            for name, recommender in self.recommenders.items():
                weight = self.weights[name]
                recommendations = recommender.recommend(
                    user_id,
                    n_items=n_items,
                    exclude_items=exclude_items
                )
                
                # Aggregate scores with weights
                for item_id, score in recommendations:
                    all_recommendations[item_id] += score * weight
                    
            # Sort and return top candidates
            candidates = [
                (item_id, score)
                for item_id, score in all_recommendations.items()
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates[:n_items]
            
        except Exception as e:
            logging.error(f"Error generating candidates: {str(e)}")
            return []
            
    def get_similar_items(
        self,
        item_id: int,
        n_items: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Find similar items using item-based CF.
        
        Args:
            item_id (int): Target item ID
            n_items (int): Number of similar items to return
            exclude_items (Optional[List[int]]): Items to exclude
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, similarity) tuples
        """
        if 'item_cf' not in self.recommenders:
            return []
            
        return self.recommenders['item_cf'].get_similar_items(
            item_id,
            n_items=n_items,
            exclude_items=exclude_items
        )
        
    def add_recommender(
        self,
        name: str,
        recommender: BaseRecommender,
        weight: float
    ) -> None:
        """
        Add a new recommender to the ensemble.
        
        Args:
            name (str): Name of the recommender
            recommender (BaseRecommender): Recommender instance
            weight (float): Weight for this recommender
        """
        self.recommenders[name] = recommender
        self.weights[name] = weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight 