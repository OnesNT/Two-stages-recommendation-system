import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Any
import logging

class EmotionAwareRanker:
    def __init__(self, config: Dict):
        """
        Initialize the emotion-aware ranker.
        
        Args:
            config (Dict): Configuration dictionary containing parameters
        """
        self.config = config
        self.model = None
        self.feature_names = None
        
    def _create_feature_vector(
        self,
        user_features: Dict[str, Any],
        item_features: Dict[str, Any],
        retrieval_score: float,
        emotion_state: Dict[str, float]
    ) -> np.ndarray:
        """
        Create a feature vector for ranking.
        
        Args:
            user_features (Dict[str, Any]): User features
            item_features (Dict[str, Any]): Item features
            retrieval_score (float): Score from retrieval stage
            emotion_state (Dict[str, float]): Current emotional state
            
        Returns:
            np.ndarray: Feature vector
        """
        features = []
        
        # User features
        features.extend([
            user_features.get('avg_rating', 0.0),
            user_features.get('n_ratings', 0),
            user_features.get('rating_variance', 0.0)
        ])
        
        # Item features
        features.extend([
            item_features.get('avg_rating', 0.0),
            item_features.get('n_ratings', 0),
            item_features.get('rating_variance', 0.0)
        ])
        
        # Retrieval score
        features.append(retrieval_score)
        
        # Emotion features
        features.extend(list(emotion_state.values()))
        
        return np.array(features)
        
    def prepare_training_data(
        self,
        user_features: Dict[int, Dict],
        item_features: Dict[int, Dict],
        interactions: List[Tuple[int, int, float, Dict[str, float]]],
        retrieval_scores: Dict[Tuple[int, int], float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the ranking model.
        
        Args:
            user_features (Dict[int, Dict]): Dictionary of user features
            item_features (Dict[int, Dict]): Dictionary of item features
            interactions (List[Tuple]): List of (user_id, item_id, rating, emotion) tuples
            retrieval_scores (Dict[Tuple, float]): Dictionary of retrieval scores
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels for training
        """
        X, y = [], []
        
        for user_id, item_id, rating, emotion in interactions:
            try:
                feature_vector = self._create_feature_vector(
                    user_features[user_id],
                    item_features[item_id],
                    retrieval_scores.get((user_id, item_id), 0.0),
                    emotion
                )
                
                X.append(feature_vector)
                y.append(rating)
                
            except Exception as e:
                logging.warning(f"Error preparing training data: {str(e)}")
                continue
                
        return np.array(X), np.array(y) 

    def fit(
        self,
        user_features: Dict[int, Dict],
        item_features: Dict[int, Dict],
        interactions: List[Tuple[int, int, float, Dict[str, float]]],
        retrieval_scores: Dict[Tuple[int, int], float]
    ) -> None:
        """
        Train the ranking model.
        
        Args:
            user_features (Dict[int, Dict]): Dictionary of user features
            item_features (Dict[int, Dict]): Dictionary of item features
            interactions (List[Tuple]): List of (user_id, item_id, rating, emotion) tuples
            retrieval_scores (Dict[Tuple, float]): Dictionary of retrieval scores
        """
        try:
            X, y = self.prepare_training_data(
                user_features,
                item_features,
                interactions,
                retrieval_scores
            )
            
            # Define feature names
            self.feature_names = [
                'user_avg_rating', 'user_n_ratings', 'user_rating_variance',
                'item_avg_rating', 'item_n_ratings', 'item_rating_variance',
                'retrieval_score',
                'joy', 'sadness', 'anger', 'fear', 'surprise', 'love'
            ]
            
            # Train LightGBM model
            train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': self.config.get('learning_rate', 0.05),
                'num_leaves': self.config.get('num_leaves', 31),
                'feature_fraction': self.config.get('feature_fraction', 0.9),
                'bagging_fraction': self.config.get('bagging_fraction', 0.8),
                'bagging_freq': self.config.get('bagging_freq', 5),
                'verbose': -1
            }
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.config.get('num_boost_round', 100),
                valid_sets=[train_data],
                early_stopping_rounds=self.config.get('early_stopping_rounds', 10),
                verbose_eval=False
            )
            
        except Exception as e:
            logging.error(f"Error training ranking model: {str(e)}")
            raise
            
    def rank_candidates(
        self,
        user_id: int,
        candidates: List[Tuple[int, float]],
        user_features: Dict[int, Dict],
        item_features: Dict[int, Dict],
        emotion_state: Dict[str, float]
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate items for a user.
        
        Args:
            user_id (int): Target user ID
            candidates (List[Tuple]): List of (item_id, retrieval_score) tuples
            user_features (Dict[int, Dict]): Dictionary of user features
            item_features (Dict[int, Dict]): Dictionary of item features
            emotion_state (Dict[str, float]): Current emotional state
            
        Returns:
            List[Tuple[int, float]]: Ranked list of (item_id, score) tuples
        """
        try:
            if not self.model:
                raise ValueError("Model not trained")
                
            features = []
            item_ids = []
            
            for item_id, retrieval_score in candidates:
                feature_vector = self._create_feature_vector(
                    user_features[user_id],
                    item_features[item_id],
                    retrieval_score,
                    emotion_state
                )
                features.append(feature_vector)
                item_ids.append(item_id)
                
            features = np.array(features)
            scores = self.model.predict(features)
            
            # Combine items with their scores and sort
            ranked_items = list(zip(item_ids, scores))
            ranked_items.sort(key=lambda x: x[1], reverse=True)
            
            return ranked_items
            
        except Exception as e:
            logging.error(f"Error ranking candidates: {str(e)}")
            return [(item_id, score) for item_id, score in candidates]  # Fallback to retrieval scores
            
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model:
            self.model.save_model(model_path)
            
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = lgb.Booster(model_file=model_path) 