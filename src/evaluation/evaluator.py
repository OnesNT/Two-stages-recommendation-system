import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
from sklearn.metrics import ndcg_score

class RecommenderEvaluator:
    def __init__(self, metrics: List[str], k_values: List[int]):
        """
        Initialize the evaluator with specified metrics and k values.
        
        Args:
            metrics: List of metric names to compute
            k_values: List of k values for evaluation
        """
        self.metrics = metrics
        self.k_values = k_values
        self.metric_functions = {
            'hit_rate': self._compute_hit_rate,
            'map': self._compute_map,
            'ndcg': self._compute_ndcg
        }
        
    def _compute_hit_rate(self, predictions: List[int], ground_truth: List[int], k: int) -> float:
        """Compute Hit Rate@K"""
        return 1.0 if any(pred in ground_truth for pred in predictions[:k]) else 0.0
    
    def _compute_map(self, predictions: List[int], ground_truth: List[int], k: int) -> float:
        """Compute Mean Average Precision@K"""
        if not ground_truth:
            return 0.0
            
        ap = 0.0
        hits = 0
        
        for i, pred in enumerate(predictions[:k]):
            if pred in ground_truth:
                hits += 1
                ap += hits / (i + 1)
                
        return ap / min(len(ground_truth), k)
    
    def _compute_ndcg(self, predictions: List[int], ground_truth: List[int], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain@K"""
        if not ground_truth:
            return 0.0
            
        dcg = 0.0
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        
        for i, pred in enumerate(predictions[:k]):
            if pred in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
                
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(self, model, test_data: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained recommendation model
            test_data: DataFrame containing test data with columns ['user_id', 'item_id']
            
        Returns:
            Dictionary of metrics and their values at different k
        """
        results = {metric: {k: [] for k in self.k_values} for metric in self.metrics}
        
        # Group test data by user
        user_items = test_data.groupby('user_id')['item_id'].apply(list).to_dict()
        
        print("\n📊 Evaluating model performance...")
        for user_id, ground_truth in tqdm(user_items.items(), desc="Users"):
            try:
                # Get model predictions
                predictions = model.recommend(user_id, top_n=max(self.k_values))
                
                # Compute metrics
                for metric in self.metrics:
                    metric_func = self.metric_functions[metric]
                    for k in self.k_values:
                        score = metric_func(predictions, ground_truth, k)
                        results[metric][k].append(score)
                        
            except Exception as e:
                print(f"Warning: Error processing user {user_id}: {str(e)}")
                continue
        
        # Compute average scores
        final_results = {}
        for metric in self.metrics:
            final_results[metric] = {
                k: np.mean(scores) for k, scores in results[metric].items()
            }
            
        return final_results
    
    def save_results(self, results: Dict[str, Dict[int, float]], output_dir: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary of evaluation results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        
        # Convert numpy types to Python native types
        results_json = {
            metric: {str(k): float(v) for k, v in scores.items()}
            for metric, scores in results.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=4)
            
        print(f"\n✅ Evaluation results saved to {output_file}")
        
        # Print results in a readable format
        print("\n📈 Evaluation Results:")
        for metric, scores in results.items():
            print(f"\n{metric.upper()}:")
            for k, score in scores.items():
                print(f"  @{k}: {score:.4f}")

    def calculate_hit_rate(
        self,
        recommended_items: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@k.
        
        Args:
            recommended_items (List[int]): List of recommended item IDs
            relevant_items (List[int]): List of relevant (ground truth) item IDs
            k (int): Number of items to consider
            
        Returns:
            float: Hit Rate@k
        """
        if not relevant_items:
            return 0.0
            
        hits = len(set(recommended_items[:k]) & set(relevant_items))
        return float(hits > 0)
        
    def calculate_map(
        self,
        recommended_items: List[int],
        relevant_items: List[int],
        k: int
    ) -> float:
        """
        Calculate Mean Average Precision@k.
        
        Args:
            recommended_items (List[int]): List of recommended item IDs
            relevant_items (List[int]): List of relevant (ground truth) item IDs
            k (int): Number of items to consider
            
        Returns:
            float: MAP@k
        """
        if not relevant_items:
            return 0.0
            
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended_items[:k], 1):
            if item in relevant_items:
                hits += 1
                sum_precisions += hits / i
                
        return sum_precisions / min(len(relevant_items), k)
        
    def calculate_ndcg(
        self,
        recommended_items: List[int],
        item_ratings: Dict[int, float],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.
        
        Args:
            recommended_items (List[int]): List of recommended item IDs
            item_ratings (Dict[int, float]): Dictionary of item ratings
            k (int): Number of items to consider
            
        Returns:
            float: NDCG@k
        """
        if not item_ratings:
            return 0.0
            
        # Create relevance lists for recommended items
        y_true = np.zeros(k)
        y_pred = np.zeros(k)
        
        # Fill relevance scores for recommended items
        for i, item in enumerate(recommended_items[:k]):
            y_pred[i] = item_ratings.get(item, 0.0)
            
        # Sort true relevance scores for ideal DCG
        y_true = np.sort(y_pred)[::-1]
        
        return ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1))
        
    def evaluate_recommendations(
        self,
        recommendations: List[Tuple[List[int], List[int], Dict[int, float]]],
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate recommendations using multiple metrics.
        
        Args:
            recommendations (List[Tuple]): List of (recommended_items, relevant_items, ratings) tuples
            
        Returns:
            Dict[str, Dict[int, float]]: Dictionary of metric scores for different k values
        """
        try:
            results = {
                'hit_rate': {k: [] for k in self.k_values},
                'map': {k: [] for k in self.k_values},
                'ndcg': {k: [] for k in self.k_values}
            }
            
            for rec_items, rel_items, ratings in recommendations:
                for k in self.k_values:
                    # Calculate Hit Rate
                    hr = self.calculate_hit_rate(rec_items, rel_items, k)
                    results['hit_rate'][k].append(hr)
                    
                    # Calculate MAP
                    map_score = self.calculate_map(rec_items, rel_items, k)
                    results['map'][k].append(map_score)
                    
                    # Calculate NDCG
                    ndcg = self.calculate_ndcg(rec_items, ratings, k)
                    results['ndcg'][k].append(ndcg)
                    
            # Calculate averages
            final_results = {
                metric: {
                    k: np.mean(scores) for k, scores in k_results.items()
                }
                for metric, k_results in results.items()
            }
            
            return final_results
            
        except Exception as e:
            logging.error(f"Error in evaluation: {str(e)}")
            return {
                'hit_rate': {k: 0.0 for k in self.k_values},
                'map': {k: 0.0 for k in self.k_values},
                'ndcg': {k: 0.0 for k in self.k_values}
            }
            
    def evaluate_emotion_alignment(
        self,
        recommended_emotions: List[Dict[str, float]],
        user_emotions: List[Dict[str, float]]
    ) -> float:
        """
        Evaluate emotional alignment of recommendations.
        
        Args:
            recommended_emotions (List[Dict[str, float]]): Emotions of recommended items
            user_emotions (List[Dict[str, float]]): User's emotional states
            
        Returns:
            float: Average emotional alignment score
        """
        try:
            alignment_scores = []
            
            for rec_emotion, user_emotion in zip(recommended_emotions, user_emotions):
                # Convert emotion dictionaries to vectors
                rec_vector = np.array(list(rec_emotion.values()))
                user_vector = np.array(list(user_emotion.values()))
                
                # Calculate cosine similarity
                similarity = np.dot(rec_vector, user_vector) / (
                    np.linalg.norm(rec_vector) * np.linalg.norm(user_vector)
                )
                
                alignment_scores.append(similarity)
                
            return np.mean(alignment_scores)
            
        except Exception as e:
            logging.error(f"Error in emotion alignment evaluation: {str(e)}")
            return 0.0 