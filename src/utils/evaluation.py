import numpy as np

class RecommendationEvaluator:
    """Class for evaluating recommendation models using standard ranking metrics."""

    def __init__(self, k=10):
        """
        Initialize evaluator with a specific cutoff (k).
        :param k: Top-K items to consider for metrics.
        """
        self.k = k

    def precision_at_k(self, recommended, relevant):
        """Calculate Precision@K."""
        recommended = recommended[:self.k]
        return len(set(recommended) & set(relevant)) / self.k

    def recall_at_k(self, recommended, relevant):
        """Calculate Recall@K."""
        recommended = recommended[:self.k]
        return len(set(recommended) & set(relevant)) / len(relevant) if relevant else 0

    def ndcg_at_k(self, recommended, relevant):
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
        def dcg(scores):
            return sum((2**scores[i] - 1) / np.log2(i + 2) for i in range(len(scores)))
        
        relevance_scores = [1 if item in relevant else 0 for item in recommended[:self.k]]
        ideal_scores = sorted(relevance_scores, reverse=True)
        
        actual_dcg = dcg(relevance_scores)
        ideal_dcg = dcg(ideal_scores)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

    def mean_average_precision(self, recommended, relevant):
        """Calculate Mean Average Precision (MAP)."""
        avg_precisions = []
        hits = 0
        for i, item in enumerate(recommended[:self.k]):
            if item in relevant:
                hits += 1
                avg_precisions.append(hits / (i + 1))
        return np.mean(avg_precisions) if avg_precisions else 0

    def mean_reciprocal_rank(self, recommended, relevant):
        """Calculate Mean Reciprocal Rank (MRR)."""
        for i, item in enumerate(recommended[:self.k]):
            if item in relevant:
                return 1 / (i + 1)
        return 0

    def evaluate(self, recommended, relevant):
        """Evaluate a recommendation list using multiple metrics."""
        return {
            "Precision@K": self.precision_at_k(recommended, relevant),
            "Recall@K": self.recall_at_k(recommended, relevant),
            "NDCG@K": self.ndcg_at_k(recommended, relevant),
            "MAP": self.mean_average_precision(recommended, relevant),
            "MRR": self.mean_reciprocal_rank(recommended, relevant),
        }

