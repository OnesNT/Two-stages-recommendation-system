import numpy as np
from collections import defaultdict

class SwingRecommender:
    def __init__(self, alpha=1):
        """
        Initializes the Swing recommender.
        :param alpha: Smoothing coefficient to control weight distribution.
        """
        self.alpha = alpha
        self.user_item_dict = defaultdict(set)  # Maps users to items
        self.item_user_dict = defaultdict(set)  # Maps items to users
        
    def fit(self, interactions):
        """
        Fits the model with user-item interactions.
        :param interactions: List of (user, item) interactions.
        """
        for user, item in interactions:
            self.user_item_dict[user].add(item)
            self.item_user_dict[item].add(user)
        
    def compute_swing_scores(self):
        """
        Computes Swing scores for item pairs.
        :return: Dictionary with (item_i, item_j) as keys and Swing scores as values.
        """
        swing_scores = defaultdict(float)
        
        for item_i in self.item_user_dict:
            for item_j in self.item_user_dict:
                if item_i >= item_j:
                    continue  # Avoid duplicate calculations
                
                common_users = self.item_user_dict[item_i] & self.item_user_dict[item_j]
                if not common_users:
                    continue
                
                score = 0
                for user_a in common_users:
                    for user_b in common_users:
                        if user_a == user_b:
                            continue
                        
                        common_items = self.user_item_dict[user_a] & self.user_item_dict[user_b]
                        score += 1 / (self.alpha + len(common_items))
                
                swing_scores[(item_i, item_j)] = score
                swing_scores[(item_j, item_i)] = score  # Symmetric
                
        return swing_scores
    
    def recommend(self, item, top_n=5):
        """
        Recommends items based on Swing scores.
        :param item: Target item to find recommendations for.
        :param top_n: Number of recommendations to return.
        :return: List of recommended items sorted by Swing score.
        """
        swing_scores = self.compute_swing_scores()
        related_items = {k[1]: v for k, v in swing_scores.items() if k[0] == item}
        return sorted(related_items.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Example usage:
data = [(1, 'A'), (1, 'B'), (2, 'A'), (2, 'C'), (3, 'B'), (3, 'C'), (4, 'A'), (4, 'D')]
recommender = SwingRecommender(alpha=1)
recommender.fit(data)
scores = recommender.compute_swing_scores()
print(scores)
print(recommender.recommend('A'))