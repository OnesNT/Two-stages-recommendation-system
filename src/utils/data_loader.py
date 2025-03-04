import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class InteractionDataset(Dataset):
    def __init__(self, interaction_matrix):
        self.user_item_pairs = []
        self.ratings = []
    
        for user in range(interaction_matrix.shape[0]):
            for item in range(interaction_matrix.shape[1]):
                if interaction_matrix[user, item] > 0:  
                    self.user_item_pairs.append((user, item))
                    self.ratings.append(interaction_matrix[user, item])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user, item = self.user_item_pairs[idx]
        rating = self.ratings[idx]
        return user, item, rating


class UserDataset(Dataset):
    def __init__(self, user_features):
        self.user_features = user_features

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, idx):
        return self.user_features[idx]
    

class PostDataset(Dataset):
    def __init__(self, posts, labels):
        """
        Args:
            posts (list of str): List of user posts (text data).
            labels (list of int): Corresponding labels for each post.
        """
        self.posts = posts
        self.labels = labels

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        post = self.posts[idx]
        label = self.labels[idx]
        return post, label


class ItemDataset(Dataset):
    def __init__(self, item_features, item_labels=None):
        """
        Args:
            item_features (np.ndarray): 2D array where each row represents an item's features.
            item_labels (list of int, optional): Corresponding labels for each item.
        """
        self.item_features = item_features
        self.item_labels = item_labels

    def __len__(self):
        return len(self.item_features)

    def __getitem__(self, idx):
        features = self.item_features[idx]
        label = self.item_labels[idx] if self.item_labels is not None else None
        return features, label