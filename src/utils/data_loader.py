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
