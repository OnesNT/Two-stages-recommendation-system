import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        return (user_embedded * item_embedded).sum(1)

    def get_user_embedding(self, user_id):
        with torch.no_grad():
            return self.user_embedding(torch.LongTensor([user_id])).numpy()

    def get_item_embedding(self, item_id):
        with torch.no_grad():
            return self.item_embedding(torch.LongTensor([item_id])).numpy()