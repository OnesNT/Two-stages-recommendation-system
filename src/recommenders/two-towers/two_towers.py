import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import faiss  # ANN Library for fast similarity search
import numpy as np

class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, embedding_dim=128, hidden_layers=[256, 128]):
        super(TwoTowerModel, self).__init__()
        
        # User tower
        self.user_tower = self.build_tower(user_input_dim, embedding_dim, hidden_layers)
        
        # Item tower
        self.item_tower = self.build_tower(item_input_dim, embedding_dim, hidden_layers)
        
    def build_tower(self, input_dim, embedding_dim, hidden_layers):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))  # Final embedding layer
        return nn.Sequential(*layers)
    
    def forward(self, user_features, item_features):
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        
        # Normalize embeddings
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        
        # Compute similarity (cosine similarity is dot product after normalization)
        scores = (user_embedding * item_embedding).sum(dim=1)
        return scores, user_embedding, item_embedding

# Loss function based on softmax
class TwoTowerLoss(nn.Module):
    def forward(self, scores, positive_idx):
        return F.cross_entropy(scores, positive_idx)

# FAISS-based ANN Retrieval Class
class ANNRetriever:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
        self.item_embeddings = None
        self.item_ids = None
    
    def build_index(self, item_embeddings, item_ids):
        self.item_embeddings = np.array(item_embeddings)
        self.item_ids = np.array(item_ids)
        self.index.add(self.item_embeddings)
    
    def search(self, user_embedding, k=10):
        user_embedding = np.array(user_embedding).reshape(1, -1)
        distances, indices = self.index.search(user_embedding, k)
        return [self.item_ids[i] for i in indices[0]]

# Example usage
def train_example():
    batch_size = 32
    user_dim, item_dim = 50, 50  # Example feature dimensions
    model = TwoTowerModel(user_dim, item_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = TwoTowerLoss()
    
    # Dummy data
    user_features = torch.randn(batch_size, user_dim)
    item_features = torch.randn(batch_size, item_dim)
    negative_samples = torch.randn(batch_size * 5, item_dim)  # 5 negatives per positive
    
    # Compute scores
    scores, user_emb, item_emb = model(user_features, item_features)
    neg_scores, _, _ = model(user_features.repeat_interleave(5, dim=0), negative_samples)
    
    # Softmax over positives & negatives
    all_scores = torch.cat([scores.unsqueeze(1), neg_scores.view(batch_size, -1)], dim=1)
    positive_idx = torch.zeros(batch_size, dtype=torch.long)  # Positive sample at index 0
    
    # Compute loss
    loss = loss_fn(all_scores, positive_idx)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Loss:", loss.item())

    # Store item embeddings in FAISS
    ann = ANNRetriever(embedding_dim=128)
    item_ids = np.arange(len(item_emb.detach().numpy()))
    ann.build_index(item_emb.detach().numpy(), item_ids)
    
    # Retrieve top-5 items for a user
    user_embedding = user_emb[0].detach().numpy()
    recommended_items = ann.search(user_embedding, k=5)
    print("Recommended Items:", recommended_items)

# Run example training step
train_example()