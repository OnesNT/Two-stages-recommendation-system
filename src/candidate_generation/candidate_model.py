import numpy as np
import pandas as pd
import torch
from src.candidate_generation.matrix_factorization import MatrixFactorization
from src.utils import helpers
from src.utils.data_loader import InteractionDataset
from src.candidate_generation.embeddings import train_embeddings, save_embeddings, load_embeddings  
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = helpers.load_config("config.yaml")
candidate_config = helpers.load_config("candidate_config.yaml")
file_path = candidate_config['data']['interaction_data_path']

# Load interaction matrix
# Load the CSV file without specifying dtype, and set the first column as the index
interaction_matrix = pd.read_csv(file_path, index_col=0)

# Convert the DataFrame to numeric, handling any non-numeric values
interaction_matrix = interaction_matrix.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

num_users, num_items = interaction_matrix.shape
embedding_dim = candidate_config['model']['embedding_dim']

# Prepare dataset and model
dataset = InteractionDataset(interaction_matrix.values)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim).to(device)

# Paths to save embeddings
user_embedding_path = "data/embeddings/user_embeddings.npy"
item_embedding_path = "data/embeddings/item_embeddings.npy"

# Check if embeddings are precomputed and saved
try:
    user_embeddings, item_embeddings = load_embeddings(user_embedding_path, item_embedding_path)
except FileNotFoundError:
    print("Precomputed embeddings not found. Training model to generate embeddings.")
    train_embeddings(model, dataloader, num_epochs=100, learning_rate=0.01, reg=0.1)
    save_embeddings(model, user_embedding_path, item_embedding_path)
    user_embeddings = model.user_embedding.weight.detach().cpu().numpy()
    item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

# Candidate generation using precomputed embeddings
def generate_candidates_from_mf(user_id, top_n=100):
    user_embedding = user_embeddings[user_id]
    
    # Calculate scores for each item by taking the dot product with item embeddings
    scores = np.dot(item_embeddings, user_embedding)

    # Sort items by score and get the top-N items
    top_item_indices = np.argsort(scores)[-top_n:][::-1]
    top_items = [(index, scores[index]) for index in top_item_indices]

    # Return top-N item IDs as candidates
    return top_items

# Main candidate generation function (currently using only MF)
def generate_candidates(user_id, top_n=100):
    mf_candidates = generate_candidates_from_mf(user_id, top_n)
    combined_candidates = list(set(mf_candidates))
    return combined_candidates[:top_n]
