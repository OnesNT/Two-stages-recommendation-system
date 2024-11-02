import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


def train_embeddings(model, interactions, num_epochs=100, learning_rate=0.01, reg=0.1):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=reg)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for user, item, rating in interactions:
            user_tensor = torch.LongTensor([user])
            item_tensor = torch.LongTensor([item])
            rating_tensor = torch.FloatTensor([rating])

            optimizer.zero_grad()
            prediction = model(user_tensor, item_tensor)
            loss = loss_fn(prediction, rating_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

def save_embeddings(model, user_save_path, item_save_path):
    np.save(user_save_path, model.user_embedding.weight.detach().numpy())
    np.save(item_save_path, model.item_embedding.weight.detach().numpy())
    print("User and item embeddings saved.")

def load_embeddings(user_save_path, item_save_path):
    user_embeddings = np.load(user_save_path)
    item_embeddings = np.load(item_save_path)
    print("User and item embeddings loaded.")
    return user_embeddings, item_embeddings