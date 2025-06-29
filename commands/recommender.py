import sys
import os
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from recommenders.itemCF.itemCF import ItemCF


class RecommenderCommands:
    def __init__(self):
        self.models = {
            'itemcf': ItemCF  
        }

    def train_model(self, model_type, matrix_path):
        """
        Train a model from a preprocessed sparse matrix (.pkl) file.
        """
        start_time = time.time()
        print("\n🚀 Starting model training process...")

        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")

        # ✅ Load preprocessed sparse matrix
        print(f"\n📥 Loading sparse matrix from {matrix_path}...")
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)

        user_item_matrix = data['matrix']
        user_encoder = data['user_encoder']
        item_encoder = data['item_encoder']

        print(f"✅ Matrix loaded successfully:")
        print(f"   - Shape: {user_item_matrix.shape}")
        print(f"   - Users: {len(user_encoder.classes_)}")
        print(f"   - Items: {len(item_encoder.classes_)}")
        print(f"   - Non-zero elements: {user_item_matrix.nnz}")
        print(f"   - Sparsity: {(user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.8f}%")

        # ✅ Train model
        try:
            print(f"\n🤖 Initializing {model_type} model...")
            model = self.models[model_type]()
            
            print("\n⚙️ Starting model training...")
            model.fit(user_item_matrix)
            
            training_time = time.time() - start_time
            print(f"\n✅ Model training completed in {training_time:.2f} seconds")
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")

        # ✅ Save model
        print("\n💾 Saving model...")
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{model_type}_model.pkl")
        model_bundle = {
            'model': model,
            'user_encoder': user_encoder,
            'item_encoder': item_encoder
        }

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_bundle, f)
            print(f"✅ Model saved successfully to {model_path}")
            return model_path
        except Exception as e:
            raise IOError(f"Failed to save model: {str(e)}")
