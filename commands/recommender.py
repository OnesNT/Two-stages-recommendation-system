import sys
import os
import pickle
import pandas as pd
from scipy.sparse import csr_matrix

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

        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")

        # ✅ Load preprocessed sparse matrix
        print(f"📥 Loading sparse matrix from {matrix_path}...")
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)

        user_item_matrix = data['matrix']
        user_encoder = data['user_encoder']
        item_encoder = data['item_encoder']

        print(f"✅ Matrix shape: {user_item_matrix.shape}")
        print(f"🧑 Users: {len(user_encoder.classes_)}, 📚 Items: {len(item_encoder.classes_)}")

        # ✅ Train model
        try:
            model = self.models[model_type]()
            model.fit(user_item_matrix)
            print("🤖 Model training completed.")
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")

        # ✅ Save model
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
            print(f"✅ Model saved to {model_path}")
            return model_path
        except Exception as e:
            raise IOError(f"Failed to save model: {str(e)}")
