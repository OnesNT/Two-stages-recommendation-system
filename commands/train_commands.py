from .recommender import RecommenderCommands

def execute(args):
    """Handle training commands"""
    print(f"🚀 Training {args.model_type} model with {args.matrix_path}")

    cmd = RecommenderCommands()
    model_path = cmd.train_model(args.model_type, args.matrix_path)
    print(f"✅ Model saved to {model_path}")