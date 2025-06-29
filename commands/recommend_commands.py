import pickle

def execute(args):
    """Handle recommendation commands"""
    print(f"🔍 Loading model from {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✨ Generating {args.top_n} recommendations for item {args.item_id}")
    recommendations = model.recommend(args.item_id, args.top_n)
    
    print("\n📋 Recommendations:")
    for i, item in enumerate(recommendations, 1):
        print(f"{i}. {item}")