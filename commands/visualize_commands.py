import json
from src.visualization.visualize_results import visualize_results

def execute(args):
    """Handle recommendation commands"""
    # print(f"🔍 Loading model from {args.model_path}")
    print(f"🔍 Loading model's results from {args.json_file}")
    with open(args.json_file, 'rb') as f:
        results = json.load(f)
    print(type(results))
    visualize_results(results, args.output_dir, args.plot_type)

