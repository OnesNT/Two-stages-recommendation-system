import argparse
from .train_commands import execute as train_execute
from .evaluate_commands import execute as evaluate_execute
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from .visualize_commands import execute as visualize_execute

def setup_parser():
    parser = argparse.ArgumentParser(description="Two-Stage Recommendation System")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ➤ Train command
    train_parser = subparsers.add_parser('train', help='Train a recommendation model')
    train_parser.add_argument('--model_type', type=str, default='itemcf', help='Type of model (e.g., itemcf)')
    train_parser.add_argument('--matrix_path', type=str, required=True, help='Path to preprocessed .pkl sparse matrix')
    train_parser.set_defaults(func=train_execute)

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pkl)')
    evaluate_parser.add_argument('--test_data', type=str, required=True, help='Path to the test dataset file')
    evaluate_parser.add_argument('--metrics', type=str, nargs='+', default=['hit_rate', 'map', 'ndcg'], help='Evaluation metrics to compute')
    evaluate_parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20], help='K values for evaluation metrics')
    evaluate_parser.add_argument('--top_n', type=int, default=10, help='Number of top recommendations to generate')
    evaluate_parser.add_argument('--output_dir', type=str, default='results', help='Directory to save evaluation results')
    evaluate_parser.set_defaults(func=evaluate_execute)

    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize evaluation results')
    visualize_parser.add_argument('--json_file', type=str, required=True, help='Path to the evaluation results JSON file')
    visualize_parser.add_argument('--output_dir', type=str, default='results/visualizations', help='Directory to save visualization outputs')
    visualize_parser.add_argument('--plot_type', type=str, choices=['all', 'metrics', 'distribution', 'summary'], default='all', help='Type of visualization to generate')
    visualize_parser.set_defaults(func=visualize_execute)

    return parser