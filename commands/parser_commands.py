import argparse
from commands.train_commands import execute as train_execute

def setup_parser():
    parser = argparse.ArgumentParser(description="Two-Stage Recommendation System")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ➤ Train command
    train_parser = subparsers.add_parser('train', help='Train a recommendation model')
    train_parser.add_argument('--model_type', type=str, default='itemcf', help='Type of model (e.g., itemcf)')
    train_parser.add_argument('--matrix_path', type=str, required=True, help='Path to preprocessed .pkl sparse matrix')
    train_parser.set_defaults(func=train_execute)

    return parser
