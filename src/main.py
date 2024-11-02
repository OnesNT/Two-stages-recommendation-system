import yaml
from src.candidate_generation.candidate_model import MatrixFactorization
# from src.ranking.ranking_model import RankingModel
from src.utils.data_loader import load_data
from src.utils.evaluation import evaluate

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load config
    config = load_config('configs/config.yaml')
    
    # Load candidate generation and ranking configurations
    candidate_config = load_config('configs/candidate_config.yaml')
    ranking_config = load_config('configs/ranking_config.yaml')

    # Initialize model parameters from config
    num_users = 1000
    num_items = 5000
    embedding_dim = candidate_config['model']['embedding_dim']
    learning_rate = candidate_config['model']['learning_rate']
    num_epochs = candidate_config['model']['num_epochs']
    
    # Stage 1: Candidate Generation
    print("Running Candidate Generation Stage...")
    candidate_model = MatrixFactorization(
        num_users=config['num_users'],
        num_items=config['num_items'],
        embedding_dim=config['embedding_dim']
    )
    candidates = candidate_model.generate_candidates() 
    

if __name__ == "__main__":
    main()