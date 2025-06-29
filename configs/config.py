"""
Configuration file for the emotion-aware recommendation system.
"""

config = {
    # Data processing
    'data': {
        'dataset_path': 'datasets/amazon_books.json',
        'test_size': 0.2,
        'validation_size': 0.1,
        'random_seed': 42,
        'min_user_interactions': 5,
        'min_item_interactions': 10
    },
    
    # Retrieval stage
    'retrieval': {
        'n_factors': 50,  # Number of latent factors for matrix factorization
        'n_candidates': 100,  # Number of candidates to retrieve
        'emotion_weight': 0.3,  # Weight for emotion-based similarity
        'cf_weight': 0.7,  # Weight for collaborative filtering scores
        'n_similar_users': 10  # Number of similar users to consider
    },
    
    # Ranking stage
    'ranking': {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_boost_round': 100,
        'early_stopping_rounds': 10
    },
    
    # Evaluation
    'evaluation': {
        'k_values': [5, 10, 20],  # k values for metrics
        'metrics': ['hit_rate', 'map', 'ndcg']
    },
    
    # Emotion detection
    'emotion': {
        'model_name': 'j-hartmann/emotion-english-distilroberta-base',
        'emotions': ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
    },
    
    # Training
    'training': {
        'batch_size': 128,
        'epochs': 10,
        'learning_rate': 0.001,
        'validation_interval': 1
    },
    
    # Logging
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/recommender.log'
    }
} 