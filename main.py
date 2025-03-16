import sys
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Add the 'src' directory to the Python path
current_dir = os.getcwd()  # Get the current working directory
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Get the absolute path to the 'src' directory
sys.path.append(src_dir)  # Add 'src' to the Python path

# Debug: Print the current working directory and Python path
print("Current Working Directory:", current_dir)
print("Python Path:", sys.path)

# Import the KNN classes
from recommenders.knn.item_knn import ItemKNN
from recommenders.knn.user_knn import UserKNN


# Load the MovieLens dataset
def load_movielens_data(data_path):
    """
    Load the MovieLens dataset from the specified path.
    """
    ratings = pd.read_csv(os.path.join(data_path, 'u.data'), sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv(os.path.join(data_path, 'u.item'), sep='|', encoding='latin-1', 
                         names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
                                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    users = pd.read_csv(os.path.join(data_path, 'u.user'), sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    return ratings, movies, users

# Preprocess the data
def preprocess_data(ratings):
    """
    Preprocess the ratings data to create a user-item interaction matrix.
    """
    # Create a user-item interaction matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
    return user_item_matrix_sparse

# Main script
if __name__ == "__main__":
    # Path to the MovieLens dataset 
    data_path = '/home/quang/Two-stages-recommendation-system/dataset/movielens_dataset/ml-100k'

    # Load and preprocess the data
    ratings, movies, users = load_movielens_data(data_path)
    user_item_matrix_sparse = preprocess_data(ratings)

    # Initialize and fit the UserKNN model
    user_knn = UserKNN(k=5, similarity_metric='cosine')
    user_knn.fit(user_item_matrix_sparse)

    # Generate recommendations for user 1
    user_recommendations = user_knn.recommend(user_id=1, top_n=5)
    print("UserKNN Recommendations for User 1:", user_recommendations)

    # Initialize and fit the ItemKNN model
    item_knn = ItemKNN(k=5, similarity_metric='cosine')
    item_knn.fit(user_item_matrix_sparse)

    # Generate recommendations for user 1
    item_recommendations = item_knn.recommend(user_id=1, top_n=5)
    print("ItemKNN Recommendations for User 1:", item_recommendations)

    # Map recommended movie IDs to titles
    def get_movie_titles(movie_ids, movies):
        """
        Map movie IDs to their titles.
        """
        return movies[movies['movie_id'].isin(movie_ids)]['title']

    # Get movie titles for UserKNN recommendations
    user_recommended_titles = get_movie_titles(user_recommendations, movies)
    print("UserKNN Recommended Movies for User 1:")
    print(user_recommended_titles)

    # Get movie titles for ItemKNN recommendations
    item_recommended_titles = get_movie_titles(item_recommendations, movies)
    print("ItemKNN Recommended Movies for User 1:")
    print(item_recommended_titles)