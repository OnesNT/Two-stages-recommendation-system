import pandas as pd
import numpy as np
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging
from typing import Dict, List, Tuple

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataProcessor:
    def __init__(self, config: Dict):
        """
        Initialize the DataProcessor with configuration parameters.
        
        Args:
            config (Dict): Configuration dictionary containing parameters
        """
        self.config = config
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the Amazon book reviews dataset.
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_json(file_path, lines=True)
            required_columns = ['reviewerID', 'asin', 'reviewText', 'overall']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess review text by tokenizing, removing stop words, and lemmatizing.
        
        Args:
            text (str): Raw review text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""
            
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]
        
        return " ".join(tokens)
        
    def detect_emotion(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in the review text using DistilRoBERTa.
        
        Args:
            text (str): Preprocessed review text
            
        Returns:
            Dict[str, float]: Dictionary of emotion scores
        """
        try:
            if not text.strip():
                return {
                    "joy": 0.0, "sadness": 0.0, "anger": 0.0,
                    "fear": 0.0, "surprise": 0.0, "love": 0.0
                }
                
            emotions = self.emotion_classifier(text)[0]
            return {item['label']: item['score'] for item in emotions}
        except Exception as e:
            logging.error(f"Error in emotion detection: {str(e)}")
            return {
                "joy": 0.0, "sadness": 0.0, "anger": 0.0,
                "fear": 0.0, "surprise": 0.0, "love": 0.0
            }
            
    def create_user_item_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Create user-item interaction matrix from the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe with user-item interactions
            
        Returns:
            Tuple[np.ndarray, Dict, Dict]: User-item matrix, user mapping, item mapping
        """
        user_ids = df['reviewerID'].unique()
        item_ids = df['asin'].unique()
        
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        
        matrix = np.zeros((len(user_ids), len(item_ids)))
        
        for _, row in df.iterrows():
            user_idx = user_to_idx[row['reviewerID']]
            item_idx = item_to_idx[row['asin']]
            matrix[user_idx, item_idx] = row['overall']
            
        return matrix, user_to_idx, item_to_idx
        
    def process_dataset(self, file_path: str) -> Tuple[pd.DataFrame, np.ndarray, Dict, Dict]:
        """
        Main function to process the dataset.
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray, Dict, Dict]: 
                Processed dataframe, user-item matrix, user mapping, item mapping
        """
        # Load data
        df = self.load_data(file_path)
        
        # Preprocess review text
        df['processed_review'] = df['reviewText'].apply(self.preprocess_text)
        
        # Detect emotions
        emotions = df['processed_review'].apply(self.detect_emotion)
        emotion_df = pd.DataFrame(emotions.tolist())
        df = pd.concat([df, emotion_df], axis=1)
        
        # Create user-item matrix
        matrix, user_to_idx, item_to_idx = self.create_user_item_matrix(df)
        
        return df, matrix, user_to_idx, item_to_idx 