import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from typing import List, Optional

class EmotionDetector:
    def __init__(self, model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion", 
                 max_length: int = 512, batch_size: int = 8):
        """
        Initialize the emotion detector with a pre-trained model.
        
        Args:
            model_name: HuggingFace model path
            max_length: Maximum token length for input texts
            batch_size: Number of samples to process at once
        """
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,  
            truncation=True,
            max_length=max_length,
            top_k=1  
        )
        self.batch_size = batch_size
        self.labels_map = {
            'sadness': 'negative',
            'joy': 'positive',
            'love': 'positive',
            'anger': 'negative',
            'fear': 'negative',
            'surprise': 'neutral'
        }

    def detect_emotion(self, text: str) -> Optional[str]:
        """Detect emotion for a single text"""
        try:
            if not isinstance(text, str) or not text.strip():
                return None
                
            result = self.classifier(text[:self.max_length])[0]
            return self.labels_map.get(result['label'].lower(), result['label'])
        except Exception as e:
            print(f"Error processing text: {e}")
            return None

    def detect_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Process multiple texts efficiently"""
        try:
            # Clean and truncate texts
            processed_texts = [str(t)[:self.max_length] for t in texts if isinstance(t, str) and t.strip()]
            if not processed_texts:
                return [None] * len(texts)
                
            results = self.classifier(processed_texts, batch_size=self.batch_size)
            return [self.labels_map.get(r[0]['label'].lower(), r[0]['label']) if r else None for r in results]
        except Exception as e:
            print(f"Batch processing error: {e}")
            return [None] * len(texts)

    def add_emotions_to_df(self, df: pd.DataFrame, text_column: str = 'review/text', 
                          emotion_column: str = 'emotion', sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Add emotion column to DataFrame with progress tracking
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            emotion_column: New column name for emotions
            sample_size: Number of rows to process (None for all)
        """
        if sample_size:
            df = df.copy().head(sample_size)
            
        tqdm.pandas(desc="Detecting emotions")
        df[emotion_column] = df[text_column].progress_apply(self.detect_emotion)
        return df
    

