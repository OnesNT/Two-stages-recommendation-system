# Two-Stage Emotion-Aware Book Recommendation System

This project implements a two-stage recommendation system for books using the Amazon Book Reviews dataset. The system incorporates users' emotional states derived from review text to enhance recommendation quality.

## Features
- Stage 1: Emotion-aware retrieval using collaborative and content-based filtering
- Stage 2: Learning-to-rank with emotional context
- Sentiment and emotion analysis from review text
- Multiple evaluation metrics (HR, MAP, NDCG)

## Project Structure
```
.
├── configs/            # Configuration files
├── datasets/          # Data processing and loading
├── models/           # Model implementations
├── src/             # Core source code
│   ├── preprocessing/  # Data preprocessing modules
│   ├── retrieval/     # First stage retrieval
│   ├── ranking/       # Second stage ranking
│   └── evaluation/    # Evaluation metrics
├── testing/         # Unit tests
└── requirements.txt  # Project dependencies
```

## Setup
1. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Amazon Book Reviews dataset and place it in the `datasets` directory.

## Usage
1. Preprocess the data:
```bash
python src/preprocessing/preprocess.py
```

2. Train the model:
```bash
python main.py --mode train
```

3. Make recommendations:
```bash
python main.py --mode predict
```

## Model Architecture
1. Stage 1 (Retrieval):
   - Emotion-aware collaborative filtering
   - Content-based filtering
   
2. Stage 2 (Ranking):
   - Learning-to-rank with emotional context
   - Feature fusion of user, item, and emotional signals

## Evaluation
The system is evaluated using:
- Hit Rate (HR)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

## License
MIT License