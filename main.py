from src.utils import helpers
from src.candidate_generation import candidate_model

def main():
    user_id = 0  # Replace with the target user's ID
    top_n = 10  # Number of recommendations you want to generate
    
    # Generate candidates
    candidates = candidate_model.generate_candidates(user_id, top_n)
    # print(f"Top {top_n} candidates for user {user_id}:", candidates)
    # Convert the top candidates to standard Python types for printing
    top_candidates = [(int(item), float(score)) for item, score in candidates]
    print(f"Top 10 candidates for user 0: {top_candidates}")
    
if __name__ == "__main__":
    main()
