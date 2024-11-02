


def generate_candidates(user_id, top_n=100):
    # Generate candidates using Matrix Factorization
    mf_candidates = matrix_factorization_model.generate_candidates(user_id, top_n)
    
    # Generate candidates using Neural Collaborative Filtering
    ncf_candidates = neural_collab_filter_model.generate_candidates(user_id, top_n)
    
    # Generate candidates using Content-Based Filtering
    content_candidates = content_based_model.generate_candidates(user_id, top_n)
    
    # Generate candidates using Two-Tower Model
    two_tower_candidates = two_tower_model.generate_candidates(user_id, top_n)
    
    # Combine and deduplicate candidates
    combined_candidates = list(set(mf_candidates + ncf_candidates + content_candidates + two_tower_candidates))
    
    # Apply diversity and weighting if needed
    final_candidates = apply_diversity_and_weighting(combined_candidates)
    
    # Return top-K final candidates
    return final_candidates[:top_n]
