import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def build_svd_embeddings():
    NUM_ITEMS = 22884
    HIDDEN_DIM = 128
    
    print("1. Loading User Ratings...")
    ratings = pd.read_csv('data/raw/rating.csv')
    
    # ⚡ THE PURIFICATION HACK: Only learn from movies people actually liked!
    print("2. Purifying data (keeping ratings >= 3.5)...")
    good_ratings = ratings[ratings['rating'] >= 3.5]
    
    print("3. Building Sparse User-Item Matrix...")
    # Using category codes to quickly build the sparse matrix without blowing up RAM
    users = good_ratings['userId'].astype('category')
    movies = good_ratings['movieId'].astype('category')
    
    user_idx = users.cat.codes
    movie_idx = movies.cat.codes
    
    # Sparse matrix: rows = users, cols = movies
    sparse_matrix = csr_matrix(
        (good_ratings['rating'].values, (user_idx, movie_idx)), 
        shape=(users.cat.categories.size, movies.cat.categories.size)
    )
    
    print(f"Matrix shape: {sparse_matrix.shape}")
    
    print(f"4. Running TruncatedSVD to extract {HIDDEN_DIM} latent factors...")
    # SVD mathematically extracts the hidden collaborative DNA of the movies
    svd = TruncatedSVD(n_components=HIDDEN_DIM, random_state=42)
    svd.fit(sparse_matrix)
    
    # Item embeddings are the transposed components
    item_factors = svd.components_.T 
    movie_id_map = movies.cat.categories.values
    
    print("5. Mapping Latent Factors to BSARec Token IDs...")
    with open('data/processed/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        movie_to_idx = mappings['movie_to_idx']
        
    pretrained_svd_matrix = np.random.normal(scale=0.01, size=(NUM_ITEMS + 1, HIDDEN_DIM)).astype(np.float32)
    pretrained_svd_matrix[0] = 0.0 # Padding
    
    found_count = 0
    for i, movie_id in enumerate(movie_id_map):
        if movie_id in movie_to_idx:
            token_id = movie_to_idx[movie_id]
            pretrained_svd_matrix[token_id] = item_factors[i]
            found_count += 1
            
    save_path = "data/processed/svd_embeddings.npy"
    np.save(save_path, pretrained_svd_matrix)
    print(f"✅ Collaborative SVD Vectors mapped for {found_count} out of {NUM_ITEMS} items!")
    print(f"💾 Saved to {save_path}")

if __name__ == '__main__':
    build_svd_embeddings()