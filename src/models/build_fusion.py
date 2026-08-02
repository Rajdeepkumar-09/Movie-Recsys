import numpy as np
from sklearn.decomposition import PCA

def build_fusion_embeddings():
    print("1. Loading Semantic (Transformer) and Collaborative (SVD) Embeddings...")
    # Load the matrices you already generated!
    transformer_emb = np.load('data/processed/transformer_embeddings.npy')
    svd_emb = np.load('data/processed/svd_embeddings.npy')

    print("2. Concatenating vectors into a 256-dimensional Hybrid DNA...")
    # We slice [1:] to temporarily ignore the zero-padding token at index 0
    combined_emb = np.concatenate((transformer_emb[1:], svd_emb[1:]), axis=1)

    print("3. Compressing 256D -> 128D using PCA to fit BSARec...")
    pca = PCA(n_components=128, random_state=42)
    fused_reduced = pca.fit_transform(combined_emb)

    print("4. Re-applying padding and saving...")
    final_matrix = np.zeros((transformer_emb.shape[0], 128), dtype=np.float32)
    final_matrix[1:] = fused_reduced

    save_path = 'data/processed/fused_embeddings.npy'
    np.save(save_path, final_matrix)
    print(f"✅ SOTA Fused Embeddings saved to {save_path}!")

if __name__ == '__main__':
    build_fusion_embeddings()