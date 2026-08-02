import os
import sys
import time
import torch
import psutil
import numpy as np
import pickle
import __main__

# Ensure models can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MovieMetadataDB:
    """Minimal reconstruction of the training-side MovieMetadataDB."""
    def get(self, key, default=None): return self.data.get(key, default)
    def __getitem__(self, key): return self.data[key]
    def __contains__(self, key): return key in self.data
    def __iter__(self): return iter(self.data)
    def __len__(self): return len(self.data)
    def keys(self): return self.data.keys()
    def items(self): return self.data.items()
    def values(self): return self.data.values()

setattr(__main__, 'MovieMetadataDB', MovieMetadataDB)

from src.models.sasrec import SASRec
from src.models.manushri.manushri_model import Model1 as ManushriModel
from src.models.suhas.suhas_model import Model2Recommender as SuhasModel

def get_file_size_mb(filepath):
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0

def get_dir_size_mb(directory):
    total_size = 0
    if os.path.exists(directory):
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmarks():
    print("=" * 60)
    print("📊 FULL SUPER BACKEND SYSTEM BENCHMARK")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. ARTIFACT SIZES
    # ---------------------------------------------------------
    print("\n[1] ARTIFACT SIZES (Disk Space)")
    
    sasrec_size = get_file_size_mb('models/weights/sasrec_final_model.pth')
    manushri_size = get_dir_size_mb('src/models/manushri/artifacts')
    suhas_size = get_dir_size_mb('src/models/suhas')
    shared_data_size = get_file_size_mb('data/processed/mappings.pkl') + get_file_size_mb('data/processed/movie_meta_dict.pkl')

    print(f" - Rajdeep's Model (SASRec Weights): {sasrec_size:.2f} MB")
    print(f" - Manushri's Model (ALS Artifacts): {manushri_size:.2f} MB")
    print(f" - Suhas's Model (Network + FAISS):  {suhas_size:.2f} MB")
    print(f" - Shared Dictionaries / Meta:       {shared_data_size:.2f} MB")
    print(f" - TOTAL DISK FOOTPRINT:             {sasrec_size + manushri_size + suhas_size + shared_data_size:.2f} MB")

    # ---------------------------------------------------------
    # 2. MEMORY / RAM CONSUMPTION
    # ---------------------------------------------------------
    print("\n[2] MEMORY CONSUMPTION (RAM)")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    print(" -> Loading all 3 models into Memory...")
    
    # Load Mappings
    with open('data/processed/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        movie_to_idx = mappings['movie_to_idx']
        idx_to_movie = {v: k for k, v in mappings['movie_to_idx'].items()}
    with open('data/processed/movie_meta_dict.pkl', 'rb') as f:
        movie_dict = pickle.load(f)

    # Load Manushri
    manushri = ManushriModel(artifacts_dir="src/models/manushri/artifacts")
    
    # Load Suhas
    suhas = SuhasModel(device='cpu')
    
    # Load Rajdeep
    device = torch.device("cpu")
    sasrec = SASRec(
        num_items=22884, max_seq_len=50, hidden_dim=128, 
        num_heads=4, num_blocks=3, dropout_rate=0.0, device=device
    )
    sasrec.load_state_dict(torch.load('models/weights/sasrec_final_model.pth', map_location=device))
    sasrec.eval()

    mem_after = process.memory_info().rss / (1024 * 1024)
    print(f" - Base Python Memory:      {mem_before:.2f} MB")
    print(f" - Memory after model load: {mem_after:.2f} MB")
    print(f" - PEAK MODEL RAM USAGE:    {mem_after - mem_before:.2f} MB")

    # ---------------------------------------------------------
    # 3. LATENCY & PROCESSING SPEED (FULL PIPELINE)
    # ---------------------------------------------------------
    print("\n[3] INFERENCE LATENCY (Processing Speed)")
    
    # Create a perfectly valid dummy payload by picking 15 real movies that Manushri's model knows
    valid_movie_ids = list(manushri.movie_to_idx.keys())[:15]
    user_ratings = [(int(mid), 5.0) for mid in valid_movie_ids]
    
    # Pre-allocate tensors for SASRec
    user_rated_mids = [mid for mid, _ in user_ratings]
    token_history = [movie_to_idx[mid] for mid in user_rated_mids if mid in movie_to_idx]
    seq = token_history[-50:]
    padded_seq = np.zeros(50, dtype=np.int64)
    padded_seq[-len(seq):] = seq
    tensor_seq = torch.LongTensor(np.array([padded_seq]))
    all_items = torch.arange(1, 22884 + 1)

    print(" -> Running Warmup passes...")
    for _ in range(5):
        m1 = manushri.recommend_from_movie_ratings(user_ratings, top_n=15, candidate_k=200)
        m2 = suhas.run_production_demo(user_history=[], candidate_movie_ids=m1["top_200_movie_ids"], n_syw=15, n_got=15)
        _ = sasrec.predict(tensor_seq, all_items)

    iterations = 100
    m1_times, m2_times, m3_times = [], [], []

    print(f" -> Benchmarking entire homepage pipeline over {iterations} requests...")
    for _ in range(iterations):
        # Time Model 1 (Manushri)
        t0 = time.perf_counter()
        m1_output = manushri.recommend_from_movie_ratings(user_ratings, top_n=15, candidate_k=200)
        t1 = time.perf_counter()
        m1_times.append(t1 - t0)
        
        top_200_mids = m1_output["top_200_movie_ids"]

        # Time Model 2 (Suhas)
        t2 = time.perf_counter()
        m2_output = suhas.run_production_demo(
            user_history=[], # Empty list just for raw speed benchmarking
            candidate_movie_ids=top_200_mids,
            n_syw=15,
            n_got=15
        )
        t3 = time.perf_counter()
        m2_times.append(t3 - t2)

        # Time Model 3 (Rajdeep - SASRec)
        t4 = time.perf_counter()
        with torch.no_grad():
            logits = sasrec.predict(tensor_seq, all_items)
            _, top_indices = torch.topk(logits[0], k=30)
        t5 = time.perf_counter()
        m3_times.append(t5 - t4)

    avg_m1 = (sum(m1_times) / iterations) * 1000
    avg_m2 = (sum(m2_times) / iterations) * 1000
    avg_m3 = (sum(m3_times) / iterations) * 1000
    total_avg = avg_m1 + avg_m2 + avg_m3

    print(f" - Hardware used:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Production Standard)'}")
    print(f" - Benchmarking Method:     Averaged over {iterations} full-pipeline requests")
    print("-" * 40)
    print(f" 1. Manushri (ALS Candidate Gen) : {avg_m1:.2f} ms")
    print(f" 2. Suhas (Multi-Modal Routing)  : {avg_m2:.2f} ms")
    print(f" 3. Rajdeep (SASRec Prediction)  : {avg_m3:.2f} ms")
    print("-" * 40)
    print(f" TOTAL HOMEPAGE LATENCY:           {total_avg:.2f} ms")
    
    # ---------------------------------------------------------
    # 4. EFFECTIVENESS METRICS
    # ---------------------------------------------------------
    print("\n[4] SYSTEM EFFECTIVENESS (Accuracy & Ranking)")
    print(" Since the Super Backend fuses 3 different models together, you should report")
    print(" your personal SASRec hit metrics to the judges using your eval script:")
    print(" -> python src/eval_sasrec.py")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmarks()