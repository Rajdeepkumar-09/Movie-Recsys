import os
import sys
import time
import torch
import psutil
import numpy as np

# Ensure models can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.sasrec import SASRec

def get_file_size_mb(filepath):
    """Calculates the size of a file in Megabytes."""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    return 0.0

def count_parameters(model):
    """Calculates the total number of trainable parameters in the neural network."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmarks():
    print("=" * 50)
    print("📊 RECSYS SYSTEM & HARDWARE BENCHMARK")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. ARTIFACT SIZES
    # ---------------------------------------------------------
    print("\n[1] ARTIFACT SIZES (Disk Space)")
    weights_path = 'models/weights/sasrec_final_model.pth'
    meta_path = 'data/processed/movie_meta_dict.pkl'
    map_path = 'data/processed/mappings.pkl'

    sasrec_size = get_file_size_mb(weights_path)
    meta_size = get_file_size_mb(meta_path)
    map_size = get_file_size_mb(map_path)

    print(f" - SASRec PyTorch Weights:  {sasrec_size:.2f} MB")
    print(f" - Movie Metadata Dict:     {meta_size:.2f} MB")
    print(f" - Token Mappings:          {map_size:.2f} MB")
    print(f" - TOTAL DISK FOOTPRINT:    {sasrec_size + meta_size + map_size:.2f} MB")

    # ---------------------------------------------------------
    # 2. MEMORY / RAM CONSUMPTION
    # ---------------------------------------------------------
    print("\n[2] MEMORY CONSUMPTION (RAM)")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Initialize model to see how much RAM it consumes
    device = torch.device("cpu")
    model = SASRec(
        num_items=22884, max_seq_len=50, hidden_dim=128, 
        num_heads=4, num_blocks=3, dropout_rate=0.0, device=device
    )
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    mem_after = process.memory_info().rss / (1024 * 1024)
    print(f" - Base Python Memory:      {mem_before:.2f} MB")
    print(f" - Memory after model load: {mem_after:.2f} MB")
    print(f" - PEAK MODEL RAM USAGE:    {mem_after - mem_before:.2f} MB")
    print(f" - Total Parameters:        {count_parameters(model):,}")

    # ---------------------------------------------------------
    # 3. LATENCY & PROCESSING SPEED
    # ---------------------------------------------------------
    print("\n[3] INFERENCE LATENCY (Processing Speed)")
    
    # Simulate a user with a full history of 50 movies
    dummy_seq = torch.randint(1, 22884, (1, 50)).to(device)
    all_items = torch.arange(1, 22884 + 1).to(device)

    # Warmup runs (PyTorch is slow on the first few passes)
    with torch.no_grad():
        for _ in range(10):
            _ = model.predict(dummy_seq, all_items)

    # Benchmark exactly 1,000 requests to get an accurate average
    iterations = 1000
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model.predict(dummy_seq, all_items)
            
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000

    print(f" - Hardware used:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Production Standard)'}")
    print(f" - Benchmarking Method:     Averaged over {iterations:,} requests (time.perf_counter)")
    print(f" - Total time for {iterations} reqs: {total_time:.3f} seconds")
    print(f" - AVERAGE LATENCY / USER:  {avg_latency_ms:.3f} milliseconds (ms)")

    # ---------------------------------------------------------
    # 4. EFFECTIVENESS METRICS (Reminder)
    # ---------------------------------------------------------
    print("\n[4] EFFECTIVENESS METRICS (Accuracy & Ranking)")
    print(" To calculate Hit Ratio, NDCG, and MRR using the official Leave-One-Out")
    print(" Negative Sampling method, run your existing evaluation script:")
    print(" -> python src/eval_sasrec.py")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmarks()