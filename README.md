RecFlix: 3-Model AI Recommendation System

RecFlix is a production-ready movie recommendation system utilizing a 3-Model Ensemble (ALS, Multi-Modal Network, and SASRec Transformer) served via a high-performance FastAPI backend to a responsive Netflix-style UI.

Architecture Highlights

Model 1 (Candidate Gen): Manushri's ALS Model: Uses Alternating Least Squares Collaborative Filtering to generate highly personalized baseline recommendations from the user's initial taste profile.

Model 2 (Similarity & Discovery): Suhas's Multi-Modal Network: A Deep Learning embedding network paired with FAISS vector search. It analyzes genres, directors, cast, and genome tags to power the "Because You Watched..." and "Go Off Trail" rows.

Model 3 (Sequential Momentum): Rajdeep's SASRec Transformer: A PyTorch-based Self-Attentive Sequential Recommendation model. It uses multi-head attention to analyze a user's chronological watch history and predict their immediate next intent, catching nuance like genre fatigue or sequel-binging.

The Super Backend: A high-performance FastAPI orchestrator that manages all three model weights in memory and routes requests dynamically in under 300ms.

The Frontend: A responsive, dark-themed HTML/JS Netflix clone that integrates with the TMDB API to fetch high-resolution posters and metadata in real-time.

Project Structure

Ensure your project contains the following data files before running (heavy datasets and model weights are excluded via .gitignore to save space):

Movie-Recsys/
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│   ├───processed
│   │       mappings.pkl
│   │       movie_meta_dict.pkl
│   │       (various .npy and .index files for training)
│   │       
│   └───raw
│           link.csv
│           genome-scores.csv
│           genome-tags.csv
│           (heavy raw .csv files)
│
├───frontend
│       index.html           <-- The Netflix Clone UI
│
├───models
│   └───weights
│           sasrec_final_model.pth
│
└───src
    │   __init__.py
    │   benchmark_system.py  <-- Full System Latency/RAM Evaluation
    │   eval_sasrec.py       <-- 1-vs-999 Negative Sampling Evaluation
    │
    ├───api
    │       super_backend.py <-- The 3-Model FastAPI Orchestrator
    │
    └───models
            sasrec.py        <-- PyTorch SASRec Architecture
            suhas/           <-- Multi-Modal Network
            manushri/        <-- ALS Matrix Factorization


How to Run

1. Install Dependencies

Ensure you have Python installed. Create a virtual environment and install the required packages:

pip install -r requirements.txt


2. Start the FastAPI Super Backend

Run the backend server. This will load all PyTorch models and FAISS matrices into memory.

python -m uvicorn src.api.super_backend:app


Wait until the terminal reads ✅ Super Backend Online!

3. Open the UI

Because the frontend is a pure HTML/JS implementation, no additional web server is required.
Simply navigate to the frontend folder and double-click index.html to open it in your web browser.

Onboarding: Select 5 movies you enjoy to build your initial Taste Profile.

Dashboard: The 3-Model Ensemble will generate your personalized homepage.

Real-Time Sequence: Hover over a movie and click Watch. Scroll to the top to observe the SASRec engine and Similarity engine instantly updating the Up Next queues based on your exact chronological sequence!

