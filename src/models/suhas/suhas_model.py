"""
Standalone inference module for movie recommendations.
Modified for local execution (no ZIP extraction, loads directly from directory).
"""

import os
import json
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Determine the absolute path to this folder (src/models/suhas/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = 'cpu' # Forced to CPU for stability in the combined backend
EMBEDDING_DIM = 32
GENOME_INPUT_DIM = 1128
TOP_K_CAST = 5
MAX_PROD_COMPANIES = 3
NUM_GENRES = 19

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
RATING_TO_WEIGHT = {5: 1.0, 4: 0.6, 3: 0.0, 2: -0.6, 1: -1.0}

# ------------------------------------------------------------------------------
# Pickle compatibility shim
# ------------------------------------------------------------------------------
class MovieMetadataDB:
    """Minimal reconstruction of the training-side MovieMetadataDB."""
    def get(self, key, default=None):
        return self.data.get(key, default)
    def __getitem__(self, key):
        return self.data[key]
    def __contains__(self, key):
        return key in self.data
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def keys(self):
        return self.data.keys()
    def items(self):
        return self.data.items()
    def values(self):
        return self.data.values()

# ------------------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------------------
class NumericalEncoder(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, embedding_dim))
    def forward(self, x):
        return self.net(x)

class GenomeEncoder(nn.Module):
    def __init__(self, input_dim=GENOME_INPUT_DIM, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    def forward(self, x):
        return self.net(x)

class MovieRepresentation(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.genre_embeddings = nn.Parameter(torch.randn(NUM_GENRES, embedding_dim) * 0.01)
        self.director_encoder = nn.Embedding(vocab_sizes["director"], embedding_dim, padding_idx=0)
        self.cast_encoder = nn.Embedding(vocab_sizes["actor"], embedding_dim, padding_idx=0)
        self.collection_encoder = nn.Embedding(vocab_sizes["collection"], embedding_dim, padding_idx=0)
        self.language_encoder = nn.Embedding(vocab_sizes["language"], embedding_dim, padding_idx=0)
        self.company_encoder = nn.Embedding(vocab_sizes["company"], embedding_dim, padding_idx=0)
        self.avg_rating_encoder = NumericalEncoder(embedding_dim)
        self.runtime_encoder = NumericalEncoder(embedding_dim)
        self.year_encoder = NumericalEncoder(embedding_dim)
        self.genome_encoder = GenomeEncoder(GENOME_INPUT_DIM, embedding_dim)

    def forward(self, meta):
        device = self.genre_embeddings.device
        genre_mask = meta["genre_mask"].to(device)
        genre_block = genre_mask.unsqueeze(1) * self.genre_embeddings
        genome_emb = self.genome_encoder(meta["genome"].to(device))
        director_emb = self.director_encoder(meta["director_id"].to(device))
        cast_embs = self.cast_encoder(meta["cast_ids"].to(device))
        cast_mask = (meta["cast_ids"].to(device) != 0).float().unsqueeze(1)
        cast_emb = (cast_embs * cast_mask).sum(dim=0) / (cast_mask.sum() + 1e-8)
        collection_emb = self.collection_encoder(meta["collection_id"].to(device))
        language_emb = self.language_encoder(meta["language_id"].to(device))
        company_embs = self.company_encoder(meta["company_ids"].to(device))
        company_mask = (meta["company_ids"].to(device) != 0).float().unsqueeze(1)
        company_emb = (company_embs * company_mask).sum(dim=0) / (company_mask.sum() + 1e-8)
        avg_rating_emb = self.avg_rating_encoder(meta["avg_rating"].to(device).unsqueeze(0))
        runtime_emb = self.runtime_encoder(meta["runtime"].to(device).unsqueeze(0))
        year_emb = self.year_encoder(meta["year"].to(device).unsqueeze(0))
        return {
            "movie_id": meta["movie_id"],
            "genre_block": genre_block,
            "genome": genome_emb,
            "director": director_emb,
            "cast": cast_emb,
            "collection": collection_emb,
            "language": language_emb,
            "company": company_emb,
            "avg_rating": avg_rating_emb,
            "runtime": runtime_emb,
            "year": year_emb,
            "genre_mask": genre_mask,
            "cast_ids": meta["cast_ids"],
            "director_id": meta["director_id"],
            "collection_id": meta["collection_id"],
            "language_id": meta["language_id"],
            "company_ids": meta["company_ids"],
            "avg_rating_raw": meta["avg_rating"].to(device),
            "runtime_raw": meta["runtime"].to(device),
            "year_raw": meta["year"].to(device),
        }

class UserEmbeddingNetwork(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        genre_flat = NUM_GENRES * embedding_dim
        other_cats = 9 * embedding_dim
        total_in = genre_flat + other_cats
        self.mlp = nn.Sequential(
            nn.Linear(total_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, movie_reprs, weights):
        if len(movie_reprs) == 0:
            return torch.zeros(self.embedding_dim, device=weights.device)
        w = weights.unsqueeze(1)
        abs_sum = weights.abs().sum() + 1e-8
        genres = torch.stack([m["genre_block"] for m in movie_reprs], dim=0)
        genre_acc = (genres * w.view(-1, 1, 1)).sum(dim=0) / abs_sum

        def acc(key):
            vals = torch.stack([m[key] for m in movie_reprs], dim=0)
            return (vals * w).sum(dim=0) / abs_sum

        concat = torch.cat([
            genre_acc.view(-1), acc("genome"), acc("director"), acc("cast"),
            acc("collection"), acc("language"), acc("company"), acc("avg_rating"),
            acc("runtime"), acc("year")
        ], dim=0)
        return self.mlp(concat)

class LikeabilityNetwork(nn.Module):
    def __init__(self, num_categories=10, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_categories, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def compute_affinity_vector(self, user_emb, movie_repr, cast_encoder):
        affinities, affinity_dict = [], {}
        device = user_emb.device
        present = movie_repr["genre_mask"] > 0
        if present.sum() > 0:
            sims = torch.cosine_similarity(movie_repr["genre_block"][present], user_emb.unsqueeze(0), dim=1)
            affinities.append(sims.mean())
            affinity_dict["genre"] = sims.mean().item()
        else:
            affinities.append(torch.tensor(0.0, device=device))
            affinity_dict["genre"] = 0.0
        for key in ["genome", "director", "avg_rating", "runtime", "year", "language", "collection", "company"]:
            affinities.append(torch.cosine_similarity(movie_repr[key].unsqueeze(0), user_emb.unsqueeze(0), dim=1).squeeze())
            affinity_dict[key] = affinities[-1].item()
        valid = movie_repr["cast_ids"] != 0
        if valid.sum() > 0:
            cast_embs = cast_encoder(movie_repr["cast_ids"][valid])
            sims = torch.cosine_similarity(cast_embs, user_emb.unsqueeze(0), dim=1)
            affinities.append(sims.mean())
            affinity_dict["cast"] = sims.mean().item()
        else:
            affinities.append(torch.tensor(0.0, device=device))
            affinity_dict["cast"] = 0.0
        affinity_vec = torch.stack(affinities)
        return self.mlp(affinity_vec), affinity_vec, affinity_dict

    def forward(self, affinity_vector):
        return self.mlp(affinity_vector)

class Model2(nn.Module):
    def __init__(self, vocab_sizes):
        super().__init__()
        self.movie_repr = MovieRepresentation(vocab_sizes)
        self.user_embed_net = UserEmbeddingNetwork()
        self.likeability_net = LikeabilityNetwork(num_categories=10)
        self.cast_encoder = self.movie_repr.cast_encoder

    def build_user_embedding(self, rated_movie_reprs, weights):
        return self.user_embed_net(rated_movie_reprs, weights)

    def compute_likeability(self, user_emb, movie_repr):
        return self.likeability_net.compute_affinity_vector(user_emb, movie_repr, self.cast_encoder)

# ------------------------------------------------------------------------------
# Similarity & Exposure
# ------------------------------------------------------------------------------
class MovieSimilarityEngine:
    CATEGORY_WEIGHTS = {k: 1.0 for k in ["genre", "director", "cast", "collection", "language", "company", "genome", "avg_rating", "runtime", "year"]}

    @staticmethod
    def compute_similarity(repr_a, repr_b, verbose=False):
        sim = {}
        sim["genre"] = float((repr_a["genre_mask"] * repr_b["genre_mask"]).sum().item())
        sim["genome"] = max(0.0, torch.cosine_similarity(repr_a["genome"].unsqueeze(0), repr_b["genome"].unsqueeze(0), dim=1).item())
        sim["director"] = 1.0 if (repr_a["director_id"] == repr_b["director_id"]) and (repr_a["director_id"].item() != 0) else 0.0
        sim["cast"] = float(len((set(repr_a["cast_ids"].cpu().numpy()) & set(repr_b["cast_ids"].cpu().numpy())) - {0}))
        sim["collection"] = 1.0 if (repr_a["collection_id"] == repr_b["collection_id"]) and (repr_a["collection_id"].item() != 0) else 0.0
        sim["language"] = 1.0 if (repr_a["language_id"] == repr_b["language_id"]) and (repr_a["language_id"].item() != 0) else 0.0
        sim["company"] = float(len((set(repr_a["company_ids"].cpu().numpy()) - {0}) & (set(repr_b["company_ids"].cpu().numpy()) - {0})))
        sim["avg_rating"] = max(0.0, 1.0 - abs(repr_a["avg_rating_raw"].item() - repr_b["avg_rating_raw"].item()))
        sim["runtime"] = max(0.0, 1.0 - abs(repr_a["runtime_raw"].item() - repr_b["runtime_raw"].item()))
        sim["year"] = max(0.0, 1.0 - abs(repr_a["year_raw"].item() - repr_b["year_raw"].item()))
        weighted_sim = {k: v * MovieSimilarityEngine.CATEGORY_WEIGHTS[k] for k, v in sim.items()}
        final = sum(weighted_sim.values())
        return (sim, weighted_sim, final) if verbose else (sim, final)

class ExposureNetwork:
    def __init__(self, movie_db):
        self.movie_db = movie_db

    def compute_exposure(self, user_rated_mids, candidate_repr):
        if len(user_rated_mids) == 0:
            return {k: 0.0 for k in ["genre", "director", "cast", "collection", "language", "company", "genome", "avg_rating", "runtime", "year"]}, 0.0
        rated_metas = [self.movie_db.get(mid) for mid in user_rated_mids if mid in self.movie_db]
        if len(rated_metas) == 0:
            return {k: 0.0 for k in ["genre", "director", "cast", "collection", "language", "company", "genome", "avg_rating", "runtime", "year"]}, 0.0
        cand_mid = candidate_repr["movie_id"].item()
        cand_raw = self.movie_db.get(cand_mid)
        exp = {}
        cand_genres = set(i for i, v in enumerate(candidate_repr["genre_mask"].cpu().numpy()) if v > 0)
        exp["genre"] = np.mean([sum(1 for g in cand_genres if m["genre_mask"][g] > 0) for m in rated_metas])
        d_id = candidate_repr["director_id"].item()
        exp["director"] = sum(1 for m in rated_metas if m["director_id"] == d_id and d_id != 0)
        cand_cast = set(candidate_repr["cast_ids"].cpu().numpy()) - {0}
        exp["cast"] = np.mean([len(cand_cast & (set(m["cast_ids"]) - {0})) for m in rated_metas])
        c_id = candidate_repr["collection_id"].item()
        exp["collection"] = sum(1 for m in rated_metas if m["collection_id"] == c_id and c_id != 0)
        l_id = candidate_repr["language_id"].item()
        exp["language"] = sum(1 for m in rated_metas if m["language_id"] == l_id and l_id != 0)
        cand_comp = set(candidate_repr["company_ids"].cpu().numpy()) - {0}
        exp["company"] = np.mean([len(cand_comp & (set(m["company_ids"]) - {0})) for m in rated_metas])
        cand_genome = cand_raw["genome"] if cand_raw else np.zeros(GENOME_INPUT_DIM, dtype=np.float32)
        exp["genome"] = np.mean([np.dot(cand_genome, m["genome"]) / (np.linalg.norm(cand_genome) * np.linalg.norm(m["genome"]) + 1e-8) for m in rated_metas])
        for key in ["avg_rating", "runtime", "year"]:
            cand_val = cand_raw[key] if cand_raw else 0.0
            exp[key] = np.mean([1.0 - abs(cand_val - m[key]) for m in rated_metas])

        def norm(v, div):
            return min(v / div, 1.0)
        def norm_sim(v):
            return max(0.0, min(1.0, v))

        normed = {
            "genre": norm(exp["genre"], 5.0),
            "director": norm(exp["director"], 10.0),
            "cast": norm(exp["cast"], 5.0),
            "collection": norm(exp["collection"], 10.0),
            "language": norm(exp["language"], 10.0),
            "company": norm(exp["company"], 5.0),
            "genome": norm_sim(exp["genome"]),
            "avg_rating": norm_sim(exp["avg_rating"]),
            "runtime": norm_sim(exp["runtime"]),
            "year": norm_sim(exp["year"]),
        }
        return normed, np.mean(list(normed.values()))

# ------------------------------------------------------------------------------
# Exportable recommender — Reads directly from the local directory
# ------------------------------------------------------------------------------
class Model2Recommender:
    def __init__(self, device: str = DEVICE):
        self.device = device
        
        # Determine the absolute path to this folder (src/models/suhas/)
        self.components_dir = os.path.dirname(os.path.abspath(__file__))

        self.config_path = self._require_file("config.json")
        self.movie_db_path = self._require_file("model2_movie_db.pkl")
        self.movie_emb_path = self._require_file("movie_embeddings.pt")

        with open(self.config_path, 'r') as f:
            config = json.load(f)
        vocab_sizes = config["vocab_sizes"]
        self.vocab = config.get("vocab", {})
        self.model = Model2(vocab_sizes).to(device)

        self.model.user_embed_net.load_state_dict(torch.load(
            self._require_file("user_embed_net.pt"), map_location=device, weights_only=True))
        self.model.likeability_net.load_state_dict(torch.load(
            self._require_file("likeability_net.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.genome_encoder.load_state_dict(torch.load(
            self._require_file("genome_encoder.pt"), map_location=device, weights_only=True))
            
        # Support either avg_rating or imdb encoder
        try:
            self.model.movie_repr.avg_rating_encoder.load_state_dict(torch.load(
                self._require_file("avg_rating_encoder.pt"), map_location=device, weights_only=True))
        except FileNotFoundError:
            self.model.movie_repr.avg_rating_encoder.load_state_dict(torch.load(
                self._require_file("imdb_encoder.pt"), map_location=device, weights_only=True))
                
        self.model.movie_repr.runtime_encoder.load_state_dict(torch.load(
            self._require_file("runtime_encoder.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.year_encoder.load_state_dict(torch.load(
            self._require_file("year_encoder.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.genre_embeddings = nn.Parameter(torch.load(
            self._require_file("genre_embeddings.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.director_encoder.load_state_dict(torch.load(
            self._require_file("director_embeddings.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.cast_encoder.load_state_dict(torch.load(
            self._require_file("actor_embeddings.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.language_encoder.load_state_dict(torch.load(
            self._require_file("language_embeddings.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.company_encoder.load_state_dict(torch.load(
            self._require_file("company_embeddings.pt"), map_location=device, weights_only=True))
        self.model.movie_repr.collection_encoder.load_state_dict(torch.load(
            self._require_file("collection_embeddings.pt"), map_location=device, weights_only=True))
        self.model.eval()

        with open(self.movie_db_path, 'rb') as f:
            self.movie_db = pickle.load(f)

        raw_embeddings = torch.load(self.movie_emb_path, map_location='cpu', weights_only=True)
        self.movie_embeddings = {}
        for mid, repr_dict in raw_embeddings.items():
            self.movie_embeddings[mid] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in repr_dict.items()
            }

        self.exposure_net = ExposureNetwork(self.movie_db)
        self.similarity_engine = MovieSimilarityEngine()
        self._title_to_mid = self._build_title_index()
        self._director_id_to_name = self._build_director_index()

        print(f"[Model 2 Ready] Loaded directly from local directory.")

    def _require_file(self, filename: str) -> str:
        """Helper to find the file in the local directory."""
        path = os.path.join(self.components_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required Model 2 component: {path}")
        return path

    def _build_title_index(self) -> Dict[str, int]:
        index = {}
        for mid, meta in self.movie_db.items():
            title = meta.get("title") if isinstance(meta, dict) else None
            if title:
                index[title.strip().lower()] = mid
        return index

    def _build_director_index(self) -> Dict[int, str]:
        index: Dict[int, str] = {}
        director_vocab = self.vocab.get("director", {})
        for name, d_id in director_vocab.items():
            if d_id != 0:
                index[d_id] = name
        return index

    def _director_name(self, director_id: int) -> str:
        if director_id == 0:
            return "Unknown Director"
        return self._director_id_to_name.get(director_id, "Unknown Director")

    def _get_user_embedding(self, rated_items: List[Tuple[int, float]]) -> torch.Tensor:
        reprs, weights = [], []
        for mid, rating in rated_items:
            if mid not in self.movie_embeddings:
                continue
            reprs.append(self.movie_embeddings[mid])
            weights.append(RATING_TO_WEIGHT.get(int(round(rating)), 0.0))
        if len(reprs) == 0:
            return torch.zeros(EMBEDDING_DIM, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model.build_user_embedding(reprs, w)

    def _get_candidate_pool(self, top_mids: List[int], rated_mids: set) -> List[int]:
        return [m for m in top_mids if m in self.movie_embeddings and m not in rated_mids]

    def recommend(
        self,
        user_rated_items: List[Tuple[int, float]],
        top_400_mids: List[int],
        syw_target_mid: Optional[int] = None,
        n_syw: int = 5,
        n_got: int = 5,
    ) -> Dict:
        user_emb = self._get_user_embedding(user_rated_items)
        rated_mids = set(mid for mid, _ in user_rated_items)
        candidates = self._get_candidate_pool(top_400_mids, rated_mids)

        # Since You Watched
        syw_results, syw_mids = [], set()
        if syw_target_mid is not None and syw_target_mid in self.movie_embeddings:
            target_repr = self.movie_embeddings[syw_target_mid]
            scores = []
            for mid in candidates:
                if mid == syw_target_mid:
                    continue
                _, final = self.similarity_engine.compute_similarity(target_repr, self.movie_embeddings[mid])
                scores.append((mid, final))
            scores.sort(key=lambda x: x[1], reverse=True)
            for mid, score in scores[:n_syw]:
                syw_mids.add(mid)
                syw_results.append({"movie_id": mid})

        # Go Off Trail
        got_results, got_mids = [], set()
        scores = []
        for mid in candidates:
            m_repr = self.movie_embeddings[mid]
            likeability, _, _ = self.model.compute_likeability(user_emb, m_repr)
            _, exposure = self.exposure_net.compute_exposure(list(rated_mids), m_repr)
            scores.append((mid, likeability.item() / (1.0 + 10*exposure), likeability.item(), exposure))
        scores.sort(key=lambda x: x[1], reverse=True)
        for mid, score, like, exp in scores[:n_got]:
            got_mids.add(mid)
            got_results.append({"movie_id": mid})

        # Movie Marathon
        mm_results = []
        excluded = syw_mids | got_mids
        scored = []
        for mid in candidates:
            m_repr = self.movie_embeddings[mid]
            likeability, _, _ = self.model.compute_likeability(user_emb, m_repr)
            scored.append({'mid': mid, 'likeability': likeability.item(), 'repr': m_repr})
        scored.sort(key=lambda x: x['likeability'], reverse=True)
        filtered = [s for s in scored[:15] if s['mid'] not in excluded]

        if len(filtered) >= 2:
            m_best = filtered[0]
            m_second = filtered[1]

            movie2 = None
            best_sim = -float('inf')
            for cand in scored:
                if cand['mid'] in (m_best['mid'], m_second['mid']):
                    continue
                _, sim = self.similarity_engine.compute_similarity(m_best['repr'], cand['repr'])
                if sim > best_sim:
                    best_sim = sim
                    movie2 = cand

            for m in [m_second, movie2, m_best]:
                if m is not None:
                    mm_results.append({"movie_id": m['mid']})

        return {
            "since_you_watched": syw_results,
            "go_off_trail": got_results,
            "movie_marathon": mm_results,
        }

    def run_production_demo(
        self,
        user_history: List[Tuple[str, float]],
        candidate_movie_ids: List[int],
        syw_target_title: Optional[str] = None,
        n_syw: int = 15,
        n_got: int = 15,
    ) -> Dict:
        # Map titles to IDs for backward compatibility with Suhas's pipeline
        resolved_ratings = []
        for title, rating in user_history:
             mid = self._title_to_mid.get(str(title).strip().lower())
             if mid:
                 resolved_ratings.append((mid, rating))

        if syw_target_title is not None:
             syw_target = self._title_to_mid.get(str(syw_target_title).strip().lower())
        else:
             syw_target = max(resolved_ratings, key=lambda pair: pair[1])[0] if resolved_ratings else None

        return self.recommend(
            user_rated_items=resolved_ratings,
            top_400_mids=candidate_movie_ids,
            syw_target_mid=syw_target,
            n_syw=n_syw,
            n_got=n_got
        )