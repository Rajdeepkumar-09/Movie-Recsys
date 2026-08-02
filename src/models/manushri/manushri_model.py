from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class Model1:
    def __init__(self, artifacts_dir: str | Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)

        self.user_factors = np.load(self.artifacts_dir / "user_factors.npy").astype(
            np.float32
        )
        self.item_factors = np.load(self.artifacts_dir / "item_factors.npy").astype(
            np.float32
        )

        user_map = pd.read_parquet(self.artifacts_dir / "user_map.parquet")
        item_map = pd.read_parquet(self.artifacts_dir / "item_map.parquet")
        train_seen = pd.read_parquet(self.artifacts_dir / "train_seen.parquet")

        self.user_to_idx = dict(zip(user_map["userId"], user_map["user_idx"]))
        self.idx_to_user = dict(zip(user_map["user_idx"], user_map["userId"]))
        self.movie_to_idx = dict(zip(item_map["movieId"], item_map["item_idx"]))
        self.idx_to_movie = dict(zip(item_map["item_idx"], item_map["movieId"]))

        self.seen_by_user: dict[int, set[int]] = (
            train_seen.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )

        item_counts = train_seen.groupby("item_idx").size().sort_values(ascending=False)
        self.popular_item_indices = item_counts.index.to_numpy(dtype=np.int64)
        self.popular_item_scores = item_counts.to_numpy(dtype=np.float32)

        self.extra_user_factors: dict[int, np.ndarray] = {}
        self.extra_seen_by_user: dict[int, set[int]] = {}
        self.mean_user_vector = self.user_factors.mean(axis=0).astype(np.float32)

    def _raw_movie_id(self, item_idx: int) -> int:
        return int(self.idx_to_movie[int(item_idx)])

    @staticmethod
    def _format_recommendations(
        recommendations: list[tuple[int, float]]
    ) -> list[dict[str, int | float]]:
        return [
            {"movieId": movie_id, "score": score}
            for movie_id, score in recommendations
        ]

    def _user_vector(self, user_id: int) -> tuple[np.ndarray | None, int | None]:
        if user_id in self.user_to_idx:
            user_idx = int(self.user_to_idx[user_id])
            return self.user_factors[user_idx], user_idx
        if user_id in self.extra_user_factors:
            return self.extra_user_factors[user_id], None
        return None, None

    def _popular_candidates(self, k: int) -> list[tuple[int, float]]:
        item_indices = self.popular_item_indices[:k]
        scores = self.popular_item_scores[:k]
        return [
            (self._raw_movie_id(item_idx), float(score))
            for item_idx, score in zip(item_indices, scores)
        ]

    def _rank_from_vector(
        self, user_vector: np.ndarray, seen_item_indices: set[int], k: int
    ) -> list[tuple[int, float]]:
        scores = self.item_factors @ user_vector
        if seen_item_indices:
            scores[list(seen_item_indices)] = -np.inf

        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [
            (self._raw_movie_id(item_idx), float(scores[item_idx]))
            for item_idx in top_indices
            if np.isfinite(scores[item_idx])
        ]

    def _vector_from_movie_ratings(
        self, movie_ratings: list[tuple[int, float]], alpha: float = 0.7
    ) -> tuple[np.ndarray, set[int]]:
        user_vector = self.mean_user_vector.copy()
        seen_item_indices: set[int] = set()

        for movie_id, rating in movie_ratings:
            if movie_id not in self.movie_to_idx:
                raise ValueError(f"Unknown movie_id: {movie_id}")

            item_idx = int(self.movie_to_idx[movie_id])
            seen_item_indices.add(item_idx)

            if rating < 3.0:
                continue

            item_vector = self.item_factors[item_idx]
            strength = min((float(rating) - 3.0) / 2.0, 1.0)
            effective_alpha = 1.0 - ((1.0 - alpha) * strength)
            user_vector = (
                effective_alpha * user_vector
                + (1.0 - effective_alpha) * item_vector
            ).astype(np.float32)

        return user_vector, seen_item_indices

    def get_candidates(self, user_id: int, k: int = 200) -> list[tuple[int, float]]:
        user_vector, user_idx = self._user_vector(user_id)
        if user_vector is None:
            return self._popular_candidates(k)

        if user_idx is not None:
            seen = self.seen_by_user.get(user_idx, set())
        else:
            seen = self.extra_seen_by_user.get(user_id, set())
        return self._rank_from_vector(user_vector, seen, k)

    def get_recommended_for_you(self, user_id: int, k: int = 10) -> list[tuple[int, float]]:
        return self.get_candidates(user_id, k=k)

    def recommend_from_movie_ratings(
        self,
        movie_ratings: list[tuple[int, float]],
        top_n: int = 10,
        candidate_k: int = 200,
    ) -> dict[str, list]:
        if len(movie_ratings) != 15:
            raise ValueError("Model 1 requires exactly 15 movie-rating pairs.")

        user_vector, seen_item_indices = self._vector_from_movie_ratings(movie_ratings)
        candidates = self._rank_from_vector(user_vector, seen_item_indices, candidate_k)

        return {
            "recommended_for_you": self._format_recommendations(candidates[:top_n]),
            "top_200_candidates": self._format_recommendations(candidates[:candidate_k]),
            "top_200_movie_ids": [movie_id for movie_id, _ in candidates[:candidate_k]],
        }

    def update_taste(
        self, user_id: int, movie_id: int, rating: float, alpha: float = 0.7
    ) -> None:
        if movie_id not in self.movie_to_idx:
            raise ValueError(f"Unknown movie_id: {movie_id}")

        if rating < 3.0:
            return

        item_idx = int(self.movie_to_idx[movie_id])
        item_vector = self.item_factors[item_idx]
        strength = min((float(rating) - 3.0) / 2.0, 1.0)
        effective_alpha = 1.0 - ((1.0 - alpha) * strength)

        user_vector, user_idx = self._user_vector(user_id)
        if user_vector is None:
            user_vector = self.mean_user_vector.copy()
            self.extra_user_factors[user_id] = user_vector

        new_vector = effective_alpha * user_vector + (1.0 - effective_alpha) * item_vector

        if user_idx is None:
            self.extra_user_factors[user_id] = new_vector.astype(np.float32)
            self.extra_seen_by_user.setdefault(user_id, set()).add(item_idx)
        else:
            self.user_factors[user_idx] = new_vector.astype(np.float32)
            self.seen_by_user.setdefault(user_idx, set()).add(item_idx)


if __name__ == "__main__":
    model = Model1()
    sample_user_id = next(iter(model.user_to_idx.keys()))
    sample_movie_id = next(iter(model.movie_to_idx.keys()))

    before = model.get_candidates(sample_user_id, k=5)
    model.update_taste(sample_user_id, sample_movie_id, rating=5.0)
    after = model.get_candidates(sample_user_id, k=5)

    print(f"Sample user: {sample_user_id}")
    print("Top 5 before update:")
    print(before)
    print("Top 5 after update:")
    print(after)