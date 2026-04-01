"""
seed_data.py
Insert sample/demo voters into MongoDB for testing.
Generates random (but realistic) iris feature vectors since no real camera is needed.

Run:
    python seed_data.py
"""

import numpy as np
from database.db import Database, register_voter


SAMPLE_VOTERS = [
    {"voter_id": "AP-2024-00001", "name": "Arjun Reddy",       "age": 34, "constituency": "Hyderabad Central"},
    {"voter_id": "AP-2024-00002", "name": "Priya Sharma",      "age": 29, "constituency": "Secunderabad"},
    {"voter_id": "AP-2024-00003", "name": "Mohammed Farhan",   "age": 45, "constituency": "Charminar"},
    {"voter_id": "AP-2024-00004", "name": "Lakshmi Devi",      "age": 52, "constituency": "Jubilee Hills"},
    {"voter_id": "AP-2024-00005", "name": "Rahul Varma",       "age": 27, "constituency": "Kukatpally"},
    {"voter_id": "AP-2024-00006", "name": "Sita Anand",        "age": 38, "constituency": "LB Nagar"},
    {"voter_id": "AP-2024-00007", "name": "Venkat Rao",        "age": 61, "constituency": "Malkajgiri"},
    {"voter_id": "AP-2024-00008", "name": "Deepika Nair",      "age": 23, "constituency": "Ameerpet"},
    {"voter_id": "AP-2024-00009", "name": "Suresh Babu",       "age": 44, "constituency": "Uppal"},
    {"voter_id": "AP-2024-00010", "name": "Ananya Krishnan",   "age": 31, "constituency": "Dilsukhnagar"},
]


def random_iris_features(seed: int) -> list:
    """Generate a deterministic pseudo-iris embedding for demo purposes."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(128).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec.tolist()


def seed():
    Database.connect()
    print("Seeding sample voters into MongoDB…\n")

    for i, voter in enumerate(SAMPLE_VOTERS):
        features = random_iris_features(seed=i + 42)
        result   = register_voter(
            voter["voter_id"], voter["name"],
            voter["age"],      voter["constituency"],
            features,
        )
        status = "✓" if result["success"] else "⚠ (already exists)"
        print(f"  {status}  {voter['voter_id']} — {voter['name']}")

    print(f"\nDone. {len(SAMPLE_VOTERS)} voters processed.")


if __name__ == "__main__":
    seed()
