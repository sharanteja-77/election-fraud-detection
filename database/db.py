"""
database/db.py
MongoDB connection and all database operations for the Election Fraud Detection system.
"""

import os
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from dotenv import load_dotenv
import numpy as np

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("DB_NAME",   "election_fraud_db")


class Database:
    """Singleton MongoDB connection wrapper."""

    _client: MongoClient = None
    _db = None

    @classmethod
    def connect(cls):
        if cls._client is None:
            try:
                cls._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                cls._client.admin.command("ping")          # Verify connection
                cls._db = cls._client[DB_NAME]
                cls._create_indexes()
                print(f"[DB] Connected to MongoDB: {DB_NAME}")
            except ConnectionFailure as e:
                print(f"[DB ERROR] Cannot connect to MongoDB: {e}")
                raise
        return cls._db

    @classmethod
    def get_db(cls):
        if cls._db is None:
            cls.connect()
        return cls._db

    @classmethod
    def _create_indexes(cls):
        db = cls._db
        # Unique voter ID index
        db.voters.create_index([("voter_id", ASCENDING)], unique=True)
        # Voting log index for fast duplicate lookups
        db.voting_logs.create_index([("voter_id", ASCENDING)])
        db.voting_logs.create_index([("timestamp", ASCENDING)])

    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db     = None
            print("[DB] Connection closed.")


# ─────────────────────────────────────────────
# Voter operations
# ─────────────────────────────────────────────

def register_voter(voter_id: str, name: str, age: int,
                   constituency: str, iris_features: list) -> dict:
    """
    Register a new voter with their iris feature vector.

    Parameters
    ----------
    voter_id      : Unique Aadhaar / voter card number
    name          : Full name
    age           : Age
    constituency  : Electoral constituency
    iris_features : 1-D list / numpy array of iris feature embeddings

    Returns
    -------
    dict with 'success' bool and 'message' string
    """
    db = Database.get_db()
    try:
        # Convert numpy array → list for BSON serialisation
        if isinstance(iris_features, np.ndarray):
            iris_features = iris_features.tolist()

        doc = {
            "voter_id":      voter_id,
            "name":          name,
            "age":           age,
            "constituency":  constituency,
            "iris_features": iris_features,
            "has_voted":     False,
            "registered_at": datetime.utcnow(),
        }
        db.voters.insert_one(doc)
        return {"success": True, "message": f"Voter '{name}' registered successfully."}

    except DuplicateKeyError:
        return {"success": False, "message": f"Voter ID '{voter_id}' already exists."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_voter_by_id(voter_id: str) -> dict | None:
    """Return a voter document or None."""
    db = Database.get_db()
    return db.voters.find_one({"voter_id": voter_id}, {"_id": 0})


def get_all_voters() -> list:
    """Return all registered voters (without MongoDB _id)."""
    db = Database.get_db()
    return list(db.voters.find({}, {"_id": 0, "iris_features": 0}))


def clear_all_voters() -> int:
    """Delete all registered voters and return the number of removed documents."""
    db = Database.get_db()
    result = db.voters.delete_many({})
    return result.deleted_count


def mark_voter_voted(voter_id: str) -> bool:
    """Set has_voted=True for a voter. Returns True on success."""
    db = Database.get_db()
    result = db.voters.update_one(
        {"voter_id": voter_id},
        {"$set": {"has_voted": True, "voted_at": datetime.utcnow()}}
    )
    return result.modified_count > 0


def get_all_iris_features() -> list[dict]:
    """
    Return all voters' (voter_id, iris_features) for in-memory matching.
    """
    db = Database.get_db()
    return list(db.voters.find({}, {"_id": 0, "voter_id": 1, "iris_features": 1}))


# ─────────────────────────────────────────────
# Voting log operations
# ─────────────────────────────────────────────

def log_voting_attempt(voter_id: str, status: str,
                       confidence: float, ip_address: str = "") -> None:
    """
    Insert a record into voting_logs for every attempt (success or fraud).

    status: 'success' | 'duplicate' | 'unrecognised' | 'error'
    """
    db = Database.get_db()
    db.voting_logs.insert_one({
        "voter_id":   voter_id,
        "status":     status,
        "confidence": round(float(confidence), 4),
        "ip_address": ip_address,
        "timestamp":  datetime.utcnow(),
    })


def get_voting_logs(limit: int = 100) -> list:
    """Return the most recent voting logs."""
    db = Database.get_db()
    return list(
        db.voting_logs.find({}, {"_id": 0})
                      .sort("timestamp", -1)
                      .limit(limit)
    )


def get_fraud_attempts() -> list:
    """Return all duplicate / fraud attempts."""
    db = Database.get_db()
    return list(db.voting_logs.find(
        {"status": {"$in": ["duplicate", "unrecognised"]}},
        {"_id": 0}
    ).sort("timestamp", -1))


# ─────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────

def get_dashboard_stats() -> dict:
    db = Database.get_db()

    total_voters    = db.voters.count_documents({})
    voted_count     = db.voters.count_documents({"has_voted": True})
    total_logs      = db.voting_logs.count_documents({})
    fraud_count     = db.voting_logs.count_documents(
                          {"status": {"$in": ["duplicate", "unrecognised"]}})
    success_count   = db.voting_logs.count_documents({"status": "success"})

    return {
        "total_voters":  total_voters,
        "voted_count":   voted_count,
        "pending_count": total_voters - voted_count,
        "total_attempts": total_logs,
        "successful_votes": success_count,
        "fraud_attempts": fraud_count,
        "turnout_pct":   round((voted_count / total_voters * 100), 2) if total_voters else 0,
    }
