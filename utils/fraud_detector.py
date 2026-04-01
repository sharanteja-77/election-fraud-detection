"""
utils/fraud_detector.py
Orchestrates preprocessing → feature extraction → DB matching → fraud decision.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from utils.iris_preprocessor import IrisPreprocessor, preprocess_base64_frame
from models.iris_model import get_model
from database.db import (
    get_voter_by_id,
    get_all_iris_features,
    mark_voter_voted,
    log_voting_attempt,
)


class VerificationResult:
    """Structured result returned from verify_voter()."""

    def __init__(self,
                 status: str,
                 voter_id: Optional[str],
                 voter_name: Optional[str],
                 confidence: float,
                 message: str,
                 timestamp: str):
        self.status     = status        # 'success' | 'duplicate' | 'unrecognised' | 'error'
        self.voter_id   = voter_id
        self.voter_name = voter_name
        self.confidence = confidence
        self.message    = message
        self.timestamp  = timestamp

    def to_dict(self) -> dict:
        return {
            "status":     self.status,
            "voter_id":   self.voter_id,
            "voter_name": self.voter_name,
            "confidence": round(self.confidence * 100, 2),   # percentage
            "message":    self.message,
            "timestamp":  self.timestamp,
            "is_fraud":   self.status in ("duplicate", "unrecognised"),
        }


# ── Main verification function ───────────────────────────────────────────────

def verify_voter(b64_frame: str, ip_address: str = "") -> VerificationResult:
    """
    Full pipeline: webcam frame → iris preprocessing → embedding → DB match →
    duplicate check → vote logging.

    Parameters
    ----------
    b64_frame  : base64-encoded JPEG/PNG from the webcam (data-URL or raw)
    ip_address : client IP for audit logging

    Returns
    -------
    VerificationResult
    """
    ts = datetime.utcnow().isoformat()

    # 1. Preprocess iris from the frame
    iris_img = preprocess_base64_frame(b64_frame)
    if iris_img is None:
        return VerificationResult(
            status="error", voter_id=None, voter_name=None,
            confidence=0.0,
            message="Could not detect or preprocess iris. Please adjust lighting and try again.",
            timestamp=ts,
        )

    # 2. Extract features
    try:
        model    = get_model()
        features = model.extract_features(iris_img)
    except Exception as e:
        return VerificationResult(
            status="error", voter_id=None, voter_name=None,
            confidence=0.0,
            message=f"Feature extraction failed: {str(e)}",
            timestamp=ts,
        )

    # 3. Match against stored iris embeddings
    stored = get_all_iris_features()
    matched_id, confidence = model.match(features, stored)

    if matched_id is None:
        log_voting_attempt("UNKNOWN", "unrecognised", confidence, ip_address)
        return VerificationResult(
            status="unrecognised", voter_id=None, voter_name=None,
            confidence=confidence,
            message="Iris not recognised. Voter is not registered in the system.",
            timestamp=ts,
        )

    # 4. Fetch full voter record
    voter = get_voter_by_id(matched_id)
    if voter is None:
        log_voting_attempt(matched_id, "error", confidence, ip_address)
        return VerificationResult(
            status="error", voter_id=matched_id, voter_name=None,
            confidence=confidence,
            message="Voter record missing from database.",
            timestamp=ts,
        )

    # 5. Duplicate vote check
    if voter.get("has_voted", False):
        log_voting_attempt(matched_id, "duplicate", confidence, ip_address)
        return VerificationResult(
            status="duplicate",
            voter_id=matched_id,
            voter_name=voter.get("name"),
            confidence=confidence,
            message=(
                f"⚠️  FRAUD DETECTED: Voter '{voter['name']}' (ID: {matched_id}) "
                f"has already cast a vote."
            ),
            timestamp=ts,
        )

    # 6. Mark as voted and log success
    mark_voter_voted(matched_id)
    log_voting_attempt(matched_id, "success", confidence, ip_address)

    return VerificationResult(
        status="success",
        voter_id=matched_id,
        voter_name=voter.get("name"),
        confidence=confidence,
        message=(
            f"✅  Identity verified. Welcome, {voter['name']}! "
            f"Vote recorded successfully."
        ),
        timestamp=ts,
    )


# ── Registration helper ──────────────────────────────────────────────────────

def extract_iris_features_from_b64(b64_frame: str) -> Optional[list]:
    """
    Extract and return iris features (as a Python list) from a base64 frame.
    Used during voter registration to store the iris embedding.

    Returns None on failure.
    """
    iris_img = preprocess_base64_frame(b64_frame)
    if iris_img is None:
        return None
    try:
        features = get_model().extract_features(iris_img)
        return features.tolist()
    except Exception:
        return None


def extract_iris_features_from_file(filepath: str) -> Optional[list]:
    """Same as above but reads from a saved image file."""
    import cv2
    img = cv2.imread(filepath)
    if img is None:
        return None
    processed = IrisPreprocessor().preprocess(img)
    if processed is None:
        return None
    try:
        features = get_model().extract_features(processed)
        return features.tolist()
    except Exception:
        return None
