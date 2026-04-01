"""
app.py
Flask application — Election Fraud Detection System
Routes: voter registration, iris verification, dashboard, logs
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from database.db import (
    Database,
    register_voter,
    get_all_voters,
    clear_all_voters,
    get_voting_logs,
    get_fraud_attempts,
    get_dashboard_stats,
)
from utils.fraud_detector import (
    verify_voter,
    extract_iris_features_from_b64,
    extract_iris_features_from_file,
)

load_dotenv()

# ── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"]      = os.getenv("SECRET_KEY", "change-me")
app.config["UPLOAD_FOLDER"]   = os.getenv("UPLOAD_FOLDER", "static/uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Connect to MongoDB on startup
with app.app_context():
    try:
        Database.connect()
    except Exception as e:
        print(f"[WARN] MongoDB connection failed at startup: {e}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "")


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Dashboard / home page."""
    return render_template("index.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/verify")
def verify_page():
    return render_template("verify.html")


@app.route("/logs")
def logs_page():
    return render_template("logs.html")


# ── API: Dashboard stats ──────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return dashboard statistics."""
    try:
        stats = get_dashboard_stats()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── API: Register voter ───────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    """
    Register a new voter.

    Accepts multipart/form-data OR application/json.

    JSON body:
        { voter_id, name, age, constituency, iris_image_b64 }

    Form data:
        voter_id, name, age, constituency  +  file: iris_image (image upload)
    """
    try:
        # ── JSON path ──────────────────────────────────────────────
        if request.is_json:
            data         = request.get_json()
            voter_id     = data.get("voter_id", "").strip()
            name         = data.get("name", "").strip()
            age          = int(data.get("age", 0))
            constituency = data.get("constituency", "").strip()
            b64_iris     = data.get("iris_image_b64", "")

            if not all([voter_id, name, age, constituency, b64_iris]):
                return jsonify({"success": False,
                                "message": "All fields including iris image are required."}), 400

            features = extract_iris_features_from_b64(b64_iris)
            if features is None:
                return jsonify({"success": False,
                                "message": "Could not extract iris features from image."}), 422

        # ── Form/file upload path ──────────────────────────────────
        else:
            voter_id     = request.form.get("voter_id", "").strip()
            name         = request.form.get("name", "").strip()
            age          = int(request.form.get("age", 0))
            constituency = request.form.get("constituency", "").strip()

            if not all([voter_id, name, age, constituency]):
                return jsonify({"success": False,
                                "message": "voter_id, name, age and constituency are required."}), 400

            if "iris_image" not in request.files:
                return jsonify({"success": False,
                                "message": "iris_image file is required."}), 400

            file = request.files["iris_image"]
            if file.filename == "" or not allowed_file(file.filename):
                return jsonify({"success": False,
                                "message": "Please upload a valid PNG/JPG image."}), 400

            filename = secure_filename(f"{voter_id}_{file.filename}")
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            features = extract_iris_features_from_file(filepath)
            if features is None:
                return jsonify({"success": False,
                                "message": "Could not extract iris features from uploaded image."}), 422

        # ── Persist to MongoDB ─────────────────────────────────────
        result = register_voter(voter_id, name, age, constituency, features)
        status_code = 201 if result["success"] else 409
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


# ── API: Verify voter (live webcam frame) ─────────────────────────────────────

@app.route("/api/verify", methods=["POST"])
def api_verify():
    """
    Verify a voter by iris recognition.

    JSON body:
        { "frame": "<base64 encoded image>" }
    """
    try:
        data  = request.get_json()
        frame = (data or {}).get("frame", "")

        if not frame:
            return jsonify({"success": False,
                            "message": "No frame provided."}), 400

        result = verify_voter(frame, ip_address=client_ip())
        return jsonify({"success": True, "result": result.to_dict()})

    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Verification error: {str(e)}"}), 500


# ── API: Voters list ──────────────────────────────────────────────────────────

@app.route("/api/voters", methods=["GET"])
def api_voters():
    """Return all registered voters (no iris data)."""
    try:
        voters = get_all_voters()
        return jsonify({"success": True, "data": voters, "count": len(voters)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/clear_registrations", methods=["POST"])
def api_clear_registrations():
    """Clear all registered voters from the database."""
    try:
        deleted = clear_all_voters()
        return jsonify({"success": True, "deleted": deleted, "message": f"{deleted} voter registrations cleared."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── API: Voting logs ──────────────────────────────────────────────────────────

@app.route("/api/logs", methods=["GET"])
def api_logs():
    """Return recent voting logs."""
    try:
        limit = int(request.args.get("limit", 100))
        logs  = get_voting_logs(limit)
        # Convert datetime objects to ISO strings
        for log in logs:
            if hasattr(log.get("timestamp"), "isoformat"):
                log["timestamp"] = log["timestamp"].isoformat()
        return jsonify({"success": True, "data": logs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/fraud", methods=["GET"])
def api_fraud():
    """Return all detected fraud / duplicate attempts."""
    try:
        fraud = get_fraud_attempts()
        for f in fraud:
            if hasattr(f.get("timestamp"), "isoformat"):
                f["timestamp"] = f["timestamp"].isoformat()
        return jsonify({"success": True, "data": fraud, "count": len(fraud)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Election Fraud Detection API"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "True").lower() == "true",
    )
