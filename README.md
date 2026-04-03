# 🗳️ Election Fraud Detection System
## Iris Recognition–Based Duplicate Vote Prevention

**Team:** M Kalyanteja · K Sharan Teja · Pream Joel · T Balaram Singh · M Varshitha · V Hasini  
**Stack:** Python · Flask · TensorFlow · OpenCV · MongoDB

---

## 📁 Project Structure

```
election_fraud_detection/
│
├── app.py                   # Flask application (routes + API)
├── train_model.py           # Standalone model training script
├── seed_data.py             # Insert demo voters into MongoDB
├── requirements.txt         # Python dependencies
├── .env                     # Configuration (edit before running)
│
├── models/
│   ├── __init__.py
│   └── iris_model.py        # TensorFlow CNN — 128-D iris embedding
│
├── utils/
│   ├── __init__.py
│   ├── iris_preprocessor.py # OpenCV iris detection + normalisation
│   └── fraud_detector.py    # Full verification pipeline
│
├── database/
│   ├── __init__.py
│   └── db.py                # MongoDB connection + all DB operations
│
├── templates/               # Jinja2 HTML pages
│   ├── base.html            # Shared layout + dark civic design
│   ├── index.html           # Dashboard
│   ├── register.html        # Voter registration + webcam capture
│   ├── verify.html          # Live iris verification
│   └── logs.html            # Audit logs + fraud table
│
└── static/
    └── uploads/             # Uploaded iris images (auto-created)
```

---

## ⚙️ Setup & Installation


### 1. Create virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Edit `.env`:

```
MONGO_URI=mongodb://localhost:27017/
DB_NAME=election_fraud_db
SECRET_KEY=change-this-to-a-long-random-string
```

### 4. Start MongoDB

```bash
# Make sure MongoDB is running locally
mongod --dbpath /data/db
```

### 5. (Optional) Seed demo voters

```bash
python seed_data.py
```

### 6. Run the application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🚀 Deployment with Free Services

### 1. Use Free MongoDB Atlas
1. Create a free MongoDB Atlas account.
2. Create a free cluster.
3. Create a database user and whitelist your IP (or use 0.0.0.0/0 for deployment).
4. Copy the connection string and set it in `.env` or use `MONGO_URI` in your hosting provider.

Example `.env` values:

```
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-address>/?retryWrites=true&w=majority
DB_NAME=election_fraud_db
SECRET_KEY=change-this-to-a-long-random-string
FLASK_DEBUG=False
UPLOAD_FOLDER=static/uploads
```

### 2. Deploy to a free Python host

#### Render (recommended free tier)
1. Sign up at https://render.com.
2. Create a new Web Service from your GitHub repo.
3. Set the build command to:

```bash
pip install -r requirements.txt
```

4. Set the start command to:

```bash
python app.py
```

5. Add environment variables:
- `MONGO_URI`
- `DB_NAME`
- `SECRET_KEY`
- `UPLOAD_FOLDER=static/uploads`

#### Railway
1. Sign up at https://railway.app.
2. Link your GitHub repo.
3. Add the same environment variables.
4. Use `python app.py` as the start command.

### 3. Push your code

Make sure your changes are committed and pushed:

```bash
git add .
git commit -m "Prepare deployment with .env.example and Procfile"
git push origin main
```

### 4. Access your deployed app

Once the host finishes building, open the assigned URL and verify the app loads.

---

## 🧠 How It Works

### Iris Recognition Pipeline

```
Webcam Frame
     │
     ▼
[IrisPreprocessor]
  • Grayscale conversion
  • CLAHE contrast enhancement
  • HoughCircles — detect iris + pupil boundaries
  • Doughnut mask → isolate iris band
  • Resize to 64×64, normalise [0,1]
     │
     ▼
[IrisModel — CNN Embedder]
  Conv2D(32) → BN → MaxPool
  Conv2D(64) → BN → MaxPool
  Conv2D(128)→ BN → MaxPool
  Conv2D(256)→ BN → GlobalAvgPool
  Dense(128) → L2-Normalise
     │
     ▼
[128-D Unit Embedding Vector]
     │
     ▼
[Cosine Similarity vs. All Stored Voters]
     │
     ├─ Score < 0.75  → Unrecognised (new/unregistered person)
     │
     └─ Score ≥ 0.75  → Best match found
                           │
                           ├─ has_voted = True  → ⚠️  FRAUD — Duplicate vote
                           │
                           └─ has_voted = False → ✅  VERIFIED — Vote recorded
```

---

## 🌐 API Endpoints

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/`             | Dashboard page                     |
| GET    | `/register`     | Voter registration page            |
| GET    | `/verify`       | Live verification page             |
| GET    | `/logs`         | Audit logs page                    |
| GET    | `/api/stats`    | Dashboard statistics (JSON)        |
| POST   | `/api/register` | Register new voter (JSON or form)  |
| POST   | `/api/verify`   | Verify voter from webcam frame     |
| GET    | `/api/voters`   | List all registered voters         |
| GET    | `/api/logs`     | Voting attempt logs                |
| GET    | `/api/fraud`    | Fraud / duplicate alerts           |
| GET    | `/api/health`   | Health check                       |

### POST `/api/register` — JSON body
```json
{
  "voter_id":      "AP-2024-00001",
  "name":          "Arjun Reddy",
  "age":           34,
  "constituency":  "Hyderabad Central",
  "iris_image_b64": "<base64 JPEG>"
}
```

### POST `/api/verify` — JSON body
```json
{
  "frame": "<base64 JPEG from webcam>"
}
```

---

## 🏋️ Training the Model

### Dataset structure

```
data/iris_dataset/
    voter_001/
        iris_1.jpg
        iris_2.jpg
    voter_002/
        iris_1.jpg
    ...
```

### Run training

```bash
python train_model.py --data_dir data/iris_dataset --epochs 50 --batch 32 --augment 4
```

Trained model is saved to `models/iris_model.h5`.

---

## 🗄️ MongoDB Collections

### `voters`
```json
{
  "voter_id":      "AP-2024-00001",
  "name":          "Arjun Reddy",
  "age":           34,
  "constituency":  "Hyderabad Central",
  "iris_features": [0.021, -0.134, ...],   // 128-D embedding
  "has_voted":     false,
  "registered_at": "2024-11-01T10:30:00Z"
}
```

### `voting_logs`
```json
{
  "voter_id":   "AP-2024-00001",
  "status":     "success",          // success | duplicate | unrecognised | error
  "confidence": 0.9312,
  "ip_address": "192.168.1.10",
  "timestamp":  "2024-11-15T09:45:00Z"
}
```

---

## 🔮 Future Enhancements (as per PPT)

- National central voter database integration
- Deep learning (ResNet / EfficientNet backbone)
- Multi-biometric: iris + fingerprint + face
- Liveness detection (anti-spoofing)
- Mobile app for field officers
- Encrypted iris templates (privacy-preserving biometrics)

---

## 📄 License

Academic project — developed for educational purposes.
