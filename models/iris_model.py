import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Optional, Tuple, List, Dict

# ── Constants
INPUT_SHAPE     = (64, 64, 1)
EMBEDDING_DIM   = 128
MATCH_THRESHOLD = 0.75

MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.h5")


# ── Custom Layer
class L2Normalize(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


# ── Model Definition
def build_embedding_model() -> tf.keras.Model:
    inp = layers.Input(shape=INPUT_SHAPE)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(EMBEDDING_DIM)(x)
    x = L2Normalize(name="l2_norm")(x)

    return tf.keras.Model(inp, x, name="IrisEmbedder")


def build_classifier(num_classes: int) -> tf.keras.Model:
    base = build_embedding_model()
    out = layers.Dense(num_classes, activation="softmax")(base.output)
    return tf.keras.Model(base.input, out, name="IrisClassifier")


# ── IrisModel
class IrisModel:

    def __init__(self):
        self.embedder: Optional[tf.keras.Model] = None
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.embedder = tf.keras.models.load_model(
                    MODEL_PATH,
                    custom_objects={"L2Normalize": L2Normalize}
                )
                print("✅ Model loaded")
                return
            except Exception as e:
                print(f"⚠️ Load failed: {e}")

        self.embedder = build_embedding_model()
        print("🆕 New model initialized")

    # ── Training
    def train(self, X, y, epochs=30, batch_size=32, val_split=0.15):

        X = X.astype(np.float32)

        num_classes = len(np.unique(y))
        model = build_classifier(num_classes)

        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        cb = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=3),
            callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
        ]

        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=cb
        )

        # Extract embedding model
        self.embedder = tf.keras.Model(
            model.input,
            model.get_layer("l2_norm").output
        )

        self.embedder.save(MODEL_PATH)
        return history.history

    # ── Feature Extraction
    def extract_features(self, img):

        if self.embedder is None:
            raise RuntimeError("Model not loaded")

        img = img.astype(np.float32)

        if img.shape != INPUT_SHAPE:
            raise ValueError(f"Expected shape {INPUT_SHAPE}, got {img.shape}")

        img = np.expand_dims(img, axis=0)
        return self.embedder.predict(img, verbose=0)[0]

    # ── Matching
    def match(self, probe, records: List[Dict]):

        if len(records) == 0:
            return None, 0.0

        probe = probe / (np.linalg.norm(probe) + 1e-8)

        best_id = None
        best_score = -1

        for r in records:
            vec = np.array(r["iris_features"], dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)

            score = np.dot(probe, vec)

            if score > best_score:
                best_score = score
                best_id = r["voter_id"]

        if best_score >= MATCH_THRESHOLD:
            return best_id, float(best_score)

        return None, float(best_score)

    def save(self, path=MODEL_PATH):
        if self.embedder:
            self.embedder.save(path)


# ── Singleton
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = IrisModel()
    return _model_instance