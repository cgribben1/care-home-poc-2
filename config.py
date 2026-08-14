"""Map YAMNet / AudioSet labels to POC event categories."""

# AudioSet display names matched case-insensitively against yamnet_class_map.csv
# Fall is classifier-only — no YAMNet labels. Cough uses hybrid (YAMNet + classifier).
CATEGORY_LABELS: dict[str, list[str]] = {
    "cough": [
        "Cough",
        "Throat clearing",
    ],
}

# Classes where the classifier alone decides — YAMNet agreement not required
CLASSIFIER_ONLY_CLASSES: set[str] = {"fall"}

# Labels that often cause false positives in care-home-like settings
FALSE_POSITIVE_LABELS: set[str] = {
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Television",
    "Radio",
    "Music",
    "Musical instrument",
    "Child speech, kid speaking",
    "Baby cry, infant cry",
    "Laughter",
    "Giggle",
    "Chuckle, chortle",
    "Snicker",
    "Door",
    "Sliding door",
    "Cupboard open or close",
    "Drawer open or close",
    "Dishes, pots, and pans",
    "Cutlery, silverware",
    "Keys jangling",
    "Coin (dropping)",
    "Typing",
    "Computer keyboard",
    "Mouse",
    "Vacuum cleaner",
    "Mechanical fan",
    "Air conditioning",
    "Traffic noise, roadway noise",
}

# YAMNet label-mapping thresholds (live baseline) — cough only
CATEGORY_THRESHOLDS: dict[str, float] = {
    "cough": 0.08,
}

# Trained classifier thresholds for live ESP32 mic stream
CLASSIFIER_THRESHOLDS: dict[str, float] = {
    "fall": 0.88,
    "cough": 0.85,
}

# Do not run classifier below this RMS (raw waveform)
MIN_ALERT_RMS: float = 0.025

# Light sounds below this skip all preprocessing (no normalize/highpass)
MIN_PREPROCESS_RMS: float = 0.04

# Event must beat normal+other by this margin AND non-alert classes must stay below max
CLASSIFIER_NORMAL_MARGIN: float = 0.35
CLASSIFIER_MAX_NORMAL: float = 0.40

# Require this many consecutive windows before alerting (classifier/hybrid)
CLASSIFIER_ALERT_STREAK: int = 2

# Skip peak-normalize when signal is this quiet (avoids amplifying noise)
MIN_PEAK_NORMALIZE: float = 0.015

DEFAULT_THRESHOLD = 0.15

# How to aggregate YAMNet frame scores into one category score
CATEGORY_SCORING: dict[str, str] = {
    "cough": "top3_mean",
}

SUSTAINED_MIN_FRAMES: int = 2

FALSE_POSITIVE_PENALTY: float = 0.35
FALSE_POSITIVE_TRIGGER: float = 0.25

CATEGORY_DENOISE: dict[str, bool] = {
    "cough": True,
}

SAMPLE_RATE = 16000
CLASSIFIER_PATH = "models/classifier.joblib"
CLASSIFIER_META_PATH = "models/classifier_meta.json"
CLASSIFIER_CLASSES: list[str] = ["fall", "cough", "normal", "other"]

# Non-alert classes — classifier scores these but never fires an alert
NON_ALERT_CLASSES: set[str] = {"normal", "other"}
