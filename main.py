from dotenv import load_dotenv
import os
load_dotenv()

print(f"API key loaded: {bool(os.getenv('ANTHROPIC_API_KEY'))}")

import anthropic
import base64
import json as json_lib
import pickle
import random
import re
import threading
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

from utilities import analyze_sample_rate
from signal_processing import butter_lowpass_filter_df, normalize_df_period
from classification_helpers import compute_minimal_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared constants ──────────────────────────────────────────────────────────

SAMPLE_RATE_HZ = 100.0
DT_US          = int(1_000_000 / SAMPLE_RATE_HZ)   # 10 000 µs per sample

# Mapping from frontend feature key → computed DataFrame column names
FEATURE_COL_MAP: Dict[str, List[str]] = {
    "mean":         ["a_x__mean",               "a_y__mean",               "a_z__mean"],
    "std_dev":      ["a_x__standard_deviation",  "a_y__standard_deviation",  "a_z__standard_deviation"],
    "rms":          ["a_x__root_mean_square",    "a_y__root_mean_square",    "a_z__root_mean_square"],
    "peak":         ["a_x__maximum",             "a_y__maximum",             "a_z__maximum"],
    "absolute_max": ["a_x__absolute_maximum",    "a_y__absolute_maximum",    "a_z__absolute_maximum"],
    "fft_energy":   ["a_x__fft_energy",          "a_y__fft_energy",          "a_z__fft_energy"],
    "dominant_freq":["a_x__dominant_freq",       "a_y__dominant_freq",       "a_z__dominant_freq"],
    "kurtosis":     ["a_x__kurtosis",            "a_y__kurtosis",            "a_z__kurtosis"],
}

MODEL_DISPLAY = {
    "rf":  "Random Forest",
    "svm": "SVM",
    "nn":  "Neural Net",
}

# ── In-memory stores ──────────────────────────────────────────────────────────

projects:       Dict[str, dict] = {}
project_events: Dict[str, list] = {}   # project_id → list[EventData]
_saved_pipeline: Optional[dict] = None  # set after successful training

_training_status: dict = {
    "state":         "idle",   # idle | running | done | error
    "progress":      0,
    "current_model": "",
    "results":       None,
    "error":         None,
}
_training_lock = threading.Lock()


def _set_status(**kwargs):
    with _training_lock:
        _training_status.update(kwargs)


# ── /ping ─────────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "EdgeForge API running"}


# ── /project/create ───────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    name:                    str
    sensor_type:             str
    connection_type:         str
    trigger_type:            str
    trigger_config:          dict
    target_mcu:              str
    application_description: Optional[str]  = None
    hardware_preprocessing:  Optional[dict] = None


@app.post("/project/create")
def create_project(config: ProjectConfig):
    project_id = config.name.lower().replace(" ", "-")
    projects[project_id] = config.dict()
    return {"project_id": project_id, "config": config}


# ── /analyze-signal ───────────────────────────────────────────────────────────

class EventData(BaseModel):
    ax: List[float]
    ay: List[float] = []
    az: List[float] = []
    duration_ms: float
    class_label: str = ""


class AnalyzeSignalRequest(BaseModel):
    events:         List[EventData]
    sample_rate_hz: float = 100.0
    project_id:     Optional[str] = None   # if set, events are cached for /train


def _build_sample_df(samples: List[float], dt_us: int) -> pd.DataFrame:
    n = len(samples)
    return pd.DataFrame({
        "timestamp": [dt_us] * n,
        "a_x": samples,
        "a_y": [0.0] * n,
        "a_z": [0.0] * n,
    })


def _find_90pct_cutoff(signal: np.ndarray, sr: float) -> float:
    n = len(signal)
    if n < 4:
        return sr / 4.0
    fft_vals   = np.fft.rfft(signal)
    power      = np.abs(fft_vals) ** 2
    freqs      = np.fft.rfftfreq(n, d=1.0 / sr)
    total      = power.sum()
    if total == 0:
        return sr / 4.0
    cumulative = np.cumsum(power)
    idx        = int(np.searchsorted(cumulative, 0.90 * total))
    return float(freqs[min(idx, len(freqs) - 1)])


@app.post("/analyze-signal")
def analyze_signal(req: AnalyzeSignalRequest):
    if not req.events:
        raise HTTPException(status_code=400, detail="events list is empty")

    # Cache events for later training
    if req.project_id:
        project_events[req.project_id] = req.events

    sr    = req.sample_rate_hz
    dt_us = int(1_000_000 / sr)

    first = req.events[0]
    if not first.ax:
        raise HTTPException(status_code=400, detail="first event has no ax samples")

    sr_df       = _build_sample_df(first.ax, dt_us)
    measured_hz = analyze_sample_rate(sr_df)
    sr_delta    = abs(measured_hz - sr)
    sr_note     = (
        "consistent with declared rate." if sr_delta < 2
        else f"diverges by {sr_delta:.1f} Hz — verify device clock."
    )

    axis_event_cutoffs: Dict[str, List[float]] = {"ax": [], "ay": [], "az": []}
    for ev in req.events:
        for axis, samples in [("ax", ev.ax), ("ay", ev.ay), ("az", ev.az)]:
            if samples:
                axis_event_cutoffs[axis].append(
                    _find_90pct_cutoff(np.array(samples, dtype=float), sr)
                )

    axis_cutoffs_avg: Dict[str, float] = {}
    for axis, vals in axis_event_cutoffs.items():
        if vals:
            axis_cutoffs_avg[axis] = round(float(np.mean(vals)), 1)

    nyquist = sr / 2.0
    if axis_cutoffs_avg:
        max_cutoff         = max(axis_cutoffs_avg.values())
        recommended_cutoff = round(min(max_cutoff * 1.15, nyquist * 0.90), 1)
    else:
        recommended_cutoff = round(nyquist * 0.40, 1)

    durations          = np.array([ev.duration_ms for ev in req.events], dtype=float)
    dur_min            = float(durations.min())
    dur_max            = float(durations.max())
    dur_mean           = float(durations.mean())
    dur_p90            = float(np.percentile(durations, 90))
    recommended_window = int(np.ceil(dur_p90 / 50) * 50)

    return {
        "event_count": len(req.events),
        "sample_rate": {
            "measured_hz": round(measured_hz, 1),
            "declared_hz": sr,
            "explanation": (
                f"Reconstructed from {len(first.ax)}-sample event at "
                f"declared {sr} Hz ({dt_us} µs/sample). "
                f"Measured {round(measured_hz, 1)} Hz — {sr_note}"
            ),
        },
        "cutoff_frequency": {
            "recommended_hz":       recommended_cutoff,
            "energy_threshold_pct": 90,
            "axis_cutoffs_hz":      axis_cutoffs_avg,
            "explanation": (
                f"90% of signal energy across {len(req.events)} event(s) lies below: "
                + ", ".join(f"{ax}={hz} Hz" for ax, hz in axis_cutoffs_avg.items())
                + f". Recommended cutoff: {recommended_cutoff} Hz. Nyquist: {nyquist} Hz."
            ),
        },
        "normalization_window": {
            "recommended_ms": recommended_window,
            "min_ms":         round(dur_min, 1),
            "max_ms":         round(dur_max, 1),
            "mean_ms":        round(dur_mean, 1),
            "p90_ms":         round(dur_p90, 1),
            "explanation": (
                f"Event durations — min: {dur_min:.0f} ms, mean: {dur_mean:.0f} ms, "
                f"max: {dur_max:.0f} ms, p90: {dur_p90:.0f} ms ({len(req.events)} event(s)). "
                f"Recommended normalization window: {recommended_window} ms."
            ),
        },
    }


# ── /train ────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    project_id:        str
    cutoff_hz:         float       = 30.0
    window_ms:         float       = 1000.0
    interpolation:     str         = "cubic"
    selected_features: List[str]   = []
    model_type:        str         = "auto"    # auto | rf | svm | nn


def _preprocess_event(ev: EventData, cutoff_hz: float,
                      window_ms: float, interpolation: str) -> pd.DataFrame:
    n = len(ev.ax)
    df = pd.DataFrame({
        "timestamp": [DT_US] * n,
        "a_x": ev.ax,
        "a_y": ev.ay if ev.ay else [0.0] * n,
        "a_z": ev.az if ev.az else [0.0] * n,
    })

    # Butterworth low-pass (clamp cutoff to safe range)
    safe_cutoff = max(1.0, min(float(cutoff_hz), SAMPLE_RATE_HZ * 0.45))
    if n >= 20:
        try:
            df = butter_lowpass_filter_df(df, fs=SAMPLE_RATE_HZ, cutoff=safe_cutoff)
        except Exception:
            pass

    # Normalize to target window length
    try:
        df = normalize_df_period(window_ms, DT_US, df, interpolationKind=interpolation)
    except Exception:
        pass

    return df


def _freq_features(ev: EventData) -> dict:
    """FFT energy, dominant frequency, scipy kurtosis per axis."""
    feats: dict = {}
    for attr, prefix in [("ax", "a_x"), ("ay", "a_y"), ("az", "a_z")]:
        samples = getattr(ev, attr, [])
        if not samples:
            continue
        arr    = np.array(samples, dtype=float)
        fft    = np.fft.rfft(arr)
        power  = np.abs(fft) ** 2
        freqs  = np.fft.rfftfreq(len(arr), d=1.0 / SAMPLE_RATE_HZ)
        kval   = float(sp_kurtosis(arr))
        if not np.isfinite(kval):
            kval = 0.0

        feats[f"{prefix}__fft_energy"]    = float(np.sum(power))
        feats[f"{prefix}__dominant_freq"] = float(freqs[int(np.argmax(power))])
        feats[f"{prefix}__kurtosis"]      = kval

    return feats


def _train_one(model_id: str, X: np.ndarray, y: np.ndarray, le: LabelEncoder) -> dict:
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    if model_id == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    elif model_id == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=42)
    elif model_id == "nn":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    # Stratified k-fold — fall back to 2 if any class is tiny
    min_per_class = int(np.bincount(y).min())
    n_splits      = max(2, min(5, min_per_class))

    t0 = time.time()

    if n_splits >= 2 and len(y) >= n_splits * 2:
        cv         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores     = cross_val_score(clf, X_sc, y, cv=cv, scoring="accuracy")
        cv_accuracy = float(np.mean(scores))
    else:
        # Too few samples: use train accuracy for both
        clf.fit(X_sc, y)
        cv_accuracy = float((clf.predict(X_sc) == y).mean())

    training_time = round(time.time() - t0, 2)

    # Refit on all data to get train accuracy + confusion matrix
    clf.fit(X_sc, y)
    y_pred         = clf.predict(X_sc)
    train_accuracy = float((y_pred == y).mean())
    cm             = confusion_matrix(y, y_pred).tolist()

    return {
        "accuracy":         round(train_accuracy, 4),
        "cv_accuracy":      round(cv_accuracy, 4),
        "confusion_matrix": cm,
        "class_names":      le.classes_.tolist(),
        "training_time_s":  training_time,
        # Internal — used by _run_training to save the best pipeline; stripped before sending to frontend
        "_scaler":          scaler,
        "_clf":             clf,
    }


def _run_training(req: TrainRequest, events: List[EventData]):
    """Synchronous — runs inside a daemon thread."""

    _set_status(state="running", progress=0, current_model="Preprocessing",
                results=None, error=None)
    try:
        # ── 1. Preprocess events ──────────────────────────────────────────────
        processed: List[pd.DataFrame] = []
        labels:    List[str]          = []
        orig_idx:  List[int]          = []

        for i, ev in enumerate(events):
            if not ev.ax:
                continue
            try:
                df = _preprocess_event(ev, req.cutoff_hz, req.window_ms, req.interpolation)
                df["event_id"] = len(processed)
                processed.append(df)
                labels.append(ev.class_label or "unknown")
                orig_idx.append(i)
            except Exception:
                continue

        if len(processed) < 2:
            raise ValueError("Need at least 2 preprocessable events to train.")

        unique = list(set(labels))
        if len(unique) < 2:
            raise ValueError(
                f"All events share the same class label ({unique[0]!r}). "
                "Add a second class before training."
            )

        _set_status(progress=12)

        # ── 2. Feature extraction ──────────────────────────────────────────────
        combined  = pd.concat(processed, ignore_index=True)
        feat_df   = compute_minimal_features(combined, group_col="event_id")

        # Append freq-domain features
        for seq, oi in enumerate(orig_idx):
            ffeats = _freq_features(events[oi])
            for col, val in ffeats.items():
                feat_df.loc[seq, col] = val

        _set_status(progress=20)

        # ── 3. Column selection ────────────────────────────────────────────────
        selected_cols: List[str] = []
        for feat in (req.selected_features or []):
            if feat in FEATURE_COL_MAP:
                selected_cols.extend(
                    c for c in FEATURE_COL_MAP[feat] if c in feat_df.columns
                )

        if not selected_cols:        # fallback: first five time-domain feature groups
            fallback_keys = ["mean", "std_dev", "rms", "peak", "absolute_max"]
            for k in fallback_keys:
                selected_cols.extend(
                    c for c in FEATURE_COL_MAP[k] if c in feat_df.columns
                )

        X = feat_df[selected_cols].fillna(0.0).values.astype(float)
        le = LabelEncoder()
        y  = le.fit_transform(labels)

        # ── 4. Train models ────────────────────────────────────────────────────
        model_ids  = ["rf", "svm", "nn"] if req.model_type == "auto" else [req.model_type]
        n_models   = len(model_ids)
        prog_each  = (95 - 20) // n_models
        results    = {}

        for step, mid in enumerate(model_ids):
            _set_status(
                current_model=MODEL_DISPLAY.get(mid, mid),
                progress=20 + step * prog_each,
            )
            results[mid] = _train_one(mid, X, y, le)

        # ── 5. Save best pipeline for /classify ───────────────────────────────
        global _saved_pipeline
        best_mid = max(results, key=lambda m: results[m]["cv_accuracy"])
        _saved_pipeline = {
            "scaler":        results[best_mid]["_scaler"],
            "classifier":    results[best_mid]["_clf"],
            "label_encoder": le,
            "selected_cols": selected_cols,
            "config": {
                "cutoff_hz":     req.cutoff_hz,
                "window_ms":     req.window_ms,
                "interpolation": req.interpolation,
            },
        }

        # ── 6. Format results for frontend (strip internal keys) ───────────────
        model_list = [
            {
                "id":              mid,
                "name":            MODEL_DISPLAY.get(mid, mid),
                "accuracy":        res["accuracy"],
                "cv_accuracy":     res["cv_accuracy"],
                "training_time_s": res["training_time_s"],
                "is_best":         mid == best_mid,
            }
            for mid, res in results.items()
        ]
        final_results = {
            "models":           model_list,
            "best_model_id":    best_mid,
            "confusion_matrix": results[best_mid]["confusion_matrix"],
            "class_labels":     results[best_mid]["class_names"],
        }
        _set_status(state="done", progress=100, current_model="", results=final_results)

    except Exception as exc:
        _set_status(state="error", error=str(exc))


@app.post("/train")
def start_training(req: TrainRequest):
    if req.project_id not in project_events:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No events cached for project '{req.project_id}'. "
                "Run /analyze-signal with project_id first."
            ),
        )

    with _training_lock:
        if _training_status["state"] == "running":
            raise HTTPException(status_code=409, detail="Training already in progress.")
        _training_status["state"] = "running"

    events = project_events[req.project_id]
    t = threading.Thread(target=_run_training, args=(req, events), daemon=True)
    t.start()
    return {"status": "started", "project_id": req.project_id, "event_count": len(events)}


@app.get("/train/status")
def train_status():
    with _training_lock:
        return dict(_training_status)


# ── /classify ─────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    project_id: str
    event:      EventData


class SimulateRequest(BaseModel):
    project_id: str


def _do_classify(ev: EventData) -> dict:
    """Run a single event through the saved pipeline and return label + metrics."""
    if _saved_pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="No trained model available. Complete training on Screen 4 first.",
        )

    cfg = _saved_pipeline["config"]

    # Preprocess
    df = _preprocess_event(ev, cfg["cutoff_hz"], cfg["window_ms"], cfg["interpolation"])
    df["event_id"] = 0

    # Time-domain features
    feat_df = compute_minimal_features(df, group_col="event_id")

    # Frequency-domain features
    ffeats = _freq_features(ev)
    for col, val in ffeats.items():
        feat_df.loc[0, col] = val

    # Select columns (same as during training)
    selected_cols = _saved_pipeline["selected_cols"]
    available     = [c for c in selected_cols if c in feat_df.columns]
    X             = feat_df[available].fillna(0.0).values.astype(float)

    if X.shape[1] != len(selected_cols):
        # Pad missing columns with zeros so shape matches the trained scaler
        X_full = np.zeros((1, len(selected_cols)), dtype=float)
        for i, col in enumerate(selected_cols):
            if col in available:
                X_full[0, i] = X[0, available.index(col)]
        X = X_full

    scaler = _saved_pipeline["scaler"]
    clf    = _saved_pipeline["classifier"]
    le     = _saved_pipeline["label_encoder"]

    X_sc   = scaler.transform(X)
    pred   = clf.predict(X_sc)[0]
    label  = str(le.inverse_transform([pred])[0])

    if hasattr(clf, "predict_proba"):
        proba      = clf.predict_proba(X_sc)[0]
        confidence = float(np.max(proba))
        all_proba  = {str(le.inverse_transform([i])[0]): round(float(p), 4)
                      for i, p in enumerate(proba)}
    else:
        confidence = 1.0
        all_proba  = {label: 1.0}

    # Signal metrics
    ax_arr  = np.array(ev.ax, dtype=float) if ev.ax else np.array([0.0])
    ay_arr  = np.array(ev.ay, dtype=float) if ev.ay else np.zeros_like(ax_arr)
    az_arr  = np.array(ev.az, dtype=float) if ev.az else np.zeros_like(ax_arr)
    combined = np.concatenate([ax_arr, ay_arr, az_arr])
    peak_acc = float(np.max(np.abs(combined)))

    if len(ax_arr) > 1:
        fft_power = np.abs(np.fft.rfft(ax_arr)) ** 2
        freqs     = np.fft.rfftfreq(len(ax_arr), d=1.0 / SAMPLE_RATE_HZ)
        dom_freq  = float(freqs[int(np.argmax(fft_power))])
    else:
        dom_freq = 0.0

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "all_proba":  all_proba,
        "metrics": {
            "peak_acceleration": round(peak_acc, 4),
            "event_duration_ms": ev.duration_ms,
            "dominant_freq_hz":  round(dom_freq, 2),
        },
    }


@app.post("/classify")
def classify_event(req: ClassifyRequest):
    return _do_classify(req.event)


@app.post("/classify/simulate")
def classify_simulate(req: SimulateRequest):
    cached = project_events.get(req.project_id)
    if not cached:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No cached events for project '{req.project_id}'. "
                "Complete data collection on Screen 2 first."
            ),
        )
    ev     = random.choice(cached)
    result = _do_classify(ev)
    return {
        **result,
        "event": {
            "ax":          ev.ax,
            "ay":          ev.ay,
            "az":          ev.az,
            "duration_ms": ev.duration_ms,
            "class_label": ev.class_label,
        },
    }


# ── /export ───────────────────────────────────────────────────────────────────

# Fixed body of the generated session.py — no variable substitution needed here
# because all values come from Python globals defined in the header section.
_SCRIPT_BODY = '''
from scipy.stats import kurtosis as _scipy_kurtosis


# ── Signal processing ──────────────────────────────────────────────────────────

def _lowpass(sig, fs=SAMPLE_RATE, cutoff=CUTOFF_HZ, order=4):
    if len(sig) < 20:
        return np.array(sig, dtype=float)
    nyq = fs / 2.0
    b, a = butter(order, min(cutoff, nyq * 0.95) / nyq, btype="low")
    return filtfilt(b, a, np.array(sig, dtype=float))


def _normalize(sig, window_ms=WINDOW_MS, fs=SAMPLE_RATE):
    n_out = max(2, int(window_ms * fs / 1000))
    sig   = np.array(sig, dtype=float)
    if len(sig) < 2:
        return np.zeros(n_out)
    t_in  = np.linspace(0.0, 1.0, len(sig))
    t_out = np.linspace(0.0, 1.0, n_out)
    return CubicSpline(t_in, sig)(t_out)


def _extract(arr):
    """Feature dict for one normalised axis."""
    a     = np.array(arr, dtype=float)
    fft   = np.fft.rfft(a)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(a), d=1.0 / SAMPLE_RATE)
    return {
        "mean":               float(np.mean(a)),
        "standard_deviation": float(np.std(a)),
        "root_mean_square":   float(np.sqrt(np.mean(a ** 2))),
        "maximum":            float(np.max(a)),
        "absolute_maximum":   float(np.max(np.abs(a))),
        "fft_energy":         float(np.sum(power)),
        "dominant_freq":      float(freqs[int(np.argmax(power))]),
        "kurtosis":           float(_scipy_kurtosis(a)),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(ax, ay=None, az=None):
    """
    Classify a single sensor event.

    Parameters
    ----------
    ax : list[float]  x-axis samples
    ay : list[float]  y-axis samples ([] or None if unused)
    az : list[float]  z-axis samples ([] or None if unused)

    Returns
    -------
    tuple[str, float]  (predicted_label, confidence_0_to_1)
    """
    n_fill = max(2, int(WINDOW_MS * SAMPLE_RATE / 1000))
    axes   = []
    for raw in [ax, ay or [], az or []]:
        sig = np.array(raw, dtype=float) if raw else np.zeros(n_fill)
        axes.append(_normalize(_lowpass(sig) if len(sig) >= 20 else sig))

    feat_dict = {}
    for sig, prefix in zip(axes, ["a_x", "a_y", "a_z"]):
        for name, val in _extract(sig).items():
            feat_dict[f"{prefix}__{name}"] = val

    X    = np.array([feat_dict.get(c, 0.0) for c in SELECTED_FEATURES]).reshape(1, -1)
    X_sc = _scaler.transform(X)
    pred = _model.predict(X_sc)[0]
    conf = (
        float(np.max(_model.predict_proba(X_sc)[0]))
        if hasattr(_model, "predict_proba") else 1.0
    )
    label = CLASSES[pred] if pred < len(CLASSES) else str(pred)
    return label, conf


# ── Smoke-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    import random as _rng

    n  = max(2, int(WINDOW_MS * SAMPLE_RATE / 1000))
    t  = [i / SAMPLE_RATE for i in range(n)]
    ax = [0.35 * math.sin(2 * math.pi * 2.1 * ti) + _rng.gauss(0, 0.03) for ti in t]
    ay = [0.28 * math.sin(2 * math.pi * 3.3 * ti) + _rng.gauss(0, 0.02) for ti in t]
    az = [0.22 * math.sin(2 * math.pi * 1.7 * ti) + _rng.gauss(0, 0.02) for ti in t]

    label, confidence = classify(ax, ay, az)
    print(f"Predicted : {label}")
    print(f"Confidence: {confidence:.1%}")
'''


def _b64_wrap(s: str, width: int = 76) -> str:
    """Wrap a base64 string into a multi-line Python string for readability."""
    chunks = [s[i : i + width] for i in range(0, len(s), width)]
    return "\n".join(f'    "{c}"' for c in chunks)


def _generate_session_script(
    project_id: str,
    cutoff_hz: float,
    window_ms: float,
    interpolation: str,
    classes: list,
    selected_features: list,
    model_b64: str,
    scaler_b64: str,
    model_name: str = "",
    accuracy: float = None,
) -> str:
    date_str   = time.strftime("%Y-%m-%d")
    acc_str    = f"{accuracy:.1%}" if accuracy is not None else "N/A"
    classes_r  = repr(classes)
    features_r = repr(selected_features)

    header = (
        f'#!/usr/bin/env python3\n'
        f'"""\n'
        f'EdgeForge session.py — generated {date_str}\n'
        f'Project : {project_id}\n'
        f'Model   : {model_name}  |  CV accuracy: {acc_str}\n'
        f'Classes : {classes_r}\n'
        f'\n'
        f'Usage\n'
        f'-----\n'
        f'    from session import classify\n'
        f'    label, conf = classify(ax_samples, ay_samples, az_samples)\n'
        f'"""\n'
        f'\n'
        f'import base64\n'
        f'import pickle\n'
        f'import numpy as np\n'
        f'from scipy.signal import butter, filtfilt\n'
        f'from scipy.interpolate import CubicSpline\n'
        f'\n'
        f'# ── Pipeline config ───────────────────────────────────────────────────────────\n'
        f'CUTOFF_HZ         = {cutoff_hz}\n'
        f'WINDOW_MS         = {window_ms}\n'
        f'SAMPLE_RATE       = 100.0   # Hz\n'
        f'CLASSES           = {classes_r}\n'
        f'SELECTED_FEATURES = {features_r}\n'
        f'\n'
        f'# ── Embedded model & scaler (base64-encoded pickle) ──────────────────────────\n'
        f'_MODEL_B64 = (\n'
        f'{_b64_wrap(model_b64)}\n'
        f')\n'
        f'_SCALER_B64 = (\n'
        f'{_b64_wrap(scaler_b64)}\n'
        f')\n'
        f'_model  = pickle.loads(base64.b64decode(_MODEL_B64))\n'
        f'_scaler = pickle.loads(base64.b64decode(_SCALER_B64))\n'
    )
    return header + _SCRIPT_BODY


@app.get("/export/python/{project_id}")
def export_python(project_id: str):
    if _saved_pipeline is None:
        raise HTTPException(400, "No trained model. Complete training on Screen 4 first.")

    pipe   = _saved_pipeline
    cfg    = pipe["config"]
    le     = pipe["label_encoder"]
    scaler = pipe["scaler"]
    clf    = pipe["classifier"]

    model_b64  = base64.b64encode(pickle.dumps(clf)).decode()
    scaler_b64 = base64.b64encode(pickle.dumps(scaler)).decode()

    # Best model name + accuracy from last training run
    model_name = ""
    accuracy   = None
    with _training_lock:
        results = _training_status.get("results") or {}
    if results:
        best_id    = results.get("best_model_id", "")
        model_name = MODEL_DISPLAY.get(best_id, best_id)
        best       = next((m for m in results.get("models", []) if m["id"] == best_id), None)
        accuracy   = best["cv_accuracy"] if best else None

    script = _generate_session_script(
        project_id        = project_id,
        cutoff_hz         = cfg["cutoff_hz"],
        window_ms         = cfg["window_ms"],
        interpolation     = cfg["interpolation"],
        classes           = le.classes_.tolist(),
        selected_features = pipe["selected_cols"],
        model_b64         = model_b64,
        scaler_b64        = scaler_b64,
        model_name        = model_name,
        accuracy          = accuracy,
    )

    return Response(
        content    = script.encode(),
        media_type = "text/x-python",
        headers    = {"Content-Disposition": 'attachment; filename="session.py"'},
    )


@app.get("/export/efp/{project_id}")
def export_efp(project_id: str):
    if _saved_pipeline is None:
        raise HTTPException(400, "No trained model. Complete training on Screen 4 first.")

    pipe = _saved_pipeline
    cfg  = pipe["config"]
    le   = pipe["label_encoder"]

    with _training_lock:
        training_results = _training_status.get("results")

    package = {
        "format":      "edgeforge-package",
        "version":     "1.0",
        "project_id":  project_id,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline": {
            "cutoff_hz":       cfg["cutoff_hz"],
            "window_ms":       cfg["window_ms"],
            "interpolation":   cfg["interpolation"],
            "selected_features": pipe["selected_cols"],
        },
        "classes":          le.classes_.tolist(),
        "training_results": training_results,
    }

    return Response(
        content    = json_lib.dumps(package, indent=2).encode(),
        media_type = "application/json",
        headers    = {"Content-Disposition": f'attachment; filename="{project_id}.efp"'},
    )


# ── /copilot/chat ─────────────────────────────────────────────────────────────

COPILOT_SYSTEM = (
    "You are an expert embedded ML pipeline engineer helping a hardware engineer optimize "
    "their sensor classification project. You have access to real signal analysis data for "
    "their specific project — use it. Give specific actionable advice with exact numbers "
    "from their actual data, never generic ML advice. When recommending a parameter change "
    "include an action block: [ACTION: set_cutoff=24.8] or [ACTION: set_window=800] or "
    "[ACTION: set_model=random_forest] or [ACTION: add_feature=kurtosis] "
    "Keep responses to 3-5 sentences unless asked for detail. "
    "You are talking to a hardware engineer who understands signal processing but may be new to ML."
)


class PipelineDesignRequest(BaseModel):
    project_id:               str
    application_description:  str
    hardware_preprocessing:   Optional[dict] = None
    signal_analysis:          Optional[dict] = None


class CopilotChatRequest(BaseModel):
    message:         str
    project_id:      str
    screen:          Optional[str]  = None   # collect | pipeline | train | validate | export
    pipeline_config: Optional[dict] = None   # current frontend slider/feature state


_SCREEN_HINTS: Dict[str, str] = {
    "collect": (
        "User is on the DATA COLLECTION screen. "
        "Prioritise advice about data quality, class balance, event count sufficiency, "
        "and signal characteristics. Flag any class with fewer than 15 events."
    ),
    "pipeline": (
        "User is on the PIPELINE CONFIGURATION screen. "
        "Prioritise advice about low-pass cutoff frequency, normalisation window length, "
        "feature selection, and model choice. Reference the signal analysis numbers."
    ),
    "train": (
        "User is on the TRAINING screen. "
        "Prioritise advice about model performance, overfitting (train vs CV gap), "
        "which classes are most confused, and whether more data or different features would help."
    ),
    "validate": (
        "User is on the LIVE VALIDATION screen. "
        "Prioritise advice about classification confidence, which classes are hardest to "
        "distinguish in real-time inference, and deployment readiness."
    ),
    "export": (
        "User is on the EXPORT screen. "
        "Prioritise advice about deployment format, model size vs accuracy trade-offs, "
        "and MCU memory/latency compatibility."
    ),
}


def _build_copilot_context(
    project_id:      str,
    screen:          Optional[str]  = None,
    pipeline_config: Optional[dict] = None,
) -> str:
    sections: list = []

    # ── 1. Screen context ──────────────────────────────────────────────────────
    if screen and screen in _SCREEN_HINTS:
        sections.append(f"[CURRENT SCREEN] {_SCREEN_HINTS[screen]}")

    # ── 2. Project info ────────────────────────────────────────────────────────
    proj = projects.get(project_id)
    if proj:
        lines = [
            f"Project: {project_id}",
            f"Sensor: {proj.get('sensor_type', 'unknown')}",
            f"Target MCU: {proj.get('target_mcu', 'unknown')}",
            f"Connection: {proj.get('connection_type', 'unknown')}",
        ]
        app_desc = proj.get("application_description") or ""
        if app_desc:
            lines.append(f"Application context: {app_desc}")
        hw = proj.get("hardware_preprocessing") or {}
        if hw.get("type") and hw["type"] != "none":
            hw_str = hw["type"]
            if hw.get("cutoff_hz"):
                hw_str += f" at {hw['cutoff_hz']} Hz"
            elif hw.get("description"):
                hw_str += f": {hw['description']}"
            lines.append(f"Hardware preprocessing on chip: {hw_str}")
        sections.append("PROJECT:\n" + "\n".join(lines))

    # ── 3. Data quality ────────────────────────────────────────────────────────
    evts = project_events.get(project_id, [])
    if evts:
        label_counts: Dict[str, int] = {}
        for ev in evts:
            lbl = ev.class_label or "unknown"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        durations = [ev.duration_ms for ev in evts]
        lines = [f"Total events captured: {len(evts)}"]

        lines.append("Events per class:")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = cnt / len(evts) * 100
            flag = "  ⚠ low" if cnt < 15 else ""
            lines.append(f"  '{lbl}': {cnt} events ({pct:.0f}%){flag}")

        counts = list(label_counts.values())
        if len(counts) > 1:
            ratio = max(counts) / max(min(counts), 1)
            if ratio > 3:
                lines.append(
                    f"Class imbalance: {ratio:.1f}x — model will be biased toward "
                    f"'{max(label_counts, key=label_counts.get)}'. Collect more of the minority class."
                )
            else:
                lines.append(f"Class balance: {ratio:.1f}x ratio — acceptable.")
        elif len(counts) == 1:
            lines.append("WARNING: Only one class present — need ≥2 classes to train a classifier.")

        lines.append(
            f"Event duration: min={min(durations):.0f} ms, "
            f"mean={sum(durations)/len(durations):.0f} ms, "
            f"max={max(durations):.0f} ms"
        )
        sections.append("DATA:\n" + "\n".join(lines))

    # ── 4. Pipeline settings ───────────────────────────────────────────────────
    pipe_lines: list = []

    if pipeline_config:
        # Use the live frontend state (most current — before training)
        filt      = pipeline_config.get("filter", {})
        norm      = pipeline_config.get("normalize", {})
        feats_map = pipeline_config.get("features", {})
        model_id  = pipeline_config.get("model", "auto")
        selected  = [k for k, v in feats_map.items() if v]

        pipe_lines.append(
            f"Low-pass cutoff: {filt.get('cutoff', '?')} Hz  "
            f"(order: {filt.get('order', '?')})"
        )
        pipe_lines.append(
            f"Normalisation window: {norm.get('window', '?')} ms  "
            f"(interpolation: {norm.get('interpolation', '?')})"
        )
        pipe_lines.append(
            f"Selected features ({len(selected)}): "
            f"{', '.join(selected) if selected else 'none selected'}"
        )
        pipe_lines.append(f"Model: {model_id}")

    elif _saved_pipeline:
        # Fall back to post-training saved config
        cfg  = _saved_pipeline["config"]
        cols = _saved_pipeline["selected_cols"]
        le   = _saved_pipeline["label_encoder"]
        pipe_lines.append(
            f"Low-pass cutoff: {cfg['cutoff_hz']} Hz  "
            f"window: {cfg['window_ms']} ms  "
            f"interpolation: {cfg['interpolation']}"
        )
        pipe_lines.append(
            f"Feature columns ({len(cols)}): "
            f"{', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}"
        )
        pipe_lines.append(f"Classes (from training): {le.classes_.tolist()}")

    if pipe_lines:
        sections.append("PIPELINE CONFIG:\n" + "\n".join(pipe_lines))

    # ── 5. Training results ────────────────────────────────────────────────────
    with _training_lock:
        tr = _training_status.get("results")

    if tr:
        best_id = tr.get("best_model_id", "")
        models  = tr.get("models", [])
        clabs   = tr.get("class_labels", [])
        cm      = tr.get("confusion_matrix", [])
        lines   = []

        # Best model + overfitting check
        best = next((m for m in models if m["id"] == best_id), None)
        if best:
            gap = (best["accuracy"] - best["cv_accuracy"]) * 100
            lines.append(
                f"Best model: {best['name']}  "
                f"CV accuracy: {best['cv_accuracy']*100:.1f}%  "
                f"Train accuracy: {best['accuracy']*100:.1f}%"
            )
            if gap > 8:
                lines.append(
                    f"  ⚠ Overfitting: train is {gap:.0f} pp above CV — "
                    f"model may not generalise. Collect more data or reduce features."
                )
            elif gap <= 3:
                lines.append(f"  ✓ Generalisation gap: {gap:.0f} pp — good.")

        # All models ranked
        if len(models) > 1:
            lines.append("All models (ranked by CV accuracy):")
            for m in sorted(models, key=lambda x: -x.get("cv_accuracy", 0)):
                marker = " ← best" if m["id"] == best_id else ""
                lines.append(
                    f"  {m['name']}: CV={m['cv_accuracy']*100:.1f}%  "
                    f"train={m['accuracy']*100:.1f}%{marker}"
                )

        # Per-class accuracy + confused pairs
        if cm and clabs:
            lines.append(f"Classes: {clabs}")
            lines.append("Per-class accuracy:")
            error_pairs: list = []
            for ri, (lbl, row) in enumerate(zip(clabs, cm)):
                total   = sum(row)
                correct = row[ri] if ri < len(row) else 0
                if total > 0:
                    acc = correct / total * 100
                    flag = "  ⚠ poor" if acc < 70 else ""
                    lines.append(f"  '{lbl}': {correct}/{total} ({acc:.0f}%){flag}")
                    for ci, count in enumerate(row):
                        if ci != ri and count > 0 and ci < len(clabs):
                            error_pairs.append((lbl, clabs[ci], count))

            if error_pairs:
                error_pairs.sort(key=lambda x: -x[2])
                lines.append("Top confused pairs (actual → predicted as):")
                for actual, predicted, count in error_pairs[:5]:
                    lines.append(
                        f"  '{actual}' → '{predicted}': "
                        f"{count} time{'s' if count > 1 else ''}"
                    )

        sections.append("TRAINING RESULTS:\n" + "\n".join(lines))

    if not sections:
        return "No project data available yet — user is in early setup phase."

    return "\n\n".join(sections)


def _parse_actions(text: str) -> list:
    pattern = r'\[ACTION:\s*([^=\]\s]+)\s*=\s*([^\]]+)\]'
    actions = []
    for m in re.finditer(pattern, text):
        key = m.group(1).strip()
        raw = m.group(2).strip()
        try:
            value = float(raw) if "." in raw else int(raw)
        except ValueError:
            value = raw
        actions.append({"type": key, "value": value})
    return actions


def _strip_actions(text: str) -> str:
    return re.sub(r'\[ACTION:[^\]]+\]', '', text).strip()


PIPELINE_DESIGN_SYSTEM = (
    "You are an expert signal processing engineer with deep domain knowledge across "
    "industrial vibration analysis, medical wearables, sports biomechanics, and embedded systems. "
    "Design the optimal signal processing pipeline for the given application and signal data.\n\n"
    "Respond ONLY with valid JSON, no other text:\n"
    "{\n"
    '  "reasoning": "overall approach for this domain",\n'
    '  "filter": {\n'
    '    "type": "butterworth",\n'
    '    "cutoff_hz": 24.8,\n'
    '    "order": 4,\n'
    '    "skip": false,\n'
    '    "skip_reason": "only include if skip is true",\n'
    '    "reasoning": "domain-specific reason"\n'
    '  },\n'
    '  "normalize": {\n'
    '    "window_ms": 800,\n'
    '    "interpolation": "cubic",\n'
    '    "reasoning": "why this window for this application"\n'
    '  },\n'
    '  "features": {\n'
    '    "time_domain": ["rms", "peak", "std_dev", "kurtosis"],\n'
    '    "frequency_domain": ["fft_energy", "dominant_freq"],\n'
    '    "reasoning": "why these features for this domain"\n'
    '  },\n'
    '  "model": {\n'
    '    "type": "random_forest",\n'
    '    "reasoning": "why this model for this dataset"\n'
    '  }\n'
    "}\n\n"
    "Available time-domain feature IDs (use exactly): mean, std_dev, rms, peak, absolute_max\n"
    "Available frequency-domain feature IDs (use exactly): fft_energy, dominant_freq, kurtosis\n"
    "Available model types: random_forest, svm, nn, auto\n\n"
    "Domain knowledge to apply:\n"
    "- Bearing fault detection: kurtosis is critical (impulsive faults), use fft_energy and dominant_freq for frequency signatures\n"
    "- Gait analysis: longer windows (1500-3000ms), rms and std_dev for symmetry, mean for DC offset\n"
    "- Impact classification: peak and absolute_max for amplitude, kurtosis for impulsiveness, rms for energy\n"
    "- Gesture recognition: shorter windows (200-600ms), std_dev and rms, all axes important\n"
    "- Tool wear / machining: kurtosis trend (increases with wear), fft_energy for chatter frequency\n"
    "If hardware preprocessing already applied (e.g. hardware lowpass at X Hz), set filter.skip=true "
    "if X Hz is already appropriately bandlimiting the signal, or recommend a complementary software filter."
)

_ai_pipeline_designs: Dict[str, dict] = {}


@app.post("/pipeline/design")
def pipeline_design(req: PipelineDesignRequest):
    hw      = req.hardware_preprocessing or {}
    hw_type = hw.get("type", "none")
    hw_desc = {
        "none":     "None",
        "lowpass":  f"Hardware lowpass at {hw.get('cutoff_hz', '?')} Hz",
        "highpass": f"Hardware highpass at {hw.get('cutoff_hz', '?')} Hz",
        "custom":   hw.get("description", "Custom (undescribed)"),
    }.get(hw_type, "None")

    sa = req.signal_analysis or {}
    sig_parts: list = []
    if sa.get("sample_rate_hz"):
        sig_parts.append(f"Sample rate: {sa['sample_rate_hz']} Hz (Nyquist: {sa['sample_rate_hz']/2:.1f} Hz)")
    if sa.get("recommended_cutoff_hz"):
        sig_parts.append(f"Recommended cutoff (90% energy): {sa['recommended_cutoff_hz']} Hz")
    if sa.get("recommended_window_ms"):
        sig_parts.append(f"Recommended normalisation window: {sa['recommended_window_ms']} ms")
    if sa.get("event_count"):
        sig_parts.append(f"Events collected: {sa['event_count']}")

    user_content = (
        f"Application description: {req.application_description}\n\n"
        f"Hardware preprocessing on chip: {hw_desc}\n\n"
        "Signal analysis data:\n"
        + ("\n".join(sig_parts) if sig_parts else "Not yet available — design based on application description only.")
        + "\n\nDesign the optimal signal processing pipeline for this application."
    )

    try:
        resp = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 1024,
            system     = PIPELINE_DESIGN_SYSTEM,
            messages   = [{"role": "user", "content": user_content}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if model wraps in ```json ... ```
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())
        design = json_lib.loads(raw)
        _ai_pipeline_designs[req.project_id] = design
        return design
    except json_lib.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/copilot/chat")
def copilot_chat(req: CopilotChatRequest):
    context = _build_copilot_context(req.project_id, screen=req.screen, pipeline_config=req.pipeline_config)
    user_content = f"Project context:\n{context}\n\nUser question: {req.message}"
    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 512,
            system     = COPILOT_SYSTEM,
            messages   = [{"role": "user", "content": user_content}],
        )
        raw_text = response.content[0].text
        return {"message": _strip_actions(raw_text), "actions": _parse_actions(raw_text)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
