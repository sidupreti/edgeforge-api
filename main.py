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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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


# ── /upload-events ───────────────────────────────────────────────────────────

_UPLOAD_COL_MAP = {
    # timestamp aliases
    "t": "timestamp", "time": "timestamp", "ts": "timestamp",
    "time_us": "timestamp", "time_ms": "timestamp", "time_s": "timestamp",
    "sample_time": "timestamp", "elapsed": "timestamp",
    # x-axis aliases
    "x": "a_x", "ax": "a_x", "accel_x": "a_x", "acc_x": "a_x",
    "x-axis": "a_x", "xaxis": "a_x", "sensor3": "a_x",
    # y-axis aliases
    "y": "a_y", "ay": "a_y", "accel_y": "a_y", "acc_y": "a_y",
    "y-axis": "a_y", "yaxis": "a_y", "sensor4": "a_y",
    # z-axis aliases
    "z": "a_z", "az": "a_z", "accel_z": "a_z", "acc_z": "a_z",
    "z-axis": "a_z", "zaxis": "a_z", "sensor5": "a_z",
}

# WISDM-style activity labels that map to human-readable strings
_WISDM_ACTIVITY_MAP = {
    "A": "Walking", "B": "Jogging", "C": "Stairs",
    "D": "Sitting", "E": "Standing", "F": "LyingDown",
}


def _parse_csv_flexible(content: bytes) -> tuple:
    """
    Parse uploaded CSV/TXT bytes into (df, notes, detected_label).

    Returns
    -------
    df : pd.DataFrame with columns [timestamp, a_x, a_y, a_z]
    notes : list[str]  human-readable detection messages
    detected_label : str | None  class label found in data (e.g. WISDM activity)
    """
    import io

    notes: list          = []
    detected_label: str | None = None

    text = content.decode("utf-8", errors="replace").strip()
    lines = [l for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("File has fewer than 2 data rows")

    # ── Detect separator ───────────────────────────────────────────────────────
    first = lines[0]
    if "\t" in first:
        sep = "\t"
    elif ";" in first:
        sep = ";"
    elif "  " in first or (first.replace(" ", "").replace(".", "").replace("-", "").isnumeric()):
        sep = r"\s+"   # space-separated (UCI HAR style)
    else:
        sep = ","

    # ── Detect headerless numeric file ─────────────────────────────────────────
    first_vals = re.split(r"[\s,;\t]+", first.strip())
    first_is_numeric = all(
        re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", v) for v in first_vals if v
    )

    # ── Detect WISDM format: user, activity, timestamp, x, y, z (no header) ───
    is_wisdm = (
        not first_is_numeric
        and len(first_vals) >= 6
        and re.match(r"^\d+$", first_vals[0] or "")  # user ID (integer)
        and re.match(r"^[A-Za-z]", first_vals[1] or "")  # activity string
        and re.match(r"^\d{7,}$", first_vals[2] or "")  # large timestamp
    )

    if is_wisdm:
        extra = len(first_vals) - 6
        hdr = ["user", "activity", "timestamp", "a_x", "a_y", "a_z"] + [f"col{i}" for i in range(extra)]
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None, names=hdr, engine="python")
        notes.append("WISDM dataset format detected (user, activity, timestamp, x, y, z)")
    elif first_is_numeric:
        # UCI HAR: space-separated, 561 features, no header
        if sep == r"\s+" and len(first_vals) > 20:
            raise ValueError(
                "This appears to be a pre-processed feature file (561 columns). "
                "EdgeForge needs raw accelerometer time series data, not pre-extracted features."
            )
        # Headerless CSV: assume timestamp,a_x,a_y,a_z or just a_x,a_y,a_z
        n_cols = len(first_vals)
        if n_cols >= 4:
            hdr = ["timestamp", "a_x", "a_y", "a_z"] + [f"col{i}" for i in range(n_cols - 4)]
        elif n_cols == 3:
            hdr = ["a_x", "a_y", "a_z"]
        elif n_cols == 1:
            hdr = ["a_x"]
        else:
            hdr = ["a_x", "a_y"]
        notes.append(f"No header row detected — assumed columns: {', '.join(hdr[:4])}")
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None, names=hdr, engine="python")
    else:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")

    # ── Normalise column names ─────────────────────────────────────────────────
    orig_cols = list(df.columns)
    df.columns = [_UPLOAD_COL_MAP.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    norm_cols  = list(df.columns)

    detected_cols = [o for o, n in zip(orig_cols, norm_cols) if o != n]
    if detected_cols:
        notes.append(f"Columns detected: {', '.join(str(c) for c in orig_cols)} — mapped to EdgeForge format")

    # ── WISDM format: user, activity, timestamp, x, y, z ──────────────────────
    if "activity" in norm_cols:
        # Extract the most common activity label as the class
        act_col = df.columns[norm_cols.index("activity")] if "activity" in norm_cols else "activity"
        activities = df["activity"].dropna().astype(str).str.strip(";").str.strip()
        most_common = activities.mode()
        if len(most_common) > 0:
            raw_act = str(most_common.iloc[0])
            detected_label = _WISDM_ACTIVITY_MAP.get(raw_act, raw_act)
            unique_acts    = activities.unique().tolist()
            notes.append(
                f"WISDM format detected — activity column found: {unique_acts[:5]}"
                + (f" ... (+{len(unique_acts)-5} more)" if len(unique_acts) > 5 else "")
                + f". Using '{detected_label}' as class label."
            )

    # ── Require at least a_x ──────────────────────────────────────────────────
    if "a_x" not in norm_cols:
        raise ValueError(
            f"No recognised X-axis column. Got columns: {orig_cols}. "
            "Expected one of: a_x, x, ax, accel_x, acc_x"
        )

    # ── Fill missing axes ──────────────────────────────────────────────────────
    for col in ("a_y", "a_z"):
        if col not in norm_cols:
            df[col] = 0.0

    # ── Ensure timestamp column ────────────────────────────────────────────────
    if "timestamp" not in norm_cols:
        df["timestamp"] = DT_US
        notes.append(f"No timestamp column found — assuming {int(SAMPLE_RATE_HZ)} Hz sample rate")

    # ── Keep only needed columns ───────────────────────────────────────────────
    df = df[["timestamp", "a_x", "a_y", "a_z"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    if len(df) < 2:
        raise ValueError("File has fewer than 2 valid numeric data rows after parsing")

    # ── Convert absolute epoch timestamps to per-sample diffs ─────────────────
    ts = df["timestamp"].values
    if len(ts) > 1 and ts[0] > 1e9:
        diffs = np.diff(ts, prepend=ts[1])
        df["timestamp"] = np.abs(diffs)
        notes.append("Absolute timestamps converted to per-sample intervals")

    # Clamp unreasonable timestamps (< 100µs or > 1s) to DT_US
    df["timestamp"] = df["timestamp"].clip(lower=100, upper=1_000_000).fillna(DT_US)

    return df, notes, detected_label


def _label_from_filename(filename: str) -> str | None:
    """
    Extract a class label from a filename stem.
    'metal_event_003.csv' → 'metal',  'wood tap (2).txt' → 'wood'
    Takes the first non-numeric, non-empty token after splitting on separators.
    """
    base   = filename.split("/")[-1].rsplit(".", 1)[0]   # strip path + extension
    tokens = [t for t in re.split(r"[_\-\s()]+", base.lower()) if t and not t.isdigit()]
    return tokens[0] if tokens else None


def _zip_extract_csvs(content: bytes) -> list:
    """Extract (internal_path, bytes) for every CSV/TXT inside a ZIP."""
    import zipfile, io as _io
    results = []
    with zipfile.ZipFile(_io.BytesIO(content)) as zf:
        for name in sorted(zf.namelist()):
            # Skip macOS metadata dirs and non-CSV files
            if name.startswith("__MACOSX") or name.endswith("/"):
                continue
            if not name.lower().endswith((".csv", ".txt")):
                continue
            try:
                data = zf.read(name)
                results.append((name, data))
            except Exception:
                pass
    return results


@app.post("/inspect-zip")
async def inspect_zip(file: UploadFile = File(...)):
    """Return the list of CSV/TXT paths found inside a ZIP without processing them."""
    import zipfile, io as _io
    content = await file.read()
    try:
        entries = []
        with zipfile.ZipFile(_io.BytesIO(content)) as zf:
            for name in sorted(zf.namelist()):
                if name.startswith("__MACOSX") or name.endswith("/"):
                    continue
                if not name.lower().endswith((".csv", ".txt")):
                    continue
                info = zf.getinfo(name)
                entries.append({"path": name, "size_bytes": info.file_size})
        if not entries:
            raise HTTPException(status_code=400, detail="No CSV or TXT files found inside the ZIP")
        return {"files": entries}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Not a valid ZIP file")


@app.post("/upload-events")
async def upload_events(
    files:          List[UploadFile]   = File(...),
    project_id:     str                = Form(...),
    labels:         List[str]          = Form(...),
    zip_selections: Optional[str]      = Form(None),  # JSON: [{path, label}]
):
    if len(files) != len(labels):
        raise HTTPException(
            status_code=400,
            detail=f"files ({len(files)}) and labels ({len(labels)}) count mismatch",
        )

    parsed_events: List[EventData] = []
    event_meta:    list            = []
    errors:        list            = []

    # Parse optional per-path selections for ZIP files
    zip_sel_map: dict = {}  # {zip_path -> label}
    if zip_selections:
        try:
            for item in json_lib.loads(zip_selections):
                zip_sel_map[item["path"]] = item["label"]
        except Exception:
            pass

    for uf, label in zip(files, labels):
        raw      = await uf.read()
        fname    = uf.filename or "upload"
        is_zip   = fname.lower().endswith(".zip")

        # ── ZIP file: extract and process each CSV ─────────────────────────
        if is_zip:
            try:
                csv_files = _zip_extract_csvs(raw)
            except Exception as exc:
                errors.append({"filename": fname, "error": f"ZIP extraction failed: {exc}"})
                continue

            if not csv_files:
                errors.append({"filename": fname, "error": "No CSV/TXT files found inside ZIP"})
                continue

            for csv_path, csv_bytes in csv_files:
                # Priority: explicit zip_sel_map > provided label > auto-detect
                csv_label = zip_sel_map.get(csv_path, label)
                short     = csv_path.split("/")[-1]
                try:
                    df, notes, detected = _parse_csv_flexible(csv_bytes)
                    n      = len(df)
                    dur_ms = float(df["timestamp"].sum()) / 1000.0
                    if csv_label == "auto":
                        # data detection (WISDM activity) → filename → fallback
                        final_label = detected or _label_from_filename(short) or "unknown"
                    else:
                        final_label = csv_label
                    ev = EventData(
                        ax          = df["a_x"].tolist(),
                        ay          = df["a_y"].tolist(),
                        az          = df["a_z"].tolist(),
                        duration_ms = round(dur_ms, 1),
                        class_label = final_label,
                    )
                    parsed_events.append(ev)
                    event_meta.append({
                        "id":          f"upload-{len(parsed_events)}-{int(time.time()*1000)}",
                        "class_label": ev.class_label,
                        "duration_ms": round(dur_ms, 1),
                        "row_count":   n,
                        "waveform_az": df["a_z"].tolist(),
                        "filename":    short,
                        "zip_path":    csv_path,
                        "source_zip":  fname,
                        "notes":       notes,
                    })
                except Exception as exc:
                    errors.append({"filename": f"{fname}/{short}", "error": str(exc)})
        else:
            # ── Regular CSV/TXT file ───────────────────────────────────────
            try:
                df, notes, detected = _parse_csv_flexible(raw)
                n      = len(df)
                dur_ms = float(df["timestamp"].sum()) / 1000.0
                final_label = detected if (label == "auto" and detected) else label
                ev = EventData(
                    ax          = df["a_x"].tolist(),
                    ay          = df["a_y"].tolist(),
                    az          = df["a_z"].tolist(),
                    duration_ms = round(dur_ms, 1),
                    class_label = final_label,
                )
                parsed_events.append(ev)
                event_meta.append({
                    "id":          f"upload-{len(parsed_events)}-{int(time.time()*1000)}",
                    "class_label": final_label,
                    "duration_ms": round(dur_ms, 1),
                    "row_count":   n,
                    "waveform_az": df["a_z"].tolist(),
                    "filename":    fname,
                    "notes":       notes,
                })
            except Exception as exc:
                errors.append({"filename": fname, "error": str(exc)})

    if not parsed_events:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No files could be parsed.",
                "errors":  errors,
            },
        )

    # Cache events for training (append to any existing events)
    existing = project_events.get(project_id, [])
    project_events[project_id] = list(existing) + parsed_events

    # Run the same analysis as /analyze-signal
    fake_req = AnalyzeSignalRequest(
        events         = parsed_events,
        sample_rate_hz = SAMPLE_RATE_HZ,
        project_id     = None,
    )
    try:
        analysis = analyze_signal(fake_req)
    except Exception:
        analysis = None

    return {
        "events":   event_meta,
        "analysis": analysis,
        "errors":   errors,
    }


# ── /train ────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    project_id:        str
    cutoff_hz:         float       = 30.0
    filter_type:       str         = "butterworth"   # butterworth | chebyshev | bessel | moving_average | none
    window_ms:         float       = 1000.0
    interpolation:     str         = "cubic"          # cubic | linear | none (skip)
    selected_features: List[str]   = []
    model_type:        str         = "auto"           # auto | rf | svm | nn
    custom_blocks:     Optional[List[dict]] = []      # list of {id, name, code} for custom/standard blocks
    events:            Optional[List[EventData]] = None  # fallback if backend lost in-memory state


# ── Custom block execution ────────────────────────────────────────────────────

STANDARD_BLOCK_CODE: Dict[str, str] = {
    "bandpass": """\
from scipy.signal import butter, sosfilt
sig_cols = [c for c in ['a_x', 'a_y', 'a_z'] if c in df.columns]
low_norm  = max(0.01, min(1.0 / (SAMPLE_RATE_HZ / 2.0), 0.49))
high_norm = max(low_norm + 0.01, min(30.0 / (SAMPLE_RATE_HZ / 2.0), 0.49))
sos = butter(4, [low_norm, high_norm], btype='band', output='sos')
for col in sig_cols:
    arr = df[col].to_numpy(dtype=float)
    if len(arr) >= 20:
        df[col] = sosfilt(sos, arr)
""",
    "envelope": """\
from scipy.signal import hilbert
import numpy as np
sig_cols = [c for c in ['a_x', 'a_y', 'a_z'] if c in df.columns]
for col in sig_cols:
    arr = df[col].to_numpy(dtype=float)
    if len(arr) >= 4:
        df[col] = np.abs(hilbert(arr))
""",
    "derivative": """\
import numpy as np
sig_cols = [c for c in ['a_x', 'a_y', 'a_z'] if c in df.columns]
for col in sig_cols:
    arr = df[col].to_numpy(dtype=float)
    df[col] = np.gradient(arr)
""",
    "zscore": """\
import numpy as np
sig_cols = [c for c in ['a_x', 'a_y', 'a_z'] if c in df.columns]
for col in sig_cols:
    arr = df[col].to_numpy(dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma > 1e-9:
        df[col] = (arr - mu) / sigma
""",
    "peak_detector": """\
from scipy.signal import find_peaks
import numpy as np
sig_cols = [c for c in ['a_x', 'a_y', 'a_z'] if c in df.columns]
for col in sig_cols:
    arr = df[col].to_numpy(dtype=float)
    peaks, _ = find_peaks(np.abs(arr), height=0.1 * np.max(np.abs(arr)))
    indicator = np.zeros_like(arr)
    indicator[peaks] = 1.0
    df[col] = indicator
""",
    "fft_transform": """\
import numpy as np
sig_cols = [c for c in ['a_x', 'a_y', 'a_z'] if c in df.columns]
for col in sig_cols:
    arr = df[col].to_numpy(dtype=float)
    mag = np.abs(np.fft.rfft(arr, n=len(arr)))
    # Pad/trim back to original length
    result = np.zeros(len(arr))
    half = len(mag)
    result[:half] = mag[:half]
    df[col] = result
""",
    "abs_smooth": """\
from scipy.ndimage import uniform_filter1d
import numpy as np
sig_cols = [c for c in ['a_x', 'a_y', 'a_z'] if c in df.columns]
for col in sig_cols:
    arr = np.abs(df[col].to_numpy(dtype=float))
    df[col] = uniform_filter1d(arr, size=5, mode='nearest')
""",
}


def run_custom_block(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute a custom processing block; returns (possibly mutated) df."""
    from scipy.signal import filtfilt, hilbert, find_peaks
    from scipy.ndimage import uniform_filter1d

    namespace = {
        "df":               df.copy(),
        "np":               np,
        "pd":               pd,
        "SAMPLE_RATE_HZ":   SAMPLE_RATE_HZ,
        "filtfilt":         filtfilt,
        "hilbert":          hilbert,
        "find_peaks":       find_peaks,
        "uniform_filter1d": uniform_filter1d,
    }
    try:
        exec(compile(code, "<custom_block>", "exec"), namespace)  # noqa: S102
        result = namespace.get("df", df)
        if isinstance(result, pd.DataFrame):
            return result
    except Exception:
        pass  # silently skip broken custom block
    return df


def _apply_filter(df: pd.DataFrame, filter_type: str, cutoff_hz: float) -> pd.DataFrame:
    """Apply the selected filter type to signal columns in df."""
    from scipy.signal import cheby1, bessel, sosfilt, butter
    from scipy.ndimage import uniform_filter1d

    if filter_type == "none":
        return df

    safe_cutoff = max(1.0, min(float(cutoff_hz), SAMPLE_RATE_HZ * 0.45))
    nyq = SAMPLE_RATE_HZ / 2.0
    norm = safe_cutoff / nyq
    order = 4
    sig_cols = [c for c in ["a_x", "a_y", "a_z"] if c in df.columns]

    result = df.copy()
    for col in sig_cols:
        arr = df[col].to_numpy(dtype=float)
        if len(arr) < 20:
            continue
        try:
            if filter_type == "butterworth":
                sos = butter(order, norm, btype="low", analog=False, output="sos")
                result[col] = sosfilt(sos, arr)
            elif filter_type == "chebyshev":
                sos = cheby1(order, 0.5, norm, btype="low", analog=False, output="sos")
                result[col] = sosfilt(sos, arr)
            elif filter_type == "bessel":
                sos = bessel(order, norm, btype="low", analog=False, output="sos", norm="phase")
                result[col] = sosfilt(sos, arr)
            elif filter_type == "moving_average":
                # Window = sample_rate / cutoff gives ~half-power at cutoff
                win = max(2, int(round(SAMPLE_RATE_HZ / safe_cutoff)))
                result[col] = uniform_filter1d(arr, size=win, mode="nearest")
        except Exception:
            pass  # leave column unchanged on any filter failure

    return result


def _preprocess_event(ev: EventData, cutoff_hz: float,
                      window_ms: float, interpolation: str,
                      filter_type: str = "butterworth",
                      custom_blocks: Optional[List[dict]] = None) -> pd.DataFrame:
    n = len(ev.ax)
    df = pd.DataFrame({
        "timestamp": [DT_US] * n,
        "a_x": ev.ax,
        "a_y": ev.ay if ev.ay else [0.0] * n,
        "a_z": ev.az if ev.az else [0.0] * n,
    })

    # Apply filter (or skip if filter_type == "none")
    if n >= 20 and filter_type != "none":
        try:
            df = _apply_filter(df, filter_type, cutoff_hz)
        except Exception:
            pass

    # Normalize to target window length (skip if interpolation == "none")
    if interpolation != "none":
        try:
            df = normalize_df_period(window_ms, DT_US, df, interpolationKind=interpolation)
        except Exception:
            pass

    # Run any custom / standard pipeline blocks
    for block in (custom_blocks or []):
        code = block.get("code", "")
        if code:
            df = run_custom_block(df, code)

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
                df = _preprocess_event(ev, req.cutoff_hz, req.window_ms, req.interpolation, req.filter_type, req.custom_blocks or [])
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
                "filter_type":   req.filter_type,
                "window_ms":     req.window_ms,
                "interpolation": req.interpolation,
                "custom_blocks": req.custom_blocks or [],
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


@app.get("/session/{project_id}")
def get_session(project_id: str):
    """Debug endpoint — returns cached event count and class distribution."""
    evts = project_events.get(project_id, [])
    class_counts: Dict[str, int] = {}
    for ev in evts:
        class_counts[ev.class_label] = class_counts.get(ev.class_label, 0) + 1
    return {
        "project_id":   project_id,
        "event_count":  len(evts),
        "classes":      list(class_counts.keys()),
        "class_counts": class_counts,
        "has_analysis": project_id in {k for k, v in projects.items()},
    }


@app.post("/train")
def start_training(req: TrainRequest):
    # Use stored events; fall back to events sent in request body
    events = project_events.get(req.project_id)
    if not events and req.events:
        events = req.events
        project_events[req.project_id] = events  # cache for subsequent calls

    if not events:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No events found for project '{req.project_id}'. "
                "Please upload or collect data first."
            ),
        )

    with _training_lock:
        if _training_status["state"] == "running":
            raise HTTPException(status_code=409, detail="Training already in progress.")
        _training_status["state"] = "running"

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
    df = _preprocess_event(ev, cfg["cutoff_hz"], cfg["window_ms"], cfg["interpolation"],
                           cfg.get("filter_type", "butterworth"),
                           cfg.get("custom_blocks", []))
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


# ── C header export ───────────────────────────────────────────────────────────

def _c_float_arr(name: str, vals, per_line: int = 6) -> str:
    vals = [float(v) for v in vals]
    n    = len(vals)
    lines = [f"static const float {name}[{n}] = {{"]
    for i in range(0, n, per_line):
        chunk  = vals[i:i + per_line]
        row    = ", ".join(f"{v:.8g}f" for v in chunk)
        comma  = "" if i + per_line >= n else ","
        lines.append(f"    {row}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _c_int_arr(name: str, vals, per_line: int = 10) -> str:
    vals = [int(v) for v in vals]
    n    = len(vals)
    lines = [f"static const int {name}[{n}] = {{"]
    for i in range(0, n, per_line):
        chunk  = vals[i:i + per_line]
        row    = ", ".join(str(v) for v in chunk)
        comma  = "" if i + per_line >= n else ","
        lines.append(f"    {row}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _rf_c_arrays(clf, n_feat: int, max_trees: int = 8) -> str:
    try:
        from sklearn.tree._tree import TREE_UNDEFINED
    except ImportError:
        TREE_UNDEFINED = -2
    trees   = clf.estimators_[:max_trees]
    n_trees = len(trees)
    parts   = []
    for k, dt in enumerate(trees):
        t = dt.tree_
        feat_arr = []; thresh_arr = []; left_arr = []; right_arr = []; pred_arr = []
        for i in range(t.node_count):
            feat_arr.append(int(t.feature[i]))
            thresh_arr.append(float(t.threshold[i]))
            left_arr.append(int(t.children_left[i]))
            right_arr.append(int(t.children_right[i]))
            pred_arr.append(int(np.argmax(t.value[i][0])) if t.feature[i] == TREE_UNDEFINED else -1)
        parts.append(_c_int_arr(f"EF_T{k}_FEAT",  feat_arr,   12))
        parts.append(_c_float_arr(f"EF_T{k}_THR", thresh_arr,  6))
        parts.append(_c_int_arr(f"EF_T{k}_LEFT",  left_arr,   12))
        parts.append(_c_int_arr(f"EF_T{k}_RIGHT", right_arr,  12))
        parts.append(_c_int_arr(f"EF_T{k}_PRED",  pred_arr,   12))
    funcs = []
    for k in range(n_trees):
        funcs.append(
            f"static int8_t ef_tree_{k}(const float *x) {{\n"
            f"    int n = 0;\n"
            f"    while (EF_T{k}_FEAT[n] != {TREE_UNDEFINED}) {{\n"
            f"        n = (x[EF_T{k}_FEAT[n]] <= EF_T{k}_THR[n])"
            f" ? EF_T{k}_LEFT[n] : EF_T{k}_RIGHT[n];\n"
            f"    }}\n"
            f"    return (int8_t)EF_T{k}_PRED[n];\n"
            f"}}"
        )
    calls    = " ".join(f"votes[ef_tree_{k}(x)]++;" for k in range(n_trees))
    vote_fn  = (
        f"static int8_t ef_rf_predict(const float *x) {{\n"
        f"    int votes[EF_N_CLASSES] = {{0}}, i;\n"
        f"    {calls}\n"
        f"    int best = 0;\n"
        f"    for (i = 1; i < EF_N_CLASSES; i++) if (votes[i] > votes[best]) best = i;\n"
        f"    return (int8_t)best;\n"
        f"}}"
    )
    return "\n\n".join(parts) + "\n\n" + "\n\n".join(funcs) + "\n\n" + vote_fn


def _svm_c_arrays(clf, n_feat: int, n_classes: int) -> str:
    sv        = clf.support_vectors_
    dc        = clf.dual_coef_
    intercept = clf.intercept_
    n_sv      = sv.shape[0]
    if isinstance(clf.gamma, (int, float)):
        gamma_val = float(clf.gamma)
    elif hasattr(clf, "_gamma"):
        gamma_val = float(clf._gamma)
    else:
        gamma_val = 1.0 / max(n_feat, 1)
    parts = [
        f"#define EF_SVM_N_SV {n_sv}",
        f"#define EF_SVM_GAMMA {gamma_val:.8g}f",
        _c_int_arr("EF_SVM_N_SUPPORT",  clf.n_support_,  10),
        _c_float_arr("EF_SVM_SV",        sv.flatten(),    6),
        _c_float_arr("EF_SVM_DUAL_COEF", dc.flatten(),    6),
        _c_float_arr("EF_SVM_INTERCEPT", intercept,       6),
    ]
    infer_fn = """\
static int8_t ef_svm_predict(const float *x) {
    float K[EF_SVM_N_SV];
    int i, j, k;
    for (i = 0; i < EF_SVM_N_SV; i++) {
        float d = 0.0f;
        for (j = 0; j < EF_N_FEATURES; j++) {
            float diff = x[j] - EF_SVM_SV[i * EF_N_FEATURES + j];
            d += diff * diff;
        }
        K[i] = expf(-EF_SVM_GAMMA * d);
    }
    int votes[EF_N_CLASSES] = {0};
    int pair = 0, sv_i = 0;
    for (i = 0; i < EF_N_CLASSES; i++) {
        int tmp_j = sv_i + EF_SVM_N_SUPPORT[i];
        for (j = i + 1; j < EF_N_CLASSES; j++) {
            float val = EF_SVM_INTERCEPT[pair];
            for (k = sv_i; k < sv_i + EF_SVM_N_SUPPORT[i]; k++)
                val += EF_SVM_DUAL_COEF[(j - 1) * EF_SVM_N_SV + k] * K[k];
            for (k = tmp_j; k < tmp_j + EF_SVM_N_SUPPORT[j]; k++)
                val += EF_SVM_DUAL_COEF[i * EF_SVM_N_SV + k] * K[k];
            if (val > 0.0f) votes[i]++; else votes[j]++;
            tmp_j += EF_SVM_N_SUPPORT[j];
            pair++;
        }
        sv_i += EF_SVM_N_SUPPORT[i];
    }
    int best = 0;
    for (i = 1; i < EF_N_CLASSES; i++) if (votes[i] > votes[best]) best = i;
    return (int8_t)best;
}"""
    return "\n".join(parts) + "\n\n" + infer_fn


def _nn_c_arrays(clf, n_feat: int, n_classes: int) -> str:
    layer_sizes = [n_feat] + list(clf.hidden_layer_sizes) + [n_classes]
    parts       = []
    for k, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        parts.append(_c_float_arr(f"EF_NN_W{k}", W.flatten(), 8))
        parts.append(_c_float_arr(f"EF_NN_B{k}", b,           8))
    n_layers = len(clf.coefs_)
    lines    = ["static int8_t ef_nn_predict(const float *x) {"]
    for k, sz in enumerate(layer_sizes[1:]):
        lines.append(f"    float h{k}[{sz}];")
    lines.append("    int i, j;")
    for k in range(n_layers):
        in_sz   = layer_sizes[k]
        out_sz  = layer_sizes[k + 1]
        in_var  = "x" if k == 0 else f"h{k - 1}"
        out_var = f"h{k}"
        is_last = (k == n_layers - 1)
        lines.append(f"    for (j = 0; j < {out_sz}; j++) {{")
        lines.append(f"        float s = EF_NN_B{k}[j];")
        lines.append(f"        for (i = 0; i < {in_sz}; i++) s += {in_var}[i] * EF_NN_W{k}[i * {out_sz} + j];")
        if is_last:
            lines.append(f"        {out_var}[j] = s;")
        else:
            lines.append(f"        {out_var}[j] = s > 0.0f ? s : 0.0f;  /* ReLU */")
        lines.append("    }")
    last_h = f"h{n_layers - 1}"
    lines += [
        "    int best = 0;",
        f"    for (i = 1; i < EF_N_CLASSES; i++) if ({last_h}[i] > {last_h}[best]) best = i;",
        "    return (int8_t)best;",
        "}",
    ]
    return "\n\n".join(parts) + "\n\n" + "\n".join(lines)


def _c_math_helpers() -> str:
    return r"""static float ef_mean(const float *x, int n) {
    float s = 0.0f; int i;
    for (i = 0; i < n; i++) s += x[i];
    return s / n;
}
static float ef_std(const float *x, int n) {
    float mu = ef_mean(x, n), s = 0.0f; int i;
    for (i = 0; i < n; i++) { float d = x[i] - mu; s += d * d; }
    return sqrtf(s / n);
}
static float ef_rms(const float *x, int n) {
    float s = 0.0f; int i;
    for (i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s / n);
}
static float ef_peak(const float *x, int n) {
    float m = x[0]; int i;
    for (i = 1; i < n; i++) if (x[i] > m) m = x[i];
    return m;
}
static float ef_abs_max(const float *x, int n) {
    float m = fabsf(x[0]); int i;
    for (i = 1; i < n; i++) { float a = fabsf(x[i]); if (a > m) m = a; }
    return m;
}
/* FFT energy via Parseval: sum(|FFT(x)|^2) == n * sum(x^2) */
static float ef_fft_energy(const float *x, int n) {
    float s = 0.0f; int i;
    for (i = 0; i < n; i++) s += x[i] * x[i];
    return s * n;
}
/* Dominant frequency via DFT magnitude search (O(n^2)) */
static float ef_dom_freq(const float *x, int n) {
    float max_p = 0.0f; int max_k = 1, k, t;
    for (k = 1; k <= n / 2; k++) {
        float re = 0.0f, im = 0.0f, a;
        for (t = 0; t < n; t++) {
            a = 6.28318530f * k * t / n;
            re += x[t] * cosf(a); im -= x[t] * sinf(a);
        }
        float p = re * re + im * im;
        if (p > max_p) { max_p = p; max_k = k; }
    }
    return (float)max_k * EF_SAMPLE_RATE_HZ / n;
}
/* Excess kurtosis — matches scipy.stats.kurtosis(bias=True) */
static float ef_kurtosis(const float *x, int n) {
    float mu = ef_mean(x, n), var = 0.0f, m4 = 0.0f; int i;
    for (i = 0; i < n; i++) {
        float d = x[i] - mu, d2 = d * d;
        var += d2; m4 += d2 * d2;
    }
    var /= n; m4 /= n;
    if (var < 1e-10f) return 0.0f;
    return m4 / (var * var) - 3.0f;
}"""


def _c_filter_fn() -> str:
    return r"""/* IIR low-pass — Direct Form II Transposed, order 4 */
static float ef_flt_state[3][4];
static void ef_filter_reset(void) { memset(ef_flt_state, 0, sizeof(ef_flt_state)); }
static void ef_filter_axis(const float *in, float *out, int n, float *s) {
    int i;
    for (i = 0; i < n; i++) {
        float x = in[i];
        float y = EF_FILTER_B[0] * x + s[0];
        s[0] = EF_FILTER_B[1] * x - EF_FILTER_A[1] * y + s[1];
        s[1] = EF_FILTER_B[2] * x - EF_FILTER_A[2] * y + s[2];
        s[2] = EF_FILTER_B[3] * x - EF_FILTER_A[3] * y + s[3];
        s[3] = EF_FILTER_B[4] * x - EF_FILTER_A[4] * y;
        out[i] = y;
    }
}"""


def _c_feature_fn(selected_cols) -> str:
    axis_map = {"a_x": "ax", "a_y": "ay", "a_z": "az"}
    feat_map = {
        "mean":         "ef_mean({a}, n)",
        "std_dev":      "ef_std({a}, n)",
        "rms":          "ef_rms({a}, n)",
        "peak":         "ef_peak({a}, n)",
        "absolute_max": "ef_abs_max({a}, n)",
        "fft_energy":   "ef_fft_energy({a}, n)",
        "dominant_freq":"ef_dom_freq({a}, n)",
        "kurtosis":     "ef_kurtosis({a}, n)",
    }
    lines = [
        "static void ef_extract_features(",
        "    const float *ax, const float *ay, const float *az, int n, float *feat) {",
    ]
    for idx, col in enumerate(selected_cols):
        parts = col.split("__", 1)
        if len(parts) == 2 and parts[0] in axis_map and parts[1] in feat_map:
            expr = feat_map[parts[1]].replace("{a}", axis_map[parts[0]])
            lines.append(f"    feat[{idx}] = {expr};  /* {col} */")
        else:
            lines.append(f"    feat[{idx}] = 0.0f;  /* unknown: {col} */")
    lines.append("}")
    return "\n".join(lines)


def _c_scale_fn(scaler, n_feat: int) -> str:
    safe_scale = [float(s) if abs(float(s)) > 1e-10 else 1.0 for s in scaler.scale_]
    return "\n".join([
        _c_float_arr("EF_SCALER_MEAN",  scaler.mean_, 8),
        _c_float_arr("EF_SCALER_SCALE", safe_scale,   8),
        "",
        "static void ef_scale(float *feat) {",
        "    int i;",
        "    for (i = 0; i < EF_N_FEATURES; i++)",
        "        feat[i] = (feat[i] - EF_SCALER_MEAN[i]) / EF_SCALER_SCALE[i];",
        "}",
    ])


def _generate_c_header(project_id: str, chip: str = "generic") -> str:
    from scipy.signal import butter as _butter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    if _saved_pipeline is None:
        raise HTTPException(status_code=400, detail="No trained model. Run /train first.")

    pipe      = _saved_pipeline
    clf       = pipe["classifier"]
    scaler    = pipe["scaler"]
    le        = pipe["label_encoder"]
    sel       = pipe["selected_cols"]
    cfg       = pipe["config"]
    classes   = le.classes_.tolist()
    n_classes = len(classes)
    n_feat    = len(sel)
    cutoff    = float(cfg["cutoff_hz"])
    window_ms = float(cfg["window_ms"])
    n_samples = max(2, int(window_ms * SAMPLE_RATE_HZ / 1000))

    nyq  = SAMPLE_RATE_HZ / 2.0
    b, a = _butter(4, min(cutoff, nyq * 0.95) / nyq, btype="low")

    if isinstance(clf, RandomForestClassifier):
        model_type  = "Random Forest"
        n_trees     = min(8, len(clf.estimators_))
        model_sec   = (
            f"/* RF: first {n_trees} of {len(clf.estimators_)} trees */\n"
            + _rf_c_arrays(clf, n_feat, n_trees)
        )
        predict_call = "ef_rf_predict(feat)"
    elif isinstance(clf, SVC):
        n_sv = clf.support_vectors_.shape[0]
        model_type  = "SVM (RBF kernel)"
        model_sec   = (
            f"/* SVM: {n_sv} support vectors, OVO voting */\n"
            + _svm_c_arrays(clf, n_feat, n_classes)
        )
        predict_call = "ef_svm_predict(feat)"
    elif isinstance(clf, MLPClassifier):
        layers      = [n_feat] + list(clf.hidden_layer_sizes) + [n_classes]
        model_type  = f"Neural Network {layers}"
        model_sec   = _nn_c_arrays(clf, n_feat, n_classes)
        predict_call = "ef_nn_predict(feat)"
    else:
        raise HTTPException(status_code=500, detail=f"Unknown model type: {type(clf)}")

    chip_notes = {
        "esp32":   "Target: ESP32 — IRAM_ATTR on hot paths recommended",
        "stm32":   "Target: STM32 — consider CMSIS-DSP for ef_dom_freq",
        "nrf":     "Target: nRF52 — consider CMSIS-DSP for ef_dom_freq",
        "arduino": "Target: Arduino Nano 33 BLE (nRF52840)",
        "rp2040":  "Target: Raspberry Pi Pico (RP2040)",
        "generic": "Target: generic ARM Cortex-M / embedded C99",
    }
    chip_note  = chip_notes.get(chip, f"Target: {chip}")
    class_list = ", ".join(f'"{c}"' for c in classes)
    date_str   = time.strftime("%Y-%m-%d")

    header = (
        f"/*\n"
        f" * EdgeForge — auto-generated on-device classifier\n"
        f" * -----------------------------------------------\n"
        f" * Project  : {project_id}\n"
        f" * Model    : {model_type}\n"
        f" * Classes  : {classes}\n"
        f" * Features : {n_feat}\n"
        f" * Generated: {date_str}\n"
        f" * {chip_note}\n"
        f" *\n"
        f" * Usage\n"
        f" * -----\n"
        f" *   #include \"classifier.h\"\n"
        f" *   int8_t idx = ef_classify(ax, ay, az, EF_WINDOW_SAMPLES);\n"
        f" *   const char *label = EF_CLASSES[idx];\n"
        f" */\n\n"
        f"#pragma once\n"
        f"#include <stdint.h>\n"
        f"#include <math.h>\n"
        f"#include <string.h>\n\n"
        f"/* ── Pipeline config ─────────────────────────────────────────────── */\n"
        f"#define EF_SAMPLE_RATE_HZ  {int(SAMPLE_RATE_HZ)}\n"
        f"#define EF_CUTOFF_HZ       {int(cutoff)}\n"
        f"#define EF_WINDOW_MS       {int(window_ms)}\n"
        f"#define EF_WINDOW_SAMPLES  {n_samples}\n"
        f"#define EF_N_CLASSES       {n_classes}\n"
        f"#define EF_N_FEATURES      {n_feat}\n\n"
        f"/* ── Class labels ────────────────────────────────────────────────── */\n"
        f"static const char *EF_CLASSES[{n_classes}] = {{ {class_list} }};\n\n"
        f"/* ── Butterworth IIR coefficients (b, a) — order 4 ──────────────── */\n"
        f"{_c_float_arr('EF_FILTER_B', b, 5)}\n"
        f"{_c_float_arr('EF_FILTER_A', a, 5)}\n\n"
        f"/* ── Scaler ──────────────────────────────────────────────────────── */\n"
        f"{_c_scale_fn(scaler, n_feat)}\n\n"
        f"/* ── Model weights ───────────────────────────────────────────────── */\n"
        f"{model_sec}\n\n"
        f"/* ── Math helpers ────────────────────────────────────────────────── */\n"
        f"{_c_math_helpers()}\n\n"
        f"/* ── IIR filter ──────────────────────────────────────────────────── */\n"
        f"{_c_filter_fn()}\n\n"
        f"/* ── Feature extraction ──────────────────────────────────────────── */\n"
        f"{_c_feature_fn(sel)}\n\n"
        f"/* ── Public API ──────────────────────────────────────────────────── */\n"
        f"/**\n"
        f" * ef_classify — filter -> extract -> scale -> predict\n"
        f" * @param ax   Accelerometer X samples (EF_WINDOW_SAMPLES floats)\n"
        f" * @param ay   Accelerometer Y samples (NULL to reuse ax)\n"
        f" * @param az   Accelerometer Z samples (NULL to reuse ax)\n"
        f" * @param n    Sample count — must equal EF_WINDOW_SAMPLES\n"
        f" * @return     Class index [0..EF_N_CLASSES-1]; index into EF_CLASSES[]\n"
        f" */\n"
        f"static int8_t ef_classify(\n"
        f"    const float *ax, const float *ay, const float *az, uint16_t n) {{\n"
        f"    float fax[EF_WINDOW_SAMPLES], fay[EF_WINDOW_SAMPLES], faz[EF_WINDOW_SAMPLES];\n"
        f"    float feat[EF_N_FEATURES];\n"
        f"    ef_filter_reset();\n"
        f"    ef_filter_axis(ax,           fax, (int)n, ef_flt_state[0]);\n"
        f"    ef_filter_axis(ay ? ay : ax, fay, (int)n, ef_flt_state[1]);\n"
        f"    ef_filter_axis(az ? az : ax, faz, (int)n, ef_flt_state[2]);\n"
        f"    ef_extract_features(fax, fay, faz, (int)n, feat);\n"
        f"    ef_scale(feat);\n"
        f"    return {predict_call};\n"
        f"}}\n"
    )
    return header


@app.get("/export/c/{project_id}")
def export_c_header(project_id: str, chip: str = "generic"):
    if _saved_pipeline is None:
        raise HTTPException(status_code=400, detail="No trained model. Run /train first.")
    try:
        header = _generate_c_header(project_id, chip)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"C export failed: {exc}")
    return Response(
        content    = header.encode(),
        media_type = "text/plain",
        headers    = {"Content-Disposition": f'attachment; filename="{project_id}_classifier.h"'},
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


# ── /pipeline/custom-block ────────────────────────────────────────────────────

CUSTOM_BLOCK_SYSTEM = (
    "You are an expert signal processing engineer. Generate a Python code snippet that "
    "transforms an IMU/sensor DataFrame `df` in-place. "
    "The DataFrame has columns: timestamp, a_x, a_y, a_z (float64). "
    "Modify the signal columns directly on df (e.g. df['a_x'] = ...). "
    "Available imports already in scope: np (numpy), pd (pandas), SAMPLE_RATE_HZ (float), "
    "filtfilt, hilbert, find_peaks (scipy.signal), uniform_filter1d (scipy.ndimage). "
    "Do NOT import these — they are already available. "
    "Do NOT add any markdown fences, explanation, or comments. "
    "Output ONLY the raw Python code that modifies df. "
    "Keep it concise (under 20 lines). "
    "The code must work with typical IMU data (accelerometer, 100 Hz sample rate)."
)


class CustomBlockRequest(BaseModel):
    description: str
    project_id:  str


class StandardBlockRequest(BaseModel):
    block_type: str   # bandpass | envelope | derivative | zscore | peak_detector | fft_transform | abs_smooth


@app.post("/pipeline/custom-block")
def generate_custom_block(req: CustomBlockRequest):
    """Use Claude to generate Python processing code from a description."""
    try:
        resp = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 512,
            system     = CUSTOM_BLOCK_SYSTEM,
            messages   = [{"role": "user", "content": f"Generate a pipeline block that: {req.description}"}],
        )
        code = resp.content[0].text.strip()
        # Strip accidental markdown fences
        code = re.sub(r"^```[a-z]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code.strip())
        return {"code": code}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pipeline/standard-block")
def get_standard_block(req: StandardBlockRequest):
    """Return pre-written code for a standard block type."""
    code = STANDARD_BLOCK_CODE.get(req.block_type)
    if code is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown block type '{req.block_type}'. Available: {list(STANDARD_BLOCK_CODE.keys())}",
        )
    return {"code": code}


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
