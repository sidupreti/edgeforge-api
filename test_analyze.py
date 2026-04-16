"""
Synthetic test for POST /analyze-signal.

Generates a realistic accelerometer dataset:
  - 20 events at 100 Hz, ~800 ms each
  - Base signal: multi-frequency sine on each axis
  - Impact burst: short high-amplitude Gaussian transient (15-25 samples)
  - Additive white noise
  - Deliberate variation in event duration (750–900 ms) to exercise window stats
  - Two class labels: "idle" (low energy) and "impact" (burst present)

Prints a formatted summary of the returned recommendations.
"""

import json
import math
import random
import urllib.request
import urllib.error

SAMPLE_RATE = 100      # Hz
BASE_DURATION_MS = 800 # centre of the duration range
URL = "http://localhost:8000/analyze-signal"


def _sine(t, freq, amp, phase=0.0):
    return amp * math.sin(2 * math.pi * freq * t + phase)


def _gaussian(t, centre, sigma, peak):
    return peak * math.exp(-0.5 * ((t - centre) / sigma) ** 2)


def make_event(duration_ms: float, has_burst: bool, class_label: str):
    n      = int(duration_ms * SAMPLE_RATE / 1000)
    dt     = 1.0 / SAMPLE_RATE
    burst_centre = random.uniform(0.2, 0.7) * duration_ms / 1000  # seconds

    ax, ay, az = [], [], []
    for i in range(n):
        t = i * dt
        # Base signal — three overlapping sinusoids per axis
        bx = _sine(t, 2.1, 0.35) + _sine(t, 8.5, 0.12) + _sine(t, 15.0, 0.06)
        by = _sine(t, 3.3, 0.28, 1.0) + _sine(t, 11.0, 0.08, 0.5)
        bz = _sine(t, 1.7, 0.22, 2.1) + _sine(t, 6.0, 0.10, 1.3) + 0.08

        # Impact burst (Gaussian transient, 12-25 Hz equivalent)
        if has_burst:
            sigma  = random.uniform(0.015, 0.030)
            peak_x = random.uniform(0.8, 1.4)
            peak_y = peak_x * random.uniform(0.6, 0.85)
            peak_z = peak_x * random.uniform(0.4, 0.65)
            bx += _gaussian(t, burst_centre, sigma, peak_x)
            by += _gaussian(t, burst_centre, sigma, peak_y)
            bz += _gaussian(t, burst_centre, sigma, peak_z)

        # Additive noise
        bx += random.gauss(0, 0.03)
        by += random.gauss(0, 0.025)
        bz += random.gauss(0, 0.02)

        ax.append(round(bx, 5))
        ay.append(round(by, 5))
        az.append(round(bz, 5))

    return {
        "ax":          ax,
        "ay":          ay,
        "az":          az,
        "duration_ms": round(duration_ms, 1),
        "class_label": class_label,
    }


def build_payload():
    events = []
    for i in range(20):
        duration_ms = BASE_DURATION_MS + random.uniform(-50, 100)  # 750–900 ms range
        has_burst   = (i % 4 != 0)                                 # 75 % have burst
        label       = "impact" if has_burst else "idle"
        events.append(make_event(duration_ms, has_burst, label))
    return {"events": events, "sample_rate_hz": float(SAMPLE_RATE)}


def fmt(label, value, indent=4):
    pad = " " * indent
    print(f"{pad}{label}: {value}")


def run():
    payload = build_payload()
    body    = json.dumps(payload).encode()

    print(f"Sending {len(payload['events'])} synthetic events to {URL} …\n")

    try:
        req  = urllib.request.Request(
            URL, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"ERROR: could not reach API — {exc}")
        print("Is the server running?  uvicorn main:app --reload")
        return

    print("=" * 60)
    print("  /analyze-signal  results")
    print("=" * 60)

    print(f"\n  Events analysed : {result['event_count']}")

    sr = result["sample_rate"]
    print(f"\n{'─'*60}")
    print("  SAMPLE RATE")
    fmt("Declared",     f"{sr['declared_hz']} Hz")
    fmt("Measured",     f"{sr['measured_hz']} Hz")
    fmt("Note",         sr["explanation"])

    cf = result["cutoff_frequency"]
    print(f"\n{'─'*60}")
    print("  CUTOFF FREQUENCY  (FFT 90 % energy)")
    fmt("Recommended",  f"{cf['recommended_hz']} Hz")
    for axis, hz in cf["axis_cutoffs_hz"].items():
        fmt(f"  {axis} cutoff", f"{hz} Hz")
    fmt("Note",         cf["explanation"])

    nw = result["normalization_window"]
    print(f"\n{'─'*60}")
    print("  NORMALIZATION WINDOW")
    fmt("Recommended",  f"{nw['recommended_ms']} ms")
    fmt("Min / Mean / Max",
        f"{nw['min_ms']} / {nw['mean_ms']} / {nw['max_ms']} ms")
    fmt("p90",          f"{nw['p90_ms']} ms")
    fmt("Note",         nw["explanation"])

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run()
