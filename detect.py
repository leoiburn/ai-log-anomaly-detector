from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
from sklearn.ensemble import IsolationForest

from prefilter import FastLogPreFilter
from laya_verifier import LayaCPUVerifier


LOG_FILE = Path("sample_logs.txt")
OUTPUT_FILE = Path("anomalies.csv")

ANOMALY_KEYWORDS = (
    "ERROR",
    "FAILED",
    "DENIED",
    "UNAUTHORIZED",
    "CRITICAL",
    "WARNING",
    "WARN",
    "TIMEOUT",
)

HIGH_SEVERITY_KEYWORDS = (
    "ERROR",
    "FAILED",
    "DENIED",
    "UNAUTHORIZED",
    "CRITICAL",
)

MEDIUM_SEVERITY_KEYWORDS = (
    "WARNING",
    "WARN",
    "TIMEOUT",
)


def load_logs(file_path):
    """Load non-empty log entries from a text file."""
    with file_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def extract_features(line):
    """Convert a log entry into numeric features for IsolationForest."""
    upper_line = line.upper()
    line_length = max(len(line), 1)

    return [
        len(line),
        sum(char.isdigit() for char in line) / line_length,
        sum(char.isupper() for char in line) / line_length,
        sum(upper_line.count(keyword) for keyword in ANOMALY_KEYWORDS),
        line.count("="),
        line.count("/"),
    ]


def score_severity(line):
    """Assign a rule-based severity level to an anomalous log entry."""
    upper_line = line.upper()

    if any(keyword in upper_line for keyword in HIGH_SEVERITY_KEYWORDS):
        return "HIGH"

    if any(keyword in upper_line for keyword in MEDIUM_SEVERITY_KEYWORDS):
        return "MEDIUM"

    return "LOW"


def detect_anomalies(logs, contamination=0.20, threshold=0.0, use_prefilter=False, use_laya=False):
    """Detect unusual log entries using IsolationForest, with optional pre-filtering and Laya CPU verification."""
    if contamination != "auto" and (
        not isinstance(contamination, (int, float)) or not 0 < contamination <= 0.5
    ):
        raise ValueError("contamination must be 'auto' or a number in (0, 0.5].")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if len(logs) < 5:
        raise ValueError("Add at least 5 log entries before running detection.")
    if any(not isinstance(line, str) or not line.strip() for line in logs):
        raise ValueError("Log entries must be non-empty strings.")

    target_logs = logs
    if use_prefilter:
        prefilter = FastLogPreFilter()
        target_logs, _ = prefilter.filter_logs(logs)
        if len(target_logs) < 5:
            # If aggressive pre-filtering left < 5, fall back to raw logs to keep model stable
            target_logs = logs

    features = np.array(
        [extract_features(line) for line in target_logs],
        dtype=float,
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )

    model.fit(features)
    anomaly_scores = model.decision_function(features)

    laya = LayaCPUVerifier() if use_laya else None
    anomalies = []

    for line, score in zip(target_logs, anomaly_scores):
        if score < threshold:
            severity = score_severity(line)
            entry_dict = {
                "severity": severity,
                "anomaly_score": round(float(score), 4),
                "log_entry": line,
            }

            if use_laya and laya:
                decision = laya.decide(line)
                entry_dict["laya_threat_prob"] = decision["is_threat"]["probability"]
                entry_dict["laya_suppress"] = decision["suppress_alert"]
                entry_dict["severity"] = decision["severity_tier"]

            anomalies.append(entry_dict)

    return anomalies


def save_anomalies(anomalies, output_file=OUTPUT_FILE):
    """Save detected anomalies to a CSV file."""
    base_fields = ["severity", "anomaly_score", "log_entry"]
    extra_fields = []
    if anomalies:
        for k in anomalies[0].keys():
            if k not in base_fields and k not in extra_fields:
                extra_fields.append(k)
    fieldnames = base_fields + extra_fields

    with Path(output_file).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(anomalies)


def build_summary(logs, anomalies, contamination=0.20, threshold=0.0):
    """Summarize a run without claiming accuracy on unlabeled data."""
    scores = [item["anomaly_score"] for item in anomalies]
    severity_levels = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    severity_counts = {
        level: sum(item["severity"] == level for item in anomalies)
        for level in severity_levels
        if any(item["severity"] == level for item in anomalies) or level in ("HIGH", "MEDIUM", "LOW")
    }

    summary = {
        "logs_analyzed": len(logs),
        "anomalies_detected": len(anomalies),
        "anomaly_percentage": round(100 * len(anomalies) / len(logs), 2) if logs else 0.0,
        "severity_counts": severity_counts,
        "anomaly_score_range": {"min": min(scores), "max": max(scores)} if scores else None,
        "settings": {"contamination": contamination, "threshold": threshold, "random_state": 42},
    }

    if anomalies and "laya_threat_prob" in anomalies[0]:
        suppressed = sum(1 for item in anomalies if item.get("laya_suppress"))
        summary["laya_insights"] = {
            "verified_threats": len(anomalies) - suppressed,
            "suppressed_noise": suppressed,
            "engine": "Laya CPU Decision Model",
        }

    return summary


def parse_contamination(value):
    if value == "auto":
        return value
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use 'auto' or a number in (0, 0.5].") from error
    if not 0 < number <= 0.5:
        raise argparse.ArgumentTypeError("Use 'auto' or a number in (0, 0.5].")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect unusual log entries using Isolation Forest & Laya.")
    parser.add_argument("--input", type=Path, default=LOG_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--summary", type=Path, default=Path("summary.json"))
    parser.add_argument("--contamination", type=parse_contamination, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Flag decision scores below this value (default: 0).")
    parser.add_argument("--prefilter", action="store_true",
                        help="Discard routine health/200 logs before running detection.")
    parser.add_argument("--enable-laya", action="store_true",
                        help="Enable Laya CPU Decision Engine for semantic Yes/No verification.")
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in (args.input, args.output, args.summary)]
    if len(set(paths)) != 3:
        parser.error("Input, CSV output, and summary must use different paths.")
    try:
        logs = load_logs(args.input)
        anomalies = detect_anomalies(
            logs,
            args.contamination,
            args.threshold,
            use_prefilter=args.prefilter,
            use_laya=args.enable_laya,
        )
        summary = build_summary(logs, anomalies, args.contamination, args.threshold)
        save_anomalies(anomalies, args.output)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"\nScanned {len(logs)} log entries.")
    print(f"Detected {len(anomalies)} anomalies ({summary['anomaly_percentage']}%).")
    print(f"Severity counts: {summary['severity_counts']}\n")

    for anomaly in anomalies:
        laya_info = ""
        if "laya_threat_prob" in anomaly:
            laya_info = f" | Threat Prob: {anomaly['laya_threat_prob']} (Suppressed: {anomaly['laya_suppress']})"
        print(
            f"[{anomaly['severity']}] "
            f"Score: {anomaly['anomaly_score']}{laya_info} | "
            f"{anomaly['log_entry']}"
        )

    print(f"\nResults saved to {args.output}")
    print(f"Run summary saved to {args.summary}")


if __name__ == "__main__":
    main()
