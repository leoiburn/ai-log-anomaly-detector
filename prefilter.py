"""
Fast Log Pre-Filtering Engine.

Discards routine, harmless high-volume logs (health checks, 200 OKs, heartbeats)
at wire speed (<0.01ms per log) so downstream anomaly detectors and Laya
only receive meaningful candidates.
"""

from typing import List, Tuple, Dict, Any
import re

# High-frequency routine patterns to discard
DEFAULT_DROP_PATTERNS = [
    re.compile(r"(?:/healthz|/livez|/readyz|kube-probe|health check passed)", re.IGNORECASE),
    re.compile(r"status=(?:200|204|304)\b", re.IGNORECASE),
    re.compile(r"\bHTTP/[0-9.]+\"\s+(?:200|204|304)\b", re.IGNORECASE),
    re.compile(r"(?:heartbeat received|scheduled task completed|cache refresh completed)", re.IGNORECASE),
    re.compile(r"\bINFO\b.*(?:routine|ping|keepalive)", re.IGNORECASE),
]

# Critical patterns that must NEVER be dropped in pre-filtering
FORCE_KEEP_PATTERNS = [
    re.compile(r"(?:error|critical|failed|denied|unauthorized|timeout|exception|attack|exploit|breach)", re.IGNORECASE),
    re.compile(r"status=(?:40[134]|50[0234])\b", re.IGNORECASE),
]


class FastLogPreFilter:
    """High-speed pre-filter to suppress high-volume routine logs."""

    def __init__(self, custom_drop_patterns: List[re.Pattern] = None):
        self.drop_patterns = custom_drop_patterns or DEFAULT_DROP_PATTERNS
        self.stats = {
            "total_received": 0,
            "dropped_routine": 0,
            "passed_candidates": 0,
        }

    def should_keep(self, log_entry: str) -> bool:
        """
        Determines whether a log entry should be passed to anomaly detection.
        Returns True if the log is suspicious or noteworthy, False if routine noise.
        """
        self.stats["total_received"] += 1

        # Always keep if it contains explicit warning/threat indicators
        for keep_regex in FORCE_KEEP_PATTERNS:
            if keep_regex.search(log_entry):
                self.stats["passed_candidates"] += 1
                return True

        # Drop if it matches known routine health / successful queries
        for drop_regex in self.drop_patterns:
            if drop_regex.search(log_entry):
                self.stats["dropped_routine"] += 1
                return False

        # Default to keeping unclassified logs for downstream evaluation
        self.stats["passed_candidates"] += 1
        return True

    def filter_logs(self, logs: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Filters a batch of logs, returning (kept_logs, stats).
        """
        kept = [line for line in logs if self.should_keep(line)]
        reduction_pct = (
            round(100.0 * self.stats["dropped_routine"] / max(self.stats["total_received"], 1), 2)
        )
        run_stats = {
            **self.stats,
            "noise_reduction_percentage": reduction_pct,
        }
        return kept, run_stats
