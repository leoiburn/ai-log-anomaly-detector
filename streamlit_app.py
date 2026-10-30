import csv
import io
import json
from pathlib import Path

import streamlit as st

from detect import build_summary, detect_anomalies

SAMPLE_FILE = Path(__file__).with_name("sample_logs.txt")
MAX_UPLOAD_BYTES = 1_000_000
MAX_LOG_ENTRIES = 5_000


def parse_log_text(text):
    """Return stripped, non-empty log entries from pasted or uploaded text."""
    logs = [line.strip() for line in text.splitlines() if line.strip()]
    if len(logs) > MAX_LOG_ENTRIES:
        raise ValueError(f"Use {MAX_LOG_ENTRIES:,} log entries or fewer in the demo.")
    return logs


def anomalies_to_csv(anomalies):
    """Serialize anomaly rows for the download button."""
    buffer = io.StringIO()
    base_fields = ["severity", "anomaly_score", "log_entry"]
    extra_fields = []
    if anomalies:
        for k in anomalies[0].keys():
            if k not in base_fields and k not in extra_fields:
                extra_fields.append(k)
    fieldnames = base_fields + extra_fields

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(anomalies)
    return buffer.getvalue()


def main():
    st.set_page_config(page_title="AI Log Anomaly Detector", page_icon="🔎", layout="wide")

    st.title("AI Log Anomaly Detector")
    st.write(
        "Lightweight machine learning pipeline combining Isolation Forest with "
        "pre-filtering and the **Laya CPU Decision Engine** (Yes/No verifier) "
        "to surface and verify unusual system log entries."
    )
    st.caption("Enhanced Pipeline · Python · scikit-learn · Laya System 1 Decision Model · Streamlit")

    with st.sidebar:
        st.header("Detection settings")
        contamination = st.slider(
            "Expected anomaly proportion",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.05,
            help="Isolation Forest contamination controls the model's expected outlier proportion.",
        )
        threshold = st.slider(
            "Decision threshold",
            min_value=-0.20,
            max_value=0.10,
            value=0.00,
            step=0.01,
            help="Lower values are more selective. Higher values flag more entries.",
        )

        st.divider()
        st.header("Pipeline enhancements")
        use_prefilter = st.checkbox(
            "Enable Fast Pre-filter",
            value=False,
            help="Discards high-volume routine health checks and 200 OKs at wire speed before anomaly detection.",
        )
        use_laya = st.checkbox(
            "Enable Laya CPU Verifier",
            value=True,
            help="Runs the non-autoregressive Laya System 1 decision engine on CPU to verify threats and suppress benign noise.",
        )

    st.subheader("Input logs")
    input_source = st.radio(
        "Choose an input source",
        ["Synthetic sample", "Paste logs", "Upload file"],
        horizontal=True,
    )

    logs = []
    if input_source == "Synthetic sample":
        if SAMPLE_FILE.exists():
            logs = parse_log_text(SAMPLE_FILE.read_text(encoding="utf-8"))
            st.caption(f"Loaded {len(logs)} synthetic system log entries.")
        else:
            st.error("sample_logs.txt not found in repository root.")
    elif input_source == "Paste logs":
        pasted = st.text_area(
            "Paste log lines here",
            height=150,
            placeholder="2026-10-03 09:16:23 ERROR Failed login attempt...",
        )
        if pasted:
            try:
                logs = parse_log_text(pasted)
            except ValueError as error:
                st.error(str(error))
    elif input_source == "Upload file":
        uploaded = st.file_uploader(
            "Upload a .txt or .log file (max 1 MB)",
            type=["txt", "log"],
        )
        if uploaded:
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error("File exceeds 1 MB limit.")
            else:
                try:
                    content = uploaded.read().decode("utf-8")
                    logs = parse_log_text(content)
                except UnicodeDecodeError:
                    st.error("Please upload a valid UTF-8 text file.")
                except ValueError as error:
                    st.error(str(error))

    if logs:
        if len(logs) < 5:
            st.warning("Please provide at least 5 log entries to train the Isolation Forest.")
        else:
            try:
                anomalies = detect_anomalies(
                    logs,
                    contamination=contamination,
                    threshold=threshold,
                    use_prefilter=use_prefilter,
                    use_laya=use_laya,
                )
                summary = build_summary(logs, anomalies, contamination, threshold)
            except ValueError as error:
                st.error(str(error))
                return

            st.divider()
            st.subheader("Analysis overview")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total logs analyzed", f"{summary['logs_analyzed']:,}")
            col2.metric("Flagged anomalies", f"{summary['anomalies_detected']:,}")
            col3.metric("Anomaly percentage", f"{summary['anomaly_percentage']:.1f}%")

            if "laya_insights" in summary:
                verified = summary["laya_insights"]["verified_threats"]
                suppressed = summary["laya_insights"]["suppressed_noise"]
                col4.metric("Laya Verified Threats", f"{verified}", delta=f"-{suppressed} noise suppressed")
            else:
                col4.metric(
                    "Severity breakdown",
                    f"H: {summary['severity_counts']['HIGH']} | M: {summary['severity_counts']['MEDIUM']} | L: {summary['severity_counts']['LOW']}",
                )

            st.subheader("Flagged log entries")
            if anomalies:
                column_order = ["severity", "anomaly_score"]
                if use_laya and "laya_threat_prob" in anomalies[0]:
                    column_order.extend(["laya_threat_prob", "laya_suppress"])
                column_order.append("log_entry")

                column_config = {
                    "severity": st.column_config.TextColumn("Severity", width="small"),
                    "anomaly_score": st.column_config.NumberColumn("Anomaly score", format="%.4f", width="small"),
                    "laya_threat_prob": st.column_config.NumberColumn("Threat Prob (Laya)", format="%.2f", width="small"),
                    "laya_suppress": st.column_config.CheckboxColumn("Suppressed", width="small"),
                    "log_entry": st.column_config.TextColumn("Log entry", width="large"),
                }

                st.dataframe(
                    anomalies,
                    width="stretch",
                    hide_index=True,
                    column_order=column_order,
                    column_config=column_config,
                )

                with st.expander("View full flagged log details"):
                    for index, anomaly in enumerate(anomalies, start=1):
                        detail_col1, detail_col2 = st.columns([1, 4])
                        laya_meta = ""
                        if "laya_threat_prob" in anomaly:
                            laya_meta = f"  \nLaya Threat $P$: `{anomaly['laya_threat_prob']}` (Suppressed: {anomaly['laya_suppress']})"
                        detail_col1.markdown(
                            f"**{index}. {anomaly['severity']}**  \n"
                            f"Score: `{anomaly['anomaly_score']:.4f}`{laya_meta}"
                        )
                        with detail_col2:
                            st.code(
                                anomaly["log_entry"],
                                language=None,
                                wrap_lines=True,
                                height="content",
                                width="stretch",
                            )
            else:
                st.success("No log entries fell below the selected decision threshold.")

            download_col1, download_col2 = st.columns(2)
            download_col1.download_button(
                "Download anomaly CSV",
                data=anomalies_to_csv(anomalies),
                file_name="anomalies.csv",
                mime="text/csv",
                use_container_width=True,
            )
            download_col2.download_button(
                "Download run summary",
                data=json.dumps(summary, indent=2) + "\n",
                file_name="summary.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.expander("How the enhanced model works"):
        st.markdown(
            """
1. **Pre-filter (Optional)**: High-frequency routine health checks, heartbeats, and HTTP 200s are dropped at wire speed.
2. **Isolation Forest (Stage 1)**: Converts remaining log lines into numeric features and flags mathematical outliers in sub-milliseconds.
3. **Laya CPU Decision Model (Stage 2)**: Evaluates flagged candidates on CPU using non-autoregressive Yes/No questions (`is_threat`, `is_benign_noise`) to eliminate false alarms and confirm true security incidents.
            """
        )


if __name__ == "__main__":
    main()
