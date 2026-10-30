from laya_verifier import LayaCPUVerifier
from detect import detect_anomalies


def test_laya_cpu_verifier_flags_security_breach():
    verifier = LayaCPUVerifier()
    res = verifier.decide("CRITICAL Unauthorized access denied resource=/admin source=198.51.100.8")

    assert res["is_threat"]["decision"] is True
    assert res["is_threat"]["probability"] >= 0.50
    assert res["suppress_alert"] is False
    assert res["severity_tier"] in ("HIGH", "CRITICAL")
    assert res["device"] == "cpu"


def test_laya_cpu_verifier_suppresses_benign_noise():
    verifier = LayaCPUVerifier()
    # "User failed quiz" contains "failed" which normally tricks naive regex
    res = verifier.decide("INFO User failed test quiz sample on try 1")

    assert res["is_benign_noise"]["decision"] is True
    assert res["suppress_alert"] is True
    assert res["is_threat"]["probability"] < 0.20


def test_detect_anomalies_with_laya_integration():
    sample_logs = [
        "INFO Routine user login successful user=jsmith",
        "INFO Routine heartbeat received",
        "INFO Worker task completed",
        "INFO Cache refreshed items=50",
        "INFO User login successful user=agarcia",
        "CRITICAL Unauthorized access denied resource=/admin source=198.51.100.8",
        "INFO User failed quiz sample on try 1",
    ]
    anomalies = detect_anomalies(sample_logs, use_laya=True)

    assert len(anomalies) > 0
    # Every anomaly should now contain Laya verification fields
    for row in anomalies:
        assert "laya_threat_prob" in row
        assert "laya_suppress" in row
