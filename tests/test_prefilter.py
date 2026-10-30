from prefilter import FastLogPreFilter


def test_prefilter_discards_health_checks():
    filter_engine = FastLogPreFilter()
    assert not filter_engine.should_keep("2026-10-29 12:00:00 INFO health check passed service=api")
    assert not filter_engine.should_keep("2026-10-29 12:00:01 INFO GET /healthz status=200")
    assert not filter_engine.should_keep("2026-10-29 12:00:02 INFO GET /readyz HTTP/1.1 200")


def test_prefilter_retains_anomalies_and_errors():
    filter_engine = FastLogPreFilter()
    assert filter_engine.should_keep("2026-10-29 12:00:03 ERROR Failed database connection")
    assert filter_engine.should_keep("2026-10-29 12:00:04 CRITICAL Unauthorized access denied resource=/admin")
    assert filter_engine.should_keep("2026-10-29 12:00:05 WARNING Request timeout endpoint=/api/orders")


def test_prefilter_batch_reduces_noise():
    logs = [
        "2026-10-29 10:00:00 INFO health check passed service=api",
        "2026-10-29 10:00:01 INFO GET /healthz status=200",
        "2026-10-29 10:00:02 INFO Scheduled task completed task=cleanup",
        "2026-10-29 10:00:03 ERROR Database connection failed retries=5",
        "2026-10-29 10:00:04 CRITICAL Unauthorized access denied resource=/admin",
    ]
    filter_engine = FastLogPreFilter()
    kept, stats = filter_engine.filter_logs(logs)

    assert len(kept) == 2
    assert stats["dropped_routine"] == 3
    assert stats["passed_candidates"] == 2
    assert stats["noise_reduction_percentage"] == 60.0
