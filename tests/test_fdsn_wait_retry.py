from pathlib import Path

import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNServiceUnavailableException

import BayesISOLA.workflows as workflows


ORIGIN = UTCDateTime("2026-11-02T02:47:00")


class StubPayload(list):
    """Stand-in for a Stream/Inventory; file validation is patched out in these tests."""

    def write(self, filename, format=None):
        Path(filename).write_bytes(b"stub")


class FakeClient:
    """Minimal FDSN client returning queued waveform errors before succeeding."""

    base_url = "http://fake-fdsn"

    def __init__(self, errors=None):
        self.errors = {key: list(value) for key, value in (errors or {}).items()}
        self.waveform_calls = []

    def get_stations(self, **kwargs):
        return StubPayload()

    def get_waveforms(self, **kwargs):
        station = kwargs["station"]
        self.waveform_calls.append(station)
        queued = self.errors.get(station)
        if queued:
            raise queued.pop(0)
        return StubPayload([station])


def _station_df(*stations):
    return pd.DataFrame([{
        "network": "NZ", "station": station, "location": "10", "channel_prefix": "HH",
        "selected_channels": "HHZ,HHN,HHE", "sample_rate": 100.0, "station_lat": -41.0, "station_lon": 174.0,
        "distance_km": 50.0 + index, "azimuth_deg": 10.0, "back_azimuth_deg": 190.0,
    } for index, station in enumerate(stations)])


@pytest.fixture
def patched(monkeypatch):
    sleeps = []
    clock = {"now": ORIGIN + 3600}
    monkeypatch.setattr(workflows.time, "sleep", sleeps.append)
    monkeypatch.setattr(UTCDateTime, "now", classmethod(lambda cls: clock["now"]))
    monkeypatch.setattr(workflows, "_raw_inputs_match_request", lambda *args, **kwargs: (True, "ok"))
    return sleeps, clock


def _run(tmp_path, client, stations, **kwargs):
    return workflows.get_mseed_stationxml(
        "test", ORIGIN, 174.0, -41.0, 10.0, t_before=60.0, t_after=480.0, max_radius_km=200.0,
        output_dir=tmp_path, station_df=_station_df(*stations), client=client, overwrite=True, **kwargs,
    )


def test_no_wait_for_historical_event(tmp_path, patched):
    sleeps, _ = patched
    _, downloaded, _, window = _run(tmp_path, FakeClient(), ["AAA"])
    assert sleeps == []
    assert window["realtime_wait_s"] == 0.0
    assert list(downloaded["station"]) == ["AAA"]


def test_waits_for_future_window_plus_buffer(tmp_path, patched):
    sleeps, clock = patched
    clock["now"] = ORIGIN + 300
    _, _, _, window = _run(tmp_path, FakeClient(), ["AAA"])
    assert sleeps == [pytest.approx(480.0 - 300.0 + 60.0)]
    assert window["realtime_wait_s"] == pytest.approx(240.0)


def test_buffer_applies_when_window_end_just_passed(tmp_path, patched):
    sleeps, clock = patched
    clock["now"] = ORIGIN + 480 + 20
    _run(tmp_path, FakeClient(), ["AAA"])
    assert sleeps == [pytest.approx(40.0)]


def test_transient_error_is_retried_for_failed_station_only(tmp_path, patched):
    sleeps, _ = patched
    client = FakeClient({"BBB": [FDSNServiceUnavailableException("503")]})
    _, downloaded, log, _ = _run(tmp_path, client, ["AAA", "BBB"], download_retry_wait_s=60.0)
    assert client.waveform_calls == ["AAA", "BBB", "BBB"]
    assert sleeps == [60.0]
    assert sorted(downloaded["station"]) == ["AAA", "BBB"]
    assert list(log.loc[log["station"] == "BBB", "status"]) == ["download_retry", "downloaded"]


def test_no_data_is_not_retried(tmp_path, patched):
    sleeps, _ = patched
    client = FakeClient({"BBB": [FDSNNoDataException("204")]})
    _, downloaded, log, _ = _run(tmp_path, client, ["AAA", "BBB"])
    assert client.waveform_calls == ["AAA", "BBB"]
    assert sleeps == []
    assert list(downloaded["station"]) == ["AAA"]
    assert list(log.loc[log["station"] == "BBB", "status"]) == ["download_failed"]


def test_raises_after_max_attempts(tmp_path, patched):
    sleeps, _ = patched
    client = FakeClient({"AAA": [FDSNServiceUnavailableException("503")] * 5})
    with pytest.raises(ValueError, match="No station waveforms were acquired successfully"):
        _run(tmp_path, client, ["AAA"], max_download_attempts=3)
    assert client.waveform_calls == ["AAA"] * 3
    assert sleeps == [60.0, 60.0]
    log = pd.read_csv(tmp_path / "metadata" / "download_log.csv")
    assert list(log["status"]) == ["download_retry", "download_retry", "download_failed"]
    assert list(log["download_attempt"]) == [1, 2, 3]
