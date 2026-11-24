import json
import numpy as np
import pytest
from experiments.expressivity_tasks.x47_xsa_attention.cpu_repro_bench import difference, fingerprint, summary


def test_error_measurement_distinguishes_bit_identity_from_element_accuracy():
    a = np.array([0.0, 1.0, 1e-10], dtype="float32")
    b = a.copy()
    b[0] = -0.0
    b[1] = np.nextafter(b[1], np.float32(2))
    r = summary(a, b)
    assert r["unequal"] == 2 and r["max_abs"] == float(2**-23) and r["within_t1"]
    assert summary(a, a)["unequal"] == 0
    b[2] = 0.01
    assert not summary(a, b)["within_t1"]
    with pytest.raises(ValueError):
        summary(a, a.astype("float64"))
    with pytest.raises(ValueError):
        summary(a, b[:2])
    b[2] = np.nan
    with pytest.raises(ValueError):
        summary(a, b)


def test_private_diff_checks_contents_and_source_not_just_shapes(tmp_path):
    a = np.ones((2, 2), np.float32)
    for n in ("one", "two"):
        p = tmp_path / n
        p.mkdir()
        np.savez(p / "private-tensors.npz", g=a)
        (p / "identity.json").write_text(
            json.dumps(dict(source_packet="test", snapshots={"g": dict(sha=fingerprint(a), shape=[2, 2])}))
        )
    assert difference(tmp_path, "one", "two")["g"]["unequal"] == 0
    np.savez(tmp_path / "two/private-tensors.npz", g=a - 1)
    with pytest.raises(AssertionError, match="corrupted"):
        difference(tmp_path, "one", "two")
