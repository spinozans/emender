"""Independent, public receipt-only audit of the S16 descriptor/worker lifetime split."""

import argparse
import json
from pathlib import Path

from .branch_domain_verify import audit as audit_s15, price, read_panel
from .serving_verify import HERE, load, require

ROOT = HERE / "receipts/memory-program-s16"
ORDER = ("fresh-A", "image-A", "image-B", "fresh-B")


def audit(root=ROOT):
    prior = HERE / "receipts/memory-program-s15"
    ref15 = audit_s15(prior)
    require(
        ref15 == load(prior / "verified.json")
        and ref15["exact_rebinding_gate"]
        and not ref15["resident_recommended"]
        and ref15["remaining_s"] == 414,
        "unverified S15 comparison",
    )
    ref, end, metrics = read_panel(
        root, order=ORDER, gates=148, resident_labels=(), compare_to=prior / "fresh-A"
    )
    require(
        (root / "source-ref").read_text().strip() == ref
        and (root / "protocol-ref").read_text().strip() == "c4030adef46caee6c52f4a78871d1613425b0c4a",
        "unbound S16 freeze",
    )
    docs = load(HERE / "b1_mounts.json")["P100"]["docs"]
    headers = set()
    for label, v in metrics.items():
        c = load(root / label / "cell.json")
        im = c["image"]
        retained = label.startswith("image")
        require(
            (im is not None) == retained and v["opens"] == len(c["workspace_starts"]) == 4, "worker count"
        )
        image_ns = image_close = 0
        n_attested = 0
        if retained:
            require(
                im["full_verifications"] == 1
                and im["workers_started"] == im["workers_reaped"] == 4
                and im["offsets_checked"] == docs == 16119767
                and im["starts_bytes"] == 8 * docs
                and im["closed"]
                and im["checks"] > 4
                and im["open_ns"] > 0
                and im["close_ns"] > 0,
                "incomplete pinned image/attestation",
            )
            image_ns, image_close = im["open_ns"], im["close_ns"]
            n_attested = 1
        close = 0
        attest = handshake = 0
        for w, op in zip(c["workspace_starts"], c["opens"], strict=True):
            headers.add((w["root"], w["nodes"]))
            require(
                w["pid"] == op["wire"]["pid"]
                and close < w["start_ns"] < w["reaped_ns"]
                and 0 < w["root"] <= w["nodes"] == 21707038
                and w["documents"] == docs
                and w["starts_bytes"] == 8 * docs
                and w["attest_ns"] >= 0
                and (w["attest_ns"] == 0) == retained
                and w["workspace_ns"] > 0
                and w["attest_ns"] + w["workspace_ns"] <= op["open_ns"],
                "missing native launch/verify/exit",
            )
            close = w["reaped_ns"]
            attest += w["attest_ns"]
            handshake += w["workspace_ns"]
            n_attested += int(w["attest_ns"] > 0)
        require(n_attested == (1 if retained else 4), "unexpected attestation lifetime")
        v.update(
            domain_open_s=v["open_s"],
            full_verifications=n_attested,
            per_process_attest_s=attest / 1e9,
            native_launch_handshake_s=handshake / 1e9,
            image_open_s=image_ns / 1e9,
            image_close_s=image_close / 1e9,
        )
        # The first verification is not a shared free prerequisite for the candidate.
        v["open_s"] += v["image_open_s"]
        require(
            v["wall_s"]
            >= v["open_s"] + v["final_close_s"] + v["image_close_s"] + sum(w["wall_s"] for w in v["waves"]),
            "missing setup, wave work or final close",
        )
    require(len(headers) == 1, "different native tree handshake")
    ratios, ok = price(metrics, "image")
    return dict(
        ref=ref,
        s15_ref=ref15["ref"],
        card_s=0,
        cumulative_x47_upper=1386,
        remaining_s=414,
        elapsed_s=end["elapsed_ns"] / 1e9,
        direct_prefix_predictions=2048,
        sampled_prefix_predictions=2048,
        distinct_frontiers=1024,
        training_targets=0,
        generated_bytes=0,
        updates=0,
        m20_exits=16,
        private_exits=72,
        metrics=metrics,
        ratios=ratios,
        exact_descriptor_gate=True,
        descriptor_image_recommended=ok,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
