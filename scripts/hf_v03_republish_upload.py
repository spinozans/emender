#!/usr/bin/env python3
"""republish-hf-v03 step 2 — APPROVAL-GATED weights-only overwrite on @v0.3.

Author-approved public write (Erik Garrison): overwrite ONLY model.safetensors
on poietic-pbc/{emender-e88-1.3b,gdn-1.3b,m2rnn-cma-1.3b}@v0.3 with the verified
y-mode safetensors. modeling code / config.json / tokenizer are LEFT UNCHANGED
(inherited from the parent commit). v0.1 and v0.2 are NOT touched.

Mechanism (HF has immutable commits; "in place on v0.3" == v0.3 now resolves to
weights-corrected content):
  1. preflight: capture resolved SHA + ref target for v0.1/v0.2/v0.3 and main;
     hash the small (non-LFS) files on current v0.3.
  2. upload_file(model.safetensors only) to `main` (parent = current main).
  3. pre-tag readback at the new commit: model.safetensors LFS sha256 == local;
     code/config/tokenizer sha256 == preflight (UNCHANGED).
  4. move the v0.3 tag to the new commit (delete + recreate).
  5. postflight: v0.3 -> new commit; v0.1/v0.2 resolved SHA AND ref target
     byte-identical to preflight (untouched).

Adapted from scripts/publish_v03_public_hf.py (preserve-tag capture + readback).
Output: /tmp/republish-v03-upload-<agent>/summary.json
"""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RevisionNotFoundError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_v03_republish_lib as L

BUILD_ROOT = "/tmp/republish-v03-build"
BUILD_RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_v03_republish_build_result.json")
RELEASE_TAG = "v0.3"
UPLOAD_REVISION = "main"
PRESERVE_TAGS = ("v0.1", "v0.2")
SMALL_FILES = ("config.json", "configuration_ndm.py", "modeling_ndm.py",
               "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 27), b""):
            h.update(blk)
    return h.hexdigest()


def retry(label, func, attempts=8, delay=2.0):
    last = None
    for i in range(1, attempts + 1):
        try:
            return func()
        except (HfHubHTTPError, RevisionNotFoundError, EntryNotFoundError) as e:
            last = e
            if i == attempts:
                break
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts}: {last}")


def ensure_approval(args):
    if not args.approved_public_v03_weights_overwrite:
        raise SystemExit("--approved-public-v03-weights-overwrite is required for any public HF write")
    note = args.approval_note.strip().lower()
    required = [
        "overwrite model.safetensors",
        "poietic-pbc/emender-e88-1.3b",
        "poietic-pbc/gdn-1.3b",
        "poietic-pbc/m2rnn-cma-1.3b",
        "v0.3",
    ]
    missing = [t for t in required if t not in note]
    if missing:
        raise SystemExit(f"--approval-note missing required approval text: {missing}")


def refs_dict(api, repo_id):
    refs = api.list_repo_refs(repo_id, repo_type="model", token=False)
    return {"branches": [{"name": b.name, "target_commit": b.target_commit} for b in refs.branches],
            "tags": [{"name": t.name, "target_commit": t.target_commit} for t in refs.tags]}


def ref_target(refs, kind, name):
    for r in refs[kind]:
        if r["name"] == name:
            return r["target_commit"]
    return None


def info(api, repo_id, rev, files_metadata=False):
    return retry(f"model_info({repo_id},{rev})",
                 lambda: api.model_info(repo_id, revision=rev, token=False, files_metadata=files_metadata))


def small_file_sha(repo_id, rev, fn, cache_dir):
    p = retry(f"dl {repo_id}@{rev}:{fn}",
              lambda: hf_hub_download(repo_id, filename=fn, revision=rev, repo_type="model",
                                      token=False, cache_dir=str(cache_dir)))
    return sha256_file(p)


def remote_safetensors_meta(api, repo_id, rev):
    i = info(api, repo_id, rev, files_metadata=True)
    sib = {s.rfilename: s for s in i.siblings}
    st = sib["model.safetensors"]
    lfs = getattr(st, "lfs", None)
    return {"resolved_sha": i.sha,
            "size": getattr(st, "size", None),
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None}


def preflight(api, repo_id, cache_dir):
    main = info(api, repo_id, UPLOAD_REVISION)
    if getattr(main, "private", None) is not False:
        raise RuntimeError(f"{repo_id} not public before upload")
    refs = refs_dict(api, repo_id)
    preserved = {}
    for tag in PRESERVE_TAGS:
        tgt = ref_target(refs, "tags", tag)
        if tgt is None:
            raise RuntimeError(f"{repo_id} preserved tag {tag} missing before upload")
        preserved[tag] = {"resolved_sha": info(api, repo_id, tag).sha, "ref_target": tgt}
    small = {fn: small_file_sha(repo_id, RELEASE_TAG, fn, cache_dir / "pre") for fn in SMALL_FILES}
    return {"repo_id": repo_id,
            "main_sha": main.sha,
            "v03_resolved_before": info(api, repo_id, RELEASE_TAG).sha,
            "v03_ref_target_before": ref_target(refs, "tags", RELEASE_TAG),
            "preserved_before": preserved,
            "small_sha_before": small,
            "refs_before": refs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--approved-public-v03-weights-overwrite", action="store_true")
    ap.add_argument("--approval-note", required=True)
    ap.add_argument("--models", nargs="*", default=["e88", "gdn", "m2rnn"])
    ap.add_argument("--output", type=Path,
                    default=Path(f"/tmp/republish-v03-upload-{os.environ.get('WG_AGENT_ID','agent')}/summary.json"))
    args = ap.parse_args()
    ensure_approval(args)

    build = json.load(open(BUILD_RESULT))
    if not build.get("_all_gate_pass"):
        raise SystemExit("build gate did not pass — refusing to upload")

    api = HfApi()
    auth = api.whoami(token=True)
    cache_root = args.output.parent / "readback-cache"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"ok": False, "release_tag": RELEASE_TAG, "preserve_tags": list(PRESERVE_TAGS),
               "auth": {"name": auth.get("name"), "type": auth.get("type"),
                        "orgs": sorted(o.get("name") for o in auth.get("orgs", []) if o.get("name"))},
               "approval_note": args.approval_note, "models": []}
    json.dump(summary, open(args.output, "w"), indent=2)

    specs = [m for m in L.MODELS if m["key"] in args.models]
    for spec in specs:
        key, name, repo = spec["key"], spec["name"], spec["repo"]
        out_dir = os.path.join(BUILD_ROOT, name)
        local_st = os.path.join(out_dir, "model.safetensors")
        local_sha = build[name]["build"]["safetensors_sha256"]
        local_size = build[name]["build"]["safetensors_size"]
        assert sha256_file(local_st) == local_sha, f"{name} local safetensors sha drift"
        cache_dir = cache_root / key

        L.log(f"==== {repo} ====")
        pf = preflight(api, repo, cache_dir)
        L.log(f"{name} preflight: main={pf['main_sha']} v0.3={pf['v03_resolved_before']} "
              f"v0.1={pf['preserved_before']['v0.1']['resolved_sha']} v0.2={pf['preserved_before']['v0.2']['resolved_sha']}")

        # ---- upload model.safetensors ONLY to main ----
        L.log(f"{name} uploading model.safetensors ({local_size} bytes, sha {local_sha[:16]}) to main…")
        commit = api.upload_file(
            path_or_fileobj=local_st,
            path_in_repo="model.safetensors",
            repo_id=repo, repo_type="model", revision=UPLOAD_REVISION,
            parent_commit=pf["main_sha"], token=True,
            commit_message="republish v0.3: overwrite x-mode weights with verified y-mode weights",
            commit_description=(
                "Approved weights-only overwrite (republish-hf-v03). Replaces the broken "
                "schedule-free x-mode model.safetensors with the y-mode (training) weights "
                "recovered via the schedule-free optimizer.train() swap. Reproduces the "
                "live-harness held-out BPB. Modeling code, config.json and tokenizer are "
                "UNCHANGED (inherited from the parent commit). v0.1 and v0.2 are untouched."),
        )
        new_commit = commit.oid
        L.log(f"{name} uploaded -> commit {new_commit}")

        # ---- pre-tag readback at the new commit ----
        rb = remote_safetensors_meta(api, repo, new_commit)
        if rb["resolved_sha"] != new_commit:
            raise RuntimeError(f"{name} readback resolved {rb['resolved_sha']} != {new_commit}")
        if rb["size"] != local_size:
            raise RuntimeError(f"{name} remote safetensors size {rb['size']} != {local_size}")
        if rb["lfs_sha256"] != local_sha:
            raise RuntimeError(f"{name} remote safetensors LFS sha {rb['lfs_sha256']} != {local_sha}")
        small_after = {fn: small_file_sha(repo, new_commit, fn, cache_dir / "postcommit") for fn in SMALL_FILES}
        changed = [fn for fn in SMALL_FILES if small_after[fn] != pf["small_sha_before"][fn]]
        if changed:
            raise RuntimeError(f"{name} non-weight files CHANGED (must stay unchanged): {changed}")
        L.log(f"{name} pre-tag readback OK: weights overwritten, code/config/tokenizer unchanged")

        # ---- move the v0.3 tag to the new commit ----
        api.delete_tag(repo_id=repo, tag=RELEASE_TAG, repo_type="model", token=True)
        api.create_tag(repo_id=repo, tag=RELEASE_TAG, revision=new_commit, repo_type="model",
                       token=True, exist_ok=False,
                       tag_message="v0.3 (republished): verified y-mode weights")
        tagged = info(api, repo, RELEASE_TAG)
        if tagged.sha != new_commit:
            raise RuntimeError(f"{name} v0.3 tag resolved {tagged.sha} != {new_commit}")

        # ---- postflight: v0.1/v0.2 untouched ----
        refs_after = refs_dict(api, repo)
        preserved_after = {}
        for tag, before in pf["preserved_before"].items():
            r_sha = info(api, repo, tag).sha
            r_tgt = ref_target(refs_after, "tags", tag)
            if r_sha != before["resolved_sha"] or r_tgt != before["ref_target"]:
                raise RuntimeError(f"{name} {tag} CHANGED after publish: "
                                   f"sha {r_sha} vs {before['resolved_sha']}, tgt {r_tgt} vs {before['ref_target']}")
            preserved_after[tag] = {"resolved_sha": r_sha, "ref_target": r_tgt}
        L.log(f"{name} postflight OK: v0.3 now {new_commit}; v0.1/v0.2 untouched")

        summary["models"].append({
            "key": key, "repo_id": repo, "identity": spec["identity"],
            "v03_old_resolved_sha": pf["v03_resolved_before"],
            "v03_old_ref_target": pf["v03_ref_target_before"],
            "v03_new_resolved_sha": new_commit,
            "v03_new_ref_target": ref_target(refs_after, "tags", RELEASE_TAG),
            "upload_commit_url": commit.commit_url,
            "safetensors_sha256": local_sha, "safetensors_size": local_size,
            "remote_safetensors_lfs_sha256": rb["lfs_sha256"],
            "small_files_sha_before": pf["small_sha_before"],
            "small_files_sha_after": small_after,
            "small_files_unchanged": True,
            "preserved_before": pf["preserved_before"],
            "preserved_after": preserved_after,
        })
        json.dump(summary, open(args.output, "w"), indent=2)
        print(json.dumps({"ok": True, "model": key, "repo": repo,
                          "v0.3_old": pf["v03_resolved_before"], "v0.3_new": new_commit}, sort_keys=True), flush=True)

    summary["ok"] = True
    json.dump(summary, open(args.output, "w"), indent=2)
    print(f"summary: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}\n{traceback.format_exc()}", file=sys.stderr)
        raise
