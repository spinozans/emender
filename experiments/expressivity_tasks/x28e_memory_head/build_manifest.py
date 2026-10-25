"""Freeze manifest builder — the one spelling of freeze_manifest.json.

Rebuilds ``freeze_manifest.json`` from the actual package tree: every
package file (code, configs, generators, freeze receipts, tests, docs) plus
the LOAD-BEARING DEPENDENCY IDENTITIES bound by M2-review correction F4:

  * ``ndm/models/hybrid_ladder.py`` (the trainer's native import) and its
    transitive native implementation files actually used by the two native
    arms — the E97 triton kernel path (e97.py -> e88_fla_hybrid.py ->
    e88_triton_forward/backward/optimized, e97_chunked_autograd,
    e97_sequential, e97_multiquery_autograd, models/triton_ops.py,
    recurrent_precision.py, triton/pin_autotune.py) and the fla-gdn path
    (models/fla_gated_delta.py; the GatedDeltaNet implementation itself is
    the EXTERNAL pip ``fla`` package — environment identity, not a repo
    file, receipted as a version note in the manifest metadata);
  * the package ``__init__`` files on those import chains (they execute at
    import; pin_autotune is behavior-relevant);
  * ``scripts/gpu_lease.sh`` (the launcher's lease implementation).

Dependency paths are RELATIVE TO THE PACKAGE with ``../`` escapes (the
package sits three levels below the repo root, so the repo-root ``ndm/`` and
``scripts/`` trees are ``../../../ndm/...`` / ``../../../scripts/...``).
``trainer.verify_freeze`` re-hashes every entry from disk and checks every
bound byte is committed at HEAD — changes there invalidate the freeze.

The manifest is byte-stable: no timestamps; the manifest itself is excluded
from the files map (it cannot contain its own hash; ``hash_rule``).

Usage: python build_manifest.py   (run AFTER all receipts are final)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]

# F4: the load-bearing dependency identities (rel paths from the package).
# Every entry executes on a native-arm training/evaluation path or on the
# launcher path; each is verified from disk AND at HEAD by trainer.verify_freeze.
DEPENDENCIES = {
    # -- the trainer's native import and its import chain -------------------
    "../../../ndm/__init__.py":
        "executes at `import ndm.models.hybrid_ladder`",
    "../../../ndm/models/__init__.py":
        "executes at `import ndm.models.hybrid_ladder`",
    "../../../ndm/models/hybrid_ladder.py":
        "trainer.build_native_model: HybridLadderLM (both native arms)",
    "../../../ndm/models/ladder_lm.py":
        "hybrid_ladder imports RMSNorm/get_ladder_level/SwiGLUMLP/round_mlp_hidden",
    # -- the E97 native arm (e97_native): the triton kernel path -------------
    "../../../ndm/models/e97.py":
        "E97SplitEditLayer (the e97_native arm's layer)",
    "../../../ndm/models/e88_fla_hybrid.py":
        "E97's base class (E88FLAHybrid)",
    "../../../ndm/models/triton_ops.py":
        "e88_fla_hybrid fused elementwise ops",
    "../../../ndm/recurrent_precision.py":
        "recurrent_state_precision='legacy' validation + launch config",
    "../../../ndm/triton/__init__.py":
        "executes at `import ndm.triton.e97_chunked_autograd`",
    "../../../ndm/triton/pin_autotune.py":
        "pinned autotune configs (behavior-relevant at kernel launch)",
    "../../../ndm/triton/e88_triton_forward.py":
        "E97 fused forward kernel",
    "../../../ndm/triton/e88_triton_backward.py":
        "E97 fused backward kernel",
    "../../../ndm/triton/e88_triton_optimized.py":
        "E97 fused T-scan apply (training + fused_inference paths)",
    "../../../ndm/triton/e97_chunked_autograd.py":
        "E97 training-path chunked autograd kernel",
    "../../../ndm/triton/e97_sequential.py":
        "E97 fused_inference (evaluation) path",
    "../../../ndm/triton/e97_multiquery_autograd.py":
        "E97 dispatch alternative on the same path",
    # -- the GDN2 native arm (gdn2_native): the fla-gdn path -----------------
    "../../../ndm/models/fla_gated_delta.py":
        "FLAGatedDeltaNetLayer (the gdn2_native arm's layer; the "
        "GatedDeltaNet implementation itself is the EXTERNAL pip `fla` "
        "package — environment identity, receipted in manifest metadata)",
    # -- the launcher ---------------------------------------------------------
    "../../../scripts/gpu_lease.sh":
        "launcher.py lease implementation (acquire/list-free/trap release)",
}

EXCLUDE_FILES = {"freeze_manifest.json"}
EXCLUDE_DIRS = {"__pycache__"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(files: dict) -> str:
    """The hash_rule: SHA-256 over sorted, newline-terminated lines
    ``<sha256>  <relpath>`` of the files mapping (the manifest excluded)."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{files[rel]}  {rel}\n".encode())
    return h.hexdigest()


def package_files() -> dict:
    files = {}
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix != ".pyc":
            rel = path.relative_to(PACKAGE_ROOT).as_posix()
            if rel not in EXCLUDE_FILES:
                files[rel] = sha256_file(path)
    return files


def _imported_repo_files(path: Path) -> list:
    """AST walk: repo .py files imported by `path` (absolute `ndm.*` and
    relative imports), M2-review-2 finding 3: the manifest binds the ACTUAL
    executable import closure, not a hand-listed subset."""
    import ast
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    pkg_parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(pkg_parts[:-1])
                if node.level > 1:
                    base = base[:-(node.level - 1)]
                mod = ".".join(base)
                if node.module:
                    mods.append(f"{mod}.{node.module}") if False else None
                    mod = f"{mod}.{node.module}"
                    mods.append(mod)
                else:
                    mods.append(mod)
                for a in node.names:
                    mods.append(f"{mod}.{a.name}")
            elif node.module:
                mods.append(node.module)
                mods.extend(f"{node.module}.{a.name}" for a in node.names)
    files = set()
    for m in mods:
        if not m or not (m == "ndm" or m.startswith("ndm.")):
            continue
        rel = Path(*m.split("."))
        for cand in (rel.with_suffix(".py"), rel / "__init__.py"):
            p = REPO_ROOT / cand
            if p.is_file():
                files.add(p.resolve())
    return sorted(files)


def dependency_closure(dependencies: dict) -> dict:
    """Fixpoint over the hand-listed repo dependencies: every repo file they
    import (transitively) joins the manifest with an import-closure reason.
    External pip packages are NOT repo files - they are bound as environment
    identities (see `environment` in the manifest) and validated by
    trainer.verify_freeze."""
    closed = dict(dependencies)
    frontier = [(PACKAGE_ROOT / rel).resolve() for rel in dependencies]
    seen = set(frontier)
    while frontier:
        nxt = []
        for path in frontier:
            for imported in _imported_repo_files(path):
                if imported in seen:
                    continue
                seen.add(imported)
                rel = "../../../" + imported.relative_to(REPO_ROOT).as_posix()
                if rel not in closed:
                    closed[rel] = ("import closure: executes at import of the "
                                   "bound chain (M2-review-2 finding 3)")
                nxt.append(imported)
        frontier = nxt
    return closed


def main() -> int:
    files = package_files()
    n_package = len(files)
    deps = dependency_closure(DEPENDENCIES)
    for rel, why in sorted(deps.items()):
        path = (PACKAGE_ROOT / rel).resolve()
        if not path.is_file():
            raise SystemExit(f"dependency missing: {rel} ({path})")
        if rel in files:
            raise SystemExit(f"dependency path collides with a package file: {rel}")
        files[rel] = sha256_file(path)
    import importlib.metadata as _md
    environment = {
        "fla_version": _md.version("flash-linear-attention"),
        "torch_version": _md.version("torch"),
        "distribution_names": {"fla": "flash-linear-attention"},
        "note": "installed identities validated by trainer.verify_freeze "
                "(importlib.metadata read - no package import on the "
                "verify-only path)",
    }
    manifest = {
        "files": {rel: files[rel] for rel in sorted(files)},
        "hash_rule": "manifest_hash = sha256 over the sorted, newline-"
                     "terminated lines '<sha256>  <relpath>' of the 'files' "
                     "mapping; the manifest itself is excluded (it cannot "
                     "contain its own hash). trainer.verify_freeze recomputes "
                     "every file hash from disk, the manifest self-hash, and "
                     "the --frozen-freeze-hash argument, and UNCONDITIONALLY "
                     "verifies worktree AND HEAD for every bound path — the "
                     "package subtree AND the F4 dependency identities "
                     "(rel paths with ../ escapes: the ndm native "
                     "implementation files used by the two native arms and "
                     "scripts/gpu_lease.sh); changes there invalidate the "
                     "freeze (M2-review corrections F2/F4).",
        "manifest": "X28E engineering freeze (M2, post-review-correction "
                    "package) - SHA-256 of every code/config/generator/"
                    "receipt/test file in the package plus the F4 "
                    "load-bearing dependency identities",
        "dependency_binding": {
            "rule": "every '../'-prefixed entry is a load-bearing dependency "
                    "identity, verified from disk AND at HEAD by "
                    "trainer.verify_freeze and the launcher",
            "n_dependency_files": len(deps),
            "environment": environment,
            "files_why": dict(sorted(deps.items())),
        },
        "manifest_hash": manifest_hash(files),
        "registration_commit_m1": "574bf3994fee15547e1977bd6c1fa7487bb27137",
        "corrections": "M2-review phase-4 corrections applied (F1-F11 + "
                       "adjudication (a)); see REGISTRATION.md amendments and "
                       "ENGINEERING_FREEZE.md",
        "zero_optimizer_updates": True,
    }
    out = PACKAGE_ROOT / "freeze_manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest rebuilt: {n_package} package files + "
          f"{len(deps)} dependency identities; "
          f"manifest_hash={manifest['manifest_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
