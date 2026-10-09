"""Author-side seed, morph, dedupe and real-Pi solvability gates (era8)."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts.e97_diversity import (FAMILIES, MORPHS, capture_end_state,
                                   check_end_state, digest, canonical)


def admission_prescreen(spec, existing_prescreen):
    existing_prescreen(spec)
    if spec.get("family") in FAMILIES:
        # Follow-up user turns become model-visible later. Screen the sealed
        # solution/script too, without unsealing it into public fixtures.
        proxy = {**spec, "files": [*spec["files"], {"path": "sealed/era8-solution.txt", "content": canonical(spec["solution"])}]}
        existing_prescreen(proxy)


def seed_instruction(seed):
    if seed is None:
        return ""
    if (not isinstance(seed, dict) or set(seed) != {"kind", "payload", "provenance"}
            or seed["kind"] not in {"code", "config", "tree", "error", "solved_bundle"}
            or not isinstance(seed['provenance'], dict)
            or len(canonical(seed)) > 28000):
        raise ValueError("invalid bounded first-party SEED object")
    if seed['kind'] != 'solved_bundle' and seed['provenance'].get('origin') != 'invented':
        raise ValueError('fragment SEED provenance must be invented; harvested first-party seeds use solved_bundle')
    if seed['kind'] == 'solved_bundle' and not (seed['provenance'].get('split') == 'train' or 'base_sha256' in seed['provenance']):
        raise ValueError('solved SEED must be TRAIN-admitted or an already-verified internal morph base')
    return ("\nSEED (first-party or invented only; inspiration, not a verbatim copy):\n"
            + canonical(seed) + "\nInvent a NEW task around this seed. All family and admission requirements stand.\n")


def lake_dedupe_index(lake):
    prompts, identities = set(), set()
    for path in Path(lake).glob("*/tasks.jsonl"):
        for line in path.read_text().splitlines():
            task = json.loads(line)["task"]
            prompts.add(hashlib.sha256(task["prompt"].encode()).hexdigest())
            identities.add(task["identity"])
    return prompts, identities


def check_dedupe(spec, known_prompts, *, identity=None, known_identities=()):
    prompt_sha = hashlib.sha256(spec["prompt"].encode()).hexdigest()
    if prompt_sha in known_prompts:
        raise ValueError("prompt_sha256 collides with lake/base/tranche")
    if identity is not None and identity in known_identities:
        raise ValueError("task_identity collides with lake/base/tranche")
    return prompt_sha


def harvest_solved(author_work, lake, verify_admitted, count=8):
    """Only admitted TRAIN + matching actual passing local proof; no downloads."""
    seeds = []
    for tranche in sorted((Path(author_work) / "teacher-authored").glob("tranche-*"), reverse=True):
        root = Path(lake) / f"e97-firstparty-teacher-authored-{tranche.name}-admitted"
        if not root.is_dir() or not (tranche / "proofs.json").is_file():
            continue
        bundles, _ = verify_admitted(root)  # existing receipt/allowlist/overlap gates
        solved = {p["task_identity"] for p in json.loads((tranche / "proofs.json").read_bytes())
                  if p.get("status") == "pass" and all(v.get("status") == "pass" for v in p.get("validators", {}).values())}
        specs = [json.loads(p.read_bytes()) for p in sorted(tranche.glob("authored-*.json"))]
        for bundle in bundles:
            if bundle["split"] != "train" or bundle["task"]["identity"] not in solved:
                continue
            matches = [s for s in specs if s["prompt"] == bundle["task"]["prompt"]]
            if len(matches) != 1:
                raise ValueError("solved seed cannot bind unique author spec")
            spec = matches[0]
            import inject_pool
            body = inject_pool.gym_task_body(bundle, root)
            if body['workspace_files'] != {f['path']: f['content'] for f in spec['files']}:
                raise ValueError('solved seed fixture drift')
            spec = {**spec, 'split': 'train'}
            provenance = {"task_identity": bundle["task"]["identity"], "split": "train",
                          "lake_root": str(root), "bundle_sha256": digest(bundle),
                          "admission_receipt_sha256": hashlib.sha256((root / "admission-receipt.json").read_bytes()).hexdigest(),
                          "proofs_sha256": hashlib.sha256((tranche / "proofs.json").read_bytes()).hexdigest(),
                          "proofs_path": str(tranche / "proofs.json"),
                          "author_spec_sha256": digest(spec)}
            seeds.append({"kind": "solved_bundle", "payload": spec, "provenance": provenance})
            if len(seeds) >= count:
                return seeds
    if not seeds:
        raise ValueError("no admitted solved TRAIN seeds")
    return seeds


def verify_seed(seed, verify_admitted):
    provenance = seed['provenance']
    if provenance.get('split') != 'train' or digest(seed['payload']) != provenance['author_spec_sha256']:
        raise ValueError('solved seed spec/split drift')
    proof_path = Path(provenance['proofs_path'])
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != provenance['proofs_sha256']:
        raise ValueError('solved seed proof drift')
    if not any(p['task_identity'] == provenance['task_identity'] and p.get('status') == 'pass' for p in json.loads(proof_path.read_bytes())):
        raise ValueError('solved seed proof absent')
    root = Path(provenance['lake_root'])
    if hashlib.sha256((root / 'admission-receipt.json').read_bytes()).hexdigest() != provenance['admission_receipt_sha256']:
        raise ValueError('solved seed admission drift')
    bundles, _ = verify_admitted(root)
    bundle = next(b for b in bundles if b['task']['identity'] == provenance['task_identity'])
    if bundle['split'] != 'train' or digest(bundle) != provenance['bundle_sha256']:
        raise ValueError('solved seed bundle drift')
    import inject_pool
    body = inject_pool.gym_task_body(bundle, root)
    if body['prompt'] != seed['payload']['prompt'] or body['workspace_files'] != {f['path']: f['content'] for f in seed['payload']['files']}:
        raise ValueError('solved seed prompt/fixtures drift')
    return deepcopy(seed['payload'])


def morph_instruction(base, trick):
    if trick not in MORPHS:
        raise ValueError("unknown morph")
    return seed_instruction({"kind": "solved_bundle", "payload": base, "provenance": {"base_sha256": digest(base)}}) + (
        f"\nMORPH: {trick}. Emit a coherent new complete bundle with a NEW prompt, token and fixture tree. "
        "Entity rename: consistently rename entities/paths. Chain depth: add/remove a meaningful dependency step. "
        "Distractor: add plausible irrelevant files without ambiguity. Failure inversion: invert which supplied state "
        "is broken and require the corresponding repair/recovery. Never change family or split. "
        "Update reference plan AND every assertion/script to match. No cached proof may be reused.\n")


def validate_morph(base, candidate, trick, author, known_tokens, known_trees, known_prompts):
    """Pre-build gate only. Every variant still goes through build + full admission."""
    if trick not in MORPHS or candidate.get("family") != base.get("family"):
        raise ValueError("morph family drift")
    if candidate.get("split", base.get("split", "train")) != base.get("split", "train"):
        raise ValueError("morph split drift")
    candidate["split"] = base.get("split", "train")
    # A collision with the base is rejected even when it is not in the lake index.
    author.guards(candidate, known_tokens | {base["token"]}, known_trees | {
        author.fixture_tree_digest({f["path"]: f["content"] for f in base["files"]})})
    check_dedupe(candidate, known_prompts | {hashlib.sha256(base["prompt"].encode()).hexdigest()})
    candidate["morph"] = {"trick": trick, "base_sha256": digest(base)}
    return candidate


def prove_bundle(task, spec, stage):
    """Reference actions execute through actual Pi, then SAME bank grade gates."""
    import tiktoken
    import rl_loop_driver as driver
    import inject_pool
    from scripts.e97_pi_native_codec import native_turn

    body = inject_pool.gym_task_body(task, stage)
    identity = task["task"]["identity"]
    solution = spec["solution"]
    enc = tiktoken.get_encoding("p50k_base")
    curriculum = driver.load_curriculum()
    manifest = driver.tool_manifest(curriculum)
    tools = manifest["model_visible_tools"]
    actions = deepcopy(solution["actions"])
    if spec["family"] == "terminal":
        actions.append({"tool": "finish", "arguments": {"message": "done"}})
    index = 0

    def generate(prompt, budget, deadline):
        nonlocal index
        if index >= len(actions):
            raise ValueError("reference plan exhausted before native close")
        action = actions[index]
        index += 1
        message = {"role": "assistant", "content": None,
                   "reasoning_content": f"Reference step {index}: execute the supplied {action['tool']} operation against real observations.",
                   "tool_calls": [{"id": f"reference-{index}", "type": "function", "function": {
                       "name": action["tool"], "arguments": canonical(action["arguments"])}}]}
        text = native_turn(message, tools)
        ids = enc.encode_ordinary(text)
        if len(ids) > budget:
            raise ValueError("reference action exceeds generation budget")
        return text, ids, "valid"

    with tempfile.TemporaryDirectory(prefix="e97-era8-proof-") as tmp:
        tmp = Path(tmp)
        workspace = driver.prepare_workspace(tmp / "workspace", body)
        if check_end_state(solution["assertions"], capture_end_state(workspace, solution["assertions"], identity), identity):
            raise ValueError("degenerate task: end state already satisfied initially")
        episode_dir = tmp / "episode"
        episode_dir.mkdir()
        panel = driver.make_panel(curriculum.SYSTEM, tools, {**driver.TEACHER_PANEL, "max_turns": 12})
        record = driver.run_episode(episode_dir=episode_dir, workspace=workspace, prompt=body["prompt"],
                                    panel=panel, generate=generate, enc=enc,
                                    pi_bin=Path(manifest["pi_bin"]), manifest_path=driver.MANIFEST_PATH,
                                    pilot=None, curriculum=curriculum, seconds=180,
                                    user_script=solution.get("user_script"), end_state_task=True)
        if not record.get("close_verified") or record.get("status") != "finished" or index != len(actions):
            import shutil
            failure = stage.parent / f'proof-failure8-{identity}'
            shutil.copytree(episode_dir, failure)
            raise ValueError(f"real Pi reference solve failed: {record.get('reason')}; evidence={failure}")
        passed, grade = driver.grade_episode(body, workspace, None, {"task_id": identity, "task_sha256": digest(body)},
                                            record=record, episode_dir=episode_dir, stage="proof", system=curriculum.SYSTEM)
        if not passed:
            raise ValueError(f"real Pi proof failed mandatory sealed/degeneracy gates: {grade}")
        from scripts.build_e97_pi_native_curriculum import encode_candidate
        _, mask, units = encode_candidate(record['native_record'], record['generations'], 0, enc)
        if units != index or not sum(mask):
            raise ValueError('reference transcript failed canonical encode/target gate')
        projection = json.loads((episode_dir / "gym-terminal-projection-proof.json").read_bytes())
        # Negative control: same genuine transport transcript, wrong end state.
        mutant = deepcopy(projection)
        check = next(c for c in solution["assertions"] if c["kind"] in {"exact", "regex"})
        mutant["sealed_end_state"]["files"][check["path"]] = None
        mutant_path = tmp / "mutant.json"
        mutant_path.write_text(canonical(mutant))
        import subprocess, sys
        pin = body["task_lake"]["validator"]
        cp = subprocess.run([sys.executable, pin["program_path"], "--mode", "focused", "--spec", pin["spec_path"], "--terminal", str(mutant_path)], capture_output=True, timeout=30)
        if cp.returncode == 0:
            raise ValueError("end-state negative control admitted")
        evidence = stage / "era8-proofs" / identity
        evidence.mkdir(parents=True, exist_ok=False)
        for name in ("episode-private.json", "gym-terminal-projection-proof.json"):
            (evidence / name).write_bytes((episode_dir / name).read_bytes())
        (evidence / "transport-terminal.json").write_bytes((episode_dir / "pi/transport-terminal.json").read_bytes())
        (evidence / "grade.json").write_text(canonical(grade))
        return {"task_identity": identity, "status": "pass", "proof_kind": "era8-real-pi-end-state",
                "solver_steps": index, "native_close_verified": True, "canonical_targets": sum(mask),
                "terminal_sha256": digest(projection), "validator_receipt_sha256": digest(grade),
                "validators": {"focused": {"status": "pass"}, "regression": {"status": "pass"}},
                "degeneracy_screen": "pass", "negative_controls": ["missing output rejected"]}
