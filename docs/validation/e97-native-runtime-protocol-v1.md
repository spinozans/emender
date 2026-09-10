# Native runtime protocol boundary v1

Date: 2026-09-10. **Serialization/routing qualification passed. Tool execution
qualification was pending at this stage.** This boundary executes no shell/editor commands.
The subsequent [upstream execution gate](e97-openhands-native-execution-v2.md)
passed 18 checks; model-serving and training admission remain pending.

## Implementation

`scripts/e97_open_swe_native_runtime_protocol.py` implements strict generated-turn
admission and causal episode history using the published dataset codec:

- All five fields, canonical JSON, permitted action names and private budgets
  are checked before accepting a generated turn.
- Public commentary and final messages remain separate exact-valued events.
  Private reasoning and think thoughts never enter ordinary backend calls or
  public events. The source Think flag grants no permission.
- Shell/editor argument values are not normalized, clipped or semantically
  repaired. Invalid source ranges remain requests for the source executor to
  handle, not requests secretly replaced with other operations.
- Observations must follow pending calls. Completed episodes cannot generate
  another action. Context overflow fails without truncation or partial mutation.
- This is a single-task episode boundary, not a multi-episode chat service.

Twelve new tests cover these rules, including control characters, private routing,
exact whitespace, source stdin string values, long timeouts, invalid ranges,
canonical/type failures and transactional context overflow. With the eight
rebuild tests, **20 passed**.

## Real-source framing check

Frozen root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-runtime-protocol-v1`.

Managed process `proc_96c4` passed in **69 seconds**. A predeclared hash policy
selected **32 training trajectories** from dataset manifest
`255b02f3b3bbdc85eb15a89d22e7ff2c4bf69f151c2dfb5941a22468711993f5`.

At all **1,212 assistant boundaries**, runtime prompt bytes and token IDs matched
the corresponding original training prefix. This checked **37,769,734 prefix
token positions**, with repetition across prefixes—not that many distinct data
tokens. Complete histories, exact backend arguments, and public/private routing
also passed. No model was loaded and no source command was executed. The sampled
training trajectories are not independent holdout evidence.

- Frozen inventory: `f54a5e0fa84e8aaeb88f84777d255e8a367e7ddeeb08a848bed2da9a4b245406`
- Panel: `0d512b8403cc91b80bf56b2891562bd01cd02523b262203a135494280c340b43`
- Summary: `71c32e03575b337d2da538bcb930968ec9e42429d7de2147a28b00692dfb81b3`

## Publication regression

Rebuild code/docs were pushed as `b3d2c312`; 18 tests passed from a clean export
of its staged tree. Fidelity audits and numerical candidates were pushed as
`d29316e9`; **103 tests passed and one GPU-only test was skipped** from another
clean staged-tree export (`proc_4513`, 123 seconds). This included CPU fixed-world
merge/restart tests and the new opt-in numerical code. It does not replace the
separately frozen GPU evidence or authorize a full-model training configuration.

For numerical/restart work, ADR-003's applicable safety intentions remain
R07/R12 (complete committed state), R14/NDP13 (bounded execution), R16 (exact
source/evidence), and NDP15 checkpoint atomicity. Elastic/native/V21S/ISP and
Frontier/ROCm qualification are not claimed by these local regressions.

## Actual executor preparation

To avoid another approximate shell/editor implementation, the next adapter will
use upstream OpenHands components. The reference source is **OpenHands 0.53.0**,
commit `9ee704a25a331d0d2eb9a8e87a4dcff1d948855b`; its dependency lock pins
`openhands-aci==0.3.1`. This is a qualification candidate, **not a claim that the
historical dataset's exact runtime revision has been identified**.

The old documented runtime registry `docker.all-hands.dev` failed DNS resolution
in `proc_6e51`; the failed `native-runtime-provision-v1` root is retained. No image
was executed. Rather than retry that endpoint or install a different current
runtime, the exact GitHub source was acquired in `proc_672a` (2 seconds):

`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-runtime-source-v1`

Source manifest: `02da6dd136135ac947ca1729fd1e234a92ee72d4a8211f70e7acf1dcebab062a`.

The source preparation and image-build scripts are separate. Image construction
exports the upstream locked main dependencies with hashes, selects an immutable
base digest, copies verified upstream code without running its project build
hook, and records the resulting image ID and installed packages. It uses the
local Docker daemon only; no Docker socket, credentials, host workspace or GPU
is mounted into a runtime container. Package acquisition uses network; later
execution qualification must disable network and impose resource bounds.
**Image construction alone never marks execution or training qualified.**
