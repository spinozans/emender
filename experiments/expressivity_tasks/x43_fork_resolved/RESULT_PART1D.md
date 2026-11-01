# X43 part 1d: documented index leg, with two exposed limitations

The corpus → tape build → query sequence completed on the existing xsa binary.
All **1,115,394 finite-body fork labels and maximal context lengths agree** with
part 1c: 636,727 FORK, 478,667 FORCED; zero label or length disagreements.
The byte-query oracle **fails one of 13 patterns**: an absent `ff` is reported
at the virtual END position. This is a failed query correctness check, even
though the CLI exits zero. A fully conforming MediaWiki XML wrapper preserving
the original body also remains unresolved, as detailed below. This is not an
unqualified success report.

## Commands and resource sizing

Read `/home/erikg/sxgc/REPRODUCE.md` first, especially §§2,5,6,7,9. Ran:

```sh
/home/erikg/sxgc/xsa/target/release/xsa corpus --wikipedia receipts/part1d/shakespeare.xml.bz2 --out receipts/part1d/corpus --separator 0x1e
/home/erikg/sxgc/xsa/target/release/xsa build --tape --input receipts/part1d/corpus/corpus.bin --scratch receipts/part1d/build --threads 1 --memory-gb 2 --timeout 3600 --plan-only
/home/erikg/sxgc/xsa/target/release/xsa build --tape --input receipts/part1d/corpus/corpus.bin --scratch receipts/part1d/build --threads 1 --memory-gb 2 --timeout 3600
/home/erikg/sxgc/xsa/target/release/xsa query receipts/part1d/build/corpus.sxi2 --patterns receipts/part1d/patterns.bin -j 1 --output jsonl --verify
```

Actual absolute argv, exit codes, stdout and stderr are in the corresponding
`*.command.json`, `*.stdout`, `*.stderr` receipts. All Python invocations used
`/home/erikg/emender/.venv/bin/python`; the build subprocess PATH puts that
interpreter's directory first. No GPU, Slurm, service or workers were used.
The xsa checkout was read only.

Two GiB per phase, one thread, and 483.7 GiB free scratch passed the planner.
The corpus exceeds the documented ~1.104 MB tape minimum. The journal reports
66,227 rows/band, scratch-window binding, and `TAPE_BUDGET_OK ... plan=run=1`.
The CLI derives the producer's 1 GiB `--budget-bytes`; no undocumented build flag
was added. The disc planner conservatively allowed 0.1 GiB. The reported
11,340-byte intermediate peak is a 2-second sampling result on a very short
run and must **not** be treated as an accurate peak. Default chi was built by
this binary (425,221 witnesses), unlike the guide's default-absent example;
no source or feature changes were made to reconcile this version difference.

## Corpus binding and XML limitation

The unchanged input is 1,115,394 bytes, SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
One page/revision with title `TinyShakespeare` yields exactly:

```
b"TinyShakespeare\n" + original_input + b"\x1e"
```

The original body begins at corpus offset 16. `corpus.bin` has 1,115,411 bytes,
one separator, one document, SHA256
`09361d34d17be1882e28b4bb30025303779c88f0cd751285bb95e05ea7447a8e`.
`seps.bin` is 60 bytes; `titles.bin` is 41 bytes. The corpus MANIFEST receipts
all sidecar hashes and compressed wrapper provenance.

The first, escaped XML wrapper was well formed, but xsa does **not decode XML
entities** (confirmed in its source and emitted MANIFEST). Its resulting body
was 12 bytes longer: three literal ampersands became `&amp;`. The original
body-identity assertion failed. `escaped_wrapper.xml`, `escaped_corpus/`, and
`escaped_corpus.stdout` preserve that attempt; the directory was moved from
`corpus/` before the second invocation, so the original stdout names its former
path. The initial exception was:

```
    assert corpus.read_bytes()==title.encode()+b'\n'+raw+b'\x1e'
AssertionError
```

The final wrapper passes the tool's documented raw-body parser and preserves
all input bytes, but its three literal ampersands mean it is **not well-formed
XML**. This is an explicit deviation, not a claim to have solved general XML
entity handling. CDATA is not an assumed escape hatch: this parser copies
literal bytes between `<text>` tags. A strictly well-formed XML input and an
unchanged logical body require an upstream entity-aware corpus stage; the
work order forbids modifying xsa. No source bytes were silently repaired.

## Container and gates

`build/corpus.sxi2`: 950,624 bytes, 541,139 BWT runs, virtual-END n=1,115,412,
SHA256 `5b07bfa0adbda9622b1236cf4693ea51aece99188bfc4a1a997c24bc2f6bfbd9`.
The journal terminates `XSA_BUILD_DONE chi=425221 pipeline=v2tape ...` and
explicitly receipts `full_SA=0 persistent_positions=0`. The container header,
directory and every member CRC pass both the independent decoder and CLI
`--verify`. Journal provenance names tree `97eb18e...`, dirty=1, branch main;
`binding.json` additionally pins the binary and guide hashes. No claim that
that preexisting dirty tree equals a clean source revision is made.

## Query validation and failure

Seven 24-byte windows with known body offsets, the final 10-byte suffix, and a
one-byte repeated pattern match every overlapping corpus offset. Both a direct
suffix-array oracle (`direct_sa_queries.json`) and an independent byte scan
(`query_validation.json`) agree on **12/13** patterns, including absent NUL,
absent 24×01, and an accepted 65,536-byte absent pattern.

Pattern 10, one byte `ff`, is absent from the corpus. xsa returns count=1,
offset=1,115,411 (the virtual END), rather than zero. The container remap maps
this unused byte to the END code. This is an observed sentinel alias failure
for arbitrary-byte query semantics; labels over actual corpus bytes are
unaffected. The original validation assertion traceback is retained verbatim
in `validation_initial.stderr`; the final validator writes the full mismatch
receipt and exits nonzero. Do not interpret query process rc=0 as correctness.

Empty file, empty record and 65,537-byte record are refused as documented.
Missing terminal pattern separator, auto-detected FASTA and explicit FASTA
produce correct offsets. Conflicting JSON/count flags and removed `--mode`
and `--ri4` flags are refused. All refusals retain verbatim stderr and rc=1.

## Independent FM derivation

`validate_index.py` decodes m1 canonical Huffman heads and Elias-gamma lengths,
checks all CRCs, and constructs rank occurrence lists and C from the container.
A single LF cycle recovers row coordinates and verifies every BWT byte against
the remapped corpus plus virtual END. This validator deliberately uses O(n)
arrays for the tiny fixture; it is not part of the bounded-memory builder and
does not imply a full SA was persisted by that builder.

For every target i, start with the empty interval, then extend C(i,l) one byte
left using `[C[a]+rank(a,lo), C[a]+rank(a,hi))`. Enumerate until fewer than two
valid finite-body targets remain or the left boundary is reached. At each
depth remove title, separator, END and matches whose next target would fall
outside the body. Enumerate next-byte branches of the last repeated interval;
more than one branch means FORK. Only after producing all labels and lengths
are the retained part1c arrays loaded for comparison. No direct-SA lengths
seed or constrain FM enumeration. Maximum context length is 123.

`cross_validation.json` receipts exact agreement. Boundary filtering implements
DERIVATION_PART1C.md's finite target convention explicitly; this is not a claim
that raw cyclic corpus statistics or minimum suffixient cardinality chi equal
these per-position labels. The two independently computed label artifacts
have identical SHA256, recorded in that receipt.

## Residual risks and reproduction

Run `index_pipeline.py` only with fresh output/build directories, as xsa refuses
overwrite; run `validate_index.py`, `validate_index_edges.py`, and
`validate_index_direct_queries.py` with the approved interpreter to revalidate.
The primary validator intentionally exits 1 for the receipted ff mismatch.
The known XML limitation, sentinel alias, dirty xsa source provenance and
short-run peak sampling prevent a blanket pipeline correctness claim. No
changes to AGENTS.md or CLAUDE.md, no .pt files, and no resilient-training
behavior changes were involved. Intercom/contact_supervisor was not available
among the supplied tools; no supervisor coordination could be sent.
