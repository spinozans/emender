X43 C14 union extraction runs as the lingering user service
`emender-velocity-c14-union.service`. The service uses the repository venv,
128 CPU query threads, and an empty CUDA device list. It acquires no GPU lease.

Install the checked unit into `~/.config/systemd/user/`, run
`systemctl --user daemon-reload`, `loginctl enable-linger erikg`, then
`systemctl --user enable --now emender-velocity-c14-union.service`.
Check `loginctl show-user erikg -p Linger` and
`systemctl --user status emender-velocity-c14-union.service`.

Persistent state is under `/home/erikg/c14_consultation_features`:

- `union_service.jsonl`: service starts, failures, phase boundaries and exact
  boundary journal hashes/bytes.
- `union_console.jsonl`: completed windows/positions and elapsed phase seconds.
- `registered_{emender,eval,transformer}/journal.jsonl`: shard identities, hashes,
  bytes, full sparse round-trip parity, and exhaustive suffix sample checks.
- `union_stderr.txt`: process errors. A clean service exit means extraction
  complete only if `union_complete` exists in the service journal.

Phases are Emender prefix, frozen paired evaluation, then transformer suffix.
The transformer training cache is the Emender prefix followed by the transformer
suffix; do not concatenate evaluation into training. Transformer suffix window
indices are local to that suffix. Reusing the exact nested prefix avoids double
extraction: 225,880,702 + 434,204,084 + 746,495,031 = 1,406,579,817 positions.
At 5,884.18/s, remaining elapsed extraction is about 66 hours (the earlier ~64h
estimate used 6,088.69/s). The full transformer stream represents 45.9 hours,
of which 10.7 hours is shared with Emender. Times are projections, not deadlines.

Use `systemctl --user stop ...` to pause and `start ...` to resume. A kernel kill
restarts automatically, at most three starts/hour. Ordinary exceptions, including
the 150,000,000,000-byte disk floor, leave the unit failed for inspection; no
automatic exception retry. Every pending shard is guarded before allocation.
All committed shards are hash-checked on restart. Torn final journal bytes and
intent-bound uncommitted shards are retained as `.partial` / `.crash_*` files;
their windows are recomputed. Unexpected orphans and committed corruption fail
closed. No cleanup deletes evidence. Restart checks the entire native source
index against the frozen receipt, so startup includes a ~42 GB read.

At each phase boundary the service receipts completion and proceeds to the next
extraction phase. It does not launch training or acquire cards. In an attended
session, verify the Emender boundary, then use the registered consultation
protocol and broker cards 2/4. Final paired evaluation must wait for its complete
cache. Dense receipts/windows are frozen. Consultation models, paired table,
P1/P2/P3 and C13 remain pending until separately validated and committed.
