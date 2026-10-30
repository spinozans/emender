# X42 length sweep validation

Ten focused CPU tests passed using /home/erikg/emender/.venv/bin/python.
The anchor update loop still matches the registered pure loop, apart from the
pure-only detached gate objective. The registered score and training budget
remain unchanged. Additional eval runs after result.json is persisted.
Two reset contiguous windows per length reproduce X40 context_probe windowing;
this is variable token coverage, not equal-token evaluation across lengths.
The fresh pure arm uses the exact archived c06 policy, including gate cost.

Discovery: the old anchors had already failed registration because their default
registration path was absolute and git show HEAD:/absolute/path cannot resolve.
Their failure receipts are preserved as *_prior_registration_failure.json.
The default is now repository-relative; registered sources are unchanged.
The depth unit was already failed when inspected; it was not modified or restarted.
Only the existing anchor service is restarted, using the card-0-only lease broker.

The length table explicitly remains incomplete until three successful sweeps.
No TinyStories or 1.3B result is claimed. BF16 baseline versus FP32 pure precision
follows the existing anchor protocol. GPU feasibility remains unproven; failures
are receipted without reducing context, batch, or training time.
Resilient DiLoCo requirement IDs: not applicable (no resilient behavior changed).
Intercom contact_supervisor is unavailable in this session's tool catalog.
