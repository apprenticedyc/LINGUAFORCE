# Public Release Checklist

## Included and checked

- [x] Paper source, PDF, bibliography, and referenced figures.
- [x] Training and held-out release JSONL files.
- [x] Type-label release files used by T2/RQ3.
- [x] Deterministic metric verification script.
- [x] Figure regeneration script.
- [x] Reported model outputs needed by the evaluation scripts.
- [x] No API keys or local provider configuration.

## Must be completed before public redistribution

- [ ] Confirm that the source-corpus license permits redistribution of the
      released text and derived labels.
- [ ] Add the final dataset license and any required attribution notice.
- [ ] Select one authoritative human-agreement annotation set and ensure the
      reported agreement table, workbooks or aggregate evidence, and scripts
      refer to that same set.
- [ ] Fill in author emails and the target venue's review metadata.
- [ ] Re-run the package from a clean checkout after the above decisions.
