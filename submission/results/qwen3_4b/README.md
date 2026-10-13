# Matched Qwen3-4B Diagnostic

`base_test.jsonl` and `sft_test.jsonl` contain the original answer text for
all 634 test dialogues, matched to released IDs by exact dialogue text.
Only deployment/operator/trace metadata has been removed. Responses and
human labels have not been repaired. The SFT model is the checkpoint from
the recorded LoRA run, verified through the model repository's training link.

## Protocol

Both models received `prompt.txt`, the ordered dialogue array, and the suffix
`\n/no_think`. Requests used seed 1024 and `stream=false`. Neither export
contained nonempty thinking output. Other decoding parameters were not
explicitly set in the requests; backend defaults are not recorded. This
prompt does not request the confidence score used by the API baselines, so
AUROC is not reported. The two experiments should not be described as sharing
one universal prompt or output-token budget.

The managed LoRA run used 3,088 training dialogues, rank 16, alpha 32,
dropout 0.05, learning rate 0.0001, three epochs, batch size 1, accumulation
8, and sequence length 2,048. `protocol.json` records available settings and
hashes. It does not provide a complete backend training configuration.
Adapter weights and the service implementation are not included.

## Scoring

Run `python code/score_qwen4b.py` from the supplement directory. No service
access or GPU is needed. It verifies hashes, dialogue IDs, and split
membership before parsing only the final JSON answer. Optional JSON fences
and the documented strategy-key alias are normalized. Duplicate keys,
trailing prose, malformed JSON, missing labels, and out-of-range ratings
are not silently repaired.

Each task uses its own validity rule. Both models have 634 valid binary and
intensity predictions, 624 valid strategy sets, and 624 complete dimension
profiles. Common valid subsets contain 634, 634, 623, and 622 dialogues,
respectively. `metrics.json` includes the IDs and invalid-output coverage.

The paper's seven metrics use 1,000 paired dialogue bootstrap draws with
seed 2026. A single random generator advances through pressure, intensity,
strategies, then dimensions, with IDs sorted within each subset. Undefined
QWK draws are excluded and counted, never set to zero. Intervals are
pointwise, without multiple-comparison correction, and describe uncertainty
over the observed test sample for fixed predictions, not training-seed
variation. Scores show little change overall; the small dimension-QWK gain
does not establish broad improvement from fine-tuning.
