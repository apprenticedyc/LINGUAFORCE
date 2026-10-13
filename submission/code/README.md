# Benchmark Code

| Script | Purpose |
|---|---|
| `validate_release.py` | Verify schema, splits, hashes, prediction IDs |
| `train_human_baselines.py` | Train CPU text and constant baselines |
| `llm_clients.py` | API configuration and clients |
| `run_human_llm.py` | Resumable zero-shot inference |
| `score_human_llm.py` | Four-task scoring and aligned subsets |
| `bootstrap_human_pressure.py` | Paired pressure intervals |
| `analyze_human_robustness.py` | Negative-control FPR and paired intervals for other tasks |
| `write_robustness_tables.py` | Generate diagnostic and uncertainty tables |
| `test_human_robustness.py` | Verify resampling against sklearn and control selection |
| `write_human_result_tables.py` | Generate tables from metrics |
| `export_human_protocol.py` | Export prompt and protocol |
| `test_human_scoring.py` | Ordinal and invalid-output tests |
| `score_qwen4b.py` | Matched Qwen3-4B parsing, coverage, scores, and paired intervals |
| `test_qwen4b_scoring.py` | Final-answer parsing and ordinal-scale checks |

Paths are relative to the supplement, not the shell working directory.
See `../README.md` for commands. Saved predictions enable offline scoring.

## Optional API Configuration

For each provider, set `LINGUAFORCE_<PROVIDER>_BASE_URL`,
`LINGUAFORCE_<PROVIDER>_MODEL`, and `LINGUAFORCE_<PROVIDER>_API_KEY`.
Use uppercase GPT, QWEN, or DEEPSEEK. Include the endpoint's version or
compatibility prefix in the base URL. GPT uses Responses; the others use
chat completions. An optional `LINGUAFORCE_<PROVIDER>_API` overrides this.

Alternatively, `LINGUAFORCE_API_CONFIG` points to an external private JSON
file with provider objects containing `base_url`, `model`, `api_key`,
and `api`. Default: `~/.config/linguaforce/api_config.json`.
Environment values take precedence. No local application database is needed.

Recorded aliases are gpt-6-astra, qwen3.8-max, and deepseek-flash.
Access to the recorded GPT alias may require a compatible gateway.
Changing a model or endpoint creates a different configuration.
Never distribute credentials or use model responses as human labels.
