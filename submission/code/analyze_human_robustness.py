"""Post-hoc negative-control errors and paired dialogue bootstrap; no API calls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_human_llm import STRATEGIES
from score_human_llm import decode, ordinal
from validate_release import main as validate_release

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['gpt', 'qwen', 'deepseek', 'tfidf_word', 'tfidf_char', 'tfidf_constant']
NAMES = ['GPT-6 Astra', 'Qwen3.8-Max', 'DeepSeek-Flash', 'Word TF--IDF',
         'Character TF--IDF', 'Constant']
DIMS = [f'D{i}' for i in range(1, 8)]
DRAW_COUNT, SEED = 1000, 2026


def control_groups(gold):
    negative = [i for i, r in gold.items() if r['binary'] == 0]
    request = [i for i in negative if gold[i]['strategies']['A1_Request_Advice'] == 1]
    reason = [i for i in negative if gold[i]['strategies']['A3_Rational_Persuasion'] == 1]
    return {'all_negative': sorted(negative), 'request_advice': sorted(request),
            'rational_persuasion': sorted(reason), 'request_or_reason': sorted(set(request) | set(reason))}


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, den = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0., float(center - half)), min(1., float(center + half))]


def sample_weights(n):
    rng = np.random.default_rng(SEED)
    return np.vstack([np.ones(n), *[
        np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(DRAW_COUNT)]])


def ordinal_metrics(y, p, upper, weights):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    total = weights.sum(axis=1)
    mae = weights @ np.abs(y - p).mean(axis=1) / total
    kappas = []
    squared = (np.arange(upper + 1)[:, None] - np.arange(upper + 1)[None, :]) ** 2
    for d in range(y.shape[1]):
        yh = weights @ (y[:, d, None] == np.arange(upper + 1)).astype(float)
        ph = weights @ (p[:, d, None] == np.arange(upper + 1)).astype(float)
        observed = weights @ ((y[:, d] - p[:, d]) ** 2) / total
        expected = np.einsum('bi,ij,bj->b', yh, squared, ph) / total ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            kappas.append(np.where(expected > 0, 1 - observed / expected, np.nan))
    return {'mae': mae, 'qwk': np.mean(kappas, axis=0)}


def strategy_metrics(y, p, weights):
    tp = weights @ ((y == 1) & (p == 1)).astype(float)
    fp = weights @ ((y == 0) & (p == 1)).astype(float)
    fn = weights @ ((y == 1) & (p == 0)).astype(float)
    den = 2 * tp + fp + fn
    per_label = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)
    micro_den = den.sum(axis=1)
    return {'macro_f1': per_label.mean(axis=1),
            'micro_f1': np.divide(2 * tp.sum(axis=1), micro_den,
                                  out=np.zeros_like(micro_den), where=micro_den != 0)}


def summarize(values):
    draws = values[1:]
    valid = draws[np.isfinite(draws)]
    return {'estimate': float(values[0]) if np.isfinite(values[0]) else None,
            'ci95': np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            'valid_draws': len(valid), 'undefined_draws': len(draws) - len(valid)}


def evaluate_group(ids, models, gold, predictions, task):
    weights = sample_weights(len(ids))
    estimates = {}
    for model in models:
        records = predictions[model]
        if task == 'intensity':
            estimates[model] = ordinal_metrics([gold[i]['intensity'] for i in ids],
                [records[i]['intensity'] for i in ids], 5, weights)
        elif task == 'strategies':
            y = np.array([[gold[i]['strategies'][s] for s in STRATEGIES] for i in ids])
            p = np.array([[records[i]['strategies'][s] for s in STRATEGIES] for i in ids])
            estimates[model] = strategy_metrics(y, p, weights)
        elif task == 'dimensions':
            y = [[gold[i]['dimensions'][d] for d in DIMS] for i in ids]
            p = [[records[i]['dimensions'][d] for d in DIMS] for i in ids]
            estimates[model] = ordinal_metrics(y, p, 3, weights)
        else:
            correct = [sum(ordinal(records[i].get('dimensions', {}).get(d), 3) and
                           records[i]['dimensions'][d] == gold[i]['dimensions'][d]
                           for d in DIMS) / 7 for i in ids]
            estimates[model] = {'accuracy': weights @ correct / weights.sum(axis=1)}
    paired = {}
    for reference in ('tfidf_word', 'tfidf_char'):
        paired[reference] = {
            m: {metric: summarize(v - estimates[reference][metric]) for metric, v in metrics.items()}
            for m, metrics in estimates.items() if m != reference}
    return {'n': len(ids), 'ids': ids,
            'models': {m: {k: summarize(v) for k, v in metrics.items()} for m, metrics in estimates.items()},
            'paired_differences': paired}


def main():
    validate_release()
    data_path = ROOT / 'data/human/annotations.jsonl'
    gold = {r['id']: r for r in map(json.loads, data_path.read_text().splitlines()) if r['source_split'] == 'test'}
    predictions, hashes = {}, {}
    for model in MODELS:
        path = (ROOT / f'results/human_supervised/{model[6:]}_test.jsonl' if model.startswith('tfidf_')
                else ROOT / f'results/human_llm/{model}_test.jsonl')
        predictions[model] = {r['id']: decode(r) for r in map(json.loads, path.read_text().splitlines())}
        hashes[model] = hashlib.sha256(path.read_bytes()).hexdigest()
    controls = {}
    for group, ids in control_groups(gold).items():
        models = {}
        for model, records in predictions.items():
            valid = [i for i in ids if ordinal(records[i].get('binary'), 1)]
            errors = [i for i in valid if records[i]['binary'] == 1]
            models[model] = {'valid_n': len(valid), 'invalid_n': len(ids) - len(valid),
                             'false_positive_n': len(errors), 'false_positive_ids': errors,
                             'fpr': len(errors) / len(valid) if valid else None,
                             'wilson_ci95': wilson(len(errors), len(valid))}
        controls[group] = {'n': len(ids), 'ids': ids, 'models': models}
    common = json.loads((ROOT / 'results/human_llm/common_subset_metrics.json').read_text())
    groups = {}
    for task in ('intensity', 'strategies', 'dimensions'):
        groups[task] = evaluate_group(common[task]['ids'], MODELS, gold, predictions, task)
    full_models = [m for m in MODELS if all(
        set(predictions[m][i].get('dimensions', {})) == set(DIMS) and
        all(ordinal(predictions[m][i]['dimensions'][d], 3) for d in DIMS) for i in gold)]
    groups['dimensions_full_valid'] = evaluate_group(sorted(gold), full_models, gold, predictions, 'dimensions')
    groups['dimension_exact_all'] = evaluate_group(sorted(gold), MODELS, gold, predictions, 'exact')
    result = {'analysis': 'Post-hoc analysis of fixed predictions; no model refitting or threshold selection.',
              'bootstrap': {'draws': DRAW_COUNT, 'seed': SEED, 'unit': 'dialogue',
                            'interval': 'percentile 95%; pointwise, no multiplicity adjustment',
                            'pairing': 'Identical dialogue samples for all models within each group',
                            'undefined_qwk': 'Exclude undefined draws, report their count; never substitute zero.',
                            'scope': 'Test-dialogue sampling variability conditional on this split and fixed predictions; not training or API-run variability.'},
              'annotation_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
              'prediction_sha256': hashes, 'negative_controls': controls, 'task_intervals': groups}
    out = ROOT / 'results/human_llm/robustness.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for model, values in controls['request_or_reason']['models'].items():
        print(model, 'control false positives:', values['false_positive_n'], '/', values['valid_n'])
    for group, r in groups.items():
        print(group, 'GPT minus word:', r['paired_differences']['tfidf_word'].get('gpt'))
    print(out)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
