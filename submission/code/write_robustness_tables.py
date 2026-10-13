"""Generate paper-ready diagnostic and interval tables from saved analysis."""
import json
from pathlib import Path
from analyze_human_robustness import MODELS, NAMES

ROOT = Path(__file__).resolve().parents[1]


def interval(value):
    ci = value['ci95']
    return 'undefined' if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def table(label, columns, header, rows, caption, wide=False):
    env = 'table*' if wide else 'table'
    return '\n'.join([
        f'\\begin{{{env}}}[t]', '\\centering\\small', '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{@{}' + columns + '@{}}', '\\toprule', header + r'\\',
        '\\midrule', *[' & '.join(row) + r'\\' for row in rows],
        '\\bottomrule', '\\end{tabular}', '\\caption{' + caption + '}',
        '\\label{' + label + '}', f'\\end{{{env}}}', ''])


def main():
    data = json.loads((ROOT / 'results/human_llm/robustness.json').read_text())
    controls = data['negative_controls']['request_or_reason']
    rows = []
    for model, name in zip(MODELS, NAMES):
        r = controls['models'][model]
        lo, hi = r['wilson_ci95']
        rows.append([name, str(r['false_positive_n']), f"{100*r['fpr']:.1f}",
                     f'[{100*lo:.1f}, {100*hi:.1f}]'])
    text = table('tab:request_controls', 'lrrr', r'Model & FP & FPR (\%) & 95\% CI', rows,
        'False positives on 126 test dialogues labeled no pressure and request/advice or rational persuasion. '
        'Each dialogue is counted once; all systems return valid binary decisions. Intervals are Wilson intervals.')
    groups = data['task_intervals']
    rows = []
    for model, name in zip(MODELS, NAMES):
        i, s = groups['intensity']['models'][model], groups['strategies']['models'][model]
        rows.append([name, interval(i['mae']), interval(i['qwk']),
                     interval(s['macro_f1']), interval(s['micro_f1'])])
    text += table('tab:ordinal_strategy_ci', 'lrrrr',
        r'Model & Intensity MAE & Intensity QWK & Strategy macro-F1 & Strategy micro-F1', rows,
        'Pointwise 95\\% bootstrap intervals on all 634 test dialogues, with 1,000 paired dialogue resamples. '
        'All draws are defined. Point estimates appear in Tables~\\ref{tab:intensity_results} and~\\ref{tab:strategy_results}.', True)
    rows = []
    for model, name in zip(MODELS, NAMES):
        d = groups['dimensions']['models'][model]
        a = groups['dimension_exact_all']['models'][model]
        rows.append([name, interval(d['mae']), interval(d['qwk']), str(d['qwk']['valid_draws']), interval(a['accuracy'])])
    text += table('tab:dimension_ci', 'lrrrr',
        r'Model & MAE (295) & QWK (295) & Defined QWK draws & Exact accuracy (634)', rows,
        'Dimension 95\\% intervals. MAE/QWK use the same 295 complete-profile dialogues; exact rating accuracy '
        'uses all 634 dialogues, counting invalid outputs as wrong. QWK intervals exclude draws in which '
        'any dimension has undefined QWK; the resulting conditional intervals require caution. MAE and accuracy use all 1,000 draws.', True)
    full = groups['dimensions_full_valid']['models']
    rows = [[name, interval(full[m]['mae']), interval(full[m]['qwk']), str(full[m]['qwk']['valid_draws'])]
            for m, name in zip(MODELS, NAMES) if m in full]
    text += table('tab:dimension_full_ci', 'lrrr',
        r'Model & MAE & QWK & Defined QWK draws', rows,
        'Dimension 95\\% intervals for methods with complete valid profiles on all 634 dialogues. '
        'Qwen is excluded from this full-profile comparison. Undefined QWK draws are excluded and counted; '
        'MAE uses all 1,000 draws.', True)
    (ROOT / 'results/human_llm/robustness_tables.tex').write_text(text)
    print(ROOT / 'results/human_llm/robustness_tables.tex')


if __name__ == '__main__':
    main()
