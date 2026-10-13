"""Paired bootstrap AUROC/F1 intervals on the shared valid pressure subset."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
from score_human_llm import decode

ROOT=Path(__file__).resolve().parents[1]


def main():
    common=json.loads((ROOT/'results/human_llm/common_subset_metrics.json').read_text())
    ids=common['pressure']['ids']
    gold={r['id']:r for r in map(json.loads,(ROOT/'data/human/annotations.jsonl').read_text().splitlines())}
    y=np.array([gold[i]['binary'] for i in ids])
    arrays={}
    for name in ['gpt','qwen','deepseek','tfidf_word','tfidf_char','tfidf_constant']:
        path=(ROOT/f'results/human_supervised/{name[6:]}_test.jsonl' if name.startswith('tfidf_') else ROOT/f'results/human_llm/{name}_test.jsonl')
        rows={r['id']:decode(r) for r in map(json.loads,path.read_text().splitlines())}
        arrays[name]=(np.array([rows[i]['pressure_score'] for i in ids]),np.array([rows[i]['binary'] for i in ids]))
    rng=np.random.default_rng(2026);draws={name:[] for name in arrays}
    for _ in range(1000):
        ix=rng.integers(0,len(ids),len(ids))
        if len(np.unique(y[ix]))<2:continue
        for name,(prob,pred) in arrays.items():
            draws[name].append([roc_auc_score(y[ix],prob[ix]),f1_score(y[ix],pred[ix],zero_division=0)])
    result={'n':len(ids),'bootstrap_draws':1000,'seed':2026,'ci95':{},'paired_differences_vs_tfidf_word':{}}
    for name,values in draws.items():
        result['ci95'][name]={metric:np.quantile(np.array(values)[:,j],[.025,.975]).tolist() for j,metric in enumerate(['auroc','f1'])}
        if name!='tfidf_word':
            differences=np.array(values)-np.array(draws['tfidf_word'])
            result['paired_differences_vs_tfidf_word'][name]={metric:np.quantile(differences[:,j],[.025,.975]).tolist() for j,metric in enumerate(['auroc','f1'])}
    (ROOT/'results/human_llm/pressure_bootstrap.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
