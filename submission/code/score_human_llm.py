"""Score each task separately, reporting output coverage without repairing values."""
import json
from pathlib import Path
import warnings
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, mean_absolute_error,
                             cohen_kappa_score, precision_recall_fscore_support)
from run_human_llm import STRATEGIES, PROMPT
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def ordinal(value, upper):
    return type(value) is int and 0 <= value <= upper


def decode(record):
    import re
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", record.get("text", "").strip())
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            # A complete last field with one missing outer brace is unambiguous.
            if text.startswith('{') and text.endswith('}') and exc.pos==len(text) and exc.msg=="Expecting ',' delimiter":
                value=json.loads(text+'}')
            else:
                return {}
        # Normalize one unambiguous strategy-key spelling; never alter labels.
        if isinstance(value,dict) and isinstance(value.get('strategies'),dict):
            labels=value['strategies']
            alias='D3_Logical_False_Dilemma'
            canonical='D3_Logical_Trap_False_Dilemma'
            if alias in labels and canonical not in labels:
                labels[canonical]=labels.pop(alias)
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def qwk(y, p, upper):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cohen_kappa_score(y, p, labels=list(range(upper+1)), weights="quadratic")
    return float(result) if np.isfinite(result) else None


def score(records, gold):
    predictions = {r['id']:decode(r) for r in records}
    result = {"expected_n":len(gold), "returned_n":len(records)}
    def pairs(check):
        return [(g,predictions.get(i,{})) for i,g in gold.items() if check(predictions.get(i,{}))]
    binary = pairs(lambda p:ordinal(p.get('binary'),1))
    y=[g['binary'] for g,p in binary]; pred=[p['binary'] for g,p in binary]
    aucpairs=pairs(lambda p:type(p.get('pressure_score')) in (int,float) and 0<=p['pressure_score']<=1)
    result['pressure']={'n':len(y),'f1':float(f1_score(y,pred,zero_division=0)) if y else None,
                        'auroc_n':len(aucpairs),'auroc':float(roc_auc_score([g['binary'] for g,p in aucpairs],[p['pressure_score'] for g,p in aucpairs])) if aucpairs else None}
    intensity=pairs(lambda p:ordinal(p.get('intensity'),5))
    y=[g['intensity'] for g,p in intensity];pred=[p['intensity'] for g,p in intensity]
    result['intensity']={'n':len(y),'mae':float(mean_absolute_error(y,pred)) if y else None,'qwk':qwk(y,pred,5) if y else None}
    strategy=pairs(lambda p:isinstance(p.get('strategies'),dict) and set(p['strategies'])==set(STRATEGIES) and all(ordinal(p['strategies'][s],1) for s in STRATEGIES))
    y=[[g['strategies'][s] for s in STRATEGIES] for g,p in strategy];pred=[[p['strategies'][s] for s in STRATEGIES] for g,p in strategy]
    result['strategies']={'n':len(y)}
    if y:
        result['strategies'].update(macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),micro_f1=float(f1_score(y,pred,average='micro',zero_division=0)))
        pr,rc,f1,support=precision_recall_fscore_support(y,pred,zero_division=0)
        result['strategy_details']={s:{'precision':float(pr[i]),'recall':float(rc[i]),'f1':float(f1[i]),'support':int(support[i])} for i,s in enumerate(STRATEGIES)}
    dimensions={}
    for d in [f'D{i}' for i in range(1,8)]:
        valid=pairs(lambda p:isinstance(p.get('dimensions'),dict) and ordinal(p['dimensions'].get(d),3))
        y=[g['dimensions'][d] for g,p in valid]; pred=[p['dimensions'][d] for g,p in valid]
        dimensions[d]={'n':len(y),'mae':float(mean_absolute_error(y,pred)) if y else None,'qwk':qwk(y,pred,3) if y else None}
    result['dimensions']=dimensions
    complete=pairs(lambda p:isinstance(p.get('dimensions'),dict) and set(p['dimensions'])=={f'D{i}' for i in range(1,8)} and all(ordinal(v,3) for v in p['dimensions'].values()))
    result['dimension_profile_n']=len(complete)
    # Range-valid accuracy on the full denominator counts malformed ratings as failures.
    result['dimension_exact_accuracy_all']=sum(
        isinstance(predictions.get(i,{}).get('dimensions'),dict)
        and ordinal(predictions[i]['dimensions'].get(d),3)
        and predictions[i]['dimensions'][d]==g['dimensions'][d]
        for i,g in gold.items() for d in dimensions)/(len(gold)*7)
    if complete:
        result['dimension_profile_mae']=float(np.mean([abs(g['dimensions'][d]-p['dimensions'][d]) for g,p in complete for d in dimensions]))
        ks=[qwk([g['dimensions'][d] for g,p in complete],[p['dimensions'][d] for g,p in complete],3) for d in dimensions]
        result['dimension_profile_qwk']=float(np.mean(ks)) if all(k is not None for k in ks) else None
    result['usage']={k:sum(r.get('usage',{}).get(k,0) for r in records) for k in ['total_tokens','prompt_tokens','completion_tokens','input_tokens','output_tokens']}
    return result


def main():
    gold={r['id']:r for r in map(json.loads,(ROOT/'data/human/annotations.jsonl').read_text().splitlines()) if r['source_split']=='test'}
    all_results={}; all_records={}
    for provider in ['gpt','qwen','deepseek']:
        path=ROOT/f'results/human_llm/{provider}_test.jsonl'
        records=[json.loads(l) for l in path.read_text().splitlines()]
        if len({r['id'] for r in records}) != len(records): raise ValueError('Duplicate predictions')
        if {r['id'] for r in records} != set(gold): raise ValueError(f'Incomplete test run: {provider}, {len(records)}/{len(gold)}')
        for r in records:
            if r['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest():raise ValueError('Prompt changed')
            if 'text' in r:
                expected=hashlib.sha256(json.dumps(gold[r['id']]['utterances'],ensure_ascii=False).encode()).hexdigest()
                if r['dialogue_sha256']!=expected:raise ValueError('Dialogue mismatch')
        all_results[provider]=score(records,gold)
        all_records[provider]=records
    supervised=ROOT/'results/human_supervised/metrics.json'
    if supervised.exists():
        saved=json.loads(supervised.read_text())
        for name in ['word','char','constant']:
            records=list(map(json.loads,(ROOT/f'results/human_supervised/{name}_test.jsonl').read_text().splitlines()))
            if len(records) != len(gold) or {r['id'] for r in records} != set(gold):
                raise ValueError(f'Incomplete or duplicate supervised predictions: {name}')
            all_records[f'tfidf_{name}']=records
            all_results[f'tfidf_{name}']={**score(records,gold),'settings':saved[name].get('settings',{})}
    checks={
        'pressure':lambda p:ordinal(p.get('binary'),1) and type(p.get('pressure_score')) in (int,float) and 0<=p['pressure_score']<=1,
        'intensity':lambda p:ordinal(p.get('intensity'),5),
        'strategies':lambda p:isinstance(p.get('strategies'),dict) and set(p['strategies'])==set(STRATEGIES) and all(ordinal(v,1) for v in p['strategies'].values()),
        'dimensions':lambda p:isinstance(p.get('dimensions'),dict) and set(p['dimensions'])=={f'D{i}' for i in range(1,8)} and all(ordinal(v,3) for v in p['dimensions'].values())}
    common={}
    for task,check in checks.items():
        ids=set(gold)
        for records in all_records.values():ids&={r['id'] for r in records if check(decode(r))}
        subset={i:gold[i] for i in sorted(ids)}
        scores={m:score([r for r in records if r['id'] in ids],subset) for m,records in all_records.items()}
        common[task]={'n':len(ids),'ids':sorted(ids),'models':{m:({'mae':s.get('dimension_profile_mae'),'qwk':s.get('dimension_profile_qwk')} if task=='dimensions' else s[task]) for m,s in scores.items()}}
    (ROOT/'results/human_llm/common_subset_metrics.json').write_text(json.dumps(common,indent=2,allow_nan=False)+'\n')
    output=ROOT/'results/human_llm/metrics.json'
    output.write_text(json.dumps(all_results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:{k:v for k,v in r.items() if k not in ['strategy_details','dimensions']} for m,r in all_results.items()},indent=2))


if __name__=='__main__':main()
