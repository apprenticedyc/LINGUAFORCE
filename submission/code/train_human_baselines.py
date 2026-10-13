"""CPU text baselines with a fixed train/validation split and held-out scoring."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error
from threadpoolctl import threadpool_limits
from score_human_llm import score
from run_human_llm import STRATEGIES

ROOT=Path(__file__).resolve().parents[1]


def fit_binary(x,y,z,c):
    if len(set(y))==1:return np.full(z.shape[0],y[0],dtype=float)
    return LogisticRegression(C=c,max_iter=2000,random_state=2026).fit(x,y).predict_proba(z)[:,1]


def main():
    rows=list(map(json.loads,(ROOT/'data/human/annotations.jsonl').read_text().splitlines()))
    pool=sorted([r for r in rows if r['source_split']=='training'],key=lambda r:r['id'])
    test=sorted([r for r in rows if r['source_split']=='test'],key=lambda r:r['id'])
    tr,dev=train_test_split(pool,test_size=0.1,random_state=2026,stratify=[r['intensity'] for r in pool])
    splits={k:[r['id'] for r in group] for k,group in [('train',tr),('validation',dev),('test',test)]}
    (ROOT/'data/human/splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    out=ROOT/'results/human_supervised';out.mkdir(exist_ok=True)
    gold={r['id']:r for r in test};summaries={}
    pressure=float(np.mean([r['binary'] for r in tr]))
    constant={'binary':int(pressure>=.5),'pressure_score':pressure,
              'intensity':int(np.median([r['intensity'] for r in tr])),
              'dimensions':{f'D{i}':int(np.median([r['dimensions'][f'D{i}'] for r in tr])) for i in range(1,8)},
              'strategies':{s:int(np.mean([r['strategies'][s] for r in tr])>=.5) for s in STRATEGIES}}
    records=[{'id':r['id'],'text':json.dumps(constant),'prediction':constant,'valid':True} for r in test]
    (out/'constant_test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summaries['constant']=score(records,gold)
    for analyzer,ngram in [('word',(1,2)),('char',(3,5))]:
        vec=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,min_df=2,max_features=50000,sublinear_tf=True)
        text=lambda rs:['\n'.join(r['utterances']) for r in rs]
        x=vec.fit_transform(text(tr));v=vec.transform(text(dev));z=vec.transform(text(test))
        y=np.array([r['binary'] for r in tr]);yv=np.array([r['binary'] for r in dev])
        c=max([0.1,1,10],key=lambda c:roc_auc_score(yv,fit_binary(x,y,v,c)))
        pv=fit_binary(x,y,v,c)
        threshold=max(np.linspace(.1,.9,81),key=lambda t:(f1_score(yv,pv>=t),-abs(t-.5)))
        prob=fit_binary(x,y,z,c)
        sy=np.array([[r['strategies'][s] for s in STRATEGIES] for r in tr]);sv=np.array([[r['strategies'][s] for s in STRATEGIES] for r in dev])
        strategy_candidates={cc:np.column_stack([fit_binary(x,sy[:,i],v,cc) for i in range(15)]) for cc in [.1,1,10]}
        sc,st=max([(cc,t) for cc in strategy_candidates for t in [.3,.4,.5,.6]],key=lambda ct:f1_score(sv,strategy_candidates[ct[0]]>=ct[1],average='macro',zero_division=0))
        strategy=np.column_stack([fit_binary(x,sy[:,i],z,sc) for i in range(15)])>=st
        ordinal_predictions=[];alphas=[]
        for name in ['intensity']+[f'D{i}' for i in range(1,8)]:
            target=lambda r:r['intensity'] if name=='intensity' else r['dimensions'][name]
            upper=5 if name=='intensity' else 3
            yy=[target(r) for r in tr];vv=[target(r) for r in dev]
            models={a:Ridge(alpha=a,solver='lsqr').fit(x,yy) for a in [.1,1,10]}
            rounded=lambda m,mat:np.rint(np.clip(m.predict(mat),0,upper)).astype(int)
            a=min(models,key=lambda a:mean_absolute_error(vv,rounded(models[a],v)))
            ordinal_predictions.append(rounded(models[a],z));alphas.append(a)
        records=[]
        for i,r in enumerate(test):
            p={'binary':int(prob[i]>=threshold),'pressure_score':float(prob[i]),'intensity':int(ordinal_predictions[0][i]),
               'dimensions':{f'D{d}':int(ordinal_predictions[d][i]) for d in range(1,8)},
               'strategies':{s:int(strategy[i,j]) for j,s in enumerate(STRATEGIES)}}
            records.append({'id':r['id'],'text':json.dumps(p),'prediction':p,'valid':True})
        (out/f'{analyzer}_test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        summaries[analyzer]={'settings':{'seed':2026,'ngram':ngram,'pressure_C':c,'pressure_threshold':float(threshold),'strategy_C':sc,'strategy_threshold':st,'ridge_alphas':alphas},**score(records,gold)}
        print(analyzer,{k:v for k,v in summaries[analyzer].items() if k in ['pressure','intensity','strategies','dimension_profile_mae','dimension_profile_qwk']},flush=True)
    (out/'metrics.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
