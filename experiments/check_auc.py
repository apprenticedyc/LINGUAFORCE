import json
rows=[]
with open('output/dims_test_clean.jsonl',encoding='utf-8') as f:
    for l in f:
        if l.strip(): rows.append(json.loads(l))
y=[r['gold_binary'] for r in rows]
s=[int(r['intensity']) for r in rows]
pos=sorted([x for x,l in zip(s,y) if l==1]); neg=sorted([x for x,l in zip(s,y) if l==0])
auc=sum(sum(1 for t in neg if t<v)+0.5*sum(1 for t in neg if t==v) for v in pos)/(len(pos)*len(neg))
print('my AUC (integer, tie=0.5):', round(auc,4))
try:
    from sklearn.metrics import roc_auc_score
    print('sklearn AUC:', round(roc_auc_score(y,s),4))
except Exception as e:
    print('no sklearn:', e)
for r in rows[:3]:
    print(r['dialogue_id'], repr(r['intensity']))
