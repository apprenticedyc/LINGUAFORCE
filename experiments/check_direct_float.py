import json, math
from collections import Counter
DIRECT='output/direct_test.jsonl'; CLEAN='output/dims_test_clean.jsonl'
rows={}
with open(DIRECT,encoding='utf-8') as f:
    for l in f:
        if l.strip():
            r=json.loads(l); rows[r['dialogue_id']]=r
def spearman(xs,ys):
    n=len(xs)
    def rank(v):
        order=sorted(range(n),key=lambda i:v[i]); r=[0.0]*n;i=0
        while i<n:
            j=i
            while j<n and v[order[j]]==v[order[i]]: j+=1
            avg=(i+j-1)/2.0
            for k in range(i,j): r[order[k]]=avg
            i=j
        return r
    rx,ry=rank(xs),rank(ys); mx=sum(rx)/n;my=sum(ry)/n
    num=sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx=math.sqrt(sum((rx[i]-mx)**2 for i in range(n))); dy=math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0
def auc(s,y):
    pos=sorted([x for x,l in zip(s,y) if l==1]);neg=sorted([x for x,l in zip(s,y) if l==0])
    return sum(sum(1 for t in neg if t<v)+0.5*sum(1 for t in neg if t==v) for v in pos)/(len(pos)*len(neg))
ids=list(rows.keys()); gb=[rows[i]['gold_binary'] for i in ids]; gm=[rows[i]['gold_multi'] for i in ids]
pi=[float(rows[i]['intensity']) for i in ids]
print('direct n=',len(ids),'AUC(float)=',round(auc(pi,gb),4))
print('float ints?', Counter(float(rows[i]['intensity']).is_integer() for i in ids))
# direct best-F1 over float thresholds
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
best=(0,0,0)
for t in [x/10 for x in range(10,60,5)]:
    pb=[1 if x>=t else 0 for x in pi]
    tp=sum(1 for g,q in zip(gb,pb) if g==1 and q==1);fp=sum(1 for g,q in zip(gb,pb) if g==0 and q==1);fn=sum(1 for g,q in zip(gb,pb) if g==1 and q==0)
    f=prf(tp,fp,fn); acc=(tp+len(pb)-fp-fn-tp)/len(pb)
    if f>best[1]: best=(t,f,acc)
print('direct best thr=%s F1=%.3f Acc=%.3f'%best)
