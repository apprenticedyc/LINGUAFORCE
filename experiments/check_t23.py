import json, math
from collections import Counter
CLEAN='output/dims_test_clean.jsonl'; DIRECT='output/direct_test.jsonl'
def load(p):
    d={}
    with open(p,encoding='utf-8') as f:
        for l in f:
            if l.strip():
                r=json.loads(l); d[r['dialogue_id']]=r
    return d
clean=load(CLEAN); direct=load(DIRECT)
ids=[i for i in direct if i in clean]
gm=[clean[i]['gold_multi'] for i in ids]
ci=[float(clean[i]['intensity']) for i in ids]
di=[float(direct[i]['intensity']) for i in ids]

def spearman(xs,ys):
    n=len(xs)
    def rank(v):
        order=sorted(range(n),key=lambda i:v[i]);r=[0.0]*n;i=0
        while i<n:
            j=i
            while j<n and v[order[j]]==v[order[i]]:j+=1
            avg=(i+j-1)/2.0
            for k in range(i,j):r[order[k]]=avg
            i=j
        return r
    rx,ry=rank(xs),rank(ys);mx=sum(rx)/n;my=sum(ry)/n
    num=sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx=math.sqrt(sum((rx[i]-mx)**2 for i in range(n)));dy=math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0
def pearson(xs,ys):
    n=len(xs);mx=sum(xs)/n;my=sum(ys)/n
    num=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx=math.sqrt(sum((xs[i]-mx)**2 for i in range(n)));dy=math.sqrt(sum((ys[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0
def qwk_quad(y,p,ncls=6,round_pred=True):
    if round_pred: p=[round(x) for x in p]
    mat=[[0]*ncls for _ in range(ncls)]
    for a,b in zip(y,p): mat[a][b]+=1
    hr=[sum(mat[i][j] for j in range(ncls)) for i in range(ncls)]
    hc=[sum(mat[i][j] for i in range(ncls)) for j in range(ncls)]
    total=sum(hr)
    w=lambda i,j:(i-j)**2
    oe=sum(mat[i][j]*w(i,j) for i in range(ncls) for j in range(ncls))
    ee=sum((hr[i]*hc[j]/total)*w(i,j) for i in range(ncls) for j in range(ncls))
    return 1-oe/ee if ee else 1
def qwk_lin(y,p,ncls=6):
    p=[round(x) for x in p]
    mat=[[0]*ncls for _ in range(ncls)]
    for a,b in zip(y,p): mat[a][b]+=1
    hr=[sum(mat[i][j] for j in range(ncls)) for i in range(ncls)]
    hc=[sum(mat[i][j] for i in range(ncls)) for j in range(ncls)]
    total=sum(hr)
    oe=sum(mat[i][j]*abs(i-j) for i in range(ncls) for j in range(ncls))
    ee=sum((hr[i]*hc[j]/total)*abs(i-j) for i in range(ncls) for j in range(ncls))
    return 1-oe/ee if ee else 1

for name,intv in [('7DIM',ci),('DIRECT',di)]:
    print(f'--- {name} ---')
    print(f'  Spearman={spearman(intv,gm):.4f}')
    print(f'  Pearson={pearson(intv,gm):.4f}')
    print(f'  QWK(quad)={qwk_quad(gm,intv):.4f}  QWK(lin)={qwk_lin(gm,intv):.4f}')
    rp=[round(x) for x in intv]
    print(f'  AccExact={sum(1 for g,p in zip(gm,rp) if g==p)/len(gm):.4f} Acc+-1={sum(1 for g,p in zip(gm,rp) if abs(g-p)<=1)/len(gm):.4f}')
