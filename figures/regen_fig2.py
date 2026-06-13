"""fig2 재생성 — 제목 제거 + A/B/C 패널 레이블 (다른 figure와 형식 통일)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

rnd=json.load(open('random_split_results.json')); scf=json.load(open('scaffold_split_results.json'))
BROAD=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']
MED=['TAS2R43','TAS2R38','TAS2R5','TAS2R16','TAS2R7','TAS2R47','TAS2R40','TAS2R8','TAS2R44','TAS2R49','TAS2R41']
ORPHAN=['TAS2R2','TAS2R13','TAS2R50','TAS2R9','TAS2R3','TAS2R42']
def tier(r): return 'BROAD' if r in BROAD else ('MED' if r in MED else 'ORPHAN')
tcol={'BROAD':'#C0392B','MED':'#2980B9','ORPHAN':'#7F8C8D'}

def PL(ax,lab): ax.text(-0.13,1.05,lab,transform=ax.transAxes,fontsize=15,fontweight='bold',va='top',ha='right')

# 공통 수용체 (둘 다 있는)
recs=[r for r in (BROAD+MED+ORPHAN) if r in rnd and r in scf and rnd[r].get('auc') is not None and scf[r].get('auc') is not None]

fig,ax=plt.subplots(1,3,figsize=(17,5.4))
# A: random vs scaffold scatter
for r in recs:
    ax[0].scatter(rnd[r]['auc'],scf[r]['auc'],c=tcol[tier(r)],s=55,edgecolor='black',lw=0.4,zorder=3)
    ax[0].annotate(r.replace('TAS2R','T'),(rnd[r]['auc'],scf[r]['auc']),textcoords="offset points",xytext=(4,3),fontsize=7)
ax[0].plot([0.35,1.02],[0.35,1.02],'--',color='gray',alpha=0.5)
ax[0].axhline(0.5,ls=':',color='gray',alpha=0.4); ax[0].axvline(0.5,ls=':',color='gray',alpha=0.4)
ax[0].set_xlabel('AUC — Random Split'); ax[0].set_ylabel('AUC — Scaffold Split')
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=tcol[t],label=t) for t in ['BROAD','MED','ORPHAN']],fontsize=8,loc='lower right')
PL(ax[0],'A')
# B: AUC drop per subtype
drops=sorted([(r,rnd[r]['auc']-scf[r]['auc']) for r in recs],key=lambda x:-x[1])
names=[d[0].replace('TAS2R','T') for d in drops]; vals=[d[1] for d in drops]
ax[1].bar(range(len(names)),vals,color=['#E74C3C' if v>0 else '#27AE60' for v in vals])
ax[1].axhline(0.1,ls='--',color='salmon',alpha=0.7); ax[1].axhline(0,color='black',lw=0.8)
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names,rotation=90,fontsize=7)
ax[1].set_xlabel('TAS2R subtype'); ax[1].set_ylabel('AUC drop (random $-$ scaffold)'); PL(ax[1],'B')
# C: tier mean bar
def tmean(tlist,src): 
    v=[src[r]['auc'] for r in tlist if r in src and src[r].get('auc') is not None]
    return np.mean(v) if v else 0
tiers=['BROAD\n($\\geq$50)','MED\n(10–49)','ORPHAN\n($<$10)']
rmeans=[tmean(BROAD,rnd),tmean(MED,rnd),tmean(ORPHAN,rnd)]
smeans=[tmean(BROAD,scf),tmean(MED,scf),tmean(ORPHAN,scf)]
x=np.arange(3); w=0.36
b1=ax[2].bar(x-w/2,rmeans,w,label='Random split',color=['#C0392B','#2980B9','#7F8C8D'],edgecolor='black')
b2=ax[2].bar(x+w/2,smeans,w,label='Scaffold split',color=['#C0392B','#2980B9','#7F8C8D'],edgecolor='black',hatch='///',alpha=0.6)
for bars,ms in [(b1,rmeans),(b2,smeans)]:
    for bar,m in zip(bars,ms): ax[2].text(bar.get_x()+bar.get_width()/2,m+0.005,f'{m:.3f}',ha='center',fontsize=8,weight='bold')
ax[2].set_xticks(x); ax[2].set_xticklabels(tiers,fontsize=9); ax[2].set_ylabel('Mean AUC-ROC'); ax[2].set_ylim(0.4,1.0); ax[2].legend(fontsize=8); PL(ax[2],'C')
plt.tight_layout(); plt.savefig('fig2_baseline.png',dpi=200,bbox_inches='tight'); plt.close()
print("fig2 재생성 완료 (제목 제거, A/B/C 레이블)")
