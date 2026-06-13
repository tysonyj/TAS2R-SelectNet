"""fig1 재생성: 통일 PL 레이블 + 파스텔."""
import numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P_RED='#E8A7A0'; P_BLUE='#A8CCE5'; P_GRAY='#C9C9C9'
def PL(ax,l):
    ax.annotate(l, xy=(0,1), xycoords='axes fraction', xytext=(-30,12),
                textcoords='offset points', fontsize=15, fontweight='bold',
                va='bottom', ha='left', annotation_clip=False)

d=np.load('/home/claude/tas2r_work/extracted/TAS2R_v2_pipeline/dataset.npz',allow_pickle=True)
Y=d['Y']; R=list(d['receptors'])
counts={r:int(Y[:,R.index(r)].sum()) for r in R}
srt=sorted(counts.items(),key=lambda x:-x[1])
names=[k for k,v in srt]; vals=[v for k,v in srt]

def tier(v): return P_RED if v>=50 else (P_BLUE if v>=10 else P_GRAY)
cols=[tier(v) for v in vals]

fig,ax=plt.subplots(1,2,figsize=(15,6))
# A: 수평 막대
y=np.arange(len(names))
ax[0].barh(y,vals,color=cols,edgecolor='#555',lw=0.4)
ax[0].set_yticks(y); ax[0].set_yticklabels(names,fontsize=8)
ax[0].invert_yaxis()
for i,v in enumerate(vals): ax[0].text(v+4,i,str(v),va='center',fontsize=7)
ax[0].axvline(10,ls='--',color='gray',alpha=0.5); ax[0].axvline(50,ls='--',color=P_RED,alpha=0.7)
ax[0].set_xlabel('Number of unique ligands')
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=P_RED,label='Broadly tuned ($\\geq$50, n=6)'),
                      Patch(color=P_BLUE,label='Moderately tuned (10–49, n=11)'),
                      Patch(color=P_GRAY,label='Near-orphan (<10, n=6)')],
             fontsize=8,loc='lower right')
PL(ax[0],'A')
# B: 누적 곡선
cum=np.cumsum(vals)/sum(vals)*100
x=np.arange(1,len(vals)+1)
ax[1].plot(x,cum,'o-',color=P_RED,ms=5,lw=1.5)
ax[1].fill_between(x,cum,alpha=0.15,color=P_RED)
ax[1].axhline(cum[2],ls=':',color=P_BLUE,alpha=0.8); ax[1].text(13,cum[2]+1,f'Top 3 = {cum[2]:.0f}%',color=P_BLUE,fontsize=10)
ax[1].axhline(cum[0],ls=':',color=P_RED,alpha=0.8); ax[1].text(13,cum[0]+1,f'TAS2R14 = {cum[0]:.0f}%',color=P_RED,fontsize=10)
ax[1].set_xlabel('TAS2R subtypes (ranked by ligand count)'); ax[1].set_ylabel('Cumulative % of associations')
PL(ax[1],'B')
plt.tight_layout(); plt.savefig('fig1_distribution.png',dpi=200,bbox_inches='tight'); plt.close()
print(f"fig1 재생성: top1={cum[0]:.1f}%, top3={cum[2]:.1f}%")
