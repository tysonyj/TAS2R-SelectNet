"""그룹1 figure 파스텔 재생성: fig2, fig3, fig6, figS4, figR1."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
try:
    from adjustText import adjust_text; HAS_ADJ=True
except: HAS_ADJ=False

# 파스텔 팔레트
P_RED='#E8A7A0'; P_SALMON='#F0A8A0'; P_BLUE='#A8CCE5'; P_SKY='#8FB8DC'
P_GRAY='#D9D9D9'; P_DGRAY='#BFC9CA'; P_MINT='#A8D5BA'; P_PEACH='#F2C49B'
P_LAV='#C9A8D8'; P_ORANGE='#F2C49B'

gt=json.load(open('ground_truth.json'))
sn=gt['per_subtype_selectnet']; scf=gt['per_subtype_onehot_scaffold']; rnd=gt['per_subtype_onehot_random']
ab=json.load(open('ablation_multiseed.json'))
BROAD=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']
MED=['TAS2R43','TAS2R38','TAS2R5','TAS2R16','TAS2R7','TAS2R47','TAS2R40','TAS2R8','TAS2R44','TAS2R49','TAS2R41']
ORPHAN=['TAS2R2','TAS2R13','TAS2R50','TAS2R9','TAS2R3','TAS2R42']
def PL(ax,l):
    ax.annotate(l, xy=(0,1), xycoords='axes fraction', xytext=(-30,12),
                textcoords='offset points', fontsize=15, fontweight='bold',
                va='bottom', ha='left', annotation_clip=False)

# ════ fig2 baseline ════
rnd2=json.load(open('random_split_results.json')); scf2=json.load(open('scaffold_split_results.json'))
def tier(r): return 'BROAD' if r in BROAD else ('MED' if r in MED else 'ORPHAN')
tcol={'BROAD':P_RED,'MED':P_SKY,'ORPHAN':P_DGRAY}
recs=[r for r in BROAD+MED+ORPHAN if r in rnd2 and r in scf2 and rnd2[r].get('auc') is not None and scf2[r].get('auc') is not None]
fig,ax=plt.subplots(1,3,figsize=(17,5.4))
for r in recs:
    ax[0].scatter(rnd2[r]['auc'],scf2[r]['auc'],c=tcol[tier(r)],s=55,edgecolor='#555',lw=0.4,zorder=3)
    ax[0].annotate(r.replace('TAS2R','T'),(rnd2[r]['auc'],scf2[r]['auc']),textcoords="offset points",xytext=(4,3),fontsize=7)
ax[0].plot([0.35,1.02],[0.35,1.02],'--',color='gray',alpha=0.5)
ax[0].axhline(0.5,ls=':',color='gray',alpha=0.4); ax[0].axvline(0.5,ls=':',color='gray',alpha=0.4)
ax[0].set_xlabel('AUC — Random Split'); ax[0].set_ylabel('AUC — Scaffold Split')
ax[0].legend(handles=[Patch(color=tcol[t],label=t) for t in ['BROAD','MED','ORPHAN']],fontsize=8,loc='lower right'); PL(ax[0],'A')
drops=sorted([(r,rnd2[r]['auc']-scf2[r]['auc']) for r in recs],key=lambda x:-x[1])
names=[d[0].replace('TAS2R','T') for d in drops]; vals=[d[1] for d in drops]
ax[1].bar(range(len(names)),vals,color=[P_SALMON if v>0 else P_MINT for v in vals])
ax[1].axhline(0.1,ls='--',color=P_SALMON,alpha=0.9); ax[1].axhline(0,color='black',lw=0.8)
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names,rotation=90,fontsize=7)
ax[1].set_xlabel('TAS2R subtype'); ax[1].set_ylabel('AUC drop (random $-$ scaffold)'); PL(ax[1],'B')
def tmean(tl,src): v=[src[r]['auc'] for r in tl if r in src and src[r].get('auc') is not None]; return np.mean(v) if v else 0
tiers=['BROAD\n($\\geq$50)','MED\n(10–49)','ORPHAN\n($<$10)']; x=np.arange(3); w=0.36
rm=[tmean(BROAD,rnd2),tmean(MED,rnd2),tmean(ORPHAN,rnd2)]; sm=[tmean(BROAD,scf2),tmean(MED,scf2),tmean(ORPHAN,scf2)]
b1=ax[2].bar(x-w/2,rm,w,label='Random split',color=[P_RED,P_SKY,P_DGRAY],edgecolor='#555')
b2=ax[2].bar(x+w/2,sm,w,label='Scaffold split',color=[P_RED,P_SKY,P_DGRAY],edgecolor='#555',hatch='///',alpha=0.55)
for bars,ms in [(b1,rm),(b2,sm)]:
    for bar,m in zip(bars,ms): ax[2].text(bar.get_x()+bar.get_width()/2,m+0.005,f'{m:.3f}',ha='center',fontsize=8,weight='bold')
ax[2].set_xticks(x); ax[2].set_xticklabels(tiers,fontsize=9); ax[2].set_ylabel('Mean AUC-ROC'); ax[2].set_ylim(0.4,1.0); ax[2].legend(fontsize=8); PL(ax[2],'C')
plt.tight_layout(); plt.savefig('fig2_baseline.png',dpi=200,bbox_inches='tight'); plt.close(); print("fig2 OK")

# ════ fig3 3way ════
order=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10','TAS2R43','TAS2R38','TAS2R5','TAS2R16','TAS2R7','TAS2R47','TAS2R40','TAS2R8','TAS2R44','TAS2R2','TAS2R13']
order=[r for r in order if r in sn and sn[r]['auc'] is not None]
fig,ax=plt.subplots(1,3,figsize=(17,5)); x=np.arange(len(order)); w=0.27
ax[0].bar(x-w,[rnd.get(r,0) or 0 for r in order],w,label='Random (one-hot)',color=P_GRAY)
ax[0].bar(x,[scf.get(r,0) or 0 for r in order],w,label='Scaffold (one-hot)',color=P_BLUE)
ax[0].bar(x+w,[sn[r]['auc'] for r in order],w,label='Scaffold (SelectNet)',color=P_RED)
ax[0].set_xticks(x); ax[0].set_xticklabels([r.replace('TAS2R','') for r in order],fontsize=8)
ax[0].axhline(0.7,ls='--',color='gray',alpha=0.5); ax[0].set_ylabel('AUC-ROC'); ax[0].set_ylim(0.2,1.05); ax[0].set_xlabel('TAS2R subtype'); ax[0].legend(fontsize=8,loc='lower left'); PL(ax[0],'A')
gains=[(sn[r]['auc']-(scf.get(r,0) or 0)) for r in order]; sg=sorted(zip(order,gains),key=lambda v:-v[1]); so=[s[0] for s in sg]; sgv=[s[1] for s in sg]
ax[1].bar(range(len(so)),sgv,color=[P_MINT if g>0 else P_SALMON for g in sgv]); ax[1].axhline(0,color='black',lw=0.8)
ax[1].set_xticks(range(len(so))); ax[1].set_xticklabels([r.replace('TAS2R','T') for r in so],rotation=45,fontsize=8); ax[1].set_xlabel('TAS2R subtype'); ax[1].set_ylabel(r'$\Delta$AUC (SelectNet $-$ one-hot scaffold)'); PL(ax[1],'B')
ax[2].plot([0.25,1.05],[0.25,1.05],'--',color='gray',alpha=0.5); texts=[]
for r in order:
    xx=scf.get(r,0) or 0; yy=sn[r]['auc']
    ax[2].scatter(xx,yy,c=P_RED if r in BROAD else P_BLUE,s=60,edgecolor='#555',lw=0.5,zorder=3)
    texts.append(ax[2].text(xx,yy,r.replace('TAS2R','T'),fontsize=8))
if HAS_ADJ: adjust_text(texts,ax=ax[2],arrowprops=dict(arrowstyle='-',color='gray',lw=0.5))
ax[2].set_xlabel('AUC — Scaffold One-Hot'); ax[2].set_ylabel('AUC — Scaffold SelectNet'); ax[2].set_xlim(0.25,1.08); ax[2].set_ylim(0.25,1.08); PL(ax[2],'C')
plt.tight_layout(); plt.savefig('fig3_3way.png',dpi=200,bbox_inches='tight'); plt.close(); print("fig3 OK")

# ════ figS4 broad ════
fig,ax=plt.subplots(1,2,figsize=(13,4.8)); x=np.arange(len(BROAD)); w=0.27
ax[0].bar(x-w,[rnd.get(r,0) or 0 for r in BROAD],w,label='Random (one-hot)',color=P_GRAY)
ax[0].bar(x,[scf.get(r,0) or 0 for r in BROAD],w,label='Scaffold (one-hot)',color=P_BLUE)
ax[0].bar(x+w,[sn[r]['auc'] or 0 for r in BROAD],w,label='Scaffold (SelectNet)',color=P_RED)
ax[0].set_xticks(x); ax[0].set_xticklabels(BROAD,rotation=30,ha='right',fontsize=8); ax[0].set_ylabel('AUC-ROC'); ax[0].set_ylim(0.5,1.05); ax[0].legend(fontsize=8); PL(ax[0],'A')
means=[gt['broad_n6']['onehot_random'],gt['broad_n6']['onehot_scaffold'],gt['broad_n6']['selectnet']]
bars=ax[1].bar(['Random\n(one-hot)','Scaffold\n(one-hot)','Scaffold\n(SelectNet)'],means,color=[P_GRAY,P_BLUE,P_RED],edgecolor='#555')
for b,m in zip(bars,means): ax[1].text(b.get_x()+b.get_width()/2,m+0.005,f'{m:.3f}',ha='center',weight='bold')
ax[1].set_ylabel('Mean AUC-ROC (broadly-tuned)'); ax[1].set_ylim(0.6,0.9); PL(ax[1],'B')
plt.tight_layout(); plt.savefig('figS4_broad.png',dpi=200,bbox_inches='tight'); plt.close(); print("figS4 OK")

# ════ figR1 revision ════
encs=['onehot','aa_comp','protbert','esm2_full','esm2_pocket']
labels=['One-hot','AA comp.','ProtBERT','ESM-2\nfull-seq','ESM-2\npocket']
ms=[ab[e]['mean'] for e in encs]; sd=[ab[e]['std'] for e in encs]
pcols=[P_GRAY,P_PEACH,P_BLUE,P_SKY,P_RED]
ext=pd.read_csv('external_validation_predictions_v2.csv'); held=ext[~ext.in_bitterdb].sort_values('p_target')
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
b=ax[0].bar(labels,ms,yerr=sd,capsize=4,color=pcols,edgecolor='#555',lw=0.6)
ax[0].axhline(ms[0],ls='--',color='gray',lw=1,alpha=0.7); ax[0].set_ylabel('Macro AUC-ROC (broadly-tuned)'); ax[0].set_ylim(0.55,0.82)
for bar,m in zip(b,ms): ax[0].text(bar.get_x()+bar.get_width()/2,m+0.012,f'{m:.3f}',ha='center',fontsize=9,weight='bold')
PL(ax[0],'A')
comps=['pocket\nvs full-seq','pocket\nvs ProtBERT','pocket\nvs AA-comp']
deltas=[ab['esm2_pocket']['mean']-ab['esm2_full']['mean'],ab['esm2_pocket']['mean']-ab['protbert']['mean'],ab['esm2_pocket']['mean']-ab['aa_comp']['mean']]
bb=ax[1].bar(comps,deltas,color=P_MINT,edgecolor='#555'); ax[1].axhline(0,color='black',lw=0.8); ax[1].set_ylabel(r'$\Delta$ Macro AUC')
for bar,d in zip(bb,deltas): ax[1].text(bar.get_x()+bar.get_width()/2,d+0.002,f'+{d:.3f}',ha='center',fontsize=10,weight='bold')
ax[1].set_ylim(0,max(deltas)*1.25); PL(ax[1],'B')
cmap={'TAS2R14':P_RED,'TAS2R46':P_SKY,'TAS2R38':P_MINT,'TAS2R2':P_LAV}
cols=[cmap.get(r,P_GRAY) for r in held.receptor]
ax[2].barh(range(len(held)),held.p_target,color=cols,edgecolor='#555',lw=0.5)
ax[2].axvline(0.5,ls='--',color='gray',lw=1); ax[2].set_yticks(range(len(held))); ax[2].set_yticklabels(held.compound,fontsize=8)
ax[2].set_xlabel(r'Predicted $p$(target receptor)'); ax[2].set_xlim(0,1)
ax[2].legend(handles=[Patch(color=v,label=k.replace('TAS2R','R')) for k,v in cmap.items()],fontsize=8,loc='lower right'); PL(ax[2],'C')
plt.tight_layout(); plt.savefig('figR1_revision.png',dpi=200,bbox_inches='tight'); plt.close(); print("figR1 OK")

print("\n그룹1 파스텔 재생성 완료")
