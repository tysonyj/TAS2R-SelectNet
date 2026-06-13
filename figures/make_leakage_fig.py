"""figure: leakage 메커니즘 정량화 (유사도 분포 + 유사도-AUC 곡선)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P_BLUE='#A8CCE5'; P_RED='#E8A7A0'; P_SKY='#8FB8DC'
def PL(ax,l):
    ax.annotate(l, xy=(0,1), xycoords='axes fraction', xytext=(-30,12),
                textcoords='offset points', fontsize=15, fontweight='bold',
                va='bottom', ha='left', annotation_clip=False)

sim=json.load(open('leakage_sim.json'))
curve=json.load(open('leakage_curve.json'))

fig,ax=plt.subplots(1,2,figsize=(13,4.8))

# A: nearest-train 유사도 분포 (random vs scaffold)
rs=np.array(sim['random_sims']); ss=np.array(sim['scaffold_sims'])
bins=np.linspace(0,1,26)
ax[0].hist(rs,bins=bins,alpha=0.6,color=P_SKY,label=f'Random split (median {sim["random_nearest_sim_median"]:.2f})',density=True,edgecolor='white',lw=0.3)
ax[0].hist(ss,bins=bins,alpha=0.6,color=P_RED,label=f'Scaffold split (median {sim["scaffold_nearest_sim_median"]:.2f})',density=True,edgecolor='white',lw=0.3)
ax[0].axvline(sim['random_nearest_sim_median'],color=P_SKY,ls='--',lw=1.5)
ax[0].axvline(sim['scaffold_nearest_sim_median'],color=P_RED,ls='--',lw=1.5)
ax[0].set_xlabel('Nearest-train Tanimoto similarity'); ax[0].set_ylabel('Density')
ax[0].legend(fontsize=8,loc='upper right'); PL(ax[0],'A')

# B: 유사도 bin별 AUC 곡선
c=[x for x in curve['curve'] if x[1] is not None]
labels=[x[0] for x in c]; aucs=[x[1] for x in c]; ns=[x[2] for x in c]
xpos=np.arange(len(c))
ax[1].plot(xpos,aucs,'o-',color=P_RED,ms=9,lw=2,markeredgecolor='#555')
for i,(a,n) in enumerate(zip(aucs,ns)):
    ax[1].annotate(f'n={n}',(i,a),textcoords='offset points',xytext=(0,10),fontsize=7,ha='center')
ax[1].set_xticks(xpos); ax[1].set_xticklabels(labels,fontsize=8,rotation=20)
ax[1].set_xlabel('Nearest-train Tanimoto similarity bin'); ax[1].set_ylabel('Prediction AUC-ROC')
ax[1].set_ylim(0.6,1.04); ax[1].axhline(0.5,ls=':',color='gray',alpha=0.5)
ax[1].annotate(f"Spearman $r$ = {curve['spearman_r']:.2f}\n($p$ = {curve['spearman_p']:.1e})",
               xy=(0.05,0.88),xycoords='axes fraction',fontsize=9,
               bbox=dict(boxstyle='round',fc='white',ec='gray',alpha=0.9))
PL(ax[1],'B')

plt.tight_layout(); plt.savefig('fig_leakage.png',dpi=200,bbox_inches='tight'); plt.close()
print("fig_leakage.png 생성")
print(f"A: random median {sim['random_nearest_sim_median']:.3f} vs scaffold {sim['scaffold_nearest_sim_median']:.3f}, MWU p={sim['mannwhitney_p']:.1e}")
print(f"B: 유사도 bin AUC = {[round(a,3) for a in aucs]}")
