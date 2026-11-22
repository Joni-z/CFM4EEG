from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
r=Path(__file__).parent;d=np.load(r/'full/diagnostics.npz');s=json.loads((r/'full/summary.json').read_text());response=np.load(r/'responses/responses.npz')
red='#F0545C';gray='#9CA3AD';navy='#465363';light='#E7E9ED'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.45,'pdf.fonttype':42,'ps.fonttype':42})
def style(ax):
 ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2,width=.45,pad=2);ax.set_axisbelow(True)
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(r/f'{name}.{ext}',dpi=300,facecolor='white')
alpha=d['alpha'];valid=np.tril(np.ones((16,16),bool),-1);uniform=valid/np.maximum(valid.sum(-1,keepdims=True),1)
en=np.zeros_like(alpha);en[valid]=np.log2(alpha[valid]/uniform[valid]);bound=np.ceil(abs(en[valid]).max()*2)/2
centers=d['trained'].mean(-1);ratio=centers[:,None]/centers[None,:];bins=np.array([1,2,4,8,16,32,64,128]);wm=np.histogram(ratio[valid],bins,weights=alpha[valid]/15)[0];um=np.histogram(ratio[valid],bins,weights=uniform[valid]/15)[0]
fig,axs=plt.subplots(2,2,figsize=(5.5,3.45));fig.subplots_adjust(left=.095,right=.97,top=.93,bottom=.13,wspace=.56,hspace=.66)
for ax in axs.flat:style(ax)
ax=axs[0,0];b=np.arange(1,17)
for key,off,color,mark in [('initial',-.13,gray,'o'),('trained',.13,red,'s')]:
 a=d[key];ax.hlines(b+off,a[:,0],a[:,1],color=color,lw=1);ax.scatter(a.mean(-1),b+off,s=5,color=color,marker=mark)
ax.set_xscale('log');ax.set_xticks([.5,2,8,32,75],['0.5','2','8','32','75']);ax.minorticks_off();ax.set_yticks([1,4,8,12,16]);ax.set_ylim(.5,17);ax.set_xlim(.4,90);ax.set_ylabel('Band index');ax.set_xlabel('Nominal cutoff interval (Hz)');ax.set_title('Frequency organization',loc='left',pad=7)
ax.legend([Line2D([],[],color=gray,lw=1),Line2D([],[],color=red,lw=1)],['Initial','Trained'],frameon=False,fontsize=6,loc='upper left',handlelength=1)
ax=axs[0,1];cmap=LinearSegmentedColormap.from_list('cfm',[navy,'#FAFAFA',red]);cmap.set_bad('#E7E9ED')
im=ax.pcolormesh(np.arange(.5,17.5),np.arange(.5,17.5),np.ma.array(en,mask=~valid),cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
ax.set_xticks([1,4,8,12,16]);ax.set_yticks([1,4,8,12,16]);ax.set_xlim(.5,16.5);ax.set_ylim(.5,16.5);ax.set_xlabel('Source band');ax.set_ylabel('Target band');ax.set_title('Cross-band weighting',loc='left',pad=7)
cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.05);cb.ax.tick_params(labelsize=6,length=2);cb.set_label('log₂(weight / uniform)',size=6);cb.outline.set_linewidth(.4)
ax=axs[1,0];ax.stairs(um,bins,color=gray,ls='--',lw=1,label='Uniform');ax.stairs(wm,bins,color=red,lw=1.3,label='Measured');ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32,64,128]);ax.set_xticklabels(['1','2','4','8','16','32','64','128']);ax.set_xlim(1,128);ax.set_ylim(0,max(wm.max(),um.max())*1.25);ax.set_ylabel('Fraction of total weight');ax.set_xlabel('Nominal center-frequency ratio');ax.legend(fontsize=6,frameon=False,loc='upper right',handlelength=1.3);ax.set_title('Frequency separation',loc='left',pad=7)
ax=axs[1,1];order=['measured','envelope_only','weights_only','own_phase','shuffled_Z','identity'];labels=['Unchanged','Envelope only','Weights only','Own phase','Shuffled Z','Identity'];values=[s['interventions'][k]['metrics']['cohen_kappa'] for k in order]
ax.axvline(values[0],ls='--',color=gray,lw=.7)
for i,v in enumerate(values):
 ax.plot(v,i,'s',ms=3,color=red if i==0 else navy);ax.text(v+.024,i,f'{v:.3f}',va='center',size=6.5)
ax.set_yticks(range(6),labels,fontsize=6);ax.set_ylim(5.6,-.6);ax.set_xlim(-.03,.86);ax.set_xticks([0,.2,.4,.6,.8]);ax.set_xlabel('Cohen’s κ');ax.set_title('Test-time interventions',loc='left',pad=7);ax.grid(axis='x',ls=':',lw=.45,color=light)
save(fig,'tokenizer_diagnostics')
# Full finite-filter response and overlap, using power cosine, not cutoff intervals.
mag=response['trained_magnitude'];power=mag**2;pn=power/np.linalg.norm(power,axis=1,keepdims=True);overlap=pn@pn.T;adj=np.array([overlap[j,j-1] for j in range(1,16)]);jj,ii=np.where(valid);e=en[valid];ov=overlap[valid];top=np.argsort(e)[-10:];rho=float(spearmanr(ov,e).statistic)
fig,axs=plt.subplots(1,3,figsize=(5.5,1.85));fig.subplots_adjust(left=.075,right=.995,bottom=.27,top=.82,wspace=.5)
for ax in axs:style(ax)
for b in range(16):axs[0].plot(response['frequency_hz'],mag[b]/mag[b].max(),color=plt.cm.viridis(b/15),lw=.65)
axs[0].set_xlim(0,85);axs[0].set_ylim(0,1.07);axs[0].set_xlabel('Frequency (Hz)');axs[0].set_ylabel('Normalized magnitude');axs[0].set_title('Finite filter responses',pad=7)
axs[1].bar(np.arange(2,17),adj,color=gray,width=.65);axs[1].set_xticks([2,6,10,14]);axs[1].set_ylim(0,1.05);axs[1].set_xlabel('Adjacent pair ending at band');axs[1].set_ylabel('Power-response overlap');axs[1].set_title('Adjacent overlap',pad=7)
axs[2].scatter(ov,e,s=6,c=gray);axs[2].scatter(ov[top],e[top],s=12,facecolors='none',edgecolors=red,lw=.65);axs[2].axhline(0,c=light,lw=.5);axs[2].set_xlabel('Power-response overlap');axs[2].set_ylabel('log₂(weight / uniform)');axs[2].set_title('Weight versus overlap',pad=7);axs[2].text(.96,.95,f'ρ = {rho:.2f}',transform=axs[2].transAxes,ha='right',va='top',fontsize=6.5);axs[2].set_xlim(-.04,1.05)
save(fig,'filter_overlap')
fig,axs=plt.subplots(4,4,figsize=(5.5,5.6));fig.subplots_adjust(left=.08,right=.995,bottom=.075,top=.96,wspace=.35,hspace=.63)
for i,ax in enumerate(axs.flat):
 style(ax)
 for stage,col,ls in [('initial',gray,'--'),('trained',red,'-')]:
  v=response[stage+'_magnitude'][i];ax.plot(response['frequency_hz'],v/v.max(),color=col,ls=ls,lw=.7)
 ax.axvspan(*d['trained'][i],color=red,alpha=.10);ax.set_xlim(0,min(90,max(3,float(d['trained'][i,1])*1.4)));ax.set_ylim(0,1.07);ax.set_yticks([0,.5,1]);ax.set_title(f'Band {i+1}',pad=3)
 if i>=12:ax.set_xlabel('Hz')
 if i%4==0:ax.set_ylabel('Magnitude')
save(fig,'all_filter_responses')
fig,axs=plt.subplots(1,2,figsize=(5.5,2.3));fig.subplots_adjust(left=.08,right=.97,bottom=.22,top=.88,wspace=.4)
for ax in axs:style(ax)
axs[0].bar(np.arange(1,17),d['waveform_ratio'],color=red,width=.65);axs[0].set_xticks([1,4,8,12,16]);axs[0].set_xlabel('Band index');axs[0].set_ylabel('RMS(w) / RMS(a)');axs[0].set_title('Waveform pathway utilization',pad=7)
conf=np.array(s['interventions']['measured']['confusion']);cm=conf/conf.sum(1,keepdims=True);im=axs[1].imshow(cm,vmin=0,vmax=1,cmap=LinearSegmentedColormap.from_list('redwhite',['white',red]));cl=['SPSW','GPED','PLED','EYEM','ARTF','BCKG'];axs[1].set_xticks(range(6),cl,rotation=45,ha='right',fontsize=5.5);axs[1].set_yticks(range(6),cl,fontsize=5.5);axs[1].set_xlabel('Predicted class');axs[1].set_ylabel('True class');axs[1].set_title('Row-normalized confusion',pad=7);fig.colorbar(im,ax=axs[1],fraction=.045,pad=.04)
save(fig,'content_and_confusion')
(r/'derived.json').write_text(json.dumps({'adjacent_overlap':adj.tolist(),'overlap_enrichment_spearman':rho,'top10_overlap_range':[float(ov[top].min()),float(ov[top].max())],'top10_ratio_range':[float(ratio[valid][top].min()),float(ratio[valid][top].max())],'top10_source_bands':(ii[top]+1).tolist(),'top10_target_bands':(jj[top]+1).tolist(),'frequency_ratio_bins':bins.tolist(),'weight_mass':wm.tolist(),'uniform_mass':um.tolist(),'response_peaks_hz':response['frequency_hz'][mag.argmax(1)].tolist()},indent=2))
print((r/'derived.json').read_text())
