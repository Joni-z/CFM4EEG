"""Rebuild manuscript figures from frozen measurements; does not run a model."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
D = json.loads((ROOT / 'evidence/results.json').read_text())
TEAL, CORAL, INK, GREY = '#23877E', '#CD7655', '#26374B', '#798491'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=.025)
    fig.savefig(OUT / f'{name}.png', dpi=240, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    ax.set(xlim=(0, 100), ylim=(0, 29)); ax.axis('off')
    def box(x, y, w, h, title, detail, color=INK, face='#F3F5F7'):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=.6',
                          edgecolor=color, facecolor=face, linewidth=.7)
        ax.add_patch(p)
        ax.text(x+w/2, y+h*.64, title, ha='center', va='center', fontsize=8, color=color)
        ax.text(x+w/2, y+h*.28, detail, ha='center', va='center', fontsize=6.8, color=color)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
                                    linewidth=.75,color=GREY))
    ax.text(0,27.3,'Demodulate before embedding',fontsize=9,weight='bold')
    ax.text(75,27.3,'Learn interactions after embedding',fontsize=9,weight='bold',ha='center')
    box(0,10,13,12,'EEG window','C channels')
    box(17,10,15,12,'Filterbank','16 bands')
    arrow((13.5,16),(16.6,16))
    box(37,17.2,17,7,'Waveform W','signed samples',TEAL,'#EDF6F3')
    box(37,5.5,17,8,'Envelope E','modulus → log',CORAL,'#FCF3EE')
    arrow((32.3,18.5),(36.5,20.5)); arrow((32.3,13.2),(36.5,9.5))
    box(59,16.6,12,8,'Linear map','50 → 64',TEAL,'#EDF6F3')
    box(59,5.5,12,8,'Linear map','50 → 64',CORAL,'#FCF3EE')
    arrow((54.4,20.6),(58.7,20.6)); arrow((54.4,9.5),(58.7,9.5))
    # Two content blocks in one token; no additional fusion module.
    for i in range(8):
        ax.add_patch(Rectangle((77+i*.75,10),.7,11,
                              facecolor=TEAL if i%2==0 else CORAL,edgecolor='none',alpha=.85))
    arrow((71.5,20.6),(76.5,18)); arrow((71.5,9.5),(76.5,13))
    ax.text(80,23,'Paired token',ha='center',fontsize=7.8)
    ax.text(80,7.2,'128 coordinates',ha='center',fontsize=6.8)
    box(87,10,12.5,12,'Axial reader','time\nchannel · band')
    arrow((83.5,15.5),(86.5,15.5))
    ax.text(45,1.1,'Same electrode, same band, same 0.25-s interval',ha='center',fontsize=7.4)
    ax.text(93.2,5.3,'Pool → class',ha='center',fontsize=7.4)
    save(fig, 'architecture')


def recovery():
    a = json.loads((ROOT/'evidence/recovery-example.json').read_text())['samples']
    t = np.arange(50)/200
    fig, axes = plt.subplots(2,1,figsize=(3.32,2.15),sharex=True)
    for ax,key,label,color in zip(axes,['W','E'],['Band waveform','Amplitude envelope'],[TEAL,CORAL]):
        target, decoded = a[key], a[f'{key}_decoded']
        ax.plot(t,target,color=color,lw=1.5,label='Frontend signal')
        ax.plot(t,decoded,color=INK,lw=.9,ls=(0,(3,2)),label='Decoded from token')
        ax.set_ylabel('Amplitude',labelpad=3)
        ax.set_title(label,loc='left',pad=2)
        ax.tick_params(length=2)
        ax.locator_params(axis='y',nbins=3)
    axes[-1].set_xlabel('Time within token (s)',labelpad=2)
    axes[-1].set_xlim(0,.245)
    axes[-1].set_xticks([0,.1,.2])
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,1.025),
               ncol=2,frameon=False,fontsize=6.8,handlelength=2,columnspacing=1)
    fig.subplots_adjust(left=.18,right=.98,bottom=.18,top=.81,hspace=.65)
    save(fig,'recovery')


def class_f1():
    raw=np.asarray(D['per_class']['raw']['f1']); full=np.asarray(D['per_class']['full']['f1'])
    fig,ax=plt.subplots(figsize=(3.32,2.25))
    y=np.arange(6)
    ax.hlines(y,raw,full,color='#C6CFD3',lw=2,zorder=1)
    ax.scatter(raw,y,s=22,c=GREY,label='Raw-patch rows',zorder=2)
    ax.scatter(full,y,s=25,c=TEAL,label='Demodulation-first',zorder=3)
    for i in range(6):
        ax.text(max(raw[i],full[i])+.027,i,f'+{full[i]-raw[i]:.3f}',va='center',fontsize=7)
    ax.set(yticks=y,yticklabels=D['classes'],xlim=(0,1.19),ylim=(5.7,-.7),xlabel='Per-class F1')
    ax.set_xticks([0,.25,.5,.75,1]); ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0); ax.tick_params(axis='x',length=2)
    ax.grid(axis='x',alpha=.18,lw=.5);ax.set_axisbelow(True)
    ax.legend(loc='lower center',bbox_to_anchor=(.45,1.0),ncol=2,frameon=False,
              fontsize=6.8,handletextpad=.35,columnspacing=.8)
    fig.subplots_adjust(left=.16,right=.98,bottom=.20,top=.85)
    save(fig,'class_f1')


if __name__=='__main__':
    architecture(); recovery(); class_f1()
