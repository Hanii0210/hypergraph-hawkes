"""Plot saved Syn07 repeated timings; never regenerate inputs or fit models."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'experiments/results/synthetic/syn07_scalability'
LABELS={'features':'Feature preparation','iteration':'Cached EM update + score','fit':'Full EM + coordinate fit'}
COLORS={'features':'#427C52','iteration':'#176B91','fit':'#CB752F'}

def plot_results(results_root=DEFAULT):
    results_root=Path(results_root)
    summary=json.loads((results_root/'summary.json').read_text(encoding='utf-8'))
    cfg=json.loads((results_root/'config.json').read_text(encoding='utf-8'))
    destination=results_root/'figures';destination.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.9))
    for ax,sweep in zip(axes,['event_count','node_count']):
        for phase in LABELS:
            rows=[r for r in summary['rows'] if r['sweep']==sweep and r['phase']==phase and r['n_measured']==cfg['repetitions']
                  and (phase!='fit' or r['qualified_fits']==cfg['repetitions'])]
            rows.sort(key=lambda r:r['n_events'] if sweep=='event_count' else r['n_nodes'])
            if not rows:continue
            x=np.array([r['n_events'] if sweep=='event_count' else r['n_nodes'] for r in rows])
            y=np.array([r['median_seconds'] for r in rows])
            err=np.array([[r['median_seconds']-r['q25_seconds'] for r in rows],[r['q75_seconds']-r['median_seconds'] for r in rows]])
            slope=summary['event_count_slopes'][phase]['main_slope']
            label=LABELS[phase]
            if sweep=='event_count' and slope is not None:
                lx,ly=np.log(x),np.log(y)
                intercept=np.mean(ly-slope*lx)
                total=np.sum((ly-ly.mean())**2)
                r_squared=1-np.sum((ly-intercept-slope*lx)**2)/total if total>0 else float('nan')
                label+=f' (p={slope:.3f}, R²={r_squared:.3f})'
            ax.errorbar(x,y,yerr=err,color=COLORS[phase],marker='o',lw=1.4,capsize=3,label=label)
            if sweep=='event_count' and slope is not None:
                intercept=np.mean(np.log(y)-slope*np.log(x));ax.plot(x,np.exp(intercept)*x**slope,ls=':',color=COLORS[phase],lw=1)
        ax.set_yscale('log');ax.grid(alpha=.18);ax.set_axisbelow(True)
        ax.set_ylabel('Seconds (median and interquartile range)')
        ax.legend(fontsize=8,loc='best')
    axes[0].set_xscale('log');axes[0].set_xlabel('Number of events');axes[0].set_title('Event-count sweep, N = 4')
    counts=sorted({r['n_events'] for r in summary['rows'] if r['sweep']=='event_count' and r['n_events']})
    axes[0].set_xticks(counts,[str(n) for n in counts])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    rows=[r for r in summary['rows'] if r['sweep']=='node_count' and r['phase']=='features']
    axes[1].set_xticks([r['n_nodes'] for r in rows],[f"{r['n_nodes']}\n({r['n_events']})" for r in rows])
    axes[1].set_xlabel('Nodes (observed event count in parentheses)');axes[1].set_title('Node-count sweep, T = 200 seconds')
    qualification='All full fits meet the numerical criteria.' if summary.get('status')=='completed' else 'Only numerically qualified full-fit settings are shown.'
    fig.text(.5,.015,'One fixed path per scale; 10 timing repeats. '+qualification+'\n'
             'p and R² describe the five-point log–log fit; they do not establish asymptotic complexity.',ha='center',fontsize=9)
    if summary.get('status')!='completed':
        fig.text(.5,.11,'Incomplete or unqualified settings are omitted from the plot; see the saved status counts.',
                 ha='center',fontsize=8,color='#BA4B36')
    fig.tight_layout(rect=(0,.13,1,1))
    paths=[]
    for ext in ['png','svg']:
        path=destination/f'syn07_scalability.{ext}';fig.savefig(path,dpi=200,bbox_inches='tight');paths.append(path)
    plt.close(fig)
    return paths

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-root',type=Path,default=DEFAULT)
    for path in plot_results(p.parse_args().results_root):print(path)

if __name__=='__main__':main()
