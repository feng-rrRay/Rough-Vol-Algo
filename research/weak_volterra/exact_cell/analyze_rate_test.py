"""Report every measured payoff; primary display is the prespecified cos(.7x)."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/weak_volterra_matplotlib')
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scipy.stats import t as student_t
from run_rate_test import HERE,MODELS

def read_csv(path):
    return list(csv.DictReader(path.open()))

def main():
    folder=HERE/'results'/'production';meta=json.loads((folder/'metadata.json').read_text())
    assert len(meta['diagnostics'])==12,'Production run has not completed.'
    data=read_csv(folder/'differences.csv');rng=np.random.default_rng(990110)
    fitted=[];grids=np.array([16,32,64,128]);x=np.log(grids);xc=x-x.mean()
    for H in (.1,.25,.35):
        blocks=[np.load(folder/f'H{H}_N{2*n}.npz')['replicate_means'] for n in grids]
        for j,model in enumerate((1,3)):
            for k,F in enumerate(('cos07','sin07','gaussian','cos1')):
                means=np.array([b[:,j,k].mean() for b in blocks])
                se=np.array([b[:,j,k].std(ddof=1)/4 for b in blocks])
                bootstrap=np.array([b[rng.integers(0,len(b),size=(20000,len(b))),j,k].mean(axis=1) for b in blocks])
                sign=np.sign(means[0]);same=np.all(means*sign>0)
                valid=np.all(bootstrap*sign>0,axis=0)
                fail=1-float(valid.mean())
                resolved=bool(same and np.all(abs(means)>student_t.ppf(.975,15)*se) and fail<.01)
                slope=float(-xc@np.log(abs(means))/(xc@xc)) if same else None
                lo=hi=slo=shi=None
                if resolved:
                    bs=-(xc@np.log(abs(bootstrap[:,valid])))/(xc@xc)
                    lo,hi=np.quantile(bs,[.025,.975]).tolist()
                    # Bonferroni percentile endpoints for the six primary fits;
                    # still an approximate bootstrap statement, not a certificate.
                    if F=='cos07':
                        slo,shi=np.quantile(bs,[.05/12,1-.05/12]).tolist()
                tail=float(-np.polyfit(x[1:],np.log(abs(means[1:])),1)[0]) if same else None
                fitted.append(dict(H=H,model=MODELS[model],payoff=F,window='coarse N=16,32,64,128',
                    effective_order=slope,bootstrap95_low=lo,bootstrap95_high=hi,
                    primary_family_interval_low=slo,primary_family_interval_high=shi,
                    tail_window_order=tail,all_four_increments_resolved=resolved,
                    bootstrap_sign_failure_fraction=fail))
    with (folder/'effective_orders.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fitted[0]));w.writeheader();w.writerows(fitted)
    mc=read_csv(HERE/'results'/'mc_crosscheck'/'differences.csv');cross=[]
    for r in mc:
        other=next(s for s in data if all(s[k]==r[k] for k in ('H','N_fine','model','payoff')))
        denom=math.hypot(float(r['standard_error']),float(other['standard_error']))
        cross.append(dict(H=float(r['H']),model=r['model'],payoff=r['payoff'],
            standardized_difference=(float(r['difference'])-float(other['difference']))/denom if denom else 0.))
    extension=read_csv(HERE/'results'/'low_h_extension'/'differences.csv')
    assert len(extension)==8,'Exploratory extension has not completed.'
    secondary=[r for r in data+extension if float(r['H'])==.1 and r['model']=='N3' and r['payoff']=='gaussian']
    summary=dict(primary=[r for r in fitted if r['payoff']=='cos07'],mc_crosschecks=cross,
        secondary_payoff_extension=secondary,
        total_antithetic_path_evaluations=12*2*2*524288,
        production_wall_seconds=sum(d['seconds'] for d in meta['diagnostics']),
        interpretation='Finite-window self-convergence diagnostics. Neither a continuous-error certificate nor proof of a fixed-model leading expansion.',
        uncertainty='Pointwise t intervals for increments; replicate-bootstrap percentile intervals for effective orders. No within-net iid assumption.')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'axes.labelsize':8,'axes.titlesize':10,
                         'xtick.labelsize':8,'ytick.labelsize':8})
    colors=['#2363a0','#c25225'];fig,axes=plt.subplots(1,3,figsize=(7.4,2.9),sharey=True)
    for ax,H in zip(axes,(.1,.25,.35)):
        for color,model in zip(colors,('N1','N3')):
            rows=[r for r in data if float(r['H'])==H and r['model']==model and r['payoff']=='cos07']
            y=np.array([float(r['difference']) for r in rows]);lo=np.array([float(r['ci95_low']) for r in rows]);hi=np.array([float(r['ci95_high']) for r in rows])
            low_abs=np.where(lo*hi<=0,0.,np.minimum(abs(lo),abs(hi)))
            ax.errorbar(grids,abs(y),yerr=np.array([abs(y)-low_abs,np.maximum(abs(lo),abs(hi))-abs(y)]),
                        fmt='o-',color=color,capsize=3,label='No drift (N1)' if model=='N1' else 'Nonlinear drift (N3)')
        anchor=float(next(r['difference'] for r in data if float(r['H'])==H and r['model']=='N3' and r['payoff']=='cos07' and r['N_coarse']=='16'))
        if H==.25:
            profile=(1+np.log(grids)-np.log(2))/grids
            ax.plot(grids,anchor*profile/profile[0],'--',color='0.35',label='_nolegend_')
            ax.plot(grids,anchor*grids[0]/grids,':',color='0.35',label='_nolegend_')
            ax.text(.97,.96,'Dashed: log profile\nDotted: order 1',transform=ax.transAxes,ha='right',va='top',fontsize=6.5)
        else:
            exponent=min(4*H,1.)
            ax.plot(grids,anchor*(grids/grids[0])**(-exponent),'--',color='0.35',label='_nolegend_')
            ax.text(.97,.96,f'Dashed: order {exponent:g}',transform=ax.transAxes,ha='right',va='top',fontsize=6.5)
        ax.set(xscale='log',yscale='log',xlabel='Coarse grid N',title=f'H = {H:.2f}',xticks=grids)
        ax.set_xticklabels([str(n) for n in grids]);ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True,which='both',alpha=.18);ax.legend(fontsize=7,frameon=False,loc='lower left')
    axes[0].set_ylabel(r'$|\mathbb{E}[F(Y_{2N})-F(Y_N)]|$')
    fig.suptitle('Fixed nonlinear coefficients, F(x) = cos(0.7x); pointwise 95% intervals',fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE/'results'/'fixed_nonlinear_rate_test.png',dpi=180)
    fig.savefig(HERE/'results'/'fixed_nonlinear_rate_test.pdf')
    fig2,ax=plt.subplots(figsize=(7,3))
    for subset,label,color,marker in ((secondary[:4],'Initial fixed-budget run','#2363a0','o'),
                                      (secondary[4:],'Exploratory finer-grid MC','#c25225','s')):
        ns=np.array([float(r['N_coarse']) for r in subset]);ys=np.array([float(r['difference']) for r in subset])*1e4
        radius=np.array([float(r['ci95_high'])-float(r['difference']) for r in subset])*1e4
        ax.errorbar(ns,ys,yerr=radius,fmt=marker+'-',color=color,capsize=3,label=label)
    ax.axhline(0,color='0.4',lw=.8);ax.set(xscale='log',xlabel='Coarse grid N',ylabel=r'Signed mesh difference $\times 10^4$',
        title=r'Secondary payoff: $e^{-(x-0.4)^2/2}$, nonlinear drift, H=0.10')
    ax.set_xticks([16,32,64,128,256,512]);ax.set_xticklabels(['16','32','64','128','256','512'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True,alpha=.2);ax.legend(frameon=False,fontsize=8);fig2.tight_layout()
    fig2.savefig(HERE/'results'/'secondary_payoff_extension.pdf')
    fig2.savefig(HERE/'results'/'secondary_payoff_extension.png',dpi=180)
    # Table fragments are generated from the saved data, not transcribed.
    lines=[]
    for r in summary['primary']:
        interval=(f"[{r['bootstrap95_low']:.2f}, {r['bootstrap95_high']:.2f}]" if r['bootstrap95_low'] is not None else 'unresolved')
        lines.append(f"{r['H']:.2f} & {r['model']} & {r['effective_order']:.2f} & {interval} & {r['tail_window_order']:.2f} \\")
    (HERE/'orders_table.tex').write_text('\n'.join(lines).replace(' \\', ' \\\\')+'\n')
    lines=[]
    for r in data:
        if r['payoff']=='cos07':
            mean=float(r['difference'])*1e4;radius=(float(r['ci95_high'])-float(r['difference']))*1e4
            lines.append(f"{float(r['H']):.2f} & {r['model']} & {r['N_coarse']} & ${mean:.3f}\\pm{radius:.3f}$ \\")
    (HERE/'differences_table.tex').write_text('\n'.join(lines).replace(' \\', ' \\\\')+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
