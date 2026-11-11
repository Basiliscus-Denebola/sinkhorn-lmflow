"""V1 tables and figures from durable artifacts, with failures in every denominator."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io.result_io import load_result
from ..io.config import save_json

THRESHOLDS=(1e-4,1e-6,1e-8)
COUNTERS=('state_eval_count','objective_gradient_count','objective_diff_count','bregman_count',
          'logsumexp_count','matrix_vector_count','plan_build_count','log_kernel_build_count',
          'kernel_build_count','sr1_attempt_count','sr1_valid_count','sr1_accept_count',
          'sr1_skip_count','sr1_reject_count','sr1_fallback_count','sr1_exhausted_count')
COLORS={'sk_log_fast':'#2878b5','mirror_sr1':'#e07a25','sk_scaling_fast':'#37966f',
        'lbfgs_reduced':'#9654a0','mirror_amp_ablation':'#bd4b69'}
LABELS={'sk_log_fast':'Log SK','mirror_sr1':'SR1','sk_scaling_fast':'Scaling SK',
        'lbfgs_reduced':'L-BFGS','mirror_amp_ablation':'Amplitude only'}
GOOD=('success','not_applicable_kernel_underflow')
REPRESENTATIVES=('M1_N1024_s42_e0.001','M2_N1024_s42_e0.003','M3_N1024_s42_e0.003',
                 'M4_N1024_s42_a0.1_e0.003','M5_same_0_e0.001','M5_cross_0_1_e0.001',
                 'M7_1024x2048_s42_e0.003')


def fmt(x):
    return '—' if x is None or not np.isfinite(x) else f'{x:.4g}'


def median_range(values):
    v=[float(x) for x in values if x is not None and np.isfinite(x)]
    return dict(median=float(np.median(v)),minimum=min(v),maximum=max(v)) if v else dict(median=None,minimum=None,maximum=None)


def write_csv(rows,path,columns=()):
    frame=pd.DataFrame(rows)
    if frame.empty and columns:frame=pd.DataFrame(columns=columns)
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    return frame


def read_tracks(root,manifest,reaudit=False):
    cases={c['case_id']:c for c in manifest['cases']};rows=[];thresholds=[];attempts=[];audits=[];histories={}
    cached_case=None;problem=None;dataset=None
    if reaudit:
        from ..experiments.sr1_mainstream import make_input,audit_endpoint
        from ..problems.mnist import ensure_mnist
        from ..io.problem_io import problem_fingerprint
        from ..problems.weak_coupling import initial_fingerprint
        if any(c['family']=='M5' and c.get('data_status')=='ready' for c in cases.values()):
            from ..io.config import load_config
            config=load_config(root/'study_config.yaml')
            images,labels,provenance=ensure_mnist(config.get('mnist_cache','data/mnist'),download=False)
            dataset=(images,labels,provenance,json.loads((root/'mnist_pairs.json').read_text(encoding='utf-8')))
    for job in manifest['jobs']:
        case=cases[job['case_id']]
        base={k:job[k] for k in ('job_id','case_id','method','repeat','warmup_iters','group','mode','status')}
        base.update(family=case['family'],epsilon=case['epsilon'],alpha=case.get('alpha'),
                    comparison_excluded_reason=job.get('comparison_excluded_reason'),
                    marginal_type='dirichlet' if case.get('alpha') is not None else 'image' if case['family']=='M5' else 'uniform',
                    n=case.get('actual_n',case['n']),m=case.get('actual_m',case['m']),
                    problem_sha256=case.get('problem_sha256'),initial_sha256=case.get('initial_sha256'),
                    attempted=bool(job['attempts']),data_seed=case.get('data_seed'))
        for attempt in job['attempts']:
            attempts.append(dict(base,**{k:v for k,v in attempt.items() if not isinstance(v,dict)}))
        attempt=job['attempts'][-1] if job['attempts'] else {}
        row=dict(base,attempt=attempt.get('attempt'),run_path=attempt.get('run_path'),
                 peak_sampled_rss_bytes=attempt.get('peak_sampled_rss_bytes'),
                 estimated_peak_rss_bytes=attempt.get('estimated_peak_rss_bytes'),
                 memory_limit_bytes=attempt.get('memory_limit_bytes'),reason=attempt.get('error',attempt.get('reason',job.get('skip_reason',case.get('data_error','')))))
        result=None;issues=[];readback=None
        if attempt.get('run_path'):
            path=root/attempt['run_path']
            try:
                result=load_result(path);meta=result.summary
                runtime=json.loads((path/'runtime_metadata.json').read_text(encoding='utf-8'))
                row.update({k:v for k,v in meta.items() if not isinstance(v,(dict,list))})
                row.update(solver_name=result.solver_name,n_iter=result.n_iter,total_sec=result.elapsed_sec,
                    converged=result.converged,stop_reason=result.stop_reason,
                    source_commit=runtime.get('git_commit'),source_dirty=runtime.get('git_dirty'),
                    sr1_skip_reasons=json.dumps(meta.get('sr1_skip_reasons',{}),sort_keys=True))
                for key,expected in (('problem_sha256',case['problem_sha256']),('initial_sha256',case['initial_sha256']),
                                     ('implementation_sha256',manifest['implementation_sha256']),('config_sha256',manifest['config_sha256'])):
                    if meta.get(key)!=expected:issues.append(key+' mismatch')
                if result.solver_name!=job['method']:issues.append('method mismatch')
                warm=float(meta.get('warmup_elapsed_sec',0));post=float(meta.get('post_elapsed_sec',result.elapsed_sec))
                # SR1 takes adjacent perf_counter samples for total and post;
                # allow clock-read/bookkeeping latency, not algorithmic omissions.
                row['time_partition_error_sec']=warm+post-result.elapsed_sec
                if min(warm,post,result.elapsed_sec)<0 or abs(row['time_partition_error_sec'])>max(1e-5,1e-8*result.elapsed_sec):
                    issues.append('invalid time accounting')
                if result.converged and (meta.get('residual_l1',np.inf)>1e-8 or job['status']!='success'):
                    issues.append('invalid success criterion')
                if meta.get('audit',{}).get('passed') is False:issues.append('saved endpoint audit failed')
                for key in COUNTERS:
                    values=[getattr(h,key) for h in result.history]
                    if any(b<a for a,b in zip(values,values[1:])):issues.append(key+' decreases')
                    if values and meta.get(key,values[-1])<values[-1]:issues.append(key+' final count below history')
                if job['method'] in ('mirror_sr1','mirror_amp_ablation') and meta.get('performance_mode'):
                    expected=result.n_iter+1+meta.get('bridge_state_evaluations',0)+meta.get('sr1_reject_count',0)
                    if meta.get('state_eval_count')!=expected:issues.append('trial/bridge evaluation accounting mismatch')
                hashes={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ('summary.json','history.csv','final_state.npz')}
                if reaudit and meta.get('endpoint_valid',True):
                    if cached_case!=case['case_id']:
                        problem=None
                        problem,arrays,z=make_input(case,dataset);del arrays
                        if problem_fingerprint(problem)!=case['problem_sha256'] or initial_fingerprint(z)!=case['initial_sha256']:
                            raise ValueError('reconstructed input mismatch')
                        cached_case=case['case_id']
                    readback=audit_endpoint(problem,result)
                    if readback['passed'] is False:issues.append('readback endpoint audit failed')
                replay=meta.get('diagnostic_replay_check')
                if replay and not replay['matching_prefix']:issues.append('diagnostic trajectory diverged')
                audits.append(dict(job_id=job['job_id'],passed=not issues,issues=issues,artifact_sha256=hashes,
                    saved_endpoint_audit=meta.get('audit'),readback_endpoint_audit=readback,diagnostic_replay=replay))
                if job['repeat']==0 and job['mode']=='performance' and job['warmup_iters']==20 and (
                    case['case_id'] in REPRESENTATIVES or case['case_id'].startswith('smoke')):
                    histories[job['job_id']]=result.history
                row['artifact_audit_passed']=not issues
                if issues:row['status']='audit_failure'
                accepted=[h for h in result.history if h.phase=='sr1']
                row['accepted_step_distribution']=json.dumps(dict(Counter(str(h.step_size) for h in accepted)),sort_keys=True)
                for name,key in (('G','sr1_G'),('Q','sr1_Q'),('cos','sr1_cos')):
                    row[name+'_median']=median_range(getattr(h,key) for h in accepted)['median']
            except (OSError,ValueError,KeyError,TypeError) as exc:
                result=None;row['status']='invalid_artifact';row['reason']=str(exc)
                audits.append(dict(job_id=job['job_id'],passed=False,issues=[str(exc)]))
        rows.append(row)
        for threshold in THRESHOLDS:
            hit=next((h for h in result.history if h.residual_l1<=threshold),None) if result else None
            th=dict(base,terminal_status=row['status'],threshold=threshold,hit=hit is not None,
                iteration=hit.iteration if hit else None,total_sec=hit.elapsed_sec if hit else None,
                post_sec=hit.post_elapsed_sec if hit else None,
                reached_during_warmup=bool(hit and hit.iteration<=job['warmup_iters'] and job['method']!='sk_scaling_fast'),
                terminal_sec=row.get('total_sec'),terminal_residual=row.get('residual_l1'),
                warmup_sha256=row.get('warmup_sha256'),artifact_audit_passed=row.get('artifact_audit_passed'))
            # At the terminal tolerance include exports and final bookkeeping too.
            if hit and threshold==1e-8 and result.converged:th['total_sec']=result.elapsed_sec
            for key in COUNTERS:th[key]=getattr(hit,key) if hit else None
            thresholds.append(th)
    return rows,thresholds,attempts,audits,histories


def pair_rows(thresholds):
    lookup={(r['case_id'],r['method'],r['repeat'],r['warmup_iters'],r['threshold']):r for r in thresholds if r['mode']=='performance'}
    pairs=[]
    for r in thresholds:
        if r['mode']!='performance' or r['method'] in ('sk_log_fast','sk_scaling_fast'):continue
        for reference in ('sk_log_fast','sk_scaling_fast'):
            ref=lookup.get((r['case_id'],reference,r['repeat'],r['warmup_iters'],r['threshold']))
            out={k:r[k] for k in ('case_id','family','epsilon','alpha','marginal_type','method','repeat','warmup_iters','group','threshold')}
            out.update(reference=reference,method_status=r['terminal_status'],reference_status=ref['terminal_status'] if ref else 'not_planned',
                       speedup=None,bound=None,bound_type=None,reason='missing_reference',paired_success=False,
                       method_sec=r['total_sec'],reference_sec=ref['total_sec'] if ref else None,
                       shared_warmup_threshold=r['reached_during_warmup'])
            if r.get('comparison_excluded_reason') or (ref and ref.get('comparison_excluded_reason')):
                out['reason']='excluded_by_user'
            elif ref:
                comparable=(r['problem_sha256']==ref['problem_sha256'] and r['initial_sha256']==ref['initial_sha256'])
                if reference=='sk_log_fast':comparable=comparable and r['warmup_sha256']==ref['warmup_sha256']
                if not comparable:out['reason']='fingerprint_mismatch'
                elif r['artifact_audit_passed'] is False or ref['artifact_audit_passed'] is False:out['reason']='audit_failure'
                elif r['hit'] and ref['hit'] and r['total_sec']>0:
                    out.update(speedup=ref['total_sec']/r['total_sec'],reason='matched_threshold',paired_success=True)
                elif 'not_applicable' in ref['terminal_status']:out['reason']='reference_kernel_not_applicable'
                elif r['hit'] and ref['terminal_status']=='timeout':
                    out.update(bound=ref['terminal_sec']/r['total_sec'],bound_type='lower',reason='reference_timeout')
                elif ref['hit'] and r['terminal_status']=='timeout':
                    out.update(bound=ref['total_sec']/r['terminal_sec'],bound_type='upper',reason='method_timeout')
                else:out['reason']='threshold_not_reached_or_not_run'
            pairs.append(out)
    return pairs


def aggregate(rows,pairs):
    groups={}
    for r in rows:
        key=(r['group'],r['mode'],r['family'],r['epsilon'],r['marginal_type'],r.get('alpha'),r['method'],r['warmup_iters'])
        groups.setdefault(key,[]).append(r)
    output=[]
    for key,group in groups.items():
        category,mode,family,eps,marg,alpha,method,warm=key
        status=Counter(r['status'] for r in group)
        successes=[r for r in group if r['status']=='success']
        eligible=[p for p in pairs if p['group']==category and p['family']==family and p['epsilon']==eps
                  and p['marginal_type']==marg and p.get('alpha')==alpha and p['method']==method and p['warmup_iters']==warm
                  and p['reference']=='sk_log_fast' and p['threshold']==1e-8 and p['paired_success']]
        speed=median_range(p['speedup'] for p in eligible)
        item=dict(group=category,mode=mode,family=family,epsilon=eps,marginal_type=marg,alpha=alpha,method=method,warmup_iters=warm,
                  planned=len(group),attempted=sum(r['attempted'] for r in group),succeeded=status['success'],
                  timeout=status['timeout'],skipped_resource=status['skipped_resource'],
                  skipped_by_user=status['skipped_by_user'],
                  not_applicable=status['not_applicable_kernel_underflow'],pending=status['pending']+status['running']+status['interrupted'],
                  blocked_data=status['blocked_data'],library_stop=status['library_stop'],
                  failures=sum(n for s,n in status.items() if s not in GOOD+('pending','running','interrupted','timeout','skipped_resource','blocked_data','skipped_by_user')),
                  paired_success_count=len(eligible),speedup_median=speed['median'],speedup_min=speed['minimum'],speedup_max=speed['maximum'],
                  success_time_median=median_range(r.get('total_sec') for r in successes)['median'],
                  success_iteration_median=median_range(r.get('n_iter') for r in successes)['median'],statuses=json.dumps(dict(status),sort_keys=True))
        output.append(item)
    return output


def save_figure(fig,path):
    fig.savefig(path.parent/(path.name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(path.parent/(path.name+'.pdf'),bbox_inches='tight');plt.close(fig)


def make_figures(root,rows,pairs,histories,manifest):
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':8,
                        'pdf.fonttype':42,'figure.facecolor':'white'})
    figures=root/'figures';figures.mkdir(exist_ok=True)
    for label,families in [('synthetic',('M1','M2','M3','M4','M7')),('mnist',('M5',))]:
        selected=[p for p in pairs if p['family'] in families and p['group']=='main' and p['threshold']==1e-8 and p['method']=='mirror_sr1']
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for i,ref in enumerate(('sk_log_fast','sk_scaling_fast')):
            data=[p for p in selected if p['reference']==ref];vals=[p['speedup'] for p in data if p['paired_success']]
            if vals:
                axes[0].scatter(np.full(len(vals),i)+np.linspace(-.13,.13,len(vals)),vals,s=17,alpha=.65,label=LABELS[ref])
                axes[0].plot([i-.2,i+.2],[np.median(vals)]*2,color='black',lw=2)
            counts=Counter(p['reason'] for p in data);bottom=0
            for reason,color in [('matched_threshold','#37966f'),('reference_kernel_not_applicable','#adb5bd'),('reference_timeout','#e8b74a'),
                                 ('method_timeout','#d85a5a'),('threshold_not_reached_or_not_run','#7655a4'),
                                 ('missing_reference','#c9c9c9'),('audit_failure','#d60000'),('fingerprint_mismatch','#333333'),('excluded_by_user','#aaaaaa')]:
                n=counts[reason]
                if n:axes[1].bar(i,n,bottom=bottom,color=color,label=reason)
                bottom+=n
            axes[1].text(i,bottom+.8,f'{len(vals)}/{len(data)} paired',ha='center',fontsize=9)
        axes[0].axhline(1,color='black',ls='--',lw=1);axes[0].set_yscale('log');axes[0].set_ylabel('Reference time / SR1 time')
        axes[0].set_title(label+': successful pairs at 1e-8')
        axes[1].set_title('Coverage of planned comparisons');axes[1].set_ylabel('Repeats (not independent datasets)')
        axes[1].set_ylim(0,max(1,1.2*max((sum(p['reference']==ref for p in selected) for ref in ('sk_log_fast','sk_scaling_fast')),default=0)+1))
        for ax in axes:ax.set_xticks([0,1],['Log SK','Scaling SK']);ax.grid(axis='y',alpha=.2)
        handles,labels=axes[1].get_legend_handles_labels();unique=dict(zip(labels,handles));axes[1].legend(unique.values(),unique.keys(),fontsize=7)
        fig.tight_layout();save_figure(fig,figures/('speedups_'+label))
    representatives=REPRESENTATIVES
    if manifest['cases'][0]['case_id'].startswith('smoke'):representatives=[c['case_id'] for c in manifest['cases']]
    for case_id in representatives:
        tracks=[r for r in rows if r['case_id']==case_id and r['repeat']==0 and r['mode']=='performance'
                and r['warmup_iters']==20 and r['job_id'] in histories]
        if not tracks:continue
        fig,axes=plt.subplots(1,2,figsize=(11,4));fig.suptitle(case_id+' | total time includes warmup and setup')
        for r in tracks:
            history=histories[r['job_id']]
            for ax,key in zip(axes,('iteration','elapsed_sec')):
                x=[getattr(h,key) for h in history];y=[h.residual_l1 for h in history]
                ax.semilogy(x,y,color=COLORS[r['method']],label=LABELS[r['method']]+' / '+r['status'],lw=1.6)
                if x and r['status']!='success':ax.plot(x[-1],y[-1],'x',color=COLORS[r['method']],ms=9)
                ax.axhline(1e-8,color='grey',ls=':',lw=.8);ax.grid(alpha=.2)
        axes[0].set_xlabel('Accepted iterations (includes warmup)');axes[1].set_xlabel('Total seconds');axes[0].set_ylabel('Max row/column L1 residual')
        axes[1].legend();fig.tight_layout();save_figure(fig,figures/('convergence_'+case_id))
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for col,family in enumerate(('M1','M2','M3')):
        for method in ('sk_log_fast','mirror_sr1'):
            for row_index,key in enumerate(('n_iter','total_sec')):
                data=[r for r in rows if r['group']=='main' and r['family']==family and r['method']==method and r['data_seed']==42]
                eps=sorted(set(r['epsilon'] for r in data));vals=[]
                for ep in eps:
                    good=[r.get(key) for r in data if r['epsilon']==ep and r['status']=='success']
                    vals.append(median_range(good)['median'] or np.nan)
                axes[row_index,col].loglog(eps,vals,'o-',color=COLORS[method],label=LABELS[method])
                axes[row_index,col].set_title(family+' | seed 42, same input');axes[row_index,col].set_xlabel('epsilon');axes[row_index,col].grid(alpha=.2)
        axes[0,col].legend()
    axes[0,0].set_ylabel('Median accepted iterations');axes[1,0].set_ylabel('Median total seconds');fig.tight_layout();save_figure(fig,figures/'epsilon_trends')
    scale=[r for r in rows if r['family']=='M6' or r['case_id']=='M3_N1024_s42_e0.003']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for method in ('sk_log_fast','mirror_sr1','sk_scaling_fast'):
        data=[r for r in scale if r['method']==method and r['warmup_iters']==20 and r['mode']=='performance'];ns=sorted(set(r['n'] for r in data))
        times=[median_range(r.get('total_sec') for r in data if r['n']==n and r['status']=='success' and not r.get('comparison_excluded_reason'))['median'] or np.nan for n in ns]
        memory=[median_range(r.get('peak_sampled_rss_bytes') for r in data if r['n']==n and not r.get('comparison_excluded_reason'))['median'] for n in ns]
        axes[0].loglog(ns,times,'o-',color=COLORS[method],label=LABELS[method]);axes[1].plot(ns,[v/2**30 if v else np.nan for v in memory],'o-',color=COLORS[method],label=LABELS[method])
        for n in ns:
            counts=Counter(r['status'] for r in data if r['n']==n)
            if any(s!='success' for s in counts):axes[0].text(.02,.96-.07*len(axes[0].texts),f'{LABELS[method]} N={n}: {dict(counts)}',transform=axes[0].transAxes,fontsize=8,va='top')
    axes[0].set_ylabel('Success median seconds (warmup included)');axes[1].set_ylabel('Sampled process RSS peak (GiB)')
    excluded_scale=any(r.get('comparison_excluded_reason') and r['n']==5000 for r in scale)
    for ax in axes:
        ax.set_xlabel('N (1024 reuses M3)');ax.grid(alpha=.2);ax.legend(loc='lower right')
        if excluded_scale:
            ax.set_xlim(400,6000);ax.set_xticks([512,1024,2048,5000],['512','1024','2048','5000\nexcluded'])
            ax.axvline(5000,color='grey',ls=':',alpha=.5)
    fig.suptitle('Scale | no imputation for incomplete runs');fig.tight_layout();save_figure(fig,figures/'scale_time_memory')
    amplitude_cases=['T2_N256_d0.001_s42','T4_N256_d0.001_s42','M3_N1024_s42_e0.003']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for method in ('sk_log_fast','mirror_sr1','mirror_amp_ablation'):
        data=[r for r in rows if r['case_id'] in amplitude_cases and r['method']==method and r['warmup_iters']==20 and r['mode']=='performance']
        for ax,key in zip(axes,('total_sec','n_iter')):
            vals=[median_range(r.get(key) for r in data if r['case_id']==case and r['status']=='success')['median'] or np.nan for case in amplitude_cases]
            ax.semilogy(range(3),vals,'o-',color=COLORS[method],label=LABELS[method]);ax.set_xticks(range(3),['Toy 2','Toy 4','Mixture']);ax.grid(alpha=.2)
    axes[0].set_ylabel('Success median total seconds');axes[1].set_ylabel('Success median iterations');axes[0].legend();fig.suptitle('Amplitude ablation | coverage in ablation_summary.csv');fig.tight_layout();save_figure(fig,figures/'amplitude_ablation')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,case_id in zip(axes,('M3_N1024_s42_e0.003','M5_cross_0_1_e0.001')):
        for method in ('sk_log_fast','mirror_sr1'):
            vals=[];coverage=[]
            for warm in (0,5,20,50):
                data=[r for r in rows if r['case_id']==case_id and r['method']==method and r['warmup_iters']==warm and r['mode']=='performance']
                good=[r for r in data if r['status']=='success'];vals.append(median_range(r.get('total_sec') for r in good)['median'] or np.nan)
                coverage.append(f'{len(good)}/{len(data)}')
            ax.plot([0,5,20,50],vals,'o-',color=COLORS[method],label=LABELS[method]+' '+','.join(coverage))
        ax.set_title(case_id);ax.set_xlabel('Shared warmup iterations');ax.set_ylabel('Success median total seconds');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();save_figure(fig,figures/'warmup_ablation')
    image_cases={c.get('pair_name'):c for c in manifest['cases'] if c['family']=='M5'}
    if image_cases:
        nr=(len(image_cases)+1)//2
        fig,axes=plt.subplots(nr,4,figsize=(11,max(3,2*nr)),squeeze=False)
        for ax in axes.flat:ax.axis('off')
        for i,(name,case) in enumerate(image_cases.items()):
            path=root/'problems'/case['case_id']/'reconstruction.npz'
            pair_axes=axes[i//2,2*(i%2):2*(i%2)+2]
            if path.exists():
                with np.load(path,allow_pickle=False) as d:
                    for ax,key,label in zip(pair_axes,('image_a','image_b'),('Source','Target')):
                        if key in d:
                            ax.imshow(d[key],cmap='gray',vmin=0,vmax=255)
                            ax.set_title(f'{name}\n{label} | support {np.count_nonzero(d[key])}',fontsize=9)
                        ax.axis('off')
            else:
                for ax in pair_axes:ax.text(.5,.5,name+' blocked data',ha='center');ax.axis('off')
        fig.tight_layout();save_figure(fig,figures/'mnist_fixed_pairs')


def report_mainstream(run_dir,make_plots=True,reaudit=False,report_path=None):
    root=Path(run_dir).resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    tables=root/'tables';tables.mkdir(exist_ok=True)
    rows,thresholds,attempts,audits,histories=read_tracks(root,manifest,reaudit)
    pairs=pair_rows(thresholds);summary=aggregate(rows,pairs)
    write_csv(rows,tables/'runs.csv');write_csv(attempts,tables/'all_attempts.csv',('job_id','attempt','status'))
    write_csv(thresholds,tables/'threshold_costs.csv');write_csv(pairs,tables/'paired_speedups.csv',('case_id','method','reference','threshold','speedup','reason'))
    write_csv(summary,tables/'coverage.csv')
    write_csv([r for r in summary if r['group']=='main'],tables/'main_summary.csv')
    write_csv([r for r in rows if r['family'] in ('M6','M7') or r['case_id']=='M3_N1024_s42_e0.003'],tables/'scale_summary.csv')
    subset={r['case_id'] for r in rows if r['group']=='lbfgs'}
    write_csv([r for r in rows if r['case_id'] in subset and r['mode']=='performance' and r['warmup_iters']==20],tables/'baseline_subset.csv')
    ablation_cases={r['case_id'] for r in rows if r['group'] in ('amplitude','warmup')}
    write_csv([r for r in rows if r['case_id'] in ablation_cases and r['mode']=='performance' and r['method']!='sk_scaling_fast' and r['method']!='lbfgs_reduced'],tables/'ablation_summary.csv')
    write_csv([r for r in rows if r['status'] in ('pending','running','interrupted','blocked_data','data_generation_failure')],tables/'pending.csv',('job_id','status','reason'))
    write_csv([r for r in rows if r['status'] not in GOOD+('pending','running','skipped_by_user')],tables/'failures.csv',('job_id','status','reason'))
    write_csv([r for r in rows if r.get('comparison_excluded_reason')],tables/'user_exclusions.csv',('job_id','status','comparison_excluded_reason'))
    audit=dict(readback_recomputed=reaudit,planned=len(rows),persisted_artifacts=len(audits),
               all_artifact_checks_passed=all(a['passed'] for a in audits),tracks=audits)
    save_json(audit,tables/'independent_audit.json')
    if make_plots:make_figures(root,rows,pairs,histories,manifest)
    report=build_report(root,manifest,rows,pairs,summary,audit)
    (root/'report.md').write_text(report,encoding='utf-8')
    if report_path:
        destination=Path(report_path).resolve()
        figure_link=os.path.relpath(root/'figures',destination.parent).replace('\\','/')+'/'
        destination.write_text(report.replace('](figures/',']('+figure_link),encoding='utf-8')
    return dict(run_dir=str(root),planned=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
                audit_passed=audit['all_artifact_checks_passed'],report=str(root/'report.md'))


def build_report(root,manifest,rows,pairs,summary,audit):
    counts=Counter(r['status'] for r in rows);incomplete=any(r['status'] in ('pending','running','interrupted','blocked_data','skipped_resource','skipped_by_user','data_generation_failure') for r in rows)
    title='阶段结果' if incomplete else '完整计划结果（包括真实失败）'
    env=json.loads((root/'environment.json').read_text(encoding='utf-8'));budget=manifest['budget_snapshot']
    recovered_charge=sum(a.get('recovered_machine_charge_sec',0) for j in manifest['jobs'] for a in j['attempts'])
    main=[p for p in pairs if p['group']=='main' and p['method']=='mirror_sr1' and p['reference']=='sk_log_fast' and p['threshold']==1e-8]
    matched=[p for p in main if p['paired_success']];stats=median_range(p['speedup'] for p in matched)
    active=[p for p in matched if not p['shared_warmup_threshold']]
    wins=sum(p['speedup']>1 for p in active);losses=sum(p['speedup']<1 for p in active)
    ordinary=median_range(p['speedup'] for p in matched if p['family'] in ('M1','M2','M3','M4','M7'))['median']
    image_coarse=median_range(p['speedup'] for p in matched if p['family']=='M5' and p['epsilon']==.01)['median']
    image_fine=median_range(p['speedup'] for p in matched if p['family']=='M5' and p['epsilon']==.001)['median']
    lines=[f'# SR1 v1 主流基准：{title}', '',
        f'本轮计划 {len(rows)} 条轨道，当前状态为 `{dict(counts)}`。主比较在 1e-8 精度有 {len(matched)}/{len(main)} 个成功配对；'
        f'其中 {len(matched)-len(active)} 个配对在共享预热内完成，不归因于 SR1；其余按重复轨道计，SR1 快于轻量 log SK 的有 {wins} 个，慢于它的有 {losses} 个。成功配对时间比中位数为 {fmt(stats["median"])}，范围 {fmt(stats["minimum"])}–{fmt(stats["maximum"])}。'
        '该汇总仅作为覆盖概览；普通合成、图像和 toy 的结论分别见下表，三个计时重复不是独立数据样本。', '',
        '## 方法、计时与冻结约束','',
        'SR1 保持 v0 的单对正分母更新：每步采用当前 B0，只使用最近两个真实接受状态；要求 d>0 且相对分母至少 1e-8。'
        '固定六个步长 1, 1/2, …, 1/32，稳定目标差 Armijo，失败即单位 SK 回退。没有多历史、谱预条件或新的外层加速。', '',
        '主基线 sk_log_fast 使用缓存 log 核、行/列 logsumexp，在行归一化端点观测残差，不构造 P、完整目标差或 Bregman。'
        'SR1 在分叉处为自己的方向构造一次所需状态，费用记入该方法。sk_scaling_fast 从零势冷启动，exp 核的构造计费；'
        '核含零即 not_applicable_kernel_underflow，不截断或改变成本。L-BFGS 使用同一半对偶、末坐标固定为零的 m−1 变量，'
        'maxcor=20、maxls=50、ftol=0、gtol=0，以统一边际残差验收；library_stop 不能代替成功，没有 SK 收尾。','',
        '默认先共享最多 20 步真实轻量 SK，并向每种方法计入预热费用；机器账本只收取共享预热一次。'
        '提前达到 1e-8 就直接结束，共享预热内的阈值命中不解释为 SR1 的加速。总时间=warmup+post，setup 已包含在其中，不能再次相加。'
        '普通分叉轨道 setup 是 post 的子集；预热内收敛的 setup 来自预热，属于 warmup。阈值成本来自同一轨道首次命中，终点容差包括最终导出开销。'
        '性能与诊断单独计时；数据准备、输入哈希、写盘、独立审计不计入算法时间，manifest 另存这些开销。'
        'state_eval 在不同内核间不等价；实际 logsumexp、matvec、P 构造、目标差和梯度次数单列，dense_pass_proxy 不用于跨内核比较。'
        'objective_gradient_count 专指 L-BFGS 显式目标/梯度 oracle 调用；SR1 在其稠密 state_eval 中同步计算 f 和 c−b，不把该字段为零解释为没有计算目标或梯度。'
        'matrix_vector_count 对新增缩放核记录 K@v 与 K.T@u；SR1 的每个正常 Bregman 调用另含一次条件均值矩阵向量乘法，包含于 bregman_count。该字段为零不能解释为 SR1 没有矩阵向量运算。','',
        '## 输入、环境与预算','',
        'M1 为满秩 U(0,1) 随机成本；M2 为独立二维均匀点云；M3 为文档固定三团混合高斯；M4 保持 M3 几何但用独立 Dirichlet(1/0.1) 边际，无质量地板。'
        'M1–M4 为 1024×1024；M1–M3 使用 ε=0.01/0.003/0.001、种子 42/43/44。M4 用 ε=0.003。'
        'M6 为 N=512/1024/2048/5000，1024 复用 M3，同一个 5000 点源/目标池取前缀；M7 为 1024×2048。'
        '几何、成本和两侧权重使用独立 SeedSequence 流，改变 ε 或边际不改变点池。合成成本除以各实例原始最大值。','',
        'MNIST 来自 [CVDF 官方镜像](https://github.com/cvdfoundation/mnist) 的测试集。配对在求解前以种子 20261110 固定：'
        '同类 0/2/4/6/8 和跨类 0–1/2–3/4–5/6–7/8–9，共 20 张不同图像；以原始灰度作质量，删除精确零支持，'
        '像素坐标除以 27，平方欧氏成本固定除以完整网格直径平方 2。每对 ε=0.01/0.001；原始索引、掩码、质量、原图和文件 SHA256 均已保存。','',
        f'- Python：`{env["python"].splitlines()[0]}`；解释器：`{env["python_executable"]}`。',
        f'- NumPy {env["numpy"]} / SciPy {env["scipy"]}；CPU：{env["cpu"]}；OS：{env["os"]}。',
        f'- 基础提交 `{env["git_commit"]}`，dirty={env["git_dirty"]}；求解实现指纹 `{manifest["implementation_sha256"]}`。',
        f'- 启动可用内存 {env["available_memory_at_start_bytes"]/2**30:.3f} GiB，运行限制 {env["memory_limit_bytes"]/2**30:.3f} GiB（min(8 GiB, 60% 可用内存)）。',
        f'- 性能机器预算已用 {budget["performance_used_sec"]:.1f}/{budget["performance_budget_sec"]:.0f} 秒；诊断 {budget["diagnostic_used_sec"]:.1f}/{budget["diagnostic_budget_sec"]:.0f} 秒。',
        f'- 中断恢复的保守补计费用合计 {recovered_charge:.3f} 秒，已包含在累计预算中；它是防止恢复重置预算的上界记账，不是精确算法时间，不进入成功轨道的时间比。',
        '- 各次调用的启动内存上限与环境保存在 manifest.invocations；本表环境为最后一次调用，逐轨道原始环境仍各自保留。桌面后台负载与频率未隔离，三次重复的范围不代表置信区间。',
        '- 单轨道维数≤2048 时 300 秒，否则 1200 秒；最多 50000 步、统一 tol=1e-8。数值内核之间检查限时，单次内核可能轻微越界。',
        '- 内存记录为每 50 ms 抽样的进程 RSS 峰值，是瞬时峰值的下界；另保存保守工作区估计。资源跳过单列，未改小 N。','',
        '## 主结果：按族、精度和边际分开','',
        '表内时间比为 log SK 总时间 / SR1 总时间，仅对双方成功且输入、预热匹配的配对计算。小于 1 即 SR1 更慢。'
        '中位数与极值仅为描述统计，不是显著性或置信区间。主表不含 toy。','',
        '| 族 | ε | α | SR1 成功/计划 | SK 成功/计划 | 成功配对 | 时间比中位数 [范围] |','|---|---:|---:|---:|---:|---:|---|']
    if ordinary is not None and image_coarse is not None and image_fine is not None:
        lines[2:2]=[f'本轮实现没有显示 SR1 的普遍墙钟优势：普通合成的 log SK/SR1 时间比中位数为 {fmt(ordinary)}；'
            f'MNIST 在 ε=0.01 时为 {fmt(image_coarse)}，在 ε=0.001 时为 {fmt(image_fine)}。比值大于 1 才表示 SR1 更快，收益依赖具体输入与正则强度。'
            f'调整后的计划剩余 pending 为 {sum(r["status"] in ("pending","running","interrupted","blocked_data") for r in rows)}；原计划中 N=5000 的排除与未达标轨道保留在覆盖统计中。','']
    for s in summary:
        if s['group']!='main' or s['method']!='mirror_sr1':continue
        sk=next(x for x in summary if x['group']=='main' and x['method']=='sk_log_fast' and x['family']==s['family'] and x['epsilon']==s['epsilon'] and x['alpha']==s['alpha'])
        lines.append(f'| {s["family"]} | {s["epsilon"]} | {fmt(s["alpha"])} | {s["succeeded"]}/{s["planned"]} | {sk["succeeded"]}/{sk["planned"]} | {s["paired_success_count"]} | {fmt(s["speedup_median"])} [{fmt(s["speedup_min"])}–{fmt(s["speedup_max"])}] |')
    for label,families in [('普通合成（M1–M4、M7）',('M1','M2','M3','M4','M7')),('图像（M5）',('M5',))]:
        group=[p for p in matched if p['family'] in families];st=median_range(p['speedup'] for p in group)
        lines+=['',f'{label}：{len(group)} 个成功配对，时间比中位数 {fmt(st["median"])}，范围 {fmt(st["minimum"])}–{fmt(st["maximum"])}。']
    lines+=['','同一轨道首次命中的三个残差精度（不另行求解或重调参数）：','',
            '| 数据 | 残差阈值 | 成功配对/计划 | 其中预热内命中 | 时间比中位数 [范围] |',
            '|---|---:|---:|---:|---|']
    for label,families in [('普通合成',('M1','M2','M3','M4','M7')),('MNIST',('M5',))]:
        for threshold in THRESHOLDS:
            group=[p for p in pairs if p['group']=='main' and p['family'] in families and p['method']=='mirror_sr1'
                   and p['reference']=='sk_log_fast' and p['threshold']==threshold]
            good=[p for p in group if p['paired_success']];st=median_range(p['speedup'] for p in good)
            lines.append(f'| {label} | {threshold:g} | {len(good)}/{len(group)} | {sum(p["shared_warmup_threshold"] for p in good)} | {fmt(st["median"])} [{fmt(st["minimum"])}–{fmt(st["maximum"])}] |')
    slower=sorted((p for p in matched if p['speedup']<1),key=lambda p:p['speedup'])[:5]
    if slower:
        lines+=['','保留最慢的五个相对配对（不删除反例）：']
        for p in slower:lines.append(f'- `{p["case_id"]}` r{p["repeat"]}：时间比 {p["speedup"]:.3f}。')
        lines+=['','这些点说明减少接受步数不自动带来墙钟时间优势。SR1 的 P 构造、稳定目标差、Bregman 和拒绝试探有额外成本；具体次数见 runs.csv 与 threshold_costs.csv。']
    lines+=['','## 原始缩放与 L-BFGS','',
            '| 方法 | 计划 | 成功 | 下溢不适用 | 超时 | library_stop | 资源跳过 | 用户取消 | pending |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in ('sk_scaling_fast','lbfgs_reduced'):
        group=[r for r in rows if r['method']==method];c=Counter(r['status'] for r in group)
        lines.append(f'| {method} | {len(group)} | {c["success"]} | {c["not_applicable_kernel_underflow"]} | {c["timeout"]} | {c["library_stop"]} | {c["skipped_resource"]} | {c["skipped_by_user"]} | {c["pending"]} |')
    scaling=[p for p in pairs if p['group']=='main' and p['method']=='mirror_sr1' and p['reference']=='sk_scaling_fast' and p['threshold']==1e-8 and p['paired_success']]
    st=median_range(p['speedup'] for p in scaling)
    lines+=['',f'可用缩放 SK 对 SR1 的成功配对有 {len(scaling)} 个，时间比中位数 {fmt(st["median"])}，范围 {fmt(st["minimum"])}–{fmt(st["maximum"])}。'
            '核下溢是原始表示的适用性限制，不作为稳定算法求解失败。单边超时的 lower/upper bound 单列于 paired_speedups.csv，没有把超时时间当作完成时间。','',
            '| L-BFGS 子集 | L-BFGS 成功/计划 | log SK / L-BFGS 配对中位比 | SR1 / L-BFGS 配对中位比 |','|---|---:|---:|---:|']
    for case_id in dict.fromkeys(r['case_id'] for r in rows if r['method']=='lbfgs_reduced'):
        group=[r for r in rows if r['case_id']==case_id and r['method']=='lbfgs_reduced'];ratios={}
        for reference in ('sk_log_fast','mirror_sr1'):
            values=[]
            for r in group:
                ref=next((x for x in rows if x['case_id']==case_id and x['method']==reference and x['repeat']==r['repeat'] and x['warmup_iters']==r['warmup_iters'] and x['mode']=='performance'),None)
                if r['status']=='success' and ref and ref['status']=='success':values.append(ref['total_sec']/r['total_sec'])
            ratios[reference]=median_range(values)['median']
        lines.append(f'| {case_id} | {sum(r["status"]=="success" for r in group)}/{len(group)} | {fmt(ratios["sk_log_fast"])} | {fmt(ratios["mirror_sr1"])} |')
    lines+=['',
            '## 规模、边际与非方阵','',
            '| 输入 | 方法 | 成功/计划 | 成功时间中位数(s) | 抽样 RSS 最大值(GiB) | 状态 |','|---|---|---:|---:|---:|---|']
    scale=[r for r in rows if (r['family'] in ('M6','M7') or r['case_id']=='M3_N1024_s42_e0.003') and r['mode']=='performance' and r['warmup_iters']==20 and r['method'] in ('sk_log_fast','mirror_sr1','sk_scaling_fast')]
    for case_id in dict.fromkeys(r['case_id'] for r in scale):
        for method in ('sk_log_fast','mirror_sr1','sk_scaling_fast'):
            group=[r for r in scale if r['case_id']==case_id and r['method']==method];c=Counter(r['status'] for r in group)
            if not group:continue
            time=median_range(r.get('total_sec') for r in group if r['status']=='success')['median'];rss=median_range(r.get('peak_sampled_rss_bytes') for r in group)['maximum']
            dimension=f'{group[0]["n"]}×{group[0]["m"]}'+('（仅归档）' if group[0].get('comparison_excluded_reason') else '')
            lines.append(f'| {dimension} | {method} | {c["success"]}/{len(group)} | {fmt(time)} | {fmt(rss/2**30 if rss else None)} | `{dict(c)}` |')
    lines+=['','M4 的 α=0.1 保留极小正质量，不用地板把困难边际改容易。M7 保留真实非方阵维度。N=5000 若资源跳过或超时，图中显示状态，不能据此外推更大规模或声称已验证 N=10000。'
            '各规模虽使用同一固定点池的前缀，仍按自身成本最大值归一化；因此规模曲线反映整套基准随 N 的变化，不能直接当作严格固定连续问题的复杂度定律。','',
            '## 两个有限消融','',
            '幅度消融使用 α=(eᵀV_SR1)/(eᵀψ)，实际方向 αψ 与原搜索规则，使用自己的真实历史；不复用 SR1 的接受步。'
            '比较 toy 两块/四块与 M3，不把 toy 放进普通主结果。预热消融为 0/5/20/50 步，每种策略计入自身预热；20 步复用主实验。','',
            '| 输入 | 方法 | 预热 | 成功/计划 | 成功总时间中位数(s) | 成功步数中位数 |','|---|---|---:|---:|---:|---:|']
    acases={r['case_id'] for r in rows if r['group'] in ('amplitude','warmup')}
    keys=dict.fromkeys((r['case_id'],r['method'],r['warmup_iters']) for r in rows if r['case_id'] in acases and r['mode']=='performance' and r['method'] in ('sk_log_fast','mirror_sr1','mirror_amp_ablation'))
    for case_id,method,warm in keys:
        group=[r for r in rows if r['case_id']==case_id and r['method']==method and r['warmup_iters']==warm and r['mode']=='performance'];good=[r for r in group if r['status']=='success']
        lines.append(f'| {case_id} | {method} | {warm} | {len(good)}/{len(group)} | {fmt(median_range(r.get("total_sec") for r in good)["median"])} | {fmt(median_range(r.get("n_iter") for r in good)["median"])} |')
    lines+=['','幅度消融的直接配对（时间比=幅度匹配 / 完整 SR1，大于 1 表示完整 SR1 更快）：']
    for case_id in dict.fromkeys(r['case_id'] for r in rows if r['method']=='mirror_amp_ablation'):
        values=[];amp_rejects=[];sr1_rejects=[];amp_evals=[];sr1_evals=[]
        for r in rows:
            if r['case_id']!=case_id or r['method']!='mirror_amp_ablation':continue
            ref=next((x for x in rows if x['case_id']==case_id and x['method']=='mirror_sr1' and x['repeat']==r['repeat'] and x['warmup_iters']==20 and x['mode']=='performance'),None)
            if ref and r['status']=='success' and ref['status']=='success':
                values.append(r['total_sec']/ref['total_sec'])
                amp_rejects.append(r.get('sr1_reject_count'));sr1_rejects.append(ref.get('sr1_reject_count'))
                amp_evals.append(r.get('state_eval_count'));sr1_evals.append(ref.get('state_eval_count'))
        st=median_range(values)
        lines.append(f'- `{case_id}`：{len(values)} 个成功配对，中位数 {fmt(st["median"])}，范围 {fmt(st["minimum"])}–{fmt(st["maximum"])}。'
            f'这些配对的拒绝试探中位数（幅度/完整 SR1）为 {fmt(median_range(amp_rejects)["median"])}/{fmt(median_range(sr1_rejects)["median"])}，'
            f'状态评估中位数为 {fmt(median_range(amp_evals)["median"])}/{fmt(median_range(sr1_evals)["median"])}。')
    lines+=['','预热敏感性（仅在成功轨道中描述，须结合上表失败覆盖）：']
    for case_id in dict.fromkeys(r['case_id'] for r in rows if r['group']=='warmup'):
        medians={warm:median_range(r.get('total_sec') for r in rows if r['case_id']==case_id and r['method']=='mirror_sr1' and r['warmup_iters']==warm and r['mode']=='performance' and r['status']=='success')['median'] for warm in (0,5,20,50)}
        finite={k:v for k,v in medians.items() if v is not None}
        if finite:
            best=min(finite,key=finite.get);worst=max(finite,key=finite.get)
            lines.append(f'- `{case_id}`：已成功策略中预热 {best} 步的中位时间最低；最慢/最快中位时间比为 {finite[worst]/finite[best]:.3g}。这不是在未测试长度上的最优性结论。')
    lines+=['','上述消融只能支持这些固定输入、有限预热长度和预算下的比较。方向与幅度在非线性轨道中相互作用，不能由一次时间比断言唯一机制。','',
            '## 失败、诊断与适用边界','',
            f'全计划状态：`{dict(counts)}`。资源跳过、数据阻塞、超时、max_iter、library_stop、数值失败分开保存。'
            'failures.csv 包括最后残差和真实停止时间；pending.csv 列出未结束条目；all_attempts.csv 保留旧尝试。','']
    diagnostic=[a for a in audit['tracks'] if a.get('diagnostic_replay')]
    lines.append(f'诊断重放 {len(diagnostic)} 条：接受轨道前缀一致 {sum(a["diagnostic_replay"]["matching_prefix"] for a in diagnostic)} 条，完整长度一致 {sum(a["diagnostic_replay"]["same_length"] for a in diagnostic)} 条。诊断若因限时较短，只能声称共同前缀一致。')
    sr=[r for r in rows if r['method']=='mirror_sr1' and r['mode']=='performance']
    lines+=['',f'SR1 性能轨道累计尝试 {sum(r.get("sr1_attempt_count",0) or 0 for r in sr)}，合法 {sum(r.get("sr1_valid_count",0) or 0 for r in sr)}，接受 {sum(r.get("sr1_accept_count",0) or 0 for r in sr)}，'
            f'跳过 {sum(r.get("sr1_skip_count",0) or 0 for r in sr)}，拒绝试探 {sum(r.get("sr1_reject_count",0) or 0 for r in sr)}，回退 {sum(r.get("sr1_fallback_count",0) or 0 for r in sr)}。'
            '每条轨道的正分母跳过原因、步长分布和 G/Q/cos 中位数在 runs.csv；回退并不自动等于求解失败。','',
            '## 与 v0 的关系','',
            'v0 的理想化弱耦合 toy 继续作为机制例子。本轮保持数学方法，换成轻量主基线和分别计时的诊断。旧版完整诊断计时与本轮性能计时不能直接相除；不能用 toy 的巨大比值证明普通问题普遍加速。','',
            '## 验证与复现','',
            '开发前基线 92 项测试通过；包含报告、暂停恢复、用户取消与 Windows 文件占用重试的全套回归 112 项通过（OT Python 3.9）。五类 smoke 的 20 条轨道：19 成功、1 L-BFGS library_stop，所有有效终点独立审计通过。',
            f'本目录回读 {audit["persisted_artifacts"]} 份轨道产物；指纹、成功判据、时间、计数检查全部通过：{audit["all_artifact_checks_passed"]}。'
            f'是否重新从保存的势与重建输入分块计算终点：{audit["readback_recomputed"]}。详细逐轨道结果及产物 SHA256 见 independent_audit.json。','',
            f'结果目录：`{root}`。完整输入描述、源码指纹、硬件/BLAS 版本、固定图像配对、预算及调用记录均在目录中。','',
            '```powershell',
            '.\\scripts\\run_ot.cmd -m pytest -q -p no:cacheprovider',
            '.\\scripts\\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_smoke.yaml',
            '.\\scripts\\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_main.yaml',
            '.\\scripts\\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_mnist.yaml',
            '.\\scripts\\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_scale.yaml',
            '.\\scripts\\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_baselines.yaml',
            '.\\scripts\\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_ablation.yaml',
            f'.\\scripts\\run_ot.cmd scripts/run_sr1_mainstream.py --config configs/sr1_v1_main.yaml --resume "{root}" --continue-paused',
            f'.\\scripts\\run_ot.cmd scripts/report_sr1_mainstream.py --run-dir "{root}" --audit --report docs/sr1_v1_mainstream_report.md',
            '```','',
            '所有正式子配置共享同一账本，重复调用跳过完整有效结果。中断轨道新增 attempt；已保存失败若要重试需 --retry-failures，但仍消耗同一预算。'
            '用户主动暂停写入 pause.json，只有明确使用 --continue-paused 才会恢复；运行中可通过 pause_request.json 的 after_job_id 在完整轨道边界暂停。'
            '源码或配置不匹配需 --new-revision，新 revision 不重置累计 6 小时预算。达到预算后恢复不会凭空获得新预算。'
            '资源跳过不是 pending；可在资源允许时显式重试，未完成范围仍须保留阶段结果标记。','',
            '## 中期答辩四页提纲','',
            '1. **标准合成**：展示 M1–M3 的 ε 趋势与含失败覆盖的时间比分布。先解释相同成本/初值/残差口径，再给按族收益及 speedup<1 反例，说明每步开销。',
            '2. **图像数据**：展示固定十对 MNIST 缩略图、真实支持数与两种 ε 的 M5 结果；明确是像素质量运输，不是图像特征或分类任务。',
            '3. **规模与边际**：展示 M6 时间/RSS、M4 极小质量、M7 非方阵；N=5000 状态和内存限制必须随图说明，不填补未完成点。',
            '4. **消融与边界**：幅度匹配和 0/5/20/50 预热图，配合合法/接受/回退计数；区分观测事实、机制假设和待补范围。','',
            '图像均有 PNG/PDF 双格式，见 figures；CSV 表是所有数字的来源。']
    figure_titles=[('speedups_synthetic','普通合成：成功配对时间比与完整覆盖'),
        ('speedups_mnist','MNIST：成功配对时间比与完整覆盖'),
        ('epsilon_trends','固定输入的 epsilon 趋势'),('scale_time_memory','规模与进程内存'),
        ('amplitude_ablation','幅度匹配消融'),('warmup_ablation','预热长度消融'),
        ('mnist_fixed_pairs','固定十对 MNIST 输入及有效支持')]
    available=[(name,title) for name,title in figure_titles if (root/'figures'/(name+'.png')).exists()]
    if manifest.get('operator_policy',{}).get('excluded_cases'):
        lines[4:4]=['按用户后续要求，N=5000 在 SR1 资源检查未通过后停止剩余方法比较；已完成的单方法结果仅归档，不纳入配对收益。用户取消与算法失败分开计数。','']
        lines+=['','## 用户调整的比较范围','',
            '用户要求：若 SR1 无法在 N=5000 上运行成功，则该规模其他方法不再继续。由于 SR1 的工作区估计超过内存上限，已取消该规模剩余轨道；'
            '先前完成的 SK 轨道仅作归档，保留原始成功状态和实际时间，不纳入配对加速结论，也不在规模时间趋势中绘制有效比较点。'
            f'本轮共有 {counts["skipped_by_user"]} 条用户取消记录，独立列为 skipped_by_user，不计作算法失败或未响应的 pending。具体记录见 user_exclusions.csv 与 operator_policy.json。','']
    if available:
        lines+=['','## 数据图','']
        for name,title in available:
            lines += [f'![{title}](figures/{name}.png)','',f'[{title} PDF](figures/{name}.pdf)','']
    return '\n'.join(lines)+'\n'
