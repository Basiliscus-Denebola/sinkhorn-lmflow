"""SR1 reports from persisted artifacts only; never import or run a solver."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io.result_io import load_result
from ..io.config import save_json

THRESHOLDS = (1e-4,1e-6,1e-8)
METHODS = ('sinkhorn_reduced','mirror_sr1')


def _median_range(values):
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    return (float(np.median(v)),min(v),max(v)) if v else (None,None,None)


def _fmt(value):
    if value is None or not np.isfinite(value):
        return '—'
    return f'{value:.4g}'


def _save_figure(fig,path):
    fig.savefig(path.parent/(path.name+'.png'),dpi=150,bbox_inches='tight')
    fig.savefig(path.parent/(path.name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def _plot_case(root,case_id,tracks):
    colors = {'sinkhorn_reduced':'#3575ad','mirror_sr1':'#d36b23'}
    fig,axes = plt.subplots(2,2,figsize=(11,7.5))
    fig.suptitle(case_id+' | matched inputs, shared warmup')
    for name,result,meta in tracks:
        df=pd.DataFrame([asdict(x) for x in result.history])
        color=colors[name]; label=f"{name} r{meta['repeat']}"
        style=('-', '--', ':')[int(meta['repeat']) % 3]
        for ax,x,y in ((axes[0,0],'state_eval_count','residual_l1'),
                       (axes[0,1],'elapsed_sec','residual_l1'),
                       (axes[1,0],'post_elapsed_sec','residual_l1'),
                       (axes[1,1],'state_eval_count','energy_gap')):
            data=df[df.iteration>=meta.get('fork_iteration',20)] if x=='post_elapsed_sec' else df
            data=data[data[y].notna() & (data[y]>0)]
            ax.semilogy(data[x],data[y],color=color,alpha=.65,lw=1.2,label=label,linestyle=style,
                        marker='o' if y=='energy_gap' else None,ms=3)
            if not result.converged and len(data):
                last=data.iloc[-1]
                ax.plot(last[x],last[y],'rx',ms=8)
                ax.annotate(result.stop_reason,(last[x],last[y]),fontsize=7)
    for ax,title,xlab in zip(axes.flat,
        ('Marginal residual / state evaluations','Marginal residual / total time',
         'Marginal residual / continuation time','Known-solution energy (checkpoints)'),
        ('State evaluations','Total seconds','Continuation seconds','State evaluations')):
        ax.set_title(title,fontsize=10);ax.set_xlabel(xlab);ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Residual L1');axes[0,1].set_ylabel('Residual L1')
    axes[1,0].set_ylabel('Residual L1');axes[1,1].set_ylabel('Energy gap')
    if not any(r.energy_gap is not None for _,res,_ in tracks for r in res.history):
        axes[1,1].text(.5,.5,'No exact reference energy for transfer',transform=axes[1,1].transAxes,ha='center')
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig,root/'figures'/f'{case_id}_convergence')
    fig,axes=plt.subplots(2,2,figsize=(11,6.5));fig.suptitle(case_id+' | accepted-step diagnostics')
    for name,result,meta in tracks:
        if name!='mirror_sr1':continue
        df=pd.DataFrame([asdict(x) for x in result.history])
        for ax,key in zip(axes.flat,('sr1_G','sr1_Q','step_size','sr1_reason')):
            if key=='sr1_reason':
                fallback=df[df.phase=='sk_fallback']
                skipped=fallback[fallback.sr1_reason!='backtracks_exhausted']
                exhausted=fallback[fallback.sr1_reason=='backtracks_exhausted']
                ax.scatter(skipped.iteration,np.ones(len(skipped)),marker='x',s=12,color='#d36b23')
                ax.scatter(exhausted.iteration,np.full(len(exhausted),2),marker='x',s=12,color='red')
                ax.set_yticks([1,2],['skip -> SK','exhausted -> SK'])
            else:
                rows=df[df.phase=='sr1']
                ax.plot(rows.iteration,rows[key],marker='.',linestyle=('-', '--', ':')[int(meta['repeat']) % 3],
                        alpha=.5,ms=3,label=f"repeat {meta['repeat']}")
                if key=='sr1_G' and len(rows):ax.set_yscale('log')
            ax.set_title(key);ax.set_xlabel('Accepted to_iteration');ax.grid(alpha=.2)
    for ax in axes.flat:
        ax.set_xlim(min(meta.get('fork_iteration',0) for _,_,meta in tracks),
                    max(max((result.n_iter for name,result,_ in tracks if name=='mirror_sr1'),default=21),21))
    if any(name=='mirror_sr1' for name,_,_ in tracks):axes[0,0].legend(fontsize=8)
    fig.tight_layout();_save_figure(fig,root/'figures'/f'{case_id}_diagnostics')


def report_study(run_dir,make_plots=True):
    root=Path(run_dir)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    (root/'tables').mkdir(exist_ok=True);(root/'figures').mkdir(exist_ok=True)
    run_rows,threshold_rows,event_rows,paired_rows=[],[],[],[]
    plot_groups={}
    latest_thresholds=[]
    for pair in manifest['pairs']:
        for ai,attempt in enumerate(pair['attempts']):
            hits={}
            for name in METHODS:
                path=attempt.get('runs',{}).get(name)
                if path is None:
                    run_rows.append(dict(case_id=pair['case_id'],repeat=pair['repeat'],attempt=ai+1,
                                         solver=name,status='pending',converged=False))
                    continue
                result=load_result(root/path);meta=result.summary
                if (meta['problem_sha256']!=pair['problem_sha256'] or meta['initial_sha256']!=pair['initial_sha256']
                        or meta['warmup_sha256']!=attempt['warmup_sha256']):
                    raise ValueError('unmatched pairing fingerprints in saved results')
                base=dict(case_id=pair['case_id'],repeat=pair['repeat'],attempt=ai+1,solver=name,
                    initial_seed=pair['initial_seed'],data_seed=pair['data_seed'],problem_sha256=meta['problem_sha256'],
                    initial_sha256=meta['initial_sha256'],warmup_sha256=meta['warmup_sha256'],
                    warmup_protocol=meta['warmup_protocol'],tol=meta['tol'])
                run_rows.append(dict(base,status=result.stop_reason,converged=result.converged,n_iter=result.n_iter,
                    elapsed_sec=result.elapsed_sec,path=path,**{k:v for k,v in meta.items() if k not in base}))
                accepted=[r for r in result.history if r.phase=='sr1']
                attempts=meta.get('sr1_attempt_count',0)
                event_rows.append(dict(base,attempts=attempts,valid=meta.get('sr1_valid_count',0),
                    accepted=meta.get('sr1_accept_count',0),fallbacks=meta.get('sr1_fallback_count',0),
                    skips=meta.get('sr1_skip_count',0),exhausted=meta.get('sr1_exhausted_count',0),
                    rejected=meta.get('sr1_reject_count',0),accept_fraction=meta.get('sr1_accept_count',0)/attempts if attempts else None,
                    fallback_fraction=meta.get('sr1_fallback_count',0)/attempts if attempts else None,
                    G_median=_median_range([r.sr1_G for r in accepted])[0],Q_median=_median_range([r.sr1_Q for r in accepted])[0],
                    min_step=min((r.step_size for r in accepted),default=None),
                    skip_reasons=json.dumps(meta.get('sr1_skip_reasons',{}),sort_keys=True)))
                for threshold in THRESHOLDS:
                    hit=next((r for r in result.history if r.residual_l1<=threshold),None)
                    row=dict(base,threshold=threshold,success=hit is not None,terminal_status=result.stop_reason,
                        iteration=hit.iteration if hit else None,state_eval_count=hit.state_eval_count if hit else None,
                        objective_diff_count=hit.objective_diff_count if hit else None,bregman_count=hit.bregman_count if hit else None,
                        dense_pass_proxy=hit.dense_pass_proxy if hit else None,
                        total_sec=(0. if hit.iteration==0 else hit.elapsed_sec) if hit else None,
                        post_sec=hit.post_elapsed_sec if hit else None,
                        post_state_eval_count=(hit.state_eval_count-result.history[meta['fork_iteration']].state_eval_count
                            if hit.iteration>meta['fork_iteration'] else 0) if hit else None,
                        sr1_accept_fraction=hit.sr1_accept_count/hit.sr1_attempt_count if hit and hit.sr1_attempt_count else None,
                        sr1_fallback_fraction=hit.sr1_fallback_count/hit.sr1_attempt_count if hit and hit.sr1_attempt_count else None,
                        reached_during_warmup=bool(hit and hit.iteration<=meta['fork_iteration']))
                    threshold_rows.append(row);hits[(name,threshold)]=row
                    if ai==len(pair['attempts'])-1:latest_thresholds.append(row)
                if ai==len(pair['attempts'])-1:
                    group=(pair['case_id'],meta['problem_sha256'],meta['initial_sha256'],meta['warmup_protocol'],meta['tol'])
                    plot_groups.setdefault(group,[]).append((name,result,meta))
            for threshold in THRESHOLDS:
                sk, sr = hits.get((METHODS[0],threshold)),hits.get((METHODS[1],threshold))
                eligible=bool(sk and sr and sk['success'] and sr['success'] and not sk['reached_during_warmup'] and not sr['reached_during_warmup'])
                paired_rows.append(dict(case_id=pair['case_id'],repeat=pair['repeat'],attempt=ai+1,threshold=threshold,
                    eligible=eligible,speedup=sk['total_sec']/sr['total_sec'] if eligible and sr['total_sec']>0 else None,
                    post_speedup=sk['post_sec']/sr['post_sec'] if eligible and sr['post_sec']>0 else None,
                    eval_speedup=sk['state_eval_count']/sr['state_eval_count'] if eligible else None,
                    reason='matched_success' if eligible else 'missing_failed_or_warmup_threshold',
                    latest_attempt=ai==len(pair['attempts'])-1))
        if not pair['attempts']:
            for name in METHODS:
                run_rows.append(dict(case_id=pair['case_id'],repeat=pair['repeat'],solver=name,status='pending',converged=False))
    frames={name:pd.DataFrame(rows) for name,rows in [('all_runs',run_rows),('threshold_costs',threshold_rows),
            ('paired_speedups',paired_rows),('sr1_events',event_rows)]}
    for name,frame in frames.items():frame.to_csv(root/'tables'/f'{name}.csv',index=False)
    aggregate=[]
    latest=pd.DataFrame(latest_thresholds)
    if len(latest):
        for (case_id,name,threshold),group in latest.groupby(['case_id','solver','threshold']):
            row=dict(case_id=case_id,solver=name,threshold=threshold,attempted=len(group),
                     successes=int(group.success.sum()),success_rate=float(group.success.mean()))
            for key in ('iteration','state_eval_count','objective_diff_count','bregman_count','dense_pass_proxy',
                        'total_sec','post_sec','post_state_eval_count','sr1_accept_fraction','sr1_fallback_fraction'):
                median,low,high=_median_range(group[key])
                row.update({key+'_median':median,key+'_min':low,key+'_max':high})
            aggregate.append(row)
    agg=pd.DataFrame(aggregate);agg.to_csv(root/'tables'/'threshold_summary.csv',index=False)
    pair_statuses={s:sum(p['status']==s for p in manifest['pairs']) for s in ('completed','partial','pending')}
    save_json(dict(planned_pairs=len(manifest['pairs']),pair_statuses=pair_statuses,
                   saved_tracks=sum('path' in x for x in run_rows)),root/'tables'/'coverage.json')
    lines=['# SR1 v0 '+manifest['stage']+' 结果','',
        f"计划 {len(manifest['pairs'])} 个 case × repeat 配对；状态：{pair_statuses}。所有 attempt、失败和 pending 保留在 all_runs.csv。",'',
        '方法共用行归一化观测位置与 20 次 Sinkhorn 预热。受控种子仅改变初值；重复编号仅用于计时。',
        '时间包含方向、试探和诊断，排除保存与绘图；双方各承担一次共享预热。state_eval_count 包含已知参考态的一次诊断评估。',
        '稳定目标差每次单独计数；能量仅在初始、预热末、首次跨越阈值和 max_iter 检查点计算。稠密遍历代理不是 FLOPs。',
        'Bregman 核仅在数值评估中重新归一化行条件分布；实际 P、边际和停止容差不变。未使用 roundoff_residual 接受。','',
        '## 最终阈值 1e-8','',
        '| case | 方法 | 成功/已运行 | 迭代中位数 | 评估中位数 | 目标差核中位数 | 总秒中位数 [最小,最大] | 后期秒中位数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in aggregate:
        if row['threshold']!=1e-8:continue
        lines.append(f"| {row['case_id']} | {row['solver']} | {row['successes']}/{row['attempted']} | {_fmt(row['iteration_median'])} | {_fmt(row['state_eval_count_median'])} | {_fmt(row['objective_diff_count_median'])} | {_fmt(row['total_sec_median'])} [{_fmt(row['total_sec_min'])}, {_fmt(row['total_sec_max'])}] | {_fmt(row['post_sec_median'])} |")
    speeds=[x['speedup'] for x in paired_rows if x['threshold']==1e-8 and x['latest_attempt'] and x['eligible']]
    lines+=['',f"有效配对的最终总时间比 T_SK/T_SR1：中位数、最小、最大 = {tuple(_fmt(v) for v in _median_range(speeds))}。预热已达阈值或任一失败的配对不报告加速倍数。",'',
        '## 诊断与限制','',
        'G、Q 和夹角对应真正接受的 SR1 步，记录在终点行，from_iteration 指向起点。回退行不复用已拒绝方向的预测值。',
        '有效方向的同点耗散关系不是整条离散轨道加速定理；实际收益须同时看评估数、额外目标差核和时间。',
        '所有阈值均取同一轨道首次接受状态；初始已达阈值记零时间。缺失值不是零成本。',
        '图中红叉标记非收敛终点；能量图仅绘制有效正值，不裁剪非正能量。完整值在 history.csv。','',
        '[全部运行](tables/all_runs.csv) · [阈值成本](tables/threshold_costs.csv) · [中位数及范围](tables/threshold_summary.csv) · [配对比值](tables/paired_speedups.csv) · [保护统计](tables/sr1_events.csv)','',
        '## 图表','']
    current_events=[x for x in event_rows if x['solver']=='mirror_sr1' and
                    any(p['case_id']==x['case_id'] and p['repeat']==x['repeat'] and
                        len(p['attempts'])==x['attempt'] for p in manifest['pairs'])]
    totals={key:sum(row[key] for row in current_events) for key in
            ('attempts','valid','accepted','fallbacks','skips','exhausted','rejected')}
    diagnostic_lines=['## 数值诊断摘要','',f'最新 attempt 的 SR1 事件汇总：`{totals}`。','']
    if totals['attempts']:
        diagnostic_lines += [f"合法方向比例 {totals['valid']/totals['attempts']:.2%}，接受比例 {totals['accepted']/totals['attempts']:.2%}，"
                             f"SK 回退比例 {totals['fallbacks']/totals['attempts']:.2%}。",'']
    reasons={}
    for row in current_events:
        for key,value in json.loads(row['skip_reasons']).items():reasons[key]=reasons.get(key,0)+value
    diagnostic_lines += [f'跳过原因合计：`{reasons}`。','',
        '| case | 接受步 G 中位数的中位数 | 接受步 Q 中位数的中位数 | 最小接受步长 |',
        '|---|---:|---:|---:|']
    for case in sorted({x['case_id'] for x in current_events}):
        rows=[x for x in current_events if x['case_id']==case]
        diagnostic_lines.append(f"| {case} | {_fmt(_median_range([x['G_median'] for x in rows])[0])} | {_fmt(_median_range([x['Q_median'] for x in rows])[0])} | {_fmt(_median_range([x['min_step'] for x in rows])[1])} |")
    diagnostic_lines+=['','以上 G/Q 只统计接受的 SR1 步；跳过或耗尽后的 Sinkhorn 步不使用失败方向的预测量。',
        '若 G 大而 Q 小、步长缩小，说明同点增益未完全转化为有限步下降；最终成本比判断额外工作是否抵消收益。','']
    insert_at=lines.index('## 图表')
    lines[insert_at:insert_at]=diagnostic_lines
    if make_plots:
        for group,tracks in plot_groups.items():
            _plot_case(root,group[0],tracks)
            lines += [f"### {group[0]}",f"![收敛](figures/{group[0]}_convergence.png)",f"![诊断](figures/{group[0]}_diagnostics.png)",'']
    lines+=['## 复现与恢复','',
        f'` .\\scripts\\run_ot.cmd scripts/run_sr1_study.py --config "{root}/study_config.yaml" `','',
        f'`{manifest["resume_command"]}`','',
        f'`.\\scripts\\run_ot.cmd scripts/report_sr1.py --run-dir "{root}"`','',
        '恢复检查配置、数学输入、初值指纹，跳过有完整产物的成功配对；未完成或失败配对新增 attempt，保留旧结果。',
        '初值目录使用 SHA256 前 16 位防止 Windows 长路径，metadata 保存并核对完整 SHA256；不同问题的完整初始势分开保存。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return root/'report.md'
