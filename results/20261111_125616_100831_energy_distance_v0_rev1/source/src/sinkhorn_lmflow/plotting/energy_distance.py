"""Tables, independent endpoint re-audit and figures from persisted study artifacts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ..experiments.sr1_mainstream import atomic_json, audit_endpoint
from ..experiments.energy_distance_study import REPRESENTATIVES, METHODS
from ..io.config import load_config
from ..io.result_io import load_result
from ..io.problem_io import problem_fingerprint
from ..problems.instance import OTProblem
from ..problems.mainstream import rng


def restored_input(root, cid):
    path = root/'inputs'/cid
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    with np.load(path/'reconstruction.npz', allow_pickle=False) as arrays:
        if meta['family'] == 'M1':
            M = rng(meta['data_seed'], 1).uniform(0,1,(meta['n'],meta['m'])) / meta['raw_cost_max']
        else:
            M = cdist(arrays['x'],arrays['y'],metric='sqeuclidean') / meta['raw_cost_max']
        problem = OTProblem(arrays['a'], arrays['b'], M, meta['epsilon'], cid,
                            {k:v for k,v in meta.items() if k!='input_sha256'})
    if problem_fingerprint(problem) != meta['input_sha256']:
        raise ValueError('reconstructed input fingerprint mismatch: '+cid)
    return problem


def _markdown(frame, columns=None):
    if frame.empty:
        return '暂无已完成且可比较的数据。'
    if columns:
        frame = frame[columns]
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.4g}' if np.isfinite(value) else '—'
        return str(value).replace('|','\\|')
    lines = ['| '+' | '.join(map(str, frame.columns))+' |',
             '| '+' | '.join('---' for _ in frame.columns)+' |']
    lines += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def _figsave(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder/(name+'.png'), dpi=145)
    fig.savefig(folder/(name+'.pdf'))
    plt.close(fig)


def generate_report(run_dir, report_path, audit=False, ledger=None):
    start = perf_counter()
    root = Path(run_dir).resolve()
    report_path = Path(report_path).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    config = load_config(root/'config_resolved.yaml')
    selected = json.loads((root/'selected_kappa.json').read_text(encoding='utf-8'))
    rows, thresholds, audits, histories = [], [], {}, {}
    input_cache = {}
    for job in manifest['jobs']:
        row = dict(job)
        row.pop('artifact_sha256', None)
        if job.get('run_path'):
            result = load_result(root/job['run_path'])
            row.update({k:v for k,v in result.summary.items()
                        if k not in ('direction_diagnostics', 'accepted_energy_trace')})
            row.update(total_sec=result.elapsed_sec, n_iter=result.n_iter, stop_reason=result.stop_reason)
            trace = result.summary.get('accepted_energy_trace', [])
            initial_matvec = trace[0]['energy_matvec_sec'] if trace and trace[0]['energy_matvec_count']==1 else 0.
            row['iteration_energy_matvec_sec'] = max(0., result.summary.get('energy_matvec_sec',0.)-initial_matvec)
            endpoint = result.summary['endpoint_audit']
            if audit and result.summary.get('endpoint_valid',True):
                cid = job['case_id']
                # Only one input in memory; immutable persisted reconstruction.
                if cid not in input_cache:
                    input_cache = {cid: restored_input(root,cid)}
                endpoint = audit_endpoint(input_cache[cid],result)
                if endpoint['passed'] is False:
                    row['status'] = 'audit_failure'
            audits[job['job_id']] = endpoint
            if job['case_id'] in REPRESENTATIVES and job['repeat']==0 and job['stage']!='calibration':
                histories[job['job_id']] = (job, result)
            for tol in (1e-4,1e-6,1e-8):
                hit = next((h for h in result.history if h.residual_l1 <= tol),None)
                thresholds.append(dict(job_id=job['job_id'],case_id=job['case_id'],method=job['method'],
                    stage=job['stage'],repeat=job['repeat'],tol=tol,status=row['status'],
                    first_hit_sec=hit.elapsed_sec if hit else None,
                    first_hit_iteration=hit.iteration if hit else None,
                    warmup_hit=bool(hit and hit.phase in ('warmup','initial')),
                    endpoint_audit_passed=endpoint['passed']))
        else:
            case = next(c for c in manifest['cases'] if c['case_id']==job['case_id'])
            row.update(case)
        # CSV cells preserve structured diagnostics without Python repr.
        rows.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'runs.csv',index=False)
    frame[frame.stage=='calibration'].to_csv(root/'calibration.csv',index=False)
    pd.DataFrame(thresholds).to_csv(root/'threshold_costs.csv',index=False)
    pending = frame[frame.status=='pending_budget']
    pending.to_csv(root/'pending.csv',index=False)
    frame[~frame.status.isin(['success','pending_budget'])].to_csv(root/'failures.csv',index=False)
    coverage = frame.groupby(['stage','family','method','status'],dropna=False).size().reset_index(name='count')
    coverage.to_csv(root/'coverage.csv',index=False)
    atomic_json(audits,root/'endpoint_audit.json')
    main = frame[frame.stage=='main']
    comparisons = []
    for _,new in main[(main.method=='energy_distance') & (main.status=='success')].iterrows():
        for method in METHODS[:2]:
            baseline = main[(main.case_id==new.case_id)&(main['repeat']==new['repeat'])&(main.method==method)&(main.status=='success')]
            if not baseline.empty:
                b = baseline.iloc[0]
                comparisons.append(dict(case_id=new.case_id,family=new.family,epsilon=new.epsilon,
                    repeat=new['repeat'],baseline=method,speedup=b.total_sec/new.total_sec,
                    baseline_sec=b.total_sec,new_sec=new.total_sec,
                    warmup_converged=bool(new.warmup_converged)))
    ratios = pd.DataFrame(comparisons,columns=['case_id','family','epsilon','repeat','baseline','speedup',
                                             'baseline_sec','new_sec','warmup_converged'])
    ratios.to_csv(root/'paired_speedups.csv',index=False)
    ratio_summary = (ratios.groupby(['family','epsilon','baseline']).agg(pairs=('speedup','size'),
        median_speedup=('speedup','median'),min_speedup=('speedup','min'),max_speedup=('speedup','max')).reset_index()
        if len(ratios) else pd.DataFrame())
    energy = main[main.method=='energy_distance']
    successful = energy[energy.status=='success']
    unique_cases = main.case_id.nunique()
    complete_cases = sum(len(g[g.status=='success'])==6 for _,g in main.groupby('case_id'))
    touched_cases = main[main.status!='pending_budget'].case_id.nunique()
    snapshot = ledger.data if ledger else json.loads((root/'budget_snapshot.json').read_text(encoding='utf-8')) if (root/'budget_snapshot.json').exists() else {}
    machine = ledger.used() if ledger else snapshot.get('used_sec',0.)
    counts = dict(Counter(frame.status))
    evidence = []
    for method in METHODS[:2]:
        r = ratios[ratios.baseline==method]
        if len(r):
            evidence.append(f'相对 {method}：{len(r)} 对验收成功的配对计时，总时间比中位数 {r.speedup.median():.3g}，范围 {r.speedup.min():.3g}–{r.speedup.max():.3g}；大于 1 表示新方法更快。')
        else:
            evidence.append(f'相对 {method}：暂无双方验收成功的完成时间配对，不能判断加速。')
    costs = []
    for field in ('distance_build_sec','energy_matvec_sec','objective_diff_sec','trial_state_sec','rejected_trials','fallback_count'):
        if field in successful and successful[field].notna().any():
            costs.append(f'{field} 中位数 {successful[field].median():.3g}')
    gamma_values = pd.to_numeric(energy.get('gamma',pd.Series(dtype=float)),errors='coerce').dropna()
    gamma_text = (f'gamma：最小 {gamma_values.min():.4g}，中位 {gamma_values.median():.4g}，最大 {gamma_values.max():.4g}；零值 {int(sum(gamma_values==0))} 条。' if len(gamma_values) else '暂无 gamma 数据。')
    worst = ratios.sort_values('speedup').head(12)
    ablation = frame[frame.stage=='ablation']
    ablation_columns = [c for c in ('case_id','method','status','total_sec','n_iter','gamma','fallback_count') if c in frame]
    direction_rows = []
    for job,result in histories.values():
        diag = result.summary.get('direction_diagnostics',[])
        cos = [d['cos'] for d in diag if d.get('cos') is not None]
        q = [d['q'] for d in diag if d.get('q') is not None]
        if cos:
            direction_rows.append(dict(case_id=job['case_id'],method=job['method'],
                median_cos=np.median(cos),min_cos=min(cos),median_q=np.median(q) if q else np.nan,
                fallbacks=result.summary.get('fallback_count',0)))
    directions = pd.DataFrame(direction_rows)
    directions.to_csv(root/'direction_summary.csv',index=False)
    text = f'''# 完整成本距离驱动 Sinkhorn：v0 实测报告

本轮实际计划 {len(frame)} 条，状态统计：{json.dumps(counts,ensure_ascii=False)}。正式输入覆盖 {touched_cases}/{unique_cases}；三方法两次均成功的输入 {complete_cases}/{unique_cases}。机器累计时间约 {machine:.1f} 秒（报告生成期间仍计费）；未完成项保留在 pending.csv。

{chr(10).join(evidence)}

开销观察：{'；'.join(costs) if costs else '暂无足够完成数据'}。接受步减少并不自动意味着总墙钟收益；下方按家族配对表包含本轮重新计时的两种 SK。目标差、D@e 是 iteration_sec 的子项，不能再次加到总时间。

方向是否改变及是否有用：见方向诊断和六例消融。cos 越接近 1，方向越接近原对数残差；幅度匹配与完整方向各走独立轨道。有限消融只有一次计时，不能据此宣称统计显著性。若消融未完成，机制问题仍未回答。

## 固定协议与参数

统一 kappa={selected['kappa']}，选择原因 `{selected['reason']}`。{gamma_text}

使用 OT Python 3.9、串行、BLAS 线程 1；环境详见 environment.json。tol=1e-8，max_iter=50000，最多 20 步共享预热。log SK 和新方法各自总时间都含预热；原始缩放 SK 冷启动并计入核构造。M5 按真实非零灰度支持计算，尺寸见 inputs/*/metadata.json 与 runs.csv。没有运行 5000 或 10000 求解。

测量实现指纹：`{manifest.get('measurement_source_sha256',manifest['source_sha256'])}`。当前报告/入口修订指纹：`{manifest['source_sha256']}`。配置指纹：`{manifest['config_sha256']}`。起点提交由 environment.json 记录；任务开始时为 61af801ed0b89a0c2b6c23ac1a29382f02a3e886，用户已有 notes 改动保留。代码快照在 source/。报告修订来源：`{manifest.get('inherited_measurements_from','本目录原始测量')}`；报告层修订不产生新求解轨道，不重置累计预算。

## 覆盖与失败

{_markdown(coverage)}

not_applicable_kernel_underflow 是原始缩放表示不适用，不当作收敛失败。timeout、资源跳过、预算未启动均保留在计划分母，不用截止时间伪装成功。终点依据保存的 mu、nu 独立分块重算；审计详见 endpoint_audit.json。

## 各家族、epsilon 的配对总时间比

{_markdown(ratio_summary)}

同一输入的两次计时不是独立数据样本；不报告由此生成的置信区间。预热内收敛的收益不归因于新驱动，配对表有 warmup_converged 标志。

## 最差反例

{_markdown(worst,['case_id','repeat','baseline','speedup','baseline_sec','new_sec'])}

## 有限消融与方向诊断

{_markdown(ablation,ablation_columns)}

{_markdown(directions)}

## 理论边界

Energy distance 是已有构造。本方法是由成本几何构造的预条件梯度方法，保持 EOT 镜像几何与原平衡点，并提供连续下降方向；离散加速没有先验证明。固定成本距离不移除实际质量响应 L_P，也不保证跨越所有弱连接。使用距离而非平方距离，不构造 Hessian、谱分解或近似几何。结果仅覆盖至 2048，不能外推更大规模。

## 产物与恢复

结果目录：`{root}`。

CSV：calibration.csv、runs.csv、threshold_costs.csv、coverage.csv、failures.csv、pending.csv、paired_speedups.csv、direction_summary.csv。

恢复执行：`scripts\\run_ot.cmd scripts/run_energy_distance_study.py --config configs/energy_distance_v0.yaml`。同一账本累计预算，匹配终态不重跑；实现变更必须使用 --new-revision，仍保留累计账本。预算已耗尽时恢复不会增加求解额度。

只重建报告：`scripts\\run_ot.cmd scripts/report_energy_distance.py --run-dir "{root}" --audit --report docs/energy_distance_v0_report.md`。

单独执行报告命令的机器开销记录到 report_rebuilds.json，不作为新增正式求解时间。首次自动报告包含在实验账本；没有复用旧 SR1 结果或账本。
'''
    # Minimal deliverable first: a plot error or reserve exhaustion cannot erase
    # the CSVs, coverage and Chinese conclusions.
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(text,encoding='utf-8')
    (root/'report.md').write_text(text,encoding='utf-8')
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    if ledger is None or ledger.remaining('report') > 0:
        plot_results(folder,main,ratios,histories,frame)
        figure_text = '\n## 图表\n\n'+'\n\n'.join(f'![{name}]({(folder/(name+".png")).as_posix()})'
                         for name in ('performance','convergence','costs','mechanisms','scale_memory'))+'\n'
        text += figure_text
    elapsed = perf_counter()-start
    if ledger:
        ledger.checkpoint()
        text += f'\n本次自动报告耗时 {elapsed:.2f} 秒；报告完成时机器累计 {ledger.used():.2f} 秒。\n'
    else:
        p = root/'report_rebuilds.json'
        rebuilds = json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
        rebuilds.append(dict(elapsed_sec=elapsed,audit=audit))
        atomic_json(rebuilds,p)
    report_path.write_text(text,encoding='utf-8')
    (root/'report.md').write_text(text,encoding='utf-8')
    resume = report_path.with_name('energy_distance_v0_resume.md')
    resume.write_text(f'# v0 恢复说明\n\n结果：`{root}`\n\n账本：`{config["budget_ledger"]}`\n\n'
        '重复主命令将校验实现、配置、输入、预热与产物指纹，并跳过已验证终态；不重置预算。\n\n'
        '实现修改须 --new-revision：新目录、新指纹、同一累计账本，旧产物保留。\n\n'
        '校准开始主比较后冻结，校准不完整则 kappa=1。进程意外中断按最近每秒检查点恢复，并保守追加 1 秒；关闭期间不计机器时间。\n\n'
        '运行中单核允许有限越界；预算耗尽不再启动轨道。timeout 保留最后有效端点，属于本轮终态，不隐式追加预算。\n\n'
        f'剩余未启动条目：{len(pending)}。查看 pending.csv、failures.csv 和 manifest.json。\n',encoding='utf-8')
    print(f'REPORT={report_path} report_sec={elapsed:.2f}',flush=True)
    return frame


def plot_results(folder,main,ratios,histories,frame):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax,mnist in zip(axes,(False,True)):
        sub = ratios[(ratios.family=='M5')==mnist]
        for method in METHODS[:2]:
            data = sub[sub.baseline==method].reset_index(drop=True)
            ax.scatter(np.arange(len(data)),data.speedup,label=method,s=18,alpha=.7)
        ax.axhline(1,color='black',lw=.8)
        ax.set_yscale('log');ax.set_ylabel('baseline / energy total time')
        subset = main[(main.family=='M5')==mnist]
        ax.set_title(('MNIST' if mnist else 'Synthetic')+f': successful {sum(subset.status=="success")}/{len(subset)} planned')
        ax.set_xlabel('paired completed timing');ax.legend(fontsize=8)
    _figsave(fig,folder,'performance')
    fig,axes = plt.subplots(6,2,figsize=(12,18))
    colors = dict(sk_log_fast='C0',sk_scaling_fast='C1',energy_distance='C2',energy_zero='C3',energy_amp='C4')
    for i,cid in enumerate(REPRESENTATIVES):
        for job,result in histories.values():
            if job['case_id']!=cid or job['stage']!='main' or not result.history:
                continue
            for col,attr in enumerate(('elapsed_sec','iteration')):
                axes[i,col].semilogy([getattr(h,attr) for h in result.history],
                    [max(h.residual_l1,1e-18) for h in result.history],label=job['method'],color=colors[job['method']])
        for col in range(2):
            axes[i,col].set_title(cid,fontsize=9);axes[i,col].set_xlabel('total seconds' if col==0 else 'accepted iteration')
            axes[i,col].set_ylabel('marginal L1');axes[i,col].axhline(1e-8,color='gray',lw=.5)
            if axes[i,col].get_legend_handles_labels()[0]:axes[i,col].legend(fontsize=7)
    _figsave(fig,folder,'convergence')
    fig,ax=plt.subplots(figsize=(12,5))
    data=main[(main.method=='energy_distance')&(main.repeat==0)&main.case_id.isin(REPRESENTATIVES)&main.total_sec.notna()].copy() if 'total_sec' in main else pd.DataFrame()
    if len(data):
        x=np.arange(len(data));bottom=np.zeros(len(data))
        parts=[('warmup_sec','warmup'),('distance_build_sec','distance'),('gamma_setup_sec','gamma'),
               ('bridge_setup_sec','bridge'),('objective_diff_sec','stable objective trials'),
               ('iteration_energy_matvec_sec','iteration D matvec')]
        for field,label in parts:
            vals=data.get(field,pd.Series(0.,index=data.index)).fillna(0).to_numpy()
            ax.bar(x,vals,bottom=bottom,label=label);bottom+=vals
        ax.bar(x,np.maximum(0,data.total_sec.to_numpy()-bottom),bottom=bottom,label='other iteration/export')
        ax.set_xticks(x);ax.set_xticklabels(data.case_id,rotation=25,ha='right',fontsize=8);ax.legend(fontsize=8)
    ax.set_ylabel('seconds (exclusive components)');ax.set_title('Full-driver total cost, repeat 0')
    _figsave(fig,folder,'costs')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for job,result in histories.values():
        if not job['method'].startswith('energy'):continue
        label=job['case_id']+'/'+job['method']
        if job['case_id'] in REPRESENTATIVES[:3]:
            axes[0,0].semilogy([h.elapsed_sec for h in result.history],[max(h.residual_l1,1e-18) for h in result.history],label=label,alpha=.7)
        diag=result.summary.get('direction_diagnostics',[])
        if job['method']=='energy_distance':
            axes[0,1].plot([d['iteration'] for d in diag],[np.nan if d['cos'] is None else d['cos'] for d in diag],label=job['case_id'])
            h=[h for h in result.history if h.phase=='energy_distance']
            axes[1,0].plot([r.iteration for r in h],[r.step_size for r in h],alpha=.65)
            axes[1,1].plot([r.iteration for r in h],[r.safeguard_count for r in h],alpha=.65)
    for ax,title in zip(axes.flat,('Ablation trajectories (first 3 cases)','Full vs log direction cosine','Accepted step size','Cumulative fallback count')):
        ax.set_title(title,fontsize=10);ax.set_xlabel('total seconds' if ax is axes[0,0] else 'iteration')
    if axes[0,0].lines:axes[0,0].legend(fontsize=5)
    if axes[0,1].lines:axes[0,1].legend(fontsize=6)
    _figsave(fig,folder,'mechanisms')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    scale_ids=['M6_N512_s42_e0.003','M3_N1024_s42_e0.003','M6_N2048_s42_e0.003','M7_1024x2048_s42_e0.003']
    for method in METHODS:
        data=main[(main.method==method)&main.case_id.isin(scale_ids)&(main.status=='success')]
        for col,field in enumerate(('total_sec','process_peak_rss')):
            vals=[data[data.case_id==cid][field].median() if field in data and len(data[data.case_id==cid]) else np.nan for cid in scale_ids]
            if col:vals=np.asarray(vals)/1024**2
            axes[col].plot(range(4),vals,'o-',label=method)
    for ax in axes:
        ax.set_xticks(range(4));ax.set_xticklabels(['512','1024','2048','1024x2048']);ax.legend(fontsize=8)
    axes[0].set_ylabel('completed total seconds');axes[1].set_ylabel('sampled process peak RSS (MiB)')
    _figsave(fig,folder,'scale_memory')
