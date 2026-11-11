"""Persist final interpretation and disclosure, without changing measured data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import pandas as pd
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.experiments.energy_distance_study import BudgetLedger, REPRESENTATIVES
from sinkhorn_lmflow.experiments.sr1_mainstream import ledger_lock, atomic_json
from sinkhorn_lmflow.plotting.energy_distance import _markdown

base=Path(__file__).resolve().parents[1]
cfg=load_config(base/'configs/energy_distance_v0.yaml')
with ledger_lock(base/cfg['budget_ledger']):
    ledger=BudgetLedger(base/cfg['budget_ledger'],cfg)
    try:
        ledger.stage('report')
        root=Path(ledger.data['active_run_dir'])
        runs=pd.read_csv(root/'runs.csv')
        pairs=pd.read_csv(root/'paired_speedups.csv')
        mechanism=[]
        for cid in REPRESENTATIVES:
            main=runs[(runs.case_id==cid)&(runs.stage=='main')&(runs['repeat']==0)]
            full=main[main.method=='energy_distance'].iloc[0]
            log=main[main.method=='sk_log_fast'].iloc[0]
            ab=runs[(runs.case_id==cid)&(runs.stage=='ablation')]
            zero=ab[ab.method=='energy_zero'].iloc[0]
            amp=ab[ab.method=='energy_amp'].iloc[0]
            mechanism.append(dict(case_id=cid,log_sec=log.total_sec,full_sec=full.total_sec,
                zero_sec=zero.total_sec,amp_sec=amp.total_sec,amp_over_full=amp.total_sec/full.total_sec,
                log_updates=log.n_iter,full_updates=full.n_iter,
                stable_difference_sec=full.objective_diff_sec,rejected_trials=full.rejected_trials))
        mechanism=pd.DataFrame(mechanism)
        mechanism.to_csv(root/'mechanism_comparison.csv',index=False)
        intro='''**结论：完整成本距离驱动相对 log SK 在多数几何输入上获得了总墙钟收益，但没有胜过任何可用的原始缩放 SK。** 78 对 log SK 配对中 64 对更快（中位时间比 1.48）；66 对可用缩放 SK 配对中 0 对更快（中位比 0.209）。这些是两次重复计时的描述统计，不是独立样本推断。

按家族看，M2、M3、M4 相对 log SK 的中位时间比分别为 1.63、1.39、1.51；MNIST 为 1.58（38/40 对更快）。M1 全部变慢，中位比 0.510；2048 规模与 1024×2048 非方阵也变慢。不能把 512、1024 的收益外推至更大规模。

主要开销是稳定目标差及试探，而非构造 D 或 D@e。例如 M1(seed=42, epsilon=0.001, repeat=0) 只有小幅接受步减少，30.83 秒总时间却有 22.16 秒用于稳定目标差；509 次试探中拒绝 304 次，D 构造加全部 D@e 不到 0.12 秒。2048 首轮的 101.95 秒中，稳定目标差占 71.25 秒，超过对应 log SK 的总成本所能抵消的范围。

六个代表例的方向余弦中位数为 0.821–0.924，方向确实发生了变化；几何代表例中完整方法均快于幅度匹配，随机成本代表例则反而更慢。有限消融支持“方向变化在部分几何问题上有用”，不支持普遍有效或统计显著的结论。正式完整方法没有触发 SK 回退，但随机成本有大量拒绝试探。

'''
        path=base/'docs/energy_distance_v0_report.md'
        text=path.read_text(encoding='utf-8')
        heading,body=text.split('\n',1)
        # Idempotent supplement: preserve generated body, replace prior analysis.
        body=body[body.index('本轮实际计划'):].split('\n## 六例完整方法与消融的直接比较')[0]
        text=heading+'\n\n'+intro+body.lstrip()
        text=text.replace('目标差、D@e 是 iteration_sec 的子项，不能再次加到总时间。',
            '目标差属于 iteration_sec；energy_matvec_sec 包含 gamma 初始化的一次乘法，图中已将初始化计入 gamma、其余计入 iteration D matvec，互斥分项不重复累加。')
        text+='\n## 六例完整方法与消融的直接比较\n\n'+_markdown(mechanism)+'\n'
        text+='''
## 实现、验证和运行修订记录

新增模块为 core/cost_profile_distance.py、solvers/energy_distance.py、experiments/energy_distance_study.py 和 plotting/energy_distance.py；新增主入口、报告入口、固定 YAML 配置和三组测试。没有修改已有两种 SK 的实现。距离一般路径和精确几何路径、一般等质量问题、稳定分块目标差、固定 gamma、回退、计划数量和恢复均有测试。

核心及相关回归共 31 项通过；报告层修复后 18 项核心门禁通过，并加入配对表非空以及“仅报告修订禁止启动任何求解器”的回归断言。final_verification.json 记录再次独立审计的 250 个有效端点、12 个不适用端点、144 对时间比、786 条阈值记录，以及原始和最终修订逐轨道文件哈希完全一致。MNIST 实际支持范围为 n=132–267、m=115–206，原始索引、质量和掩码保存在输入描述与 reconstruction.npz 中。

必须披露一次执行偏差：最初报告的 pandas 行索引错误造成配对图表为空；修订时 Windows 路径分隔符检查又误启动了 rev1 的重复实验。发现后停止，rev1 中 9 条已完成和 1 条中断尝试全部排除，不用于选参或结论；其机器时间仍计入同一账本，没有重置预算。rev0 的完整 262 条计划保持不变，最终 rev2 只继承 rev0 的原始测量并修正报告。详情见 excluded_revision.json；rev1 保留 EXCLUDED_FROM_FORMAL_STUDY.json，不能将它的条目合并进本轮正式比较。修复后的显式 --report-from 入口只接受全部完成且计算源码、配置未改变的计划，并禁止启动求解。

最初完整测量和自动报告耗时 2271.04 秒（37 分 51 秒）；最终累计还包括上述排除尝试、报告修订和独立复核，见下方及独立预算账本。没有推送、合并远端或触碰旧 SR1 账本。正式计划没有未完成条目；上述额外尝试是执行偏差，不能称为严格只启动了 262 条轨道。

报告主体由持久化产物自动生成，本节解读由 tmp/energy_finalize_report.py 从同一产物补充。完整复核脚本为 tmp/energy_final_verify.py。重复生成基础报告后可运行补充脚本恢复这份解读。
'''
        # Replace the first preliminary budget value with the final value after
        # the budget context closes below.
        (root/'report.md').write_text(text,encoding='utf-8')
        path.write_text(text,encoding='utf-8')
        resume=base/'docs/energy_distance_v0_resume.md'
        resume.write_text(resume.read_text(encoding='utf-8')+f'''
最终交付为 rev2，测量来自原始完整 rev0，rev1 为已排除的误启动修订，勿合并。正式 262 条均为终态。

仅报告修订的安全入口：

```powershell
.\\scripts\\run_ot.cmd scripts/run_energy_distance_study.py --config configs/energy_distance_v0.yaml --report-from "{root}"
```

此入口禁止任何求解器调用。通常直接运行 report_energy_distance.py --audit 即可重新出图；随后运行 `.\\scripts\\run_ot.cmd tmp/energy_finalize_report.py` 可补回定量解读和修订披露。
''',encoding='utf-8')
    finally:
        ledger.close()
        root=Path(ledger.data['active_run_dir'])
        atomic_json(ledger.data,root/'budget_snapshot.json')
        for target in (base/'docs/energy_distance_v0_report.md',root/'report.md'):
            text=target.read_text(encoding='utf-8')
            import re
            text=re.sub(r'机器累计时间约 [\d.]+ 秒（报告生成期间仍计费）',
                        f'最终机器累计时间 {ledger.data["used_sec"]:.2f} 秒（含排除尝试与最终复核）',text,count=1)
            text=re.sub(r'最终机器累计时间 [\d.]+ 秒（含排除尝试与最终复核）',
                        f'最终机器累计时间 {ledger.data["used_sec"]:.2f} 秒（含排除尝试与最终复核）',text,count=1)
            text+=f'\n最终累计机器时间：{ledger.data["used_sec"]:.2f} 秒，预算 3600 秒，超额 0 秒。\n'
            target.write_text(text,encoding='utf-8')
        print('FINAL_MACHINE_SEC',ledger.data['used_sec'])
