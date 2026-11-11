# v0 恢复说明

结果：`D:\sinkhorn-lmflow\results\20261111_125856_524318_energy_distance_v0_rev2`

账本：`results/energy_distance_v0_budget.json`

重复主命令将校验实现、配置、输入、预热与产物指纹，并跳过已验证终态；不重置预算。

实现修改须 --new-revision：新目录、新指纹、同一累计账本，旧产物保留。

校准开始主比较后冻结，校准不完整则 kappa=1。进程意外中断按最近每秒检查点恢复，并保守追加 1 秒；关闭期间不计机器时间。

运行中单核允许有限越界；预算耗尽不再启动轨道。timeout 保留最后有效端点，属于本轮终态，不隐式追加预算。

剩余未启动条目：0。查看 pending.csv、failures.csv 和 manifest.json。

最终交付为 rev2，测量来自原始完整 rev0，rev1 为已排除的误启动修订，勿合并。正式 262 条均为终态。

仅报告修订的安全入口：

```powershell
.\scripts\run_ot.cmd scripts/run_energy_distance_study.py --config configs/energy_distance_v0.yaml --report-from "D:\sinkhorn-lmflow\results\20261111_125856_524318_energy_distance_v0_rev2"
```

此入口禁止任何求解器调用。通常直接运行 report_energy_distance.py --audit 即可重新出图；随后运行 `.\scripts\run_ot.cmd tmp/energy_finalize_report.py` 可补回定量解读和修订披露。
