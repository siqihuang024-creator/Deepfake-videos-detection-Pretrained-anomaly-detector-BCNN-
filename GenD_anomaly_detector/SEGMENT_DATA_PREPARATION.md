# 试跑后的候选数据准备（2026-11-26）

此阶段承接作业 279 的时长核查及 281、285、288 的预处理试跑。它扩大候选数据准备规模，仍不训练 GenD、不拟合贝叶斯头、不定分数阈值、不处理最终测试视频画面。正式片段训练清单及分组加权训练、独立校准/视频评分的接通在此后完成，不能将本阶段的候选 CSV 直接交给旧整视频训练程序。

## 已检查到的边界

作业 288 包含 36 个中段窗口，33 个达到暂定持续覆盖条件。1,138 张裁剪的可读性、224×224 尺寸、表格/记录哈希、检测数组和时间戳检查通过。目视抽查中 HDTF 与 DFDC 裁剪基本正常；TalkingHead 有动物特写、模糊插入镜头、极暗画面、较大对齐边角和人物切换。三段低覆盖窗口均需保留作诊断。33 个覆盖标记通过并不等于人脸语义、亮度和人物连续性都合格，不能作为全量质量保证。

## 本阶段固定范围与预算

- 保留原整视频 train/calibration/dev/test 来源组划分；source_only 仅表示已有来源组隔离，不能声称人物隔离。
- A 为 HDTF、TalkingHead-1KH 的原 train/calibration 来源组。每个全局来源组按 seed 42 的路径哈希选一个原视频，最多生成 4 个 10 秒中段候选，避免同一来源多个文件或长时长主导数量。原 train/calibration 不交换，也不从 dev/test 补入。
- 开发诊断先取 DFDC dev 的 100 个可配对来源组，每组一个真实文件、一个伪造文件，共 200 个视频；来源组和文件均按预先固定哈希选取。其余 dev 仍标为未使用的开发数据，不能改称最终测试。
- 所有原始 test 元数据保存为 reserved_test.csv；不生成它们的候选，不解码、不看图、不提特征或评分。PolyGlotFake 保留在这里。
- 候选总数硬上限 3,000；超过上限先报错，不自动扩充任务或改变数据角色。实际数量由远程完整时长表决定，不保证达到上限。
- 检测固定为 pad_on_empty_v1，0.5 阈值、0.4 NMS、640 输入；224×224 人脸图，每候选最多均匀选 32 帧。维持原五格覆盖诊断标记，但保留所有窗口和失败记录；本阶段不按标记发布训练样本或过滤测试结果。

当前原清单共 573 个 A train 原视频、133 个 A calibration 原视频。它们分别含约 455、115 个已有来源组。来源组限额会减少重复来源原件的处理数量；程序报告实际入选原视频、来源组、候选数，不把片段数当成独立人物数。

## 登记与运行

从仓库根目录执行；先更新代码。`--reference-pilot` 使用已目视核查的中段试跑登记。程序核对原 CSV、时长审计及参考试跑的哈希，检查入选原视频的实际大小和 mtime，并将父表、选中原视频、保留测试元数据和未使用开发元数据全部存入新目录。

```bash
py=work/.venv-gend/bin/python
"$py" GenD_anomaly_detector/scripts/prepare_segment_data.py \
  --manifest "$HOME/gend_steps34_manifest_pilot.csv" \
  --duration-audit work/video_duration_audit_279 \
  --reference-pilot work/segment_interior_10s_20261125_222208/segment_pilot.json \
  --video-root /data/datasets/shuang --out "$DATA_PREPARATION" \
  --dev-source-groups 100 --max-candidate-clips 3000 --seed 42

sbatch GenD_anomaly_detector/slurm/segment_data.sbatch \
  "$DATA_PREPARATION/segment_data_plan.json" "$DATA_PREPARATION/faces"
```

GPU 任务申请 1 GPU、4 CPU、16 GiB 内存，时间上限 8 小时。这是调度上限，不是耗时预测。小批观测支持小时级而非分钟级的扩大准备，实际速度还受视频尺寸、关键帧间隔、存储 I/O、功率和节点负载影响。先处理有界候选及 200 个开发视频，而非全量三万余测试视频。

日志为 `work/logs/gend_segment_data_<jobid>.out`，每个候选显示进度；结果写入 `faces/pilot_summary.json`、`pilot_measurements.csv`、逐片段 records/detections/crops。每个已完成窗口有独立原子记录。错误会被汇总并以非零状态退出；修复原因后可在同一命令末尾添加 `resume`。恢复必须使用同一代码、运行时、文件、检测政策和登记；代码/输入有变化则用新目录。

后续将根据实际覆盖和抽查结果确定每个来源最多保留的训练片段数、校准/评估单位及质量处理规则，然后生成冻结特征缓存并训练第三/四步的贝叶斯头。测试集必须等方法及这些规则锁定后才打开。
