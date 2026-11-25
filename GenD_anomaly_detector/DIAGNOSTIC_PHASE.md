# 第二步完成后的贝叶斯主线与平行诊断（2026-11-25）

这是当前后续工作的入口。按用户最新决定，先完成贝叶斯神经网络头实验；统计基线做后续或平行实验。旧 `external_anomaly.sbatch` 和早期实验方案保留追溯，不作为本轮入口。

## 范围与判断边界

- 第一步为官方 GenD 直接推理；第二步为已完成的九次微调。第三步为冻结官方 GenD + 贝叶斯神经网络头，第四步为冻结已选微调 GenD + 重新训练的贝叶斯神经网络头。**两步都不继续微调 GenD**。NIW / Ledoit–Wolf 由独立入口运行，不作为贝叶斯主线的前置条件。
- 先用现有 `identity_ablation.py --action select` 锁定九个 checkpoint 和第四步 LN-b128 胜出者，再补官方同协议验证对照。完整九模型测试仍按 STEP2_ABLATIONS.md 后补，不改报告规则、不重新训练。
- `official_validation.sbatch` 现在对官方模型、已选 LN-b128 模型都调用同一个 `predict_frames.py` 和 `evaluate.py`，使用相同验证帧、batch、精度和指标。`comparison.json` 给出重推理后的 AUROC 差值；训练期 Lightning 的最佳验证成绩只作参考。胜出模型原本就在该验证集选出，因此该差值仍不是独立测试集上的无偏提升，不能重新据此挑模型。
- 第一轮已更新参数，不能代表官方初始权重。`W1-conf-real` 是无方向分布距离，不能据此断言真假倾向或特征塌缩。应检查假概率分布、固定阈值误报率及特征。
- `gad/metric_compat.py` 在外层修复今后训练日志的 EER 阈值，不改上游文件。已有 checkpoint 和日志保留；历史 NaN 阈值对应的训练准确率/F1 无效，不能直接恢复。验证 AUROC 选模规则不变。
- 官方 GenD 开发使用过 CDFv3 和 DeepSpeak。开发暴露和本项目重划分测试视频的具体重叠是两回事；正式第一、二步测试仍须核对视频清单并披露暴露。没有完整官方清单就写“未核实”，不能写“确认无重叠”。参见 [GenD 论文 §3.2](https://arxiv.org/html/2508.06248v4)。
- DeepSpeak-v2 / CoCoVideo 本轮暂缓，不自动解压或加入实验。DeepSpeak 有真有假，且存在编码差异，不能默认当作全新真实来源，见[官方说明](https://huggingface.co/datasets/faridlab/deepspeak_v2)。
- 风格建模、patch 检测仍是待文献和诊断支持的候选。本次不实现这些新方法。12 月 3 日是检查点，不保证正结果。

## 数据角色登记

准备 UTF-8 CSV，每行一个视频。必须有 `split,dataset,rel_path,label,source_clip,method`；可补 `target_id,donor_id,driver_id,language`。

| 字段/角色 | 约定 |
| --- | --- |
| `label` | 0=视觉真实，1=视觉伪造，与旧项目标签相反 |
| `rel_path` | 相对统一视频根目录的完整 POSIX 路径，含数据集目录 |
| `source_clip` | 经核对的全局原始来源组，原视频及其全部伪造衍生物同组 |
| 人物 ID | 有可靠元数据才填，同一人物跨数据集也应统一；不能造 ID |
| `train` | HDTF / TalkingHead-1KH 真实视频，仅拟合异常头 |
| `calibration` | 同来源独立留出的真实视频，仅定阈值，不参与拟合或调参 |
| `dev` | DFDC 一部分，以及已用于选模的 CelebDF 验证视频用于诊断；后者不是独立泛化证据 |
| `test` | 保留 PolyGlotFake 和 DFDC 其余部分，方法锁定前不评估 |

四份清单先固定。DFDC 分片编号不是人物 ID，程序不自动按分片划分。默认要求至少来源组隔离；可靠身份可用 `provided_ids` 并检查 target/donor/driver 跨角色重叠。无法建立人物身份时显式使用 `source_only`，只能声称来源组隔离。程序检查已填的组，不能替代元数据、人脸和跨数据集来源审计。

当前 A 明确仅为 HDTF / TalkingHead-1KH；不把 CelebDF 训练身份的真视频加入异常头拟合或阈值校准。这个范围与原 `build_anomaly_table.py` 排除 CelebDF 的规则一致，不能声称本次删掉了一批已经使用过的 CelebDF 拟合数据。CelebDF 在本阶段是异常头未拟合过的来源，但微调编码器见过其训练集，官方编码器也有该数据集的开发暴露。

DFDC 的 `original` 元数据可在字段有效且映射唯一时用于建立原始来源组，需核对缺失引用、分片路径和重复原件；它不提供人物隔离证明。仍需先生成并人工核对实际 CSV。按来源组 bootstrap 未覆盖同一演员在多个原视频之间的相关性，区间可能偏窄，报告中会显式注明。

从包含 `GenD_anomaly_detector/` 的仓库根目录运行以下 Bash 命令。环境变量必须指向实际审计清单和模型文件。

```bash
py=work/.venv-gend/bin/python
gad=GenD_anomaly_detector
id=work/celeb_identity_seed42
d=work/gend_bayesian_v1

# 已有锁定文件时保留，不重复执行 select；不要重新 prepare/submit/fit。
"$py" "$gad/scripts/identity_ablation.py" --experiment "$id" --action select
mkdir -p work/logs
sbatch "$gad/slurm/official_validation.sbatch" "$id"

"$py" "$gad/scripts/prepare_diagnostic.py" \
  --manifest "$AUDITED_MANIFEST" --video-root /data/datasets/shuang \
  --selection "$id/runs/ablation_selection.json" --out "$d" \
  --identity-policy source_only --audit-note "Describe the actual source audit and identity limitations"
```

`--out` 必须是新目录。可靠完整人物元数据才使用 `--identity-policy provided_ids`。

## 预处理、缓存与 CPU 诊断

登记协议 v3 同时声明 bayesian 主线、baseline 平行实验及贝叶斯早停规则。train/calibration/dev/test 全部从检测成功帧中均匀抽最多 32 帧，不做图像训练增强。这里的 train 仅指拟合异常头，与第二步 GenD 训练不同；第二步原有随机抽帧和结果不变。旧登记不自动迁移，须另建协议和预处理目录，避免实验定义或帧缓存混用。

先用独立试跑目录、小批视频计时，再估算正式规模。正式预处理使用新目录，不能混入旧角色记录。下面预处理命令必须在已分配的 GPU 作业中运行，也支持分片参数。

```bash
"$py" "$gad/scripts/preprocess_videos.py" --videos "$d/tables/videos.csv" \
  --video-root /data/datasets/shuang --work "$d/faces" --retinaface "$RETINAFACE_WEIGHTS" \
  --splits train calibration dev
"$py" "$gad/scripts/write_frame_lists.py" --diagnostic-plan "$d/diagnostic_plan.json" \
  --tables "$d/tables" --work "$d/faces" --out "$d/lists"

features_job=$(sbatch --parsable "$gad/slurm/diagnostic_features.sbatch" \
  "$d" "$d/faces" "$id/runs/ablation_selection.json")
sbatch --dependency="afterok:${features_job%%;*}" "$gad/slurm/bayesian_heads.sbatch" "$d"
# 可稍后或另行并行提交统计对照，不是主线的前置任务：
# sbatch --dependency="afterok:${features_job%%;*}" "$gad/slurm/diagnostic_cpu.sbatch" "$d"
```

两个冻结编码器串行提取，每个编码器一遍。主线的贝叶斯小网络直接用缓存，在独立 CPU 作业中训练；不加载 GenD、无需分配 GPU。`--device cuda` 也受支持，但必须显式使用 GPU 作业，不能在 CPU 分配中使用。具体耗时以试跑为准。`development.txt` 合并 train/calibration/dev，绝不含 test。`frame_lists.json` 绑定清单哈希及不可评分视频，开发输出使用新目录。

## 贝叶斯神经网络主线

现有实现是无偏置 Bayesian Deep SVDD，全连接宽度通常为 `1024 → 512 → 128`，不是卷积 BCNN。保留原有目标：真实视频特征先减去拟合子集均值，学习到固定非零中心的距离，并加变分权重的 KL 正则。不能把它写成已实现另一篇论文的卷积网络。

`train` 角色内部按来源组与所有已知人物 ID 的连通分量固定划分，seed 42，向上取整留出约 20% 的组用于早停；其余组更新网络权重。实际视频比例可能不等于 80/20。`calibration` 角色不参与梯度、中心估计或早停，只在网络确定后定阈值；`dev` 只用于诊断。这些内部划分在两个编码器间相同，并写入结果。至少需要两个独立组及足够拟合视频。

网络按原设置最多 100 轮、patience 12、Adam 学习率 1e-3、batch 128、KL 权重 0.001；早停比较用 8 组权重样本。评分时固定抽取 50 组权重，用距离平方的均值作为异常分数、方差作为不确定性。它不是已校准的误判概率。异常头只用真实视频，但 GenD 已接受过真假监督，不能声称整个系统从未见过伪造数据。

`bayesian_heads.sbatch` 对官方/微调缓存各训练一次，保存 `training_progress.jsonl`、`training_history.csv`、`heads.npz`、逐视频分数和报告，并自动分别锁定到 `locks/bayesian/<encoder>_<jobid>/`。进度日志包含轮次、训练目标、早停距离和当前最佳轮次；报告保留塌缩诊断、不确定性与错误的关联。最终测试加载锁定网络，不重新训练。

平行 NIW / Ledoit–Wolf 会复用相同特征缓存，但无需早停，默认用完整 train 拟合；贝叶斯网络会留出内部早停组，因此数据用法的差异须在报告中注明。若以后要严格隔离“贝叶斯化”的收益，还需同架构确定性网络等相应对照，当前不扩展该工作。

缓存位于 `work/feature_cache/<signature>/`，`cache_receipts/` 指向具体缓存。签名包含实际模型 state、checkpoint、预处理、精度、上游提交、帧清单、裁剪文件大小/修改时间与时间戳。后者不是像素内容审计；完整文件另记录 SHA256，不完整缓存不能发布或复用。

| 缓存内容 | 规则 |
| --- | --- |
| `frame_l2.npy` | 逐帧 L2 特征，FP16，通常 1024 维 |
| `frame_norms.npy` | 归一化前幅值，FP32；可近似恢复原始特征，有量化误差 |
| `frame_indices.npy` / `frame_times_seconds.npy` / `frame_video_index.npy` | 原帧号、实际解码 PTS、视频归属；旧记录时间未知存 NaN，不推测 FPS |
| `frame_fake_prob.npy` | 同次前向的分类头假概率 |
| `video_features.npy` | FP16 存储量化前的逐帧 L2 特征均值，不再次归一化视频均值，保持旧基线口径 |
| `video_fake_prob.npy` / `metadata.json` | 帧概率均值；推理耗时、吞吐、显存峰值与来源指纹 |

每视频最多 32 帧，不足不补。帧不当作独立训练视频；不全量存 patch。旧均值缓存不能伪造成逐帧缓存，两个编码器需要各重新推理一次。

开发报告含 AUROC、双向 AP、阈值下实际 FPR/TPR、按来源/语言真实误报率、人物/来源组 bootstrap 95% 区间、误报清单和最多 20 个案例裁剪图。脸部裁剪不代表完整背景，风格检查仍须回看原视频。

NIW 默认精确 Student-t 预测分数，关闭 100 次后验抽样，不宣称得到了误判概率。可显式增加 `--posterior-draws` 做后验距离离散程度诊断，它也不是校准错误概率。

阈值取独立真实校准视频分数的 95%/99% 线性分位数，`score >= threshold` 判异常。报告校准实际 FPR、`1/n` 粒度和目标域实际 FPR，不保证名义 1%/5%。少量样本或同分时偏差可能很大。ROC 插值 TPR 单独命名，不等于部署阈值实测检出率。bootstrap 只反映已有样本，零误报时区间可能退化，不能证明未知来源零误报。

## 最终测试门禁

贝叶斯批处理已为两套网络写出锁定记录。手动运行主线时可用 `--freeze-to` 保存网络参数、阈值、代码和缓存指纹；例如：

```bash
"$py" "$gad/scripts/evaluate_bayesian.py" --plan "$d/diagnostic_plan.json" \
  --lists "$d/lists/frame_lists.json" --development-cache "$d/cache_receipts/official.json" \
  --out "$d/bayesian/official_manual" --freeze-to "$d/locks/bayesian/official_manual"
```

最终视频预处理后，在**新**列表目录加 `write_frame_lists.py --with-test`，保留原开发清单；提取 test 缓存仍带 `--list-registration`。微调模型用原 selection；官方模型用预先登记的 `{"checkpoint":"official"}` selection。

```bash
"$py" "$gad/scripts/evaluate_bayesian.py" --phase test \
  --plan "$d/diagnostic_plan.json" --lists "$d/final_lists/frame_lists.json" \
  --development-cache "$d/cache_receipts/official.json" \
  --test-cache "$d/cache_receipts/official_test.json" --lock "$d/locks/bayesian/official_manual" \
  --out "$d/final/bayesian_official"
```

测试加载已锁头和阈值，不重新拟合。代码/配置变化会拒绝旧锁。主线与平行实验各有独立锁和 receipt，因此可以分别评估一次，共用缓存；换输出目录不能重复评估同一实验/编码器。若要正式报告平行结果，须在打开任何最终测试前把相关配置一并冻结；开始最终测试后脚本拒绝生成新锁。失败保留 started 状态供排查，不自动清理。软件门禁不能替代测试纪律。

## 官方选模重叠审计与本地验证

`audit_official_overlap.py` 接受人工统一命名的两张 CSV：`dataset,canonical_video_id`，候选另有 `split`。ID 使用同一套数据集内完整相对视频名/可靠映射，不能仅用 basename。

```bash
"$py" "$gad/scripts/audit_official_overlap.py" --official "$OFFICIAL_CANONICAL_LIST" \
  --candidate "$OUR_CANONICAL_LIST" --out "$id/checks/official_selection_overlap.json"
```

只有确实取得完整官方选模清单才加 `--official-list-complete`。映射本身须人工核对；部分清单零匹配仍报告 unresolved。开发阶段官方验证对照可以先做并注明暴露；第一、二步正式测试仍须处理该审计。

`tests/test_diagnostic.py` 检查指标、量化/时间戳/缓存损坏、冻结提取复用、来源/人物泄漏、独立校准、模型匹配和测试锁等。既有训练批量、优化器、监控、选模与统计头测试继续保留。CPU 检查不下载模型；真实 H200、官方权重和完整视频预处理仍需在远程做小样本验证。
