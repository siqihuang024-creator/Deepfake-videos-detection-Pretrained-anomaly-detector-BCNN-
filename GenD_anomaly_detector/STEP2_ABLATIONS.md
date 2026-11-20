# CelebDFv3 身份隔离微调：预先登记的扩展实验

日期：2026-11-20。登记于本轮测试结果产生之前。现有单次微调目录保留，矩阵使用独立运行名。

## 固定数据与研究问题

所有运行使用 `work/celeb_identity_seed42` 中同一份身份隔离 41/9/9 清单、训练/验证帧清单和预处理缓存。划分种子固定为 42；训练种子取 42、43、44。每视频最多 32 帧，不增加 clip、连续帧组或每视频取帧数。

| 设置 | 可训练参数 | 单次前向批量 | 预设学习率 |
| --- | --- | ---: | --- |
| ln_b128 | LayerNorm 与分类头 | 128 | 3e-4 |
| ln_b256 | LayerNorm 与分类头 | 256 | 3e-4 |
| full_b128 | 整个编码器与分类头 | 128 | 非 LayerNorm 编码器参数 1e-5；LayerNorm 与分类头 3e-4 |

共 9 次训练，只用一张 GPU 依次运行：先 ln-b128 三个种子，再 ln-b256，最后 full-b128。每种设置回答批量或参数更新范围对检测及资源需求的影响，不使用占位张量或空转计算。LN-b128 保留 GenD 的训练配方，但本项目的数据划分和检测后取帧规则与论文并非完全相同，因此不称为完整论文复现。全参数组还使用不同的编码器学习率，结果差异不能仅归因于参数数量。

损失系数、增强、bf16、30 epoch、仅初始一轮 warmup、10 epoch cyclic cosine、验证视频 AUROC 选模保持当前配置。训练批量改变时，同样 epoch 数的优化步数不同，不做线性学习率缩放。梯度累积禁用，防止批内度量损失含义和调度计数发生混淆。

## 全参数学习率与资源测量

`gad/finetuning.py` 在项目内子类化 GenD，保留原权重衰减分组条件（名称包含 `bias` 或 `norm` 时无衰减）、AdamW betas，并增加两档学习率。每个参数恰好属于一个组。实际 LayerNorm 模块参数与分类头使用高档，其余编码器使用低档。

全参数组使用与上游相同的 warmup 和 cyclic/cosine 时间坐标；对各组按相同比例计算最小学习率，即 `min_lr / lr × group_lr`。默认低档下限为 3.333e-7，高档为 1e-5。LN 基线仍使用原上游优化器和调度器。

正式矩阵前，先在 full-b128 上运行 20 个实际优化步骤，单独目录保存结果，不做验证、选 checkpoint 或测试。此试跑用于检查能否运行、峰值显存和初步吞吐；20 步含初始化和热身开销，不能直接作为稳态性能结论。如需梯度检查点，先用独立试跑确认，再在 `prepare` 时为三个 full 运行统一登记。试跑完成后才提交正式矩阵，两者不同时占用 GPU。

训练日志包括：

- `run_meta.json`：训练种子、参数范围、参数组学习率/数量、前向和更新批量、累积次数、代码提交、帧清单哈希及性能信息。
- `performance.json`：分 epoch 图像数、训练时间、images/s、GPU 型号及峰值 allocated/reserved 显存；吞吐包含数据等待，排除验证时间。
- `gpu_log.csv`：每 5 秒记录 GPU UUID、功耗、功率上限、利用率和显存；监控进程在训练结束或报错时清理。

这些是实际研究计算的观测数据；代码不修改设备功率或风扇设置。显存需求、吞吐和功耗须分别解读。若要测功率限制的因果影响，需要在获准的不同功率设置下对同一工作负载做受控比较，单次资源日志本身不能证明降功率会慢多少。

## 按顺序运行

从包含 `work/` 的仓库根目录执行。先确认远端代码更新、依赖测试和权重一致性检查通过；创建 `work/logs`。以下命令不自动提交测试阶段。

```bash
# 资源试跑：仅一个 full-b128 任务、20 个训练步骤。
sbatch --array=6 GenD_anomaly_detector/slurm/identity_ablation.sbatch work/celeb_identity_seed42 probe

# 试跑完成后查看 runs/full_b128_s42_probe/{run_meta.json,performance.json}。
# 若需要试梯度检查点，可另提交：
# sbatch --array=6 --export=ALL,GEND_GRAD_CHECKPOINTING=1 \
#   GenD_anomaly_detector/slurm/identity_ablation.sbatch work/celeb_identity_seed42 probe

# 在试跑成功后登记矩阵；文件已存在会报错，不覆盖。
work/.venv-gend/bin/python GenD_anomaly_detector/scripts/identity_ablation.py \
  --experiment work/celeb_identity_seed42 --action prepare
# 如果决定 full 全部启用梯度检查点，在上条 prepare 命令加入 --gradient-checkpointing。

# 9 次训练：每任务一张 GPU、8 核、48 GB 主机内存、6 个加载进程。
# 提交三个依赖数组，各数组并发为 1，保证设置顺序及总共最多一个训练任务运行。
work/.venv-gend/bin/python GenD_anomaly_detector/scripts/identity_ablation.py \
  --experiment work/celeb_identity_seed42 --action submit

# 全部训练完成并生成各自 selection.json 后，仅凭验证结果锁定模型。
work/.venv-gend/bin/python GenD_anomaly_detector/scripts/identity_ablation.py \
  --experiment work/celeb_identity_seed42 --action select
```

`submit` 提交 ln-b128（索引 0–2）、ln-b256（3–5）和 full-b128（6–8）三个数组，均使用 `%1`。后一个数组依赖前一数组全部成功（`afterok`），因此设置顺序不依赖 Slurm 的数组启动顺序；同组内种子启动顺序不影响实验。任一阶段失败，后续阶段不会启动。不要重复执行提交命令，先检查打印出的作业号及队列。

全参数任务会检查先前 20 步资源试跑是否成功。运行目录已存在时拒绝覆盖；技术失败须保留日志、检查原因后处理，不自动重用不完整训练。旧版五种子登记文件与当前方案不兼容，程序会拒绝继续，不会自动覆盖旧登记或结果。

`ablation_plan.json` 固定矩阵、输入哈希、单 GPU 执行策略和选择规则。`matrix_selections.json` 锁定 9 个 checkpoint。`runs/ablation_selection.json` 是第 4 步默认提取器：仅在三个 ln-b128 运行中按验证 AUROC 取最大，平分取最小种子。其他组测试成绩不能改变此选择。

## 最终测试与报告

必须先完成官方 CelebDFv3 选模数据的重叠核查、测试集预处理及相应帧清单。任何模型的测试指标都不参与上述设置、超参数或种子的选择。

每个运行内部仅用验证视频 AUROC 选择 epoch。三个设置各自报告三个预定种子的测试 AUROC 均值和样本标准差，以及按伪造方法的均值和标准差；该标准差衡量固定划分上的训练随机性，不是跨身份或跨数据划分的不确定性，三个种子的估计精度也低于五个。所有 9 个已选 checkpoint 各评估一次，不从中按测试成绩挑选“最好模型”。

```bash
# 仅在前述核查和模型锁定完成后显式提交。
sbatch --array=0-8%1 GenD_anomaly_detector/slurm/identity_ablation.sbatch work/celeb_identity_seed42 evaluate

# 全部评估成功后汇总。
work/.venv-gend/bin/python GenD_anomaly_detector/scripts/identity_ablation.py \
  --experiment work/celeb_identity_seed42 --action summarize
```

评估目录固定为 `eval/ablations/{run_name}`，存在即拒绝重复运行。汇总核对所有模型使用同一视频表和同一验证/测试帧清单，且 checkpoint 与测试前的锁定记录一致。第 1 步的官方模型只评估一次。第 4 步传入 `runs/ablation_selection.json`，A/B 数据和异常检测设置保持原方案。

推理与特征提取默认 batch 256，可通过 `GEND_PREDICT_BATCH_SIZE` 设置；并不承诺不同 batch 的混合精度结果位级相同。当前矩阵仅包含上述三种设置，不增加 full-b256 或其他变体。
