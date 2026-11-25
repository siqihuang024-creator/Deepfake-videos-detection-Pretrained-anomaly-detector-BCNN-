# GenD anomaly detector

以 GenD（WACV 2026）为特征提取器，在其上接贝叶斯头做视频深伪检测。

**本项目选用 GenD-PE**（Perception Encoder L/14，官方权重 `yermandy/GenD_PE_L`）。
官方仓库同时支持 CLIP、PE、DINOv3 三种 backbone，CLIP 和 DINOv3 不在主线中使用。
当前第二步完成后的流程见 **[DIAGNOSTIC_PHASE.md](DIAGNOSTIC_PHASE.md)**：先完成两个冻结 GenD 编码器上的贝叶斯神经网络头实验；NIW / Ledoit–Wolf 作为后续或平行对照，共用逐帧缓存。第二步不重训，暂缓 DeepSpeak / CoCoVideo；下文早期外部全量评估命令仅作历史记录。
前一阶段 3D-CNN 实验见 `E:\PhD\Deepfake Video 3D-CNN+BCNN v2(1)`，背景摘要见上级目录 `PREVIOUS_PROJECT_CONTEXT.md`。

## 目录约定

| 路径 | 内容 | 规则 |
| --- | --- | --- |
| `third_party/GenD/` | 官方仓库 https://github.com/yermandy/GenD | **保持原样不改**。需要改动时在外层写包装或子类 |
| `EXPERIMENT_PLAN.md` | 第一阶段实验方案：哪些跟 GenD、哪些跟旧协议 | 改动要写进文末变更记录 |
| `protocols/celeb_video_disjoint_seed42/` | 旧协议第 01–30 组清单（原样复制，带 SHA256） | 不改 |
| `gad/` | 本项目代码：标签映射、抽帧规则、GenD 配置、权重加载、评估 | |
| `scripts/` | 命令行脚本，运行顺序见下 | |
| `tests/test_protocol.py` | 不需要 GPU 的协议检查（`python tests/test_protocol.py`） | |

## 第一、二步的运行顺序

以下为保留的旧 30 组协议记录。当前身份隔离实验使用下文“身份隔离与外部异常检测实验”和 [STEP2_ABLATIONS.md](STEP2_ABLATIONS.md)，不执行此处旧组训练命令。

```bash
W=<工作目录>          # 存放表格、人脸图、列表、运行结果
V="${CELEB_VIDEO_ROOT:?Set CELEB_VIDEO_ROOT to the CelebDFv3 video directory}"
python scripts/build_video_table.py --out $W/tables --video-root $V
# 人脸检测 + 裁剪；默认不处理测试集。先用 --limit-per-label 5 做小样本测试，看耗时和成功率
python scripts/preprocess_videos.py --videos $W/tables/videos.csv --membership $W/tables/membership.csv \
    --groups 1 --video-root $V --work $W/faces --retinaface <det_10g.onnx>
python scripts/write_frame_lists.py --tables $W/tables --work $W/faces --out $W/lists --groups 1
python scripts/check_weight_loading.py --frame-list $W/lists/val.txt --out $W/checks/weights.json
# 第一步：官方权重，只跑验证集
python scripts/predict_frames.py --model official --frame-list $W/lists/val.txt --out $W/step1/val.csv
python scripts/evaluate.py --videos $W/tables/videos.csv --val-pred $W/step1/val.csv --out $W/step1/val_report.json
# 第二步：第 01 组跑通后再跑满 30 组
python scripts/run_groups.py --lists $W/lists --runs-dir $W/runs/step2 --groups 1
python scripts/select_run.py --runs-dir $W/runs/step2 --out $W/runs/step2/selection.json
```

第 02–30 组用 Slurm 批量运行（在仓库根目录提交，`work/` 为工作目录）：

```bash
mkdir -p work/logs
sbatch GenD_anomaly_detector/slurm/preprocess_train.sbatch            # 8 个分片，最多同时 4 个
# 全部分片结束、没有错误后，重新生成所有组的帧清单（val.txt 和 group_01 的清单内容不变）
python GenD_anomaly_detector/scripts/write_frame_lists.py --tables work/tables --work work/faces --out work/lists
sbatch --export=ALL,STEP2_GROUPS="2 3 4 5 6 7 8 9 10 11" GenD_anomaly_detector/slurm/train_groups.sbatch
```

预处理遇到解码错误或检测器运行错误时，脚本以非零状态退出。重新提交同一命令只会重试失败的视频。

测试集只在 `select_run.py` 写出选择记录之后处理（`preprocess_videos.py --splits test`、`write_frame_lists.py --with-test`），然后由 `predict_frames.py --selection ...` 和 `evaluate.py --test-pred ... --selection ...` 评估一次。

上游版本：commit `387a42266dd385fbe8f7626c5d3aa03eea3bfaab`（2026-04-21，"Update bibtex"），克隆于 2026-11-12。

## 身份隔离与外部异常检测实验

从含有 `GenD_anomaly_detector/` 和 `work/` 的仓库根目录运行。每次实验使用新的工作目录，保存视频表、帧清单、权重和评估结果。旧的 30 组结果保留在 `work/runs/step2`。

```bash
py=work/.venv-gend/bin/python
gad=GenD_anomaly_detector
id=work/celeb_identity_seed42
celeb_root="${CELEB_VIDEO_ROOT:?Set CELEB_VIDEO_ROOT to the CelebDFv3 video directory}"
mkdir -p work/logs

# full_manifest 指向带 target_id、donor_id 和 source_clip 的完整清单。
"$py" "$gad/scripts/build_identity_split.py" --manifest "$full_manifest" --out "$id/tables" \
  --train-fakes-per-method-equal-to-real
sbatch --array=0-7%2 "$gad/slurm/preprocess_protocol.sbatch" "$id" "$celeb_root" train-val

# 训练/验证预处理完成后生成帧清单并检查权重。
"$py" "$gad/scripts/write_frame_lists.py" --identity-split \
  --tables "$id/tables" --work "$id/faces" --out "$id/lists"
"$py" "$gad/scripts/check_weight_loading.py" --frame-list "$id/lists/val.txt" --out "$id/checks/weights.json"

# 第 2 步：按 STEP2_ABLATIONS.md 完成资源试跑、prepare、submit、select。
# 第 1 步和第 2 步的正式测试均须先完成官方选模数据重叠核查与测试集预处理。
# 不要求先测试官方模型才能开始第 2 步训练。
```

第 2 步按固定种子从每种方法抽取与训练真视频相同数量的假视频。种子 42 的清单是 417 个真视频与 22 × 417 个假视频，验证及测试使用划分后的全部视频。`tables/summary.json` 记录抽样数量，`lists/coverage.json` 记录人脸裁剪后可用的真视频数与每种方法的假视频数。该规则约束方法配额。

### 256 图像微调批量的含义

每视频最多取 32 帧与训练批量是两个独立设置。默认每次前向和每次参数更新使用 128 张独立图像。`train_gend.py --batch-size 256` 会将单次前向和参数更新批量都设为 256；不增加每视频帧数，不强制每批由 8 个完整视频组成，也不引入 clip 时序维度。学习率、损失系数和最大 30 个 epoch 保持原配置。

当前通过下述三设置矩阵运行 `ln_b256_s42`、`ln_b256_s43`、`ln_b256_s44`，不再额外启动旧的 `finetuned_b256` 单次方案。

批量 256 是有记录的训练协议变更，不等同于原批量 128：每轮参数更新次数约减半，alignment 和 uniformity 的批内样本及配对数量改变，收敛轨迹和验证结果也可能改变。批量增大不保证吞吐量或指标更好。当前拒绝 `mini_batch_size != batch_size`：两个 128 图像批次的累积不等价于一次 256 图像的度量损失，上游按 DataLoader 批次数定义的调度长度也会与优化步数错位。

`run_meta.json` 保存实际前向批量、更新批量、梯度累积次数及与上游的配置差异。第 4 步只使用矩阵从三个 ln-b128 中按验证结果锁定的 `runs/ablation_selection.json`，不使用 ln-b256 的选择记录。检查批量配置可运行 `python GenD_anomaly_detector/tests/test_training_batch.py -v`。

### 三种设置、三个训练种子的单 GPU 扩展实验

预先登记的矩阵为 `ln_b128`、`ln_b256`、`full_b128`，各用训练种子 42、43、44，共 9 次；所有运行共用划分种子 42 的数据清单和现有预处理。只用一张 GPU，三个设置按上述顺序串行运行，不使用占位或空转计算。完整的学习率分组、20 步资源试跑、依赖数组提交、测试报告与第 4 步选模规则见 [STEP2_ABLATIONS.md](STEP2_ABLATIONS.md)。这些额外训练用于检验真实的研究问题，并记录实际资源需求。

全参数微调使用本项目子类实现，不修改 `third_party/GenD`。现有 LN 默认配置保留。推理与特征提取可通过 `GEND_PREDICT_BATCH_SIZE` 设置批量（Slurm 默认 256），通过 `GEND_NUM_WORKERS` 设置加载进程数。推理批量不改变模型或帧清单，但混合精度与内核选择可能导致细小数值差异，不能保证位级完全相同；元数据记录批量、吞吐量和峰值显存。

两次评估使用同一份验证和测试清单，分别用各自的验证分数确定阈值。报告与帧预测写入 `$id/eval/`，`per_method` 按伪造方法用同一批可评分真视频计算 AUROC。划分审计记录目标人脸、供体人脸和来源视频的重叠。TalkingFace 的音频驱动片段另行计数。官方模型曾用 CelebDFv3 验证数据参与选模，提交第 1 步前需核对官方验证清单和新测试集的重叠。

第 3、4 步使用同一份经审计的外部数据清单。清单字段为 `dataset,rel_path,split,label,target_id,donor_id,source_clip,method`。给真实训练集和真实验证集指定不同身份，测试集来源于另一外部数据集且包含视觉真、假视频。身份和来源 ID 应在数据集之间对齐。脚本按提供的 ID 检查外部划分间的重叠，并排除 FF++、CelebDFv3 数据集名。与上游数据共享的身份和来源视频需在准备清单时排查。外部视频根目录应包含清单列出的相对路径。

```bash
ext=work/external_anomaly_v1
"$py" "$gad/scripts/build_anomaly_table.py" \
  --manifest "$external_manifest" --video-root "$external_root" --out "$ext/tables"
sbatch "$gad/slurm/preprocess_protocol.sbatch" "$ext" "$external_root" train-val
sbatch "$gad/slurm/preprocess_protocol.sbatch" "$ext" "$external_root" test

# 两项预处理完成后执行。
"$py" "$gad/scripts/write_frame_lists.py" --anomaly --with-test \
  --tables "$ext/tables" --work "$ext/faces" --out "$ext/lists"
# 传入第 2 步的选择记录会做第 3、4 步配对实验。省略该参数会单独做第 3 步。
sbatch "$gad/slurm/external_anomaly.sbatch" "$ext" "$id/runs/ablation_selection.json"
```

冻结编码器，逐帧提取并 L2 归一化特征，再按视频取平均。对于两个编码器，以下五种方法都用 A 的训练真视频拟合，用 A 的验证真视频校准到名义 5% FPR。

- B1 用 Ledoit–Wolf 估计协方差，计算 Mahalanobis 距离平方。
- B2 用第 5 个真实近邻的余弦距离。
- M1 用 Normal–Inverse-Wishart 先验，`nu0 = d + 2`。得分是多元 Student-t 后验预测分布的负对数密度，对后验尺度下的 Mahalanobis 距离单调。额外抽取 100 组均值和协方差，导出距离均值和标准差。
- M2 用固定维数的 PCA（最多 64 维）和 Dirichlet-process 变分高斯混合。用 A 的验证真视频在对角和全协方差之间选择。
- M3 用无偏置的贝叶斯 Deep SVDD。只从 A 的训练真视频估计特征均值，训练、验证和测试统一减去这个均值，再用中心化后的训练特征初始化固定中心。网络宽度为 512、128，以 A 的验证真视频早停。KL 权重为 0.001。最终抽取 50 组网络权重，用到中心的平均距离平方评分，并导出其方差。

结果保留特征、拟合参数、验证和测试逐视频分数。报告包含 AUROC、AP、EER、按方法的 AUROC，以及验证阈值在 B 上对应的实际 FPR、TPR。M1 和 M3 还报告正确和误判视频的平均不确定性及与误判的 Spearman 相关。

第 4 步用选中的微调编码器重新提取 A/B 特征，并在 A 上重新拟合全部异常头和验证阈值，不复用第 3 步拟合的异常头。HDTF-only 与 HDTF+TalkingHead-1KH 两种 A、DFDC 与 PolyGlotFake 两种 B，应分别准备清单和工作目录；当前脚本不会自动按 B 的数据集拆分报告。官方 GenD 分类头在 B 上的概率分数对照属于计划中的补充，当前 `external_anomaly.sbatch` 仅执行特征提取与五种异常头，尚未自动生成该分类对照。

M3 的 `heads.npz` 保存训练特征均值。`report.json` 记录训练视频的后验平均距离平方在样本间的方差，以及后验均值网络的输出特征方差相对初始化的比例。分数标准差低于或等于均值的 0.1%，或表示方差低于或等于初始化的 0.01%，会触发诊断提示，供检查训练是否退化。

## 官方权重与预处理配置

**权重**：Hugging Face 官方发布了三个版本：`yermandy/GenD_CLIP_L_14`、`yermandy/GenD_PE_L`、`yermandy/GenD_DINOv3_L`。
加载方式见 `third_party/GenD/README.md`（`src.hf.modeling_gend.GenD.from_pretrained`）。

**PE 版本的论文配置**（`src/exp/wacv_rebuttal.py`，配置名 `wacv-PE_L-*`）：

- backbone：`vit_pe_core_large_patch14_336`（timm），输入 `img_size=224`
- 可训练参数：只放开 LayerNorm（`norm_pre`、`norm1`、`norm2`、`norm`），其余冻结
- 头：`LinearNorm`，即特征先 L2 归一化，再接 Linear(D, 2)
- 损失：交叉熵；完整 GenD 版本再加 uniformity 和 alignment 损失（论文主配置为 U=0.5、A=0.1，仓库里另有其他取值的扫描）
- 优化：AdamW（weight decay 0），lr 3e-4、最低 1e-5，bf16-mixed。下面是经 `run_exp.py` 启动时实际生效的取值：
  - 先用 `get_default_train_config()` 定基础值：batch 128，开头预热 1 个 epoch；
  - 再叠加 `get_common()`：cyclic cosine（每 10 个 epoch 一个周期），最多 30 个 epoch，早停 patience 30；
  - `Config` 类自身的默认 batch 是 512，但会被上面的基础值覆盖，不会生效；
  - 预热只在训练开头做一次，之后每个周期直接重启。论文描述的是每个周期都预热，这里以代码为准
- 增强（`config.Augmentations` 默认值）：水平翻转 0.5、仿射（±10°、平移 0.1、缩放 0.9–1.1）、高斯模糊 0.1、亮度和对比度扰动 0.1、JPEG 质量 40–100
- checkpoint 选择指标：`val/mAP_video`
- 视频级分数：帧概率取平均（`reduce_video_predictions="mean"`）
- 训练数据：FF++ 训练集，配置读取 5 套 seed 对应的 `random-32-frames/split-0..4` 子集

**人脸预处理**（`detector.py`，RetinaFace）：

```bash
python detector.py -i <视频目录> -m at_least -n 32 -o <输出目录> --det_thres 0.1 -s 1.3 --target_size none
```

即人脸框外扩 1.3 倍、检测阈值 0.1、每个视频至少取 32 帧、保存原尺寸裁剪。

**正式实验配置读取的数据目录**（`wacv_rebuttal.py` 中的路径）：

- 训练：`FF-x1.3-th0.5-all/subset/random-32-frames/split-{seed}`；
- 验证 / 测试：`CDFv3-x1.3-th0.5-all/subset/uniform-32-frames`。

这些路径表明配置读取了随机或均匀的 32 帧子集，但仅凭路径不能确认子集生成时是在原视频帧号还是检测成功帧中采样。核对上游提交 `387a422` 时，未定位到生成这些特定子集的独立脚本。
官方处理后的数据集放在 https://huggingface.co/datasets/yermandy/GenD 。

### 本项目的取帧协议与官方实现的区别

每个视频合计最多使用 32 张人脸图像；Slurm 的 8 个分片用于分配视频处理任务，不代表每个视频有 8 个 clip。

| 阶段 | 本项目当前实际行为 |
| --- | --- |
| 训练 | 逐帧检测人脸，再从检测成功帧中按视频固定随机抽最多 32 帧；各 epoch 复用这批帧并施加图像增强 |
| 验证 / 测试 | 逐帧检测人脸，再从检测成功帧列表中均匀抽最多 32 帧；位置计算使用四舍五入 |
| 有效帧不足 32 | 全部使用，不重复补帧；无有效人脸的视频计入覆盖率报告 |

GenD 论文第 3.2 节描述先在原视频上均匀取 32 帧，再做人脸检测。官方 `detector.py` 已公开 `fixed_num_frames`、`fixed_stride`、`at_least` 三种模式。其中 `fixed_num_frames` 使用 `np.linspace(..., dtype=int)` 在原视频帧号上取样，再检测人脸；检测失败后不会自动从其他帧补齐。它和本项目的检测后取样并不等价；即使所有帧都检测成功，整数取样与四舍五入也可能产生不同帧号。

因此，当前实现应描述为“参考 GenD 正式实验配置制定的、从检测成功帧中抽样的固定协议”，不能称为逐帧完全复现官方 `fixed_num_frames`。这处说明修正不改变当前算法、抽帧种子或已有结果。未来若更改取样规则，应另设协议和预处理目录，并统一应用于四步实验，避免混用已有缓存。

## 需要注意的点

- 官方的验证集包含 **CDFv3 验证集、DeepSpeak v1.1 / v2 验证集和 FFIW**（`get_common()`）。
  也就是说，官方发布的权重是用这些数据**选出来的 checkpoint**。
  - 拿官方权重在 Celeb-DF v3 上测零样本结果时，要核对它的 CDFv3 划分和我们的测试集是否重叠。
  - DeepSpeak 作为测试集时同理。
- 仓库里同时包含 Effort、ForAda、FSFM 的实现（`src/model/`），以后做其他提取器时可以沿用同一框架。
