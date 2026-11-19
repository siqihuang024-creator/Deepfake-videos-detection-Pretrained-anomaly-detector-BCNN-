# GenD anomaly detector

以 GenD（WACV 2026）为特征提取器，在其上接贝叶斯头做视频深伪检测。

**本项目选用 GenD-PE**（Perception Encoder L/14，官方权重 `yermandy/GenD_PE_L`）。
官方仓库同时支持 CLIP、PE、DINOv3 三种 backbone，CLIP 和 DINOv3 不在主线中使用。
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
sbatch "$gad/slurm/preprocess_protocol.sbatch" "$id" "$celeb_root" test

# 两项预处理完成后执行。
"$py" "$gad/scripts/write_frame_lists.py" --identity-split --with-test \
  --tables "$id/tables" --work "$id/faces" --out "$id/lists"

# 第 1 步。官方权重直接评估，返回作业号。
official_job=$(sbatch --parsable "$gad/slurm/identity_supervised.sbatch" "$id" official)
# 第 2 步。官方评估成功后微调，微调成功后评估选出的权重。
fit_job=$(sbatch --parsable --dependency="afterok:${official_job%%;*}" "$gad/slurm/identity_supervised.sbatch" "$id" fit)
sbatch --dependency="afterok:${fit_job%%;*}" "$gad/slurm/identity_supervised.sbatch" "$id" fine-tuned
```

第 2 步按固定种子从每种方法抽取与训练真视频相同数量的假视频。种子 42 的清单是 417 个真视频与 22 × 417 个假视频，验证及测试使用划分后的全部视频。`tables/summary.json` 记录抽样数量，`lists/coverage.json` 记录人脸裁剪后可用的真视频数与每种方法的假视频数。该规则约束方法配额。

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
sbatch "$gad/slurm/external_anomaly.sbatch" "$ext" "$id/runs/finetuned/selection.json"
```

冻结编码器，逐帧提取并 L2 归一化特征，再按视频取平均。对于两个编码器，以下五种方法都用 A 的训练真视频拟合，用 A 的验证真视频校准到名义 5% FPR。

- B1 用 Ledoit–Wolf 估计协方差，计算 Mahalanobis 距离平方。
- B2 用第 5 个真实近邻的余弦距离。
- M1 用 Normal–Inverse-Wishart 先验，`nu0 = d + 2`。得分是多元 Student-t 后验预测分布的负对数密度，对后验尺度下的 Mahalanobis 距离单调。额外抽取 100 组均值和协方差，导出距离均值和标准差。
- M2 用固定维数的 PCA（最多 64 维）和 Dirichlet-process 变分高斯混合。用 A 的验证真视频在对角和全协方差之间选择。
- M3 用无偏置的贝叶斯 Deep SVDD。只从 A 的训练真视频估计特征均值，训练、验证和测试统一减去这个均值，再用中心化后的训练特征初始化固定中心。网络宽度为 512、128，以 A 的验证真视频早停。KL 权重为 0.001。最终抽取 50 组网络权重，用到中心的平均距离平方评分，并导出其方差。

结果保留特征、拟合参数、验证和测试逐视频分数。报告包含 AUROC、AP、EER、按方法的 AUROC，以及验证阈值在 B 上对应的实际 FPR、TPR。M1 和 M3 还报告正确和误判视频的平均不确定性及与误判的 Spearman 相关。

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
- 训练数据：FF++ 训练集，每个视频随机取 32 帧，用 5 个 seed 各自抽帧（`split-0..4`）

**人脸预处理**（`detector.py`，RetinaFace）：

```bash
python detector.py -i <视频目录> -m at_least -n 32 -o <输出目录> --det_thres 0.1 -s 1.3 --target_size none
```

即人脸框外扩 1.3 倍、检测阈值 0.1、每个视频至少取 32 帧、保存原尺寸裁剪。

**但论文实验实际用的数据目录不同**（`wacv_rebuttal.py` 中的路径）：

- 训练：`FF-x1.3-th0.5-all/subset/random-32-frames/split-{seed}`，即检测阈值 0.5，先处理所有帧，每个 seed 再各自**随机**取 32 帧；
- 验证 / 测试：`CDFv3-x1.3-th0.5-all/subset/uniform-32-frames`，即**均匀**取 32 帧。

复现时以这个为准，不用 README 示例命令里的 `at_least`（按分散顺序取到 32 张有人脸的帧为止）。
官方处理后的数据集放在 https://huggingface.co/datasets/yermandy/GenD 。

## 需要注意的点

- 官方的验证集包含 **CDFv3 验证集、DeepSpeak v1.1 / v2 验证集和 FFIW**（`get_common()`）。
  也就是说，官方发布的权重是用这些数据**选出来的 checkpoint**。
  - 拿官方权重在 Celeb-DF v3 上测零样本结果时，要核对它的 CDFv3 划分和我们的测试集是否重叠。
  - DeepSpeak 作为测试集时同理。
- 仓库里同时包含 Effort、ForAda、FSFM 的实现（`src/model/`），以后做其他提取器时可以沿用同一框架。
