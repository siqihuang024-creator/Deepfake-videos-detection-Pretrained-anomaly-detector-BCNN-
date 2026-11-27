# 342 后续：片段协方差对照与校准图片复核

本轮是开发诊断，尚未运行远程实验。保留313、316、317、342的结果，不修改旧算法或缓存。342的来源均值协方差反映来源之间的变化，给单个片段评分时可能遗漏来源内波动；小方差方向是可信的风险机制，但不是已经确认的性能下降原因。先补做固定范围对照，再决定是否开展NIW。

## 固定配方

- 完全复用313的清单、角色、两个冻结编码器缓存与第二步选模登记。实际参考为拟合679片段/352来源，早停169/88，真实校准112/112，开发200视频/100配对来源。不解码测试视频、不重训或重提特征。
- 只用 `gradient_fit` 估计统计量。对来源g的每个片段赋权 `1 / (G * n_g)`，每个来源总权重为 `1/G`。中心与旧来源均衡中心相同。
- 使用加权总体协方差 `C = sum_i w_i (x_i-mu)(x_i-mu)^T`，无独立样本自由度修正。显式验证 `C = 来源均值协方差 + 来源内片段协方差`；不将679片段宣称为679个独立来源。
- 固定两档 `alpha=0.3,0.5`：`C_alpha = (1-alpha)*C + alpha*trace(C)/d*I`。不是对相关片段调用基于iid假设的自动收缩系数估计。
- 原1024维特征与去分类方向后的1023维特征分别运行。方向复用342已锁定的冻结分类器权重，不根据开发标签重估，不做额外归一化。四种新评分全部报告，不挑最好一档作为独立测试结论。
- 每种评分分别用原112个真实校准单元的95%/99%线性分位数定阈值，`score >= threshold` 判异常。保留342全部六种旧评分，两编码器共40个操作点。AUROC按已知来源组bootstrap500次，误报率给Wilson区间；都不保证未知人物之间独立。

## 逐片段距离分解

每个空间分别从342保存的收缩协方差恢复经验协方差，提取旧经验零空间（官方原空间实际约673维）。零特征值容差按该空间最大特征值与浮点精度确定，写入参数文件。不按模型分数选取子空间。

对旧版和两档新协方差同时输出：

1. **当前经验零空间的贡献及占比**：在当前协方差自己的特征方向上，这是一项非负、可加的距离贡献，占比在0到1之间。新版零空间可能改变，不能当作同一个子空间直接比较。
2. **固定旧零空间的分解**：令原偏差分解为旧零空间分量a与正交补分量b，精确记录 `a^T P a`、`b^T P b`、`2a^T P b`，三项之和等于总距离，P为当前精度矩阵。

旧协方差的交叉项约为0，新协方差可能有正或负交叉项。因此新版的 `旧零空间自身项/总距离` 不是可解释为概率的占比，可能大于1，不能省略交叉项。贡献下降也不独自证明AUROC变化的因果机制。

相对342，新版同时改变协方差的估计范围和收缩强度；这轮结果不单独识别其中哪一项造成差异。先按这两档既定对照报告，不为追求正结果临时扩大网格。

## 校准图片复核

协方差任务登记原官方中心距离超过原95%校准阈值的全部校准片段，最多6个；实际342参考是6个TalkingHead片段。此选择只服务于排查，不改变训练、校准或开发清单。

单独的 `export_calibration_inspection.py` 每个片段从原32个已选帧中等距选5个位置，导出5张已保存的224×224裁剪PNG及5张同帧原始画面JPEG，并生成并排联系图和HTML。最多6片段、30组画面，不下载全部7万多张图，不导出完整原视频。

导出前核对请求、清单、预处理记录及视频文件大小/纳秒修改时间；重新解码时严格核对窗口内帧序号与原PTS，不用平均fps推算。画面本身尚需人工复核：是否确有人脸、裁剪是否合理、是否为字幕/静态图等。5帧不能保证整段质量。不能因为分数高直接删样本或重新分配角色。

## 资源与运行

`covariance_controls.sbatch`：默认tarasca，4 CPU、16 GiB、`--time=0`，不申请GPU，不加载GenD权重，只读取共享项目缓存。

`calibration_inspection.sbatch`：默认hefest，2 CPU、4 GiB、`--time=0`，不申请GPU。原视频在该节点的 `/data/datasets/shuang`，所以图片任务不能仅因tarasca空闲就移过去；即便只用CPU，也可能因节点预留而排队。原始视频根目录可通过 `GEND_VIDEO_ROOT` 明确覆盖。

在项目根目录执行，先确保代码已推送并在服务器更新：

```bash
mkdir -p work/logs
reference="$PWD/work/segment_bayesian_311_20261126_195021"
controls="$PWD/work/segment_controls_313_20261126_230336"
geometry="$PWD/work/geometry_diagnostics_342_tarasca"
data="$PWD/work/segment_data_v1_20261126_132052"
cache="$PWD/work/segment_features_301_20261126_192919"
selection="$PWD/work/celeb_identity_seed42/runs/ablation_selection.json"
out="$PWD/work/covariance_controls_342_$(date +%Y%m%d_%H%M%S)"
inspection="${out}_inspection"

# Submit once. Save these output paths and job IDs rather than resubmitting.
jid="$(sbatch --parsable GenD_anomaly_detector/slurm/covariance_controls.sbatch \
    "$reference" "$controls" "$geometry" "$data" "$cache" "$selection" "$out")"
jid="${jid%%;*}"
inspect_jid="$(sbatch --parsable --dependency="afterok:$jid" \
    GenD_anomaly_detector/slurm/calibration_inspection.sbatch \
    "$out" "$data" "$cache" "$inspection")"
inspect_jid="${inspect_jid%%;*}"
printf 'Covariance job: %s\nInspection job: %s\nResults: %s\nInspection: %s\n' \
    "$jid" "$inspect_jid" "$out" "$inspection"
```

第二个作业只有第一个成功完成后才运行；其等待依赖不是GPU等待。运行时长是否可无限仍由集群策略决定。

## 输出与验证

- `covariance_registration.json` 在拟合前写入固定配方、完整参考配置和旧结果登记哈希；启动核对313、316、342的锁、代码及模型方向哈希，用现有缓存重算342全部六种旧分数与阈值，差异即停止。
- 每编码器：`distance_heads.npz`、全单元/校准/开发分数、逐片段 `distance_decomposition.csv`、按角色/数据集/标签汇总的 `decomposition_summary.csv`、报告和阈值。汇总中的均值/中位数是描述性片段统计，不作为独立样本推断。
- 共同：`comparison.csv`、`inspection_targets.csv`、`inspection_request.json`；最后写入带输出文件SHA256的 `completion.json`，未完成目录不能当作完整结果。
- 图片导出另写 `index.html`、各片段联系图/小批量原图与裁剪、记录、manifest和完成锁，生成独立 `.tar.gz` 及 `.sha256`。不修改301预处理目录。
- 所有入口拒绝覆盖既有输出。不要在运行前预建最终输出目录。

本地验证：`test_covariance_controls.py` 覆盖权重、来源内分解、固定子空间交叉项、零空间谱、原/残余协方差变换、完整1024维缓存流程及旧结果篡改；`test_calibration_inspection.py` 覆盖导出数量、归档校验、禁止测试角色/路径越界、PTS漂移时不发布完成标记。实际服务器视频解码与图片内容需在正式小批量导出后确认。
