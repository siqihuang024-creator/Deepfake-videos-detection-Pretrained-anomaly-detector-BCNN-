# HDTF-only 来源组成敏感性检查

在投入全量数据准备v2之前，先用既有冻结特征检查当前统计基线对TalkingHead来源的依赖。自然人脸资格问题已在固定抽查中确认，但不据此预设它导致了模型性能问题。本轮不做按分数清洗、不重提特征、不训练新神经网络、不查看最终测试。

## 固定四格设计

| 拟合来源 | 校准来源 | 目的 |
| --- | --- | --- |
| HDTF + TalkingHead | HDTF + TalkingHead | 复现344原结果 |
| 仅HDTF | HDTF + TalkingHead | 检查拟合中心/协方差的来源敏感性 |
| HDTF + TalkingHead | 仅HDTF | 单独检查阈值来源；AUROC必须保持不变 |
| 仅HDTF | 仅HDTF | 检查整体来源限制 |

保留原313确定的角色及全部DFDC开发视频，四格之间不重抽片段、不重分拟合/早停/校准、不做开发分数筛选。拟合只用 `gradient_fit`，早停角色不用于本轮统计模型。

实际输入：混合拟合679片段/352来源；HDTF拟合308片段/162来源。混合校准112来源；HDTF校准53来源。HDTF早停91片段/48来源本轮不使用。

每套编码器分别计算：中心平方距离、来源均值Ledoit–Wolf Mahalanobis、来源等权片段协方差固定收缩0.3/0.5、原分类概率。共两编码器×两拟合来源×两校准来源×五方法×两阈值=80行。分类器不重新拟合，因此其开发AUROC四格必须完全相同。

限制来源后，继续按保留的来源等权、来源内片段等权，重新归一化权重。新阈值分别取对应真实校准集95%/99%线性分位数。53个校准来源意味着经验FPR步长约1.89个百分点，不能解读成精确1%/5%的独立保证。输出同时报告实际误报和检出数量/比例与Wilson区间。

## 解释边界

- 这是看过开发结果后的探索性来源消融，不是独立确认实验。整体去掉一个数据集仍是事后制定的选择，只是避免了按个别分数挑删。
- HDTF-only同时改变来源域、拟合样本量和质量组成。结果变化不能单独归因于卡通脸、模糊或裁剪问题；结果稳定也不能证明自然人脸资格已解决。
- 不设置“官方中心AUROC超过80%即证明质量不是瓶颈”等事后通过线。看同一开发视频上的排序与实际操作点，并保留全部结果。
- 本轮不重训贝叶斯神经头，不能用统计基线稳定来宣称贝叶斯负结果不受数据问题影响。若需要回答该问题，须另行登记神经头对照及样本量/KL变化。
- 原始与HDTF-only不是独立样本；原有来源bootstrap区间不替代方法差异的配对区间，也不保证未知演员之间独立。
- 语义规则仍须在正式最终实验前确定，不因敏感性结果稳定就把绘图脸视为已通过自然人脸检查。

## 资源、锁和输出

独立入口 `scripts/run_source_sensitivity.py`，作业模板 `slurm/source_sensitivity.sbatch`。默认tarasca、4 CPU、16 GiB、`--time=0`，不申请GPU；CPU作业仍可能排队。无需读取hefest原视频或加载GenD权重。

加载313、316的原始输入/结果锁，验证344完成标记、所有文件哈希与算法哈希，混合拟合必须重现344相应五种旧评分和混合校准阈值。登记四格、实际来源数、全部单元和代码哈希后才拟合。

输出：`sensitivity_registration.json`、原 `training_units.csv`、四格的选用单元 `selected_units.csv`；每编码器两种拟合的全单元分数、统计头参数及四格报告；总表 `comparison.csv`，最后写出带各文件SHA256的 `completion.json`。拒绝覆盖旧目录。旧算法和313—344结果均保持原样。

本地验证 `test_source_sensitivity.py` 覆盖：来源整体排除、无开发标签选择、同源角色隔离、来源等权及直接矩阵距离验证、HDTF拟合不受保留集之外特征影响、完整1024维缓存流程、344重现、80行四格输出、只换校准时AUROC不变、分类器四格AUROC不变，以及旧结果被改时拒绝启动。

## 一次提交：在tarasca更新、测试、运行

本地推送代码后，在服务器项目根目录执行下面整段。代码更新在同一个CPU作业内完成，避免单独申请更新任务。更新后作业模板会先跑新增测试，再运行检查；旧作业345无需取消。

```bash
(
set -euo pipefail
project="$HOME/projects/DeepfakeVideos Detection_Pretrained anomaly detector+BCNN/GenD pretrained"
cd "$project"
existing="$(squeue -h -u "$USER" -n gend-source-check -o '%i')"
if [[ -n "$existing" ]]; then
    echo 'A source-check job already exists. No duplicate submitted.'
    squeue -u "$USER" -n gend-source-check -o '%.12i %.12T %.12M %.40R'
    exit 0
fi
mkdir -p work/logs
jid="$(sbatch --parsable --partition=batch --nodelist=tarasca \
    --ntasks=1 --cpus-per-task=4 --mem=16G --time=0 \
    --job-name=gend-source-check \
    --output="$project/work/logs/gend_source_check_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
project="$HOME/projects/DeepfakeVideos Detection_Pretrained anomaly detector+BCNN/GenD pretrained"
cd "$project"
git pull --ff-only origin main
git log -1 --oneline
source "$project/work/covariance_latest_submission.env"
covariance="$out"
out="$project/work/source_sensitivity_${SLURM_JOB_ID}"
exec bash GenD_anomaly_detector/slurm/source_sensitivity.sbatch \
    "$project/work/segment_bayesian_311_20261126_195021" \
    "$project/work/segment_controls_313_20261126_230336" \
    "$covariance" \
    "$project/work/segment_data_v1_20261126_132052" \
    "$project/work/segment_features_301_20261126_192919" \
    "$project/work/celeb_identity_seed42/runs/ablation_selection.json" \
    "$out"
SBATCH
)"
jid="${jid%%;*}"
out="$project/work/source_sensitivity_${jid}"
{
    printf 'project=%q\n' "$project"
    printf 'jid=%q\n' "$jid"
    printf 'out=%q\n' "$out"
} > "$project/work/source_sensitivity_latest_submission.env"
printf 'Submitted: %s\nResults: %s\n' "$jid" "$out"
squeue -j "$jid" -o '%.12i %.12T %.12M %.40R'
)
```

进度与完成检查：

```bash
source "$HOME/projects/DeepfakeVideos Detection_Pretrained anomaly detector+BCNN/GenD pretrained/work/source_sensitivity_latest_submission.env"
sacct -X -j "$jid" --format=JobID,State,Elapsed,Timelimit,ExitCode
if [[ -s "$out/completion.json" ]]; then
    cat "$out/completion.json"
    cat "$out/comparison.csv"
elif [[ -f "$project/work/logs/gend_source_check_${jid}.out" ]]; then
    tail -n 40 "$project/work/logs/gend_source_check_${jid}.out"
else
    squeue -j "$jid" -o '%.12i %.12T %.19S %.50R'
fi
```

345范围说明：其既定导出目标是原六个高分校准窗口，不能用于后来固定抽查第9个拟合窗口的原图诊断。第9段若需原图对照，应单独登记目标，不能把345未包含它误认为已经检查过。
