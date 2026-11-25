# 长视频片段试跑（2026-11-25）

当前先核查实际时长，再做少量人脸预处理试跑。10 秒是试跑候选值，不是已经确定的正式训练协议。第二步九次训练、checkpoint、验证补测和原始整视频清单都保留。

## 本轮可以运行什么

1. `audit_video_durations.py` 在 CPU 上读取 HDTF、TalkingHead-1KH、DFDC、PolyGlotFake 的实际视频头，输出时长、平均/标称帧率、时间基、视频流起始时间、文件大小和修改时间。支持直接读取已有元数据包里的 `video_inventory.csv`，不解压数据集。
2. `prepare_segment_pilot.py` 从原整视频草案继承角色和来源组，在 train / calibration / dev 中按来源组确定性选一小批原视频，产生带独立编号、时间边界和原视频编号的虚拟片段；不另存或转码原视频。test 不进入片段试跑或人脸检测；时长审计只作技术元数据核查，PyAV 在起始时间未知时可能解码首帧以读取 PTS，不分析视觉内容。
3. `preprocess_segment_pilot.py` 仅在指定窗口内做人脸检测、对齐和最多 32 帧均匀抽取，输出片段耗时、人脸有效率、实际 PTS。支持同一配置断点续跑。

默认每个“角色 × 数据集 × 标签”最多选 2 个来源组，每组只选 1 个原视频，每个原视频最多选 2 个不重叠窗口；从时间轴均匀分布的候选窗口选取，尾段不足 2 秒排除，其余短尾保留并记录。原视频/来源组数量与片段数量分别统计，不能把片段数写成独立人物数。此上限仅用于试跑，不是最终训练来源权重方案。

## 第一步：实际时长核查，只申请 CPU

先把本次代码推送并在 hefest 拉取，再从远程仓库根目录执行：

```bash
cd "$HOME/projects/DeepfakeVideos Detection_Pretrained anomaly detector+BCNN/GenD pretrained"
mkdir -p work/logs
sbatch GenD_anomaly_detector/slurm/audit_video_durations.sbatch \
    "$HOME/gend_steps34_metadata.tar.gz" work/video_duration_audit_v1
```

日志为 `work/logs/gend_duration_audit_<jobid>.out`，每 50 个文件显示一次进度条。4 个 CPU 并行读取，单文件超时 30 秒；优先用 ffprobe，缺少 ffprobe 时用现有 Python 环境中的 PyAV 子进程。PyAV 最多在流起始时间未知时解码首帧，不扫描完整视频。输出目录包括：

- `durations.csv`：逐视频元数据；`errors.csv`：失败清单。
- `summary.json`：各数据集时长分位数、总小时数、帧率分位数和起始时间未知数量。
- `progress.jsonl`：断点记录；`audit_contract.json` 和 `SHA256SUMS.txt`：输入、代码和后端身份。

若文件报错，作业会在完成其他文件后以非零状态结束；先检查错误原因，再用同一参数加第三个位置参数 `resume` 重试。成功且文件大小、mtime 未变的记录复用，失败或文件已变化的记录重查。输入清单、代码或后端版本改变时须用新目录。时长是视频头估计，不等于全片解码验证；平均/标称帧率不能证明恒定帧率。

## 第二步：审阅统计后，生成少量片段

将已核查的原整视频 CSV 放到远程，保留它的原版本。以下 `AUDITED_MANIFEST` 必须替换为实际路径。**不要把本地 Windows 路径直接传给服务器。**

```bash
py=work/.venv-gend/bin/python
"$py" GenD_anomaly_detector/scripts/prepare_segment_pilot.py \
    --manifest "$AUDITED_MANIFEST" \
    --duration-audit work/video_duration_audit_v1 \
    --out work/segment_pilot_10s_v1 \
    --seconds 10 --clips-per-original 2 --groups-per-stratum 2 --minimum-seconds 2
```

这里只在时长核查没有错误时生成试跑。若 ffprobe 的起始时间未知，使用 `--backend pyav` 重做一份新目录的时长核查，而不是假设视频从零秒开始。

TalkingHead 的官方人脸 tube 区间暂不自动应用。必须先核对下载的视频版本、帧率、区间索引语义和时间对齐，才能转换为时间段；不能假定 30 fps。当前试跑也用于观察无效人脸窗口的比例。

## 第三步：单 GPU 测预处理速度

```bash
sbatch GenD_anomaly_detector/slurm/segment_pilot.sbatch \
    work/segment_pilot_10s_v1/segment_pilot.json \
    work/segment_pilot_10s_v1/faces
```

申请 1 张 GPU、4 CPU、16 GiB 内存、最多 1 小时；不加载 GenD，也不训练贝叶斯头。使用实际 PTS 和半开时间窗口定位到前一个关键帧，再解码到窗口内，避免反复扫描长视频前缀。不按 fps 乘秒数猜全局帧号；保存的 frame 编号是该窗口内的解码序号，原始 PTS 另存。

日志记录分配的 CUDA 标识、节点各 GPU 功率上限快照及 RetinaFace 的实际 ONNX Runtime provider。节点快照可能包含其他人的卡，不能把每一行都当成本作业占用。缺少 CUDA provider 时会报错，避免把静默 CPU 回退误当作 GPU 速度；CUDA provider 列表也不能保证每个算子均在 GPU 上执行。正式计时仍需结合节点负载和功率限制判断。

查看 `faces/pilot_summary.json` 和 `faces/pilot_measurements.csv`：每个角色/数据集分别统计片段数、原视频数、来源组数、无脸/失败片段、选中帧数和新处理片段的耗时。检测器加载耗时单列；重用记录不参与本次处理速度统计。中途失败可加第三个位置参数 `resume` 重试。文件、窗口、代码或检测器权重不匹配时拒绝续用旧结果。

## 试跑后的正式方案

根据实际时长、有效人脸比例和吞吐确定片段长度及原视频/来源组上限，再单独登记正式片段协议。届时需要配套实现来源均衡训练、按原视频/来源组隔离早停、独立真实校准、原视频级汇总与不确定性计算，缓存和报告同时列出片段/原视频/来源组数量。不能只增加清单行数就把片段当独立视频塞进现有头训练。

本轮 `prepare_diagnostic.py` 会拒绝片段试跑清单，防止误用旧整视频训练/校准。原整视频 Bayesian Deep SVDD 流程仍可独立运行。NIW、Ledoit–Wolf、深度和多种子对照保持后续/平行实验地位。

## 本地验证

`tests/test_segment_pilot.py` 检查有理数帧率、未知时长、错误记录与续跑、目录安全、真实 PTS 边界、非零流起点、来源抽样上限、角色继承、片段/原视频计数，以及片段试跑不能进入旧整视频训练。解码器与人脸模型在测试中使用替身；真实视频、RetinaFace CUDA 提供器和速度仍由远程小批试跑验证。

实现按 [PyAV seek 文档](https://pyav.org/docs/stable/api/container.html#av.container.InputContainer.seek) 使用流时间基和关键帧寻址，再按帧 PTS 筛选窗口。

## 检测漏检复核与回退试跑

作业 281 的 20 个片段均处理完成，其中 5 个无脸、1 个仅检出一帧。原始画面核查发现 TalkingHead 的部分窗口落在文字页、片尾和黑屏；HDTF 的一个视频则有清晰人脸却漏检。CPU 定点复测中，该 HDTF 视频三帧原图在阈值 0.5 下均漏检，加黑边后均检出正确人脸，候选分数约 0.87；正常人脸对照仍检出，文字页对照仍未检出。这是少量技术诊断证据，尚不代表全量效果。

新增显式选择的 `pad_on_empty_v1`：每帧先作原检测，仅当原检测没有任何候选时，在四周填充黑色（上下各为原高度的一半取整，左右各为原宽度的一半取整）重试。检测输入仍为 640×640，阈值 0.5、NMS 0.4；框与五点关键点减去填充偏移后映射回原图。从原图做人脸对齐裁剪，黑边不作为 GenD 的输入增强。回退候选必须有有限正面积框，框中心和五点关键点都在原图范围内；不符合者记录并排除。规则对全部数据集、角色和真假标签一致，不作标签或人物特例。

默认 `none` 保留原行为，第二步及上游 GenD 文件不改。新政策进入记录和试跑契约，包含辅助代码哈希；切换政策或代码时必须使用新 work 目录，原作业 281 的结果保留。记录和汇总新增回退尝试帧数、恢复帧数、被拒候选数、最终选中帧中的回退帧数，以及有效但少于 32 帧的片段数。`ok` 仍表示至少有一帧可裁剪，不是质量验收标准；尚未据此删除少帧片段。

先复用原来的 `segment_pilot.json` 和 20 个时间窗口，单独启动回退试跑：

```bash
sbatch --export=ALL,GEND_DETECTION_POLICY=pad_on_empty_v1 \
  --job-name=gend-pad-pilot \
  GenD_anomaly_detector/slurm/segment_pilot.sbatch \
  "$PILOT/segment_pilot.json" "$PILOT/faces_pad_on_empty_v1"
```

须比较同一批片段的检出数量、裁剪质量和耗时；回退可能产生误检，恢复的片段仍需目视核查。TalkingHead 有效时段选择是另一个待验证问题，当前回退试跑保持原窗口不变，以便分别观察两项影响。正式方案锁定后，所有参与对比的方法必须共用同一版预处理和实际取帧清单。

## 下一阶段：中间时段候选与持续人脸覆盖

作业 285 的恢复裁剪已在本地核查：HDTF 新选中的 64 张人脸裁剪正常；TalkingHead 两张回退帧是低亮度人脸。另一片尾窗口唯一一帧也是淡出过程中的人脸，并非文字误检，但不能代表持续 10 秒的人脸画面。接下来保持检测配置，集中检查候选窗口位置及人脸覆盖。

新增 `--placement interior_v1`，保留默认 `timeline` 以追溯旧试跑。新选项在原视频内部均匀布置窗口中心，时长充足时避开两端；为保持不重叠和留出边距，较短视频自动减少窗口数，短于 10 秒则保留整段（仍受最短 2 秒限制）。例如 100 秒视频最多选四段：15–25、35–45、55–65、75–85 秒。选择不读取检测分数、真假判别结果或最终测试画面。

本轮沿用相同整视频清单、来源组、角色及种子 42；每个角色/数据集/标签选 2 个来源组，每组 1 个原视频，每个原视频最多 4 个 10 秒候选窗口。对当前输入预计产生 36 个片段：HDTF 与 TalkingHead 各 16 段、DFDC 开发部分 4 段。最终测试集继续保留。预计数量以生成的清单为准，不以不足数量为由临时换人或跨角色补样。

所有窗口都保留原始记录和检测数组，并报告以下暂定诊断标记：至少选中 32 帧、检测比例至少 50%、将窗口按实际 PTS 分成 5 个等长时间格且至少 4 格的人脸检测比例达到 50%。三个条件同时成立时记为 `sustained_face_candidate`；汇总列出候选片段数及具有候选片段的原视频数。这只是预处理试跑的可解释标记，不是最终质量标准、训练样本剔除条件或测试过滤规则，也不保证亮度、人物连续性及裁剪质量。没有通过的窗口同样保存，不能隐去困难样本。

```bash
"$py" GenD_anomaly_detector/scripts/prepare_segment_pilot.py \
  --manifest "$HOME/gend_steps34_manifest_pilot.csv" \
  --duration-audit work/video_duration_audit_279 \
  --out "$NEW_PILOT" --seconds 10 --clips-per-original 4 \
  --groups-per-stratum 2 --minimum-seconds 2 --seed 42 \
  --placement interior_v1
```

该步骤仍属于预处理流程验证。完成后确定有效窗口及来源均衡规则，登记正式片段协议，并接通训练、早停、独立校准与原视频级汇总。然后才运行两个冻结 GenD 编码器提取特征、在 CPU 上训练第三/四步贝叶斯头和进行开发集比较。正式片段训练尚不能直接使用试跑 CSV。
