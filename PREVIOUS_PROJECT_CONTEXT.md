# V2 项目实验背景（2026-11-11 核对）

源项目：`E:\PhD\Deepfake Video 3D-CNN+BCNN v2(1)`。本文件根据源码、配置、训练 history 和评估 JSON 整理；源项目未修改。

## 研究目标与实现

- Phase A：用真实和伪造视频标签监督训练三层 3D CNN 特征提取器与线性分类头。CelebDF++ 使用离线 dlib 人脸框和对齐的人脸裁剪，DFD 使用整帧。主要输入为 32 连续帧，输出视频级伪造分数。
- 原计划的 Phase C：冻结 Phase A 特征提取器和 BatchNorm，只用真实视频训练 Pyro 均值场贝叶斯头（特征维度 → 256 → 64 → 1），用异常分数检测伪造。代码和配置已经存在，但现有 `实验结果` 中未发现 Phase C 的训练结果或评估报告。
- 入口：`pretrain_extractor.py`、`train_3d_bcnn.py`、`evaluate_3d_bcnn.py`；模型：`src/video_bcnn/model.py`；协议与采样：`src/video_bcnn/protocols.py`、`src/video_bcnn/experiment.py`；主要配置：`configs/v2/`。

## 已完成的不同协议，不能直接横向比较

| 协议 | 训练/评估范围 | 已存档结果 |
| --- | --- | --- |
| 原 P0.5 消融 | CelebDF++ 全量开发验证 8,205 视频；seed 42 | E0 AUROC 0.6827；E1 0.6869；E2 0.8058；E3（4×4 空间压缩）0.7819；E4（TCN+GAP）0.8064。E2 到 E4 提升仅约 0.0007。|
| DFD E2 | DFD 完整验证 527 视频；整帧输入 | AUROC 0.5747。与 CelebDF++ 的裁剪输入和数据分布不同。|
| 修订开发验证 | 训练 622 真 + 1,244 假；原 val/test 合并为 16,503 视频开发验证；没有独立 test | E2 AUROC 0.7930；固定 1:2 开发子集 801 视频 AUROC 0.7985。该结果不能作为独立测试性能。|
| 30 组 video-disjoint E2 | 30 个固定、不重叠的伪造训练组；每组 616 真 + 616 假，100 epoch；固定 val 134 真 + 134 假；最终 test 133 真 + 133 假 | 30 组最佳 val AUROC 均值 0.7827、标准差 0.0203、范围 0.7282–0.8364。按 val 选出的第 19 组最佳 val AUROC 0.8364（epoch 36），独立 test AUROC 0.7671，身份聚类 95% CI [0.7250, 0.8097]。|

30 组实验的第 19 组测试集：266 视频，阈值来自验证集；准确率/平衡准确率 0.6353，TN/FP/FN/TP = 120/13/84/49。固定 0.5 伪造概率阈值下准确率 0.7068。另有第 02 组的测试报告，AUROC 0.7536；两者不是 30 次独立测试估计。源项目 `docs/phase_a_video_disjoint_30_group.md` 记录了固定分组、复用 pilot 和选择规则。

## 后续工作应继承的边界

1. 先明确沿用哪个协议和 checkpoint。若需要独立测试结论，30 组协议的第 19 组是已有明确选择流程的基线；原 P0.5 与修订协议的验证指标不能和其 test 指标直接比较。
2. 30 组 E2 模型使用 22×22 空间输出，特征维度为 15,488；现有 Phase C 配置的 TCN/4×4 提取器输出 512 维。接贝叶斯头前需明确如何复用权重与特征维度，不能直接套用该配置。
3. 已有 CelebDF++ 人脸框单变量捷径诊断；早期完整验证结果可能受到数据集偏差影响。后续比较需保留相同视频 ID、预处理、clip 数和聚类统计口径。
4. 30 组只在验证集选一个模型；不要根据已有 test 结果重新挑模型。该 test 集已经被评估，继续据此调整设计会削弱它作为最终独立测试的作用。

## 关键存档位置（相对于源项目）

- `实验结果/E2_30runs_backup_20261029/artifacts/v2_video_disjoint/`：30 组 best/last checkpoint、配置、runtime、逐 epoch history。
- `实验结果/E2_two_model_final_tests/artifacts/v2_video_disjoint/e2_group19_formal/`：第 19 组 best checkpoint 与最终 test 报告及逐视频/逐 clip 分数。
- `实验结果/CelebDFv3_E0-E2_seed42_fullval-audit_20260923/`、`CelebDFv3_E3_spatial4x4_seed42_fullval-audit_20260924/`、`CelebDFv3_E4_TCN-gap_seed42_fullval-audit_20260925/`：早期消融实验。
- `实验结果/E2_revised_seed42_bundle/`：16,503 视频修订开发验证。
- `实验结果/dfd_e2_step3_w0_full60_bundle/`：DFD E2。

下一步开始具体实验设计时，应先核对所选 checkpoint 的保存配置、数据清单哈希、输入预处理，以及当前 GPU 环境；本轮只做了文件审阅，没有执行训练或复制大体积数据。
