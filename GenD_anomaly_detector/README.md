# GenD anomaly detector

以 GenD（WACV 2026）为特征提取器，在其上接贝叶斯头做视频深伪检测。

**本项目选用 GenD-PE**（Perception Encoder L/14，官方权重 `yermandy/GenD_PE_L`）。
官方仓库同时支持 CLIP、PE、DINOv3 三种 backbone，CLIP 和 DINOv3 不在主线中使用。
前一阶段 3D-CNN 实验见 `E:\PhD\Deepfake Video 3D-CNN+BCNN v2(1)`，背景摘要见上级目录 `PREVIOUS_PROJECT_CONTEXT.md`。

## 目录约定

| 路径 | 内容 | 规则 |
| --- | --- | --- |
| `third_party/GenD/` | 官方仓库 https://github.com/yermandy/GenD | **保持原样不改**。需要改动时在外层写包装或子类 |
| （待建）`src/`、`configs/`、`scripts/` | 本项目自己的代码：数据清单、贝叶斯头、评估 | |

上游版本：commit `387a42266dd385fbe8f7626c5d3aa03eea3bfaab`（2026-04-21，"Update bibtex"），克隆于 2026-11-12。

## 从官方代码核对到的要点

**权重**：Hugging Face 官方发布了三个版本：`yermandy/GenD_CLIP_L_14`、`yermandy/GenD_PE_L`、`yermandy/GenD_DINOv3_L`。
加载方式见 `third_party/GenD/README.md`（`src.hf.modeling_gend.GenD.from_pretrained`）。

**PE 版本的论文配置**（`src/exp/wacv_rebuttal.py`，配置名 `wacv-PE_L-*`）：

- backbone：`vit_pe_core_large_patch14_336`（timm），输入 `img_size=224`
- 可训练参数：只放开 LayerNorm（`norm_pre`、`norm1`、`norm2`、`norm`），其余冻结
- 头：`LinearNorm`，即特征先 L2 归一化，再接 Linear(D, 2)
- 损失：交叉熵；完整 GenD 版本再加 uniformity 和 alignment 损失（论文主配置为 U=0.5、A=0.1，仓库里另有其他取值的扫描）
- 优化：AdamW，lr 3e-4，cyclic cosine（每 10 epoch 一个周期），最多 30 epoch，早停 patience 30，batch 512，bf16-mixed
- 增强（`config.Augmentations` 默认值）：水平翻转 0.5、仿射（±10°、平移 0.1、缩放 0.9–1.1）、高斯模糊 0.1、亮度和对比度扰动 0.1、JPEG 质量 40–100
- checkpoint 选择指标：`val/mAP_video`
- 视频级分数：帧概率取平均（`reduce_video_predictions="mean"`）
- 训练数据：FF++ 训练集，每个视频随机取 32 帧，用 5 个 seed 各自抽帧（`split-0..4`）

**人脸预处理**（`detector.py`，RetinaFace）：

```bash
python detector.py -i <视频目录> -m at_least -n 32 -o <输出目录> --det_thres 0.1 -s 1.3 --target_size none
```

即人脸框外扩 1.3 倍、检测阈值 0.1、每个视频至少取 32 帧、保存原尺寸裁剪。
官方处理后的数据集放在 https://huggingface.co/datasets/yermandy/GenD 。

## 需要注意的点

- 官方的验证集包含 **CDFv3 验证集、DeepSpeak v1.1 / v2 验证集和 FFIW**（`get_common()`）。
  也就是说，官方发布的权重是用这些数据**选出来的 checkpoint**。
  - 拿官方权重在 Celeb-DF v3 上测零样本结果时，要核对它的 CDFv3 划分和我们的测试集是否重叠。
  - DeepSpeak 作为测试集时同理。
- 仓库里同时包含 Effort、ForAda、FSFM 的实现（`src/model/`），以后做其他提取器时可以沿用同一框架。
