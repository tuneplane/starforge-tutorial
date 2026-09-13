# pytorch-imgcls_simple-cnn_digit-images_v1

用一个 CNN 训练按类别分好目录的图片。本仓唯一一个非 LLM 的实验，也是研究员自己的数据
最常见的形态。

## 跑起来

```bash
tuneplane dataset prepare digit_images                        # 生成 2500 张 PNG（纯标准库，几秒）
tuneplane dataset push digit-images v1 datasets/digit-images  # 推成平台数据集
tuneplane submit pytorch-imgcls_simple-cnn_digit-images_v1 \
  --profile <卡型>:1 --train-dataset <你的用户名>/digit-images@v1
tuneplane job logs <job id>
```

实测 H200 单卡 5 个 epoch，`best_top1=0.9280`，约一分钟。

## 换成你自己的数据

把图片按类别分目录，替换 `datasets/digit-images/` 就行，**代码一行不用改**：

```
train/<类名>/*.png
val/<类名>/*.png
```

改 `config.yaml` 里的 `num_classes`，重新 push 一个新版本，再 submit。

要换模型或换数据管线，改 `train.py` 的 `build_model` / `build_datasets` —— 这两个函数归你；
其余部分是平台契约（参数怎么传进来、只在 rank 0 上报、产物写在哪）。

## 会得到什么

- 控制台曲线：`train/loss`、`validation/loss`、`validation/accuracy`、`train/lr`
- `checkpoints/best.pt`，以及每次评估的 epoch 快照
- `eval/report.json`

## 几个会被问到的点

- `batch_size` 是**每进程**的：多卡时全局 batch = 该值 × 卡数
- 多机会编译成真正的 torchrun 进程组（有 rendezvous，不是各节点各训各的）
- 这个方法没有 `tuneplane export` / `tuneplane eval` —— 分类器没有有意义的 HF 导出，平台也没有给它打分的基准
- 数据该放数据集还是 Volume：见 [docs/datasets-vs-volumes.md](../../docs/datasets-vs-volumes.md)
- 全部可调参数：`tuneplane methods pytorch/image-classification`
