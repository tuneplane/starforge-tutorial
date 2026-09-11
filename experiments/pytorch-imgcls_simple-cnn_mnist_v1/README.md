# pytorch-imgcls_simple-cnn_mnist_v1

原生 PyTorch CNN 示例：`simple_cnn` 在 MNIST 上做监督图像分类。本仓第一个**不是**语言模型
后训练的实验 —— 用来演示平台对普通 PyTorch 训练的支持，以及两条数据接入链路。

## 目标

证明"平台不只跑 LLM 后训练"。这个方法是 catalog 里唯一属于 `supervised` 类别的：
带标注的输入/目标对、从初始化开始训练、**spec 里没有 base model**。

同时演示完整闭环：数据准备 → 推成平台数据集 → 提交 → 控制台看曲线 → 取 checkpoint。

## 数据准备（先做）

```bash
# 在仓库根目录。只用标准库，不需要装 torch/torchvision
sf dataset prepare mnist                      # 下载到 datasets/mnist/MNIST/raw/（带 md5 校验，可重复执行）
sf dataset push mnist v1 datasets/mnist       # 推成平台数据集（进你自己的命名空间）
```

训练侧当然要 torchvision，但那是平台镜像里的事 —— 本仓 `dependencies = []` 是刻意的，
为 11MB 数据在笔记本上装 2GB 依赖不合算。

`datasets/**/raw/` 同时在 `.gitignore` 和平台的上传排除清单里 —— 原始数据既不进 git，
也不会被打进作业包。它只通过平台数据集到达集群，节点不需要出网。

## 组成

- `config.yaml` — 平铺配置，**没有 defaults 继承**。每个键都由 recipe 声明，
  `sf validate` 在本地就能查出拼错和越界。
- `train.py` — 训练循环，**归你所有**。要换模型或换数据管线，改 `build_model` /
  `build_datasets` 两个函数即可；其余部分是平台契约（adapter 传进来的参数、只在 rank 0
  上报、产物写在输出目录下）。
- `recipe.lock.json` — 固定 recipe、入口、digest 与 framework 版本（`pytorch@2.14.0`，
  即 torch 版本本身）。

## 运行

```bash
sf submit pytorch-imgcls_simple-cnn_mnist_v1 \
  --profile <卡型>:1 \
  --train-dataset <你的用户名>/mnist@v1
```

单卡就够。多卡时 `batch_size` 是**每进程**的，全局 batch = 该值 × 卡数，adapter 会把
拓扑编译成真正的 torchrun 进程组（两机及以上会建立 rendezvous，而不是各训各的）。

限时演示可以不改文件：

```bash
sf submit pytorch-imgcls_simple-cnn_mnist_v1 --profile <卡型>:1 \
  --train-dataset <你的用户名>/mnist@v1 --set max_steps=200
```

`--set` 的类型和区间在提交前就按 recipe 声明校验，拼错不会排队。

## 换一条数据链路：用 Volume

数据集是不可变、带版本的副本；Volume 是就地维护的受治理目录，没有版本，当前内容就是内容。
适合大到 / 敏感到不适合推成数据集的素材。改两处即可：

```bash
sf volume create cats-and-dogs
sf volume push ./local-images --name cats-and-dogs
```

```yaml
# config.yaml
data:
  volumes:
    - <你的用户名>/cats-and-dogs
input_pipeline: imagefolder        # <data_dir>/{train,val}/<类名>/*
data_dir: ${VOLUMES_DIR}/cats-and-dogs
num_classes: 2
arch: resnet18                     # torchvision 架构要 3 通道，MNIST 的 1 通道会被明确拒绝
```

平台把 volume 只读挂载在 `$VOLUMES_DIR/<名字>`，路径由平台决定，所以配置里写的是名字、
让 `data_dir` 去展开 —— 不要硬编码路径。忘了声明 volume 时，`data_dir` 会带着那个没设上的
变量名直接失败，而不是读到空目录拿空数据训练。

## 为什么不用 custom/custom

shell 脚本也能调 torchrun，差别在这四点：

| | `custom/custom` | 本实验 |
| --- | --- | --- |
| 多机 | 一个容器、没有协调者，各训各的 | 真正的 rendezvous，由计费拓扑编译出来 |
| 超参 | `--set` 落不到任何地方 | 声明式，进队列前校验类型与区间 |
| 镜像 | 每次提交都要传 `--image` | 平台镜像，按 `runtime_id` 解析 |
| 可观测性 | `external`，自带 URL 且镜像里要装 SDK | `platform`，开箱有曲线 |

## 产物与曲线

- 曲线：`train/loss`、`validation/loss`、`validation/accuracy`、`train/lr`，由 rank 0 通过
  `starforge.report` 上报，控制台直接可见。
- checkpoint：输出目录下 `checkpoints/`（`best.pt` + 每次评估的那个 epoch 一个）。
- 评测报告：`eval/report.json`，由 `train.py` 自己写出 —— 这个 recipe 没有 `sf export` /
  `sf eval`（分类器没有有意义的 HF 导出，这里也没有给它打分的基准 harness）。

## 前置条件（运维一次性）

`pytorch-2.14.0` 是部署侧构建的镜像，catalog 不内嵌地址：

```bash
./deploy/docker/build-runtimes.sh pytorch --push     # 在 starforge 仓库
```

然后在控制台填：**设置 → 运行时 → 镜像 → 「PyTorch 默认镜像」**，给个 tag 即可，
保存立即生效。没配的话提交会明确报 `runtime_id='pytorch-2.14.0' has no built-in OCI source`。
