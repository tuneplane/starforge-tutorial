# pytorch-imgcls_simple-cnn_digit-images_v1

训练**你自己上传的图片** —— 一张张 PNG，按类别分目录。和
`pytorch-imgcls_simple-cnn_mnist_v1` 是同一个方法、同一份 `train.py`，差别只在数据形态：
那个读 MNIST 原生的 idx 二进制，这个读 `ImageFolder`，也就是你把照片按类别分好目录的样子。

## 目标

演示真实场景里最常见的那条路：一堆分好类的图片 → 上传 → 训练 → 看曲线 → 取 checkpoint。
顺带把「图片该放数据集还是 Volume」这个选择讲清楚（见最后一节）。

## 数据准备（先做）

素材是真实的 MNIST 手写数字，解码成 PNG 并摆成 `ImageFolder` 布局 —— 和你把自己拍的
照片按类别分目录，形态完全一样。存的是 RGB 而不是灰度，因为真实上传的照片就是三通道，
让教程走真实那条路而不是绕开一个真实的坑。

```bash
sf dataset prepare mnist          # 原始 idx 文件（纯标准库下载，带 md5 校验）
sf dataset prepare digit_images   # 解码成 2000 张训练图 + 500 张验证图
sf dataset push digit-images v1 datasets/digit-images
```

```
datasets/digit-images/
├── train/0/00015.png …   每类 200 张
└── val/0/00061.png …     每类 50 张
```

`datasets/**/raw/` 同时在 `.gitignore` 和平台的上传排除清单里 —— 原始数据不进 git，
也不会被打进作业包。

## 组成

- `config.yaml` — 平铺配置，没有 defaults 继承。每个键都由 recipe 声明，
  `sf validate` 在本地就能查出拼错和越界。
- `train.py` — 训练循环，**归你所有**。换模型或换数据管线改 `build_model` /
  `build_datasets` 两个函数即可；其余是平台契约（adapter 传进来的参数、只在 rank 0 上报、
  产物写在输出目录下）。
- `recipe.lock.json` — 固定 recipe、入口、digest 与 framework 版本。

## 运行

```bash
sf submit pytorch-imgcls_simple-cnn_digit-images_v1 \
  --profile <卡型>:1 \
  --train-dataset <你的用户名>/digit-images@v1
```

单卡就够。多卡时 `batch_size` 是**每进程**的，全局 batch = 该值 × 卡数，adapter 会把拓扑
编译成真正的 torchrun 进程组（两机及以上建立 rendezvous，而不是各训各的）。

在你自己的仓库里更推荐把数据集写进 `config.yaml` 的 `data.train.dataset`，命令行参数只是
临时覆盖；教程仓库不能写死某个人的用户名，所以这里用参数。

实测（H200 单卡，5 个 epoch）：

```
[train] epoch 0 step 32  top1=0.8120 loss=0.6038
[train] epoch 4 step 160 top1=0.9280 loss=0.2097
[train] done best_top1=0.9280
```

## 数据集还是 Volume

两者不是「哪个更好」，是回答**两个不同的问题**：

| | 数据集 | Volume |
| --- | --- | --- |
| 它回答的问题 | 三个月后还能说清**训的是哪一版** | 这批材料**谁能看、能不能带走** |
| 标识 | `<owner>/<name>@<version>`，封版后不可覆盖 | `<owner>/<name>`，当前内容即内容 |
| 改数据 | 起一个新版本 | 直接增删文件 |
| 可追溯 | 版本钉死 | **事后无法回答「这次 run 读了哪些文件」**（ADR-0011 明确记下了这个取舍） |
| 到达作业 | 拉进共享缓存 + sha256 校验 | 只读挂载；有治理文件系统时**不复制** |
| 很多小文件 | 每个文件一次签名 | 500 个/次批量签名 + 已存在跳过 |
| 质量闸 | 重复率 / 与评测集重合 / 敏感值扫描 | 无（它不是训练材料） |
| 下载 | 默认允许 | **默认禁止**，开启要确认并审计 |

**一句话规则：会变的进 Volume，不会变的进数据集。**

本实验的 2500 张图是脚本生成的、不会再变，所以走数据集 —— 顺带拿到版本号和校验。
真实场景里研究员自己一直在加的原始图片属于另一边：

```bash
sf volume create my-photos
sf volume push ./photos --name my-photos
sf submit <实验> --profile <卡型>:1 --volume <你的用户名>/my-photos
```

```yaml
# config.yaml：平台把 volume 只读挂在 $VOLUMES_DIR/<名字>，路径由平台定，
# 所以写名字、让 data_dir 去展开，不要硬编码路径。
input_pipeline: imagefolder
data_dir: ${VOLUMES_DIR}/my-photos
```

**推荐的组合不是二选一**：原始图片放 Volume 就地维护 → 预处理作业读 Volume、
声明 `--output-dataset` 产出打包好的训练集 → 训练读那个数据集。一次成本，长期可复现。
平台会把治理属性顺着这条边传下去：产出的数据集继承所读 volume 里**最严**的下载策略，
并记下 `produced_from`。

## 产物与曲线

`train/loss`、`validation/loss`、`validation/accuracy`、`train/lr` 由 rank 0 通过
`starforge.report` 上报，控制台直接可见。checkpoint 在输出目录的 `checkpoints/`
（`best.pt` + 每次评估的那个 epoch 一个），`eval/report.json` 由 `train.py` 自己写出 ——
这个 recipe 没有 `sf export` / `sf eval`（分类器没有有意义的 HF 导出，也没有给它打分的基准）。
