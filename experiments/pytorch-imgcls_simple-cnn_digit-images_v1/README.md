# pytorch-imgcls_simple-cnn_digits-volume_v1

训练**你自己上传的图片**。和 `pytorch-imgcls_simple-cnn_mnist_v1` 是同一个方法、同一份
`train.py`，差别只有一处：数据从 **Volume** 来，不是平台数据集。两条链路都值得会一遍。

## 目标

演示真实场景里最常见的那条路：一堆按类别分好目录的图片 → 上传 → 训练。
顺带把「什么时候用数据集、什么时候用 Volume」这个选择讲清楚（见最后一节）。

## 数据准备（先做）

素材是真实的 MNIST 手写数字，解码成一张张 PNG，按 `ImageFolder` 布局摆好 ——
和你把自己拍的照片按类别分目录，是完全一样的形态。

```bash
sf dataset prepare mnist              # 原始 idx 文件（只用标准库下载）
sf dataset prepare digit_images       # 解码成 2000 张训练图 + 500 张验证图
```

```
datasets/digit-images/
├── train/0/00015.png …   每类 200 张
└── val/0/00061.png …     每类 50 张
```

存的是 RGB 而不是灰度，因为真实上传的照片就是三通道 —— 让教程走真实那条路，
而不是绕开一个真实的坑。

## 上传成 Volume

```bash
sf volume create digit-images
sf volume push datasets/digit-images --name digit-images
sf volume ls
```

`sf volume push` 是**批量签名**的（500 个文件一次调用），而且服务端按内容寻址：
已经在的文件直接跳过，换台机器续传也认得。这就是它适合「很多图片」的原因。

## 运行

```bash
sf submit pytorch-imgcls_simple-cnn_digits-volume_v1 \
  --profile <卡型>:1 \
  --volume <你的用户名>/digit-images
```

在你自己的仓库里，更推荐把它写进 `config.yaml` —— Volume 是实验的属性，不是某次提交的：

```yaml
data:
  volumes:
    - <你的用户名>/digit-images
```

教程仓库不能写死某个人的用户名，所以这里用命令行参数。

平台把它**只读**挂在 `$VOLUMES_DIR/digit-images`，路径由平台定；`config.yaml` 里写的是
`data_dir: ${VOLUMES_DIR}/digit-images`，让平台去展开。**不要硬编码路径** —— 换后端就变。
忘了声明 volume 时，训练会带着那个没设上的变量名直接失败，而不是读到空目录训出一个废模型。

## 数据集还是 Volume

两者不是「哪个更好」，是回答**两个不同的问题**：

| | 数据集 | Volume |
| --- | --- | --- |
| 它回答的问题 | 三个月后还能说清**训的是哪一版** | 这批材料**谁能看、能不能带走** |
| 标识 | `<owner>/<name>@<version>`，封版后不可覆盖 | `<owner>/<name>`，当前内容即内容 |
| 改数据 | 起一个新版本 | 直接增删文件 |
| 可追溯 | 版本钉死 | **事后无法回答「这次 run 读了哪些文件」**（[ADR-0011](https://github.com/wccdev/starforge/blob/main/docs/adr/0011-governed-material-is-an-unversioned-volume.md) 明确记下了这个取舍） |
| 到达作业 | 拉进共享缓存 + sha256 校验，注入 `<NAME>_DATA_DIR` | 只读挂载；有治理文件系统时**不复制** |
| 很多小文件 | 每个文件一次签名 | 500 个/次批量签名 + 已存在跳过 |
| 质量闸 | 重复率 / 与评测集重合 / 敏感值扫描 | 无（它不是训练材料） |
| 预览 | jsonl / csv / parquet 可预览 | 任何设置下都没有预览 |
| 下载 | 默认允许 | **默认禁止**，开启要确认并审计 |

按这个顺序问自己：

1. **需要「事后说清训的是哪一版」吗？** 需要 → 数据集。这是它唯一独有的能力。
2. **这些字节允许离开平台吗？** 不允许 → Volume。
3. **它是在原地维护的吗**（文件不断增删，没人想为此起版本号）？ 是 → Volume。
4. **几万个小文件 / 几百 GB？** → Volume，或者**打包后**推成数据集。
5. 都不是 → 数据集，它默认更安全。

**真正推荐的组合不是二选一**：原始图片放 Volume 就地维护 → 预处理作业读 Volume、
声明 `--output-dataset` 产出打包好的训练集 → 训练读那个数据集。一次成本，长期可复现。
而且平台会把治理属性顺着这条边传下去：产出的数据集继承所读 volume 里**最严**的下载策略，
并记下 `produced_from`。

## 产物与曲线

和 mnist 那个实验一样：`train/loss`、`validation/accuracy` 进控制台曲线，
checkpoint 在输出目录的 `checkpoints/`，`eval/report.json` 由 `train.py` 写出。
