# 数据集还是 Volume

两者不是「哪个更好」，是回答**两个不同的问题**。

| | 数据集 | Volume |
| --- | --- | --- |
| 它回答的问题 | 三个月后还能说清**训的是哪一版** | 这批材料**谁能看、能不能带走** |
| 标识 | `<owner>/<name>@<version>`，封版后不可覆盖 | `<owner>/<name>`，当前内容即内容 |
| 改数据 | 起一个新版本 | 直接增删文件 |
| 可追溯 | 版本钉死 | 事后无法回答「这次 run 读了哪些文件」 |
| 到达作业 | 拉进共享缓存 + sha256 校验 | 只读挂载；有治理文件系统时**不复制** |
| 很多小文件 | 每个文件一次签名 | 500 个/次批量签名，已存在的跳过 |
| 质量闸 | 重复率 / 与评测集重合 / 敏感值扫描 | 无（它不是训练材料） |
| 下载 | 默认允许 | **默认禁止**，开启要确认并审计 |

**一句话规则：会变的进 Volume，不会变的进数据集。**

这条线好用，是因为它自动决定了另外三件事该往哪边倒：版本（只有不变的东西才需要版本号）、
质量闸（只有不变的东西才值得扫一次就把结论记住）、下载策略（会变的、人在维护的，通常是内部材料）。

## 怎么写

数据集在提交时引用，或写进 `config.yaml` 的 `data.train.dataset`：

```bash
tuneplane dataset push my-data v1 <目录>
tuneplane submit <实验> --profile <卡型>:1 --train-dataset <你>/my-data@v1
```

Volume 只读挂在 `$VOLUMES_DIR/<名字>`，路径由平台定 —— 所以配置里写名字、让 `data_dir`
去展开，**不要硬编码路径**：

```bash
tuneplane volume create my-photos
tuneplane volume push ./photos --name my-photos
tuneplane submit <实验> --profile <卡型>:1 --volume <你>/my-photos
```

```yaml
input_pipeline: imagefolder
data_dir: ${VOLUMES_DIR}/my-photos
```

## 推荐的组合不是二选一

原始素材放 Volume 就地维护 → 预处理作业读 Volume、声明 `--output-dataset` 产出打包好的
训练集 → 训练读那个数据集。一次成本，长期可复现。

平台会把治理属性顺着这条边传下去：产出的数据集继承所读 volume 里**最严**的下载策略，
并记下 `produced_from`。
