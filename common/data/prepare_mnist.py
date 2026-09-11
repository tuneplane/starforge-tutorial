#!/usr/bin/env python
"""下载 MNIST，按 torchvision 的目录布局写到 datasets/mnist/，供 pytorch recipe 使用。

只用标准库，**不需要装 torch/torchvision**。训练侧当然要 torchvision，但那是平台镜像里的事；
为了 11MB 数据在笔记本上装 2GB 依赖不合算，演示前多一步 pip 也是多一个出错的地方。
本仓 `dependencies = []` 是刻意的，这个脚本尊重它。

和本仓其它 prepare 脚本不同的是：产物不是 jsonl，而是一个**目录**。图像语料本来就是目录，
`pytorch/image-classification` 拿到的也是挂载点而不是文件。所以下一步不是 export 一个
`*_DATA_DIR` 环境变量，而是把这个目录推成平台数据集：

    sf dataset prepare mnist
    sf dataset push mnist v1 datasets/mnist
    sf submit pytorch-imgcls_simple-cnn_mnist_v1 --profile h100:1 \
        --train-dataset <你的用户名>/mnist@v1

推上去之后，作业启动时平台把这个版本拉进共享缓存并注入目录路径，集群节点不需要出网。
`datasets/**/raw/` 在 .gitignore 里，也在平台的上传排除清单里 —— 原始数据不进 git，
也不会被打进作业包。

布局要和 torchvision 对齐：`<out>/MNIST/raw/` 下同时放 `.gz` 与解压后的文件。
训练时 `datasets.MNIST(root, download=False)` 校验的是**解压后**那四个名字，所以解压不能省。
"""
import gzip
import hashlib
import os
import shutil
import urllib.request

import typer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# torchvision 用的同一个镜像（lecun.com 原站常年 403）。
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# md5 取自 torchvision 下载并校验过的那份副本，不是抄来的常量。
# 校验和不是可选项：截断的下载会安静地变成一份训不出东西的数据，
# 而那种失败看起来像模型不行，不像数据不对。
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: str) -> str:
    digest = hashlib.md5()  # noqa: S324 - 对齐上游的校验口径，不是安全用途
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(name: str, expected_md5: str, raw_dir: str) -> None:
    archive = os.path.join(raw_dir, name)
    if os.path.isfile(archive) and _md5(archive) == expected_md5:
        print(f"  {name} 已存在且校验通过")
    else:
        url = MIRROR + name
        print(f"  下载 {name} …")
        with urllib.request.urlopen(url, timeout=60) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
        got = _md5(archive)
        if got != expected_md5:
            os.remove(archive)
            raise SystemExit(f"{name} 校验和不符：期望 {expected_md5}，实得 {got}（下载被截断？）")

    # 解压后的文件才是训练时 torchvision 认的那四个名字。
    extracted = archive[: -len(".gz")]
    if not os.path.isfile(extracted):
        with gzip.open(archive, "rb") as src, open(extracted, "wb") as dst:
            shutil.copyfileobj(src, dst)


def main(
    out: str = typer.Option(
        os.path.join(REPO_ROOT, "datasets", "mnist"),
        "--out",
        help="输出目录；文件写在其下的 MNIST/raw/",
    ),
) -> None:
    """下载 MNIST 训练集与测试集 -> <out>/MNIST/raw/。重复执行是安全的。"""
    raw_dir = os.path.join(out, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    for name, expected in FILES.items():
        _fetch(name, expected, raw_dir)

    print(f"\n写入 {len(os.listdir(raw_dir))} 个文件 -> {raw_dir}")
    print("\n完成。下一步把它推成平台数据集：")
    print(f"  sf dataset push mnist v1 {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    typer.run(main)
