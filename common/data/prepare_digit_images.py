#!/usr/bin/env python
"""准备一个按类别分目录的图片数据集 —— 研究员自己的数据长的就是这个样子。

产物是 `datasets/digit-images/{train,val}/<类名>/*.png`，即 torchvision `ImageFolder`
的布局。素材取自 MNIST（真实手写数字，不是合成噪声），存成 RGB PNG，因为真实上传的
照片就是三通道。

只用标准库：下载、校验、解 idx、编码 PNG 全在这个文件里。本仓 `dependencies = []`
是刻意的，为几 MB 数据在笔记本上装 2GB 依赖不合算。

    sf dataset prepare digit_images
    sf dataset push digit-images v1 datasets/digit-images
"""
import gzip
import hashlib
import os
import shutil
import struct
import urllib.request
import zlib

import typer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# 校验和不是可选项：截断的下载会安静地变成一份训不出东西的数据，
# 而那种失败看起来像模型不行，不像数据不对。
SOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: str) -> str:
    digest = hashlib.md5()  # noqa: S324 - 对齐上游校验口径，非安全用途
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(cache: str) -> None:
    """把四个原始文件下到 cache/ 并解压；已存在且校验通过则跳过。"""
    os.makedirs(cache, exist_ok=True)
    for name, expected in SOURCES.items():
        archive = os.path.join(cache, name)
        if not (os.path.isfile(archive) and _md5(archive) == expected):
            print(f"  下载 {name} …")
            with urllib.request.urlopen(MIRROR + name, timeout=60) as r, open(archive, "wb") as f:
                shutil.copyfileobj(r, f)
            got = _md5(archive)
            if got != expected:
                os.remove(archive)
                raise SystemExit(f"{name} 校验和不符：期望 {expected}，实得 {got}（下载被截断？）")
        plain = archive[: -len(".gz")]
        if not os.path.isfile(plain):
            with gzip.open(archive, "rb") as src, open(plain, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _png_rgb(gray: bytes, w: int, h: int) -> bytes:
    """8-bit RGB PNG。每行前面那个 0 是 filter byte，PNG 规定每条扫描线都要有。"""
    rows = [b"\x00" + b"".join(bytes((v, v, v)) for v in gray[y * w:(y + 1) * w]) for y in range(h)]

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))  # 2 = truecolour
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )


def _export(split: str, cache: str, stem: str, out_root: str, per_class: int) -> int:
    with open(os.path.join(cache, f"{stem}-images-idx3-ubyte"), "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise SystemExit(f"{stem} 图像文件格式不对（magic={magic}）")
        blob = f.read(count * rows * cols)
    with open(os.path.join(cache, f"{stem}-labels-idx1-ubyte"), "rb") as f:
        if struct.unpack(">II", f.read(8))[0] != 2049:
            raise SystemExit(f"{stem} 标签文件格式不对")
        labels = f.read(count)

    size = rows * cols
    written = dict.fromkeys(range(10), 0)
    total = 0
    for index in range(count):
        label = labels[index]
        if written[label] >= per_class:
            if all(n >= per_class for n in written.values()):
                break
            continue
        folder = os.path.join(out_root, split, str(label))
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"{index:05d}.png"), "wb") as f:
            f.write(_png_rgb(blob[index * size:(index + 1) * size], cols, rows))
        written[label] += 1
        total += 1
    return total


def main(
    out: str = typer.Option(
        os.path.join(REPO_ROOT, "datasets", "digit-images"), "--out", help="ImageFolder 根目录"
    ),
    per_class: int = typer.Option(200, "--per-class", help="每类多少张训练图"),
    val_per_class: int = typer.Option(50, "--val-per-class", help="每类多少张验证图"),
) -> None:
    """下载 MNIST 并导出成 ImageFolder 布局的 PNG。重复执行是安全的。"""
    # 原始 idx 放在输出目录**之外**：`sf dataset push <目录>` 会上传目录下的每个文件，
    # 放里面就等于把 63MB 的原始二进制混进图片数据集。
    cache = os.path.join(REPO_ROOT, "datasets", ".mnist-source")
    _fetch(cache)
    n_train = _export("train", cache, "train", out, per_class)
    n_val = _export("val", cache, "t10k", out, val_per_class)

    print(f"\n写入 {n_train} 张训练图 + {n_val} 张验证图 -> {out}")
    print("\n下一步：")
    print(f"  sf dataset push digit-images v1 {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    typer.run(main)
