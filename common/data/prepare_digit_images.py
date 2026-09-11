#!/usr/bin/env python
"""把 MNIST 解码成一张张 PNG，按 ImageFolder 布局写出 —— 用来演示「上传自己的图片」。

产物形如 `datasets/digit-images/{train,val}/<类名>/*.png`，也就是
`torchvision.datasets.ImageFolder` 认的那个布局，和你把自己拍的照片按类别分目录放好
完全一样。素材本身是真实的手写数字，不是合成噪声。

为什么存成 RGB 而不是灰度：真实场景上传的是照片，三通道。存成 RGB 让这条链路和真实
情况走同一条路（ImageFolder → ToTensor → 3 通道），而不是在教程里躲开一个真实的坑。

只用标准库 —— PNG 编码就在下面，三十行。本仓 `dependencies = []` 是刻意的。

    sf dataset prepare mnist            # 先要有 idx 原始文件
    sf dataset prepare digit_images     # 本脚本
    sf volume create digit-images
    sf volume push datasets/digit-images --name digit-images
"""
import os
import struct
import zlib

import typer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MNIST_RAW = os.path.join(REPO_ROOT, "datasets", "mnist", "MNIST", "raw")


def _png_rgb(gray: bytes, w: int, h: int) -> bytes:
    """8-bit RGB PNG。每行前面那个 0 是 filter byte，PNG 规定每条扫描线都要有。"""
    rows = []
    for y in range(h):
        line = bytearray(b"\x00")
        for value in gray[y * w:(y + 1) * w]:
            line += bytes((value, value, value))
        rows.append(bytes(line))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))  # 2 = truecolour
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )


def _read_idx(images: str, labels: str) -> tuple[list[bytes], list[int], int, int]:
    """读 MNIST 的 idx 格式：一个头，然后是连续的像素/标签。"""
    with open(images, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise SystemExit(f"{images} 不是 idx3 图像文件（magic={magic}）")
        blob = f.read(count * rows * cols)
    with open(labels, "rb") as f:
        magic, label_count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise SystemExit(f"{labels} 不是 idx1 标签文件（magic={magic}）")
        label_bytes = f.read(label_count)
    size = rows * cols
    return [blob[i * size:(i + 1) * size] for i in range(count)], list(label_bytes), rows, cols


def _export(split: str, images: str, labels: str, out_root: str, per_class: int) -> int:
    pixels, targets, rows, cols = _read_idx(images, labels)
    written = {c: 0 for c in range(10)}
    total = 0
    # strict=True：图像数与标签数对不上是数据损坏，不该安静地按短的那个截断。
    for index, (raw, label) in enumerate(zip(pixels, targets, strict=True)):
        if written[label] >= per_class:
            if all(n >= per_class for n in written.values()):
                break
            continue
        folder = os.path.join(out_root, split, str(label))
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"{index:05d}.png"), "wb") as f:
            f.write(_png_rgb(raw, cols, rows))
        written[label] += 1
        total += 1
    return total


def main(
    out: str = typer.Option(
        os.path.join(REPO_ROOT, "datasets", "digit-images"), "--out", help="ImageFolder 根目录"
    ),
    per_class: int = typer.Option(200, "--per-class", help="每类导出多少张训练图"),
    val_per_class: int = typer.Option(50, "--val-per-class", help="每类导出多少张验证图"),
) -> None:
    """MNIST idx -> <out>/{train,val}/<0-9>/*.png。"""
    train_images = os.path.join(MNIST_RAW, "train-images-idx3-ubyte")
    if not os.path.isfile(train_images):
        raise SystemExit(
            f"没找到 {train_images}\n先跑 `sf dataset prepare mnist` 下载原始文件"
        )

    n_train = _export("train", train_images, os.path.join(MNIST_RAW, "train-labels-idx1-ubyte"),
                      out, per_class)
    n_val = _export("val", os.path.join(MNIST_RAW, "t10k-images-idx3-ubyte"),
                    os.path.join(MNIST_RAW, "t10k-labels-idx1-ubyte"), out, val_per_class)

    print(f"\n写入 {n_train} 张训练图 + {n_val} 张验证图 -> {out}")
    print("\n完成。下一步把整个目录推成 Volume：")
    rel = os.path.relpath(out, REPO_ROOT)
    print("  sf volume create digit-images")
    print(f"  sf volume push {rel} --name digit-images")


if __name__ == "__main__":
    typer.run(main)
