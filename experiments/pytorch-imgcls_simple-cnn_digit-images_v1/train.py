"""Versioned PyTorch image-classification entrypoint. Edit the model and the data, not the contract.

The two functions worth replacing are `build_model` and `build_datasets`. The rest is the platform
contract: the arguments the adapter passes, reporting from rank zero only, and writing every artifact
under the output directory.

Runs under torchrun on one node or several, and as a plain `python train.py` on a laptop -- the only
difference is whether the distributed environment variables are present.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

# `tuneplane` is grouped with the third-party imports on purpose: this file's destination is a user
# repo, where the SDK *is* a third-party package, and that is where `ruff check` runs on it. Keeping
# it in a first-party block would be clean here and flagged in every repo `tuneplane new` copies it into.
from tuneplane.report import finish, init, log
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# Normalisation constants of the two built-in sets. Keeping them here rather than inline is what
# makes swapping the dataset a one-line edit.
_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "imagefolder": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class SimpleCNN(nn.Module):
    """Two convolutions and two linear layers -- the reference MNIST net.

    It is the default because it reaches ~99% in a few minutes on one card, which is what a first
    run on a new platform should do. `arch: resnet18` switches to torchvision instead.
    """

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.pool(self.drop1(x)).flatten(1)
        return self.fc2(self.drop2(F.relu(self.fc1(x))))


def build_model(config: dict[str, Any], in_channels: int) -> nn.Module:
    """Replace this with your own architecture."""
    arch = str(config.get("arch") or "simple_cnn")
    num_classes = int(config.get("num_classes") or 10)
    if arch == "simple_cnn":
        return SimpleCNN(num_classes, in_channels)
    from torchvision import models

    factory = getattr(models, arch, None)
    if factory is None:
        raise SystemExit(f"unknown arch {arch!r}: not simple_cnn and not a torchvision.models entry")
    if in_channels != 3:
        # torchvision stems are 3-channel. Saying so beats the shape error the first conv would
        # raise, which names a tensor and not the two config keys that disagree.
        raise SystemExit(
            f"arch {arch!r} expects 3-channel input but the {in_channels}-channel input_pipeline "
            "was selected; use arch: simple_cnn, or convert to RGB in build_datasets"
        )
    # weights=None keeps the run offline; a cluster node usually cannot reach the weight host, and a
    # silent 10-minute hang on a download is a worse failure than training from scratch.
    return factory(weights=None, num_classes=num_classes)


def resolve_data_root(config: dict[str, Any], data_dir: str) -> str:
    """Where the input pipeline reads, from the two ways a job is given data.

    `--data-dir` is the platform dataset the spec declared, already pulled into the shared cache;
    it wins, because naming a dataset is the more specific act. Otherwise the experiment's own
    `data_dir`, expanded against the environment so it can name a mounted volume without
    hard-coding a path the platform chose:

        data_dir: ${VOLUMES_DIR}/cats-and-dogs

    An unexpanded `${...}` means the variable is not there -- usually a volume that the spec never
    declared, so nothing mounted it. Saying so beats reading an empty directory and training on
    nothing, which is the failure that looks like a bad model rather than a bad path.
    """
    if data_dir:
        return data_dir
    declared = str(config.get("data_dir") or "").strip()
    if not declared:
        return ""
    expanded = os.path.expandvars(declared)
    if "$" in expanded:
        raise SystemExit(
            f"data_dir {declared!r} references an environment variable that is not set "
            f"(resolved to {expanded!r}); a volume reaches the job only when the experiment "
            "declares it under data.volumes, or you pass --volume at submit"
        )
    return expanded


def build_datasets(config: dict[str, Any], data_dir: str, *, may_download: bool = True):
    """Replace this with your own input pipeline. Returns (train_set, val_set, in_channels)."""
    kind = str(config.get("input_pipeline") or "mnist")
    mean, std = _STATS.get(kind, _STATS["imagefolder"])
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if kind == "imagefolder":
        root = Path(data_dir or "data")
        return (
            datasets.ImageFolder(str(root / "train"), transform=tf),
            datasets.ImageFolder(str(root / "val"), transform=tf),
            3,
        )

    factory = datasets.MNIST if kind == "mnist" else datasets.CIFAR10
    # Downloading is the fallback, never the plan: with --data-dir given the files are already
    # there, and a node without egress should fail saying so rather than hang on a socket.
    root, download = (data_dir, False) if data_dir else (str(_fallback_root(kind)), may_download)
    return (
        factory(root, train=True, download=download, transform=tf),
        factory(root, train=False, download=download, transform=tf),
        1 if kind == "mnist" else 3,
    )


def _fallback_root(kind: str) -> Path:
    """Where to put a downloaded copy when the experiment declared no dataset.

    The shared cache if the platform injected one (so the second run is free), otherwise the working
    directory, which is what a laptop wants.
    """
    if cache := os.environ.get("TUNEPLANE_DATA_CACHE", "").strip():
        return Path(cache) / kind
    return Path("data")


def build_optimizer(config: dict[str, Any], model: nn.Module):
    lr = float(config.get("learning_rate") or 1e-3)
    decay = float(config.get("weight_decay") or 0.0)
    if str(config.get("optimizer") or "adamw") == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=decay, momentum=float(config.get("momentum") or 0.9)
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)


def build_scheduler(config: dict[str, Any], optimizer, epochs: int):
    kind = str(config.get("lr_scheduler") or "constant")
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        seen += y.numel()
    model.train()
    return loss_sum / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overrides-json", required=True)
    parser.add_argument("--data-dir", default="")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    # The overrides arrive already validated against the recipe, so they win outright.
    config.update(json.loads(args.overrides_json))

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if distributed:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1

    torch.manual_seed(int(config.get("seed") or 0) + rank)

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval").mkdir(parents=True, exist_ok=True)

    # Only one rank may materialise a download and the others have to wait for it: several
    # processes unpacking the same archive into the same directory corrupt it, and the loser
    # reads a half-written file. With --data-dir given nothing downloads and both barriers are
    # free, which is the path a submitted job actually takes.
    data_root = resolve_data_root(config, args.data_dir)
    if distributed and rank != 0:
        dist.barrier()
    train_set, val_set, in_channels = build_datasets(config, data_root, may_download=rank == 0)
    if distributed and rank == 0:
        dist.barrier()
    batch_size = int(config.get("batch_size") or 64)
    workers = int(config.get("num_workers") or 0)
    sampler = DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=workers, pin_memory=use_cuda, drop_last=False,
    )
    # Evaluation runs on rank 0 over the whole set, so its loader is never sharded: an all-reduce of
    # per-rank accuracies would be one more thing to get subtly wrong for no gain at this size.
    val_loader = DataLoader(val_set, batch_size=max(batch_size * 4, 1), num_workers=workers, pin_memory=use_cuda)

    model = build_model(config, in_channels).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
    # Evaluation and checkpointing happen on rank 0 alone, so both go through the *unwrapped*
    # module. A forward through the DDP wrapper broadcasts the module's buffers -- every
    # BatchNorm carries a running mean and variance, so every torchvision architecture has some
    # -- and a collective that one rank enters by itself blocks until the process group times
    # out, half an hour of allocated GPUs later. `torchvision.models` is exactly what `arch`
    # invites, so this is the default path rather than an exotic one.
    bare_model = model.module if distributed else model
    epochs = int(config.get("epochs") or 1)
    max_steps = int(config.get("max_steps") or 0)
    opt = build_optimizer(config, model)
    sched = build_scheduler(config, opt, epochs)
    amp = bool(config.get("amp")) and use_cuda
    autocast_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and autocast_dtype is torch.float16)
    clip = config.get("max_grad_norm")
    log_interval = int(config.get("log_interval") or 20)
    eval_every = int(config.get("eval_interval_epochs") or 1)

    # Rank zero is the only reporter: every rank calling log() would multiply each point by the
    # world size on the same step, which reads as a broken curve rather than as a duplicate.
    if rank == 0:
        init(hparams={**config, "world_size": world, "global_batch_size": batch_size * world})

    step, best, stop = 0, 0.0, False
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=amp):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            scaler.step(opt)
            scaler.update()
            step += 1
            if rank == 0 and step % log_interval == 0:
                log({"train/loss": float(loss.detach()), "train/lr": sched.get_last_lr()[0]}, step=step)
            if max_steps and step >= max_steps:
                stop = True
                break
        sched.step()

        if rank == 0 and ((epoch + 1) % eval_every == 0 or stop or epoch == epochs - 1):
            val_loss, top1 = evaluate(bare_model, val_loader, device)
            log({"validation/loss": val_loss, "validation/accuracy": top1}, step=step)
            print(f"[train] epoch {epoch} step {step} top1={top1:.4f} loss={val_loss:.4f}", flush=True)
            weights = bare_model.state_dict()
            payload = {"epoch": epoch, "step": step, "top1": top1, "config": config, "model": weights}
            torch.save(payload, ckpt_dir / f"epoch{epoch}.pt")
            if top1 >= best:
                best = top1
                torch.save(payload, ckpt_dir / "best.pt")
        if stop:
            break

    if rank == 0:
        # The recipe declares eval/report.json as an artifact and this recipe has no eval action, so
        # writing it is this script's job or nobody's.
        (out_dir / "eval" / "report.json").write_text(
            json.dumps(
                {
                    "task": str(config.get("input_pipeline") or "mnist"),
                    "arch": str(config.get("arch") or "simple_cnn"),
                    "metric": "top1",
                    "value": best,
                    "steps": step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        finish()
        print(f"[train] done best_top1={best:.4f} -> {out_dir}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
