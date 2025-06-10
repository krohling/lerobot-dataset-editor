#!/usr/bin/env python3
"""Command‑line utility for **offline** edits of LeRobot datasets.

Supported actions
-----------------
• **remove‑episode**  – Delete an episode (parquet + videos) and fix metadata.
• **trim‑frames**     – Excise a contiguous span of frames from one episode.

All operations are local; nothing is pushed to / pulled from the Hub. **Make a backup first – edits are destructive.**
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import click
import datasets  # type: ignore

from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.utils import (
    INFO_PATH,
    STATS_PATH,
    EPISODES_STATS_PATH,
    write_info,
    write_episode,
    write_episode_stats,
    embed_images,
    write_json,
    write_jsonlines,
    load_episodes_stats
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

CODEBASE_VERSION = "v2.1"
###############################################################################
# Helpers
###############################################################################

def _load_meta(dataset_dir: Path) -> LeRobotDatasetMetadata:
    """Load metadata without touching the Hub (assumes dataset is already local)."""
    return LeRobotDatasetMetadata(
        repo_id=dataset_dir.name,
        root=dataset_dir,
        revision=CODEBASE_VERSION,
        force_cache_sync=False,
    )


def _parse_frame_range(expr: str, max_len: int) -> Tuple[int, int]:
    """Return *inclusive start*, *exclusive end* indices given a slice expression like
    "200-300" or "250-end". Accepts ":" or "-" as the separator for convenience."""

    expr = expr.replace(":", "-")
    if "-" not in expr:
        raise ValueError("Frame range must contain '-' (e.g. '100-200' or '300-end').")
    start_s, end_s = expr.split("-")
    start = int(start_s) if start_s else 0
    end = max_len if end_s in ("end", "") else int(end_s)

    if not (0 <= start < end <= max_len):
        raise ValueError(f"Range '{expr}' is invalid for episode length {max_len}.")
    return start, end


def _safe_write_parquet(ds: datasets.Dataset, target: Path) -> None:
    """Write **atomically**: write to temp, then replace original path."""
    tmp_path = target.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    ds.to_parquet(tmp_path)
    tmp_path.replace(target)

# updates the stats for a single episode
def _overwrite_episode_stats(
    dataset_dir: Path, episodes_stats: dict[int, dict]
) -> None:
    eps_stats = load_episodes_stats(dataset_dir)
    eps_stats.update(episodes_stats)
    # remove previous file
    (dataset_dir / EPISODES_STATS_PATH).unlink(missing_ok=True)
    
    for ep_idx, stats in eps_stats.items():
        write_episode_stats(ep_idx, stats, dataset_dir)


###############################################################################
# Core operations
###############################################################################

def remove_episode(dataset_dir: Path, ep_idx: int) -> None:
    """Remove an entire episode and patch metadata/statistics."""

    meta = _load_meta(dataset_dir)
    if ep_idx not in meta.episodes:
        click.echo(f"Episode {ep_idx} not found.", err=True)
        sys.exit(1)

    # 1️⃣  Delete parquet file
    (meta.root / meta.get_data_file_path(ep_idx)).unlink(missing_ok=True)

    # 2️⃣  Delete associated videos
    for vkey in meta.video_keys:
        (meta.root / meta.get_video_file_path(ep_idx, vkey)).unlink(missing_ok=True)

    # 3️⃣  Remove episode & stats entries
    meta.episodes.pop(ep_idx, None)
    meta.episodes_stats.pop(ep_idx, None)

    # 4️⃣  Rewrite episodes.jsonl from scratch
    (meta.root / "meta" / "episodes.jsonl").unlink(missing_ok=True)
    for ep in meta.episodes.values():
        write_episode(ep, meta.root)

    # 5️⃣  Patch info.json counters
    meta.info["total_episodes"] = len(meta.episodes)
    meta.info["total_frames"] = sum(e["length"] for e in meta.episodes.values())
    meta.info["total_videos"] = len(meta.video_keys) * meta.info["total_episodes"]
    meta.info["total_chunks"] = (
        max((idx // meta.chunks_size for idx in meta.episodes), default=-1) + 1
    )
    meta.info["splits"] = {"train": f"0:{meta.info['total_episodes']}"}
    write_info(meta.info, meta.root)

    # 6️⃣  Refresh aggregate stats
    update_stats(dataset_dir, _meta_prefetched=meta)


def trim_frames(dataset_dir: Path, ep_idx: int, range_expr: str) -> None:
    """Excise a span of frames inside one episode."""

    meta = _load_meta(dataset_dir)
    if ep_idx not in meta.episodes:
        click.echo(f"Episode {ep_idx} not found.", err=True)
        sys.exit(1)

    parquet_path = meta.root / meta.get_data_file_path(ep_idx)
    if parquet_path.stat().st_size == 0:
        click.echo(
            "Parquet file is 0‑bytes (likely left from a failed previous run). "
            "Restore from backup or delete the episode and try again.",
            err=True,
        )
        sys.exit(1)

    ds = datasets.load_dataset("parquet", data_files=str(parquet_path), split="train")
    start, end = _parse_frame_range(range_expr, len(ds))
    keep_idx: List[int] = [i for i in range(len(ds)) if not (start <= i < end)]

    if not keep_idx:
        click.echo("All frames would be deleted – aborting. Use remove-episode instead.", err=True)
        sys.exit(1)

    # 1️⃣  Build trimmed dataset
    trimmed = ds.select(keep_idx)
    trimmed = embed_images(trimmed)  # maintain LeRobot conventions

    # 2️⃣  Atomically replace parquet
    _safe_write_parquet(trimmed, parquet_path)

    # 3️⃣  Update episode length & episodes.jsonl entry
    meta.episodes[ep_idx]["length"] = len(trimmed)
    write_episode(meta.episodes[ep_idx], meta.root)

        # 4️⃣  Recompute per‑episode stats – convert lists → numpy for compute_episode_stats
    import numpy as np  # local import to keep global deps minimal

    def _to_np(val):
        if isinstance(val, list):
            # try stacking first (works for vectors/arrays), else fallback to simple np.array
            try:
                return np.stack(val)
            except Exception:
                return np.array(val)
        return val

    frame_dict = {c: _to_np(trimmed[c]) for c in trimmed.column_names}
    ep_stats = compute_episode_stats(frame_dict, meta.features)
    meta.episodes_stats[ep_idx] = ep_stats
    # write_episode_stats(ep_idx, ep_stats, meta.root)
    _overwrite_episode_stats(meta.root, meta.episodes_stats)

        # 5️⃣  Re‑encode video if dataset stores videos
    if meta.video_keys:
        from lerobot.common.datasets.video_utils import decode_video_frames, encode_video_frames
        import numpy as np
        import torch
        from torchvision.transforms.functional import to_pil_image

        kept_ts: List[float] = [float(t) for t in trimmed["timestamp"]]
        tol = 1 / meta.fps - 1e-4

        for vkey in meta.video_keys:
            vpath = meta.root / meta.get_video_file_path(ep_idx, vkey)
            if not vpath.is_file():
                continue

            tmp_dir = vpath.parent / f"_tmp_{ep_idx}_{vkey}"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            tensor_frames = decode_video_frames(vpath, kept_ts, tol, backend="pyav").squeeze(0)
            for i, frame_tensor in enumerate(tensor_frames):
                to_pil_image(frame_tensor).save(tmp_dir / f"frame_{i:06d}.png")

            encode_video_frames(tmp_dir, vpath, meta.fps, overwrite=True)
            shutil.rmtree(tmp_dir)

    # 6️⃣  Patch dataset‑level counters & stats  Patch dataset‑level counters & stats
    meta.info["total_frames"] = sum(ep["length"] for ep in meta.episodes.values())
    write_info(meta.info, meta.root)
    update_stats(dataset_dir, _meta_prefetched=meta)


def update_stats(dataset_dir: Path, _meta_prefetched: LeRobotDatasetMetadata | None = None) -> None:
    """Re‑aggregate statistics across **current** episodes and write as pure‑Python JSON."""
    import numpy as np

    def _to_py(o):
        """Recursively convert NumPy types → native Python (list/float/int)."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_py(v) for v in o]
        return o

    meta = _meta_prefetched or _load_meta(dataset_dir)
    if not meta.episodes_stats:
        click.echo("No per‑episode stats found; cannot aggregate.", err=True)
        return

    meta.stats = aggregate_stats(list(meta.episodes_stats.values()))
    py_stats = _to_py(meta.stats)
    (meta.root / STATS_PATH).unlink(missing_ok=True)
    write_jsonlines(py_stats, meta.root / STATS_PATH)

###############################################################################
# CLI definitions
###############################################################################

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:  # noqa: D401
    """Edit LeRobot datasets *in‑place*.

    Always back up your dataset **before** running destructive operations.
    """


# ---------------------------------------------------------------------------
# remove‑episode
# ---------------------------------------------------------------------------


@cli.command("remove-episode")
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--episode", "-e", required=True, type=int, help="Episode index to delete.")
def _cmd_remove_episode(dataset_dir: Path, episode: int) -> None:  # noqa: D401
    """Delete an entire episode (data + media) and fix metadata."""
    remove_episode(dataset_dir, episode)
    click.echo(f"✔ Episode {episode} removed.")


# ---------------------------------------------------------------------------
# trim‑frames
# ---------------------------------------------------------------------------


@cli.command("trim-frames")
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--episode", "-e", required=True, type=int, help="Episode index to edit.")
@click.option(
    "--frames",
    "-f",
    required=True,
    type=str,
    help="Range to delete (e.g. '200-300', '250-end').",
)
def _cmd_trim_frames(dataset_dir: Path, episode: int, frames: str) -> None:  # noqa: D401
    """Remove a contiguous range of frames from an episode."""
    trim_frames(dataset_dir, episode, frames)
    click.echo(f"✔ Frames {frames} removed from episode {episode}.")

###############################################################################

if __name__ == "__main__":
    cli()
