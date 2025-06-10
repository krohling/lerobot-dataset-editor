# LeRobot Dataset Editor

A tiny, offline‑friendly command‑line utility for **surgically editing LeRobot datasets** that live on your disk (or in the 🤗 cache) — no Hugging Face Hub access needed.

## Features

| Action           | What it does                                                                                                                                                                       |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `remove-episode` | Deletes an entire episode: parquet table, any associated videos, episode metadata & per‑episode statistics. Everything else is patched automatically.                              |
| `trim-frames`    | Removes a contiguous slice of frames *inside* an episode (`200-300`, `300-end`, …). Rewrites the parquet, re‑encodes videos only for the kept timestamps, recalculates statistics. |

Every edit ends with a **fresh aggregate `stats.json`** so your dataset is always train‑ready.

## Installation

```bash
# clone the repo
$ git clone https://github.com/your‑org/lerobot‑dataset‑editor.git
$ cd lerobot‑dataset‑editor

# create env & install deps (Python 3.9+)
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

## Quick start

```bash
# nuke episode 3 completely
$ python lerobot_dataset_editor.py remove-episode \
      ~/.cache/huggingface/lerobot/your_repo \
      --episode 3

# drop frames 250‑end from episode 7
$ python lerobot_dataset_editor.py trim-frames \
      ~/.cache/huggingface/lerobot/your_repo \
      --episode 7 --frames 250-end
```

Behind the scenes the tool:

1. Loads dataset metadata locally (no network).
2. Performs the destructive edit.
3. Recalculates per‑episode + global stats.
4. Atomically rewrites the affected files (parquet & JSON) so a crash never leaves zero‑byte junk.

## CLI reference

```text
Usage: lerobot_dataset_editor.py [COMMAND] DATASET_DIR [OPTIONS]

Commands:
  remove-episode  Delete an entire episode (parquet + videos).
  trim-frames     Remove a range of frames from one episode.
```

### `remove-episode`

```bash
python lerobot_dataset_editor.py remove-episode DATASET_DIR -e INDEX
```

* `-e`, `--episode` – episode index to delete (integer).

### `trim-frames`

```bash
python lerobot_dataset_editor.py trim-frames DATASET_DIR -e INDEX -f RANGE
```

* `-e`, `--episode` – episode index to edit.
* `-f`, `--frames`  – slice to drop. Accepted forms: `start-end`, `start-`, `-end`.

  * `end` or empty means "to last frame".
  * Indices are **0‑based**.

## Caveats & roadmap

* Video re‑encoding currently uses **PyAV → PNG → FFmpeg**; it’s CPU‑heavy. If you only store images, the edit is instant.
* No Windows testing yet.
* `ffmpeg` must be available on your system PATH for video re-encoding; install it via your OS package manager (e.g., `brew install ffmpeg`, `apt-get install ffmpeg`, etc.).

Contributions welcome – open an issue or PR! 🙂

## License

Apache 2.0, identical to the upstream LeRobot license.
