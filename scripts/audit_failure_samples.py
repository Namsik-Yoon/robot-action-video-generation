"""Decode a reproducible, purposive pilot sample. Outputs stay local."""
import json
import random
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/data/train"
OUT = ROOT / "reports/local/failure-audit"
DATASETS = [
    "Beegbrain/pick_lemon_and_drop_in_bowl",
    "Beegbrain/pick_place_green_block",
    "jpata/so100_pick_place_tangerine",
    "sixpigs1/so100_pick_cube_in_box",
    "Odog16/so100_cube_drop_pick_v1",
    "sixpigs1/so100_stack_cube_error",
    "sixpigs1/so100_stack_cube_error",  # same-scene no-error controls
]


def contact_sheet(frames, indices, title, path):
    width, height = 320, 240
    sheet = Image.new("RGB", (width * 4, (height + 24) * 3 + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for n, i in enumerate(indices):
        im = frames[i].copy()
        im.thumbnail((width, height))
        x, y = n % 4 * width, n // 4 * (height + 24) + 30
        sheet.paste(im, (x, y))
        draw.text((x + 5, y + height + 3), f"frame {i}", fill="black")
    sheet.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    rows = []
    for group, dataset in enumerate(DATASETS, 1):
        base = TRAIN / dataset
        info = json.loads((base / "meta/info.json").read_text())
        episodes = {e["episode_index"]: e for e in (
            json.loads(line) for line in (base / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
        )}
        paths = sorted(base.rglob("*.parquet"))
        if group == 7:
            paths = [p for p in paths if any("without errors" in t for t in episodes[int(p.stem.split("_")[-1])].get("tasks", []))]
        for rank, parquet in enumerate(sorted(rng.sample(paths, min(3, len(paths)))), 1):
            sample_id = f"G{group:02d}-S{rank:02d}"
            video = next(base.glob(f"videos/**/{parquet.stem}.mp4"))
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                codec = stream.codec_context.name
                frames, times = [], []
                for frame in container.decode(stream):
                    frames.append(frame.to_image())
                    times.append(float(frame.time) if frame.time is not None else None)
            table = pq.read_table(parquet)
            ep_id = int(parquet.stem.split("_")[-1])
            index = np.linspace(0, len(frames) - 1, 12, dtype=int).tolist()
            contact_sheet(frames, index, f"{sample_id} {dataset} {parquet.stem}", OUT / f"{sample_id}.jpg")
            if group == 6:
                for start in (20, 32):
                    detail = list(range(start, min(start + 12, len(frames))))
                    contact_sheet(frames, detail, f"{sample_id} consecutive frames {start} onward", OUT / f"{sample_id}-detail-{start}.jpg")
            rows.append({
                "sample_id": sample_id, "dataset": dataset,
                "episode": parquet.stem, "tasks": episodes[ep_id].get("tasks", []),
                "metadata_frames": episodes[ep_id]["length"], "video_frames": len(frames),
                "parquet_rows": table.num_rows, "fps": info["fps"], "codec": codec,
                "video": str(video.relative_to(ROOT)), "parquet": str(parquet.relative_to(ROOT)),
                "contact_frames": index, "timestamps": times,
            })
            print(sample_id, dataset, parquet.stem, len(frames), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Saved {len(rows)} pilot samples to {OUT}")


if __name__ == "__main__":
    main()
