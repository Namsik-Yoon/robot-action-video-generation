"""Read-only inventory; never imports or executes official competition code."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def inspect(root):
    train = root / "data/train"
    infos = sorted(train.rglob("meta/info.json"))
    datasets = []
    lengths = []
    schemas = collections.Counter()
    pair_errors = []
    for path in infos:
        info = json.loads(path.read_text())
        base = path.parent.parent
        parquet = sorted(base.rglob("*.parquet"))
        videos = sorted(base.rglob("*.mp4"))
        pids = {p.stem for p in parquet}
        vids = {p.stem for p in videos}
        if pids != vids:
            pair_errors.append(str(base.relative_to(train)))
        episodes_path = path.parent / "episodes.jsonl"
        if episodes_path.exists():
            for line in episodes_path.read_text().splitlines():
                if line.strip():
                    ep = json.loads(line)
                    if "length" in ep:
                        lengths.append(ep["length"])
        schema = pq.read_schema(parquet[0]) if parquet else None
        if schema:
            schemas[str(schema.remove_metadata())] += 1
        datasets.append({
            "dataset": str(base.relative_to(train)),
            "robot_type": info.get("robot_type"),
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "fps": info.get("fps"),
            "action_shape": info.get("features", {}).get("action", {}).get("shape"),
            "video_features": {k:v for k,v in info.get("features", {}).items() if v.get("dtype") == "video"},
            "parquet_count": len(parquet), "video_count": len(videos),
        })
    arrays = sorted((root / "data/eval/actions").glob("*.npy"))
    images = sorted((root / "data/eval/images").glob("*.png"))
    action_formats = collections.Counter()
    image_formats = collections.Counter()
    nonfinite = []
    action_min, action_max = None, None
    for path in arrays:
        a = np.load(path, allow_pickle=False)
        action_formats[f"{a.shape} / {a.dtype}"] += 1
        if not np.isfinite(a).all():
            nonfinite.append(path.name)
        else:
            action_min = np.minimum(action_min, a.min(axis=0)) if action_min is not None else a.min(axis=0)
            action_max = np.maximum(action_max, a.max(axis=0)) if action_max is not None else a.max(axis=0)
    for path in images:
        with Image.open(path) as im:
            image_formats[f"{im.size} / {im.mode}"] += 1
            im.verify()
    sample_path = sorted(train.rglob("*.parquet"))[0]
    sample = pq.read_table(sample_path)
    return {
        "dataset_count": len(datasets),
        "train_episodes_metadata": sum(d["episodes"] or 0 for d in datasets),
        "train_frames_metadata": sum(d["frames"] or 0 for d in datasets),
        "train_parquet_files": sum(d["parquet_count"] for d in datasets),
        "train_video_files": sum(d["video_count"] for d in datasets),
        "robot_types": dict(collections.Counter(d["robot_type"] for d in datasets)),
        "fps_by_dataset": dict(collections.Counter(d["fps"] for d in datasets)),
        "episode_length_frames": {k: float(v) for k,v in zip(["min","median","p95","max"], np.percentile(lengths,[0,50,95,100]))},
        "episode_lengths_count": len(lengths),
        "train_pair_errors": pair_errors,
        "eval": {"images": len(images), "actions": len(arrays), "ids_match": {p.stem for p in images} == {p.stem for p in arrays}, "action_formats": dict(action_formats), "image_formats": dict(image_formats), "nonfinite_actions": nonfinite, "action_min": action_min.tolist(), "action_max": action_max.tolist()},
        "parquet_schema_variants": dict(schemas),
        "sample_parquet": {"path": str(sample_path.relative_to(root)), "rows": sample.num_rows, "schema": str(sample.schema), "first_row": sample.slice(0,1).to_pylist()},
        "datasets": datasets,
        "scope": "All metadata/eval inputs and file pairs; one Parquet schema per dataset and one sample payload. Videos not decoded; training numeric contents not fully scanned.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {"datasets", "sample_parquet", "parquet_schema_variants"}}, ensure_ascii=False, indent=2))
