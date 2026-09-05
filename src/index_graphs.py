"""Build a unified index over Rome + synthetic graph corpora."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .graph_io import graph_stats, read_graphml


@dataclass
class GraphRecord:
    graph_id: str
    source: str
    path: str
    graph_class: str | None
    n_nodes: int
    n_edges: int
    params: dict | None = None


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rome_counts_from_name(path: Path) -> tuple[int, int | None]:
    """Rome files are named grafo<ID>.<n_nodes>.graphml; edge count needs loading."""
    stem = path.stem  # grafo1000.14
    if "." in stem:
        try:
            return int(stem.rsplit(".", 1)[-1]), None
        except ValueError:
            pass
    return 0, None


def _iter_rome(rome_dir: Path) -> list[GraphRecord]:
    records = []
    for i, path in enumerate(sorted(rome_dir.glob("*.graphml"))):
        try:
            n_nodes, _ = _rome_counts_from_name(path)
            if n_nodes == 0:
                g = read_graphml(path)
                n_nodes = g.number_of_nodes()
                n_edges = g.number_of_edges()
            else:
                # Edge count is filled during labeling; manifest only needs node filter.
                n_edges = 0
            records.append(
                GraphRecord(
                    graph_id=f"rome_{i:05d}",
                    source="rome",
                    path=str(path),
                    graph_class=None,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                )
            )
        except Exception:
            continue
    return records


def _iter_synthetic(synthetic_root: Path, manifest_path: Path) -> list[GraphRecord]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    records = []
    for i, entry in enumerate(manifest):
        rel = entry["file"].replace("/", "\\")
        path = synthetic_root / rel
        if not path.exists():
            path = synthetic_root.parent / entry["file"]
        records.append(
            GraphRecord(
                graph_id=f"syn_{entry['class'].replace('-', '_')}_{i:04d}",
                source="synthetic",
                path=str(path),
                graph_class=entry.get("class"),
                n_nodes=entry["n_nodes"],
                n_edges=entry["n_edges"],
                params=entry.get("params"),
            )
        )
    return records


def select_graphs(records: list[GraphRecord], cfg: dict) -> list[GraphRecord]:
    sel = cfg["selection"]
    min_n, max_n = sel["min_nodes"], sel["max_nodes"]
    filtered = [r for r in records if min_n <= r.n_nodes <= max_n]

    rome = [r for r in filtered if r.source == "rome"]
    syn = [r for r in filtered if r.source == "synthetic"]

    random.seed(42)
    if len(rome) > sel["rome_max"]:
        rome = random.sample(rome, sel["rome_max"])
    if len(syn) > sel["synthetic_max"]:
        syn = random.sample(syn, sel["synthetic_max"])

    combined = rome + syn
    target = sel["target_total"]
    if len(combined) > target:
        # Prefer keeping all synthetic (labeled classes) and subsample Rome.
        rome_budget = max(0, target - len(syn))
        if len(rome) > rome_budget:
            rome = random.sample(rome, rome_budget)
        combined = rome + syn

    combined.sort(key=lambda r: (r.source, r.graph_id))
    return combined


def build_index(config_path: str | Path) -> list[GraphRecord]:
    cfg = load_config(config_path)
    root = Path(cfg["data_root"])
    records: list[GraphRecord] = []

    rome_dir = Path(cfg["sources"]["rome"]["path"].format(data_root=root))
    if rome_dir.exists():
        records.extend(_iter_rome(rome_dir))

    syn_root = Path(cfg["sources"]["synthetic"]["path"].format(data_root=root))
    manifest = Path(cfg["sources"]["synthetic"]["manifest"].format(data_root=root))
    if manifest.exists():
        records.extend(_iter_synthetic(syn_root, manifest))

    return select_graphs(records, cfg)


def write_index(records: list[GraphRecord], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
