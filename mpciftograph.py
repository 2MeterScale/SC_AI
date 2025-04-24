#!/usr/bin/env python3
"""
Download all structures from Materials Project, save their CIF files, and convert them to PyTorch-Geometric graphs.

Usage:
    export MP_API_KEY="<your key>"
    python mp_cif_to_graph.py --out_dir ./cif_files --graph_dir ./graphs --max_workers 8

Notes:
- Requires: mp-api, pymatgen, torch, torch_geometric, tqdm, joblib
- The script streams the list of materials via paginated queries, downloads each CIF once, and
  writes a .pt file per structure containing a torch_geometric.data.Data object.
- Graph construction uses a simple r‑cut neighbourhood (8 Å) and encodes Z, Cartesian coords and
  inter‑atomic distance. Extend `structure_to_graph` as needed.
"""
import os
import argparse
import json
from pathlib import Path
from functools import partial
from typing import List

from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
import torch
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm

R_CUT = 8.0  # Å


def structure_to_graph(structure: Structure, r_cut: float = R_CUT) -> Data:
    """Convert a pymatgen Structure to a PyTorch‑Geometric Data object."""
    cnn = CrystalNN(distance_cutoffs=(r_cut,))
    edges: List[List[int]] = []
    dists: List[float] = []

    for i in range(len(structure)):
        neigh = cnn.get_nn_info(structure, i)
        for n in neigh:
            j = n["site_index"]
            d = n["weight"]  # distance Å
            if d <= r_cut:
                edges.append([i, j])
                dists.append(d)

    if not edges:  # isolated atom? create dummy self‑loop
        edges = [[0, 0]]
        dists = [0.0]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(1)

    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32).view(-1)

    return Data(x=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, u=lattice)


def download_and_process(material_id: str, out_dir: Path, graph_dir: Path, mp: MPRester):
    cif_path = out_dir / f"{material_id}.cif"
    graph_path = graph_dir / f"{material_id}.pt"

    if cif_path.exists() and graph_path.exists():
        return  # already processed

    doc = mp.get_structure_by_material_id(material_id, conventional_unit_cell=True)
    structure = doc

    # Save CIF
    cif_string = structure.to(fmt="cif")
    with open(cif_path, "w", encoding="utf‑8") as f:
        f.write(cif_string)

    # Build graph and save
    graph = structure_to_graph(structure)
    torch.save(graph, graph_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Directory to save CIF files")
    parser.add_argument("--graph_dir", required=True, help="Directory to save graph .pt files")
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    graph_dir = Path(args.graph_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("4e1LpV3Y9BIF4LLPaT8iL4coIVUnCIPF")
    if not api_key:
        raise RuntimeError("Set the environment variable MP_API_KEY with your Materials Project key.")

    with MPRester(api_key) as mp:
        # paginated query – get only IDs to minimise bandwidth
        all_ids = []
        response = mp.materials.summary.search()  # first page
        all_ids.extend([d.material_id for d in response.data])
        while response.more_data_available:
            response = mp.materials.summary.search(**response.next_query)
            all_ids.extend([d.material_id for d in response.data])

        print(f"Total materials: {len(all_ids)}")

        worker = partial(download_and_process, out_dir=out_dir, graph_dir=graph_dir, mp=mp)
        Parallel(n_jobs=args.max_workers, backend="loky")(
            delayed(worker)(mid) for mid in tqdm(all_ids)
        )

    # Persist list for future reference
    (out_dir / "processed_ids.json").write_text(json.dumps(all_ids, indent=2))


if __name__ == "__main__":
    main()
