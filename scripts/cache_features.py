#!/usr/bin/env python3
"""Cache frozen Atom-JEPA features for every conformer written by prepare_data.py.

One forward pass per molecule gives every representation the Direction-A experiments need,
so a readout experiment afterwards costs seconds instead of an encoder pass.

For each molecule we store two reductions of the encoder state, both SO(3)-invariant:

  per_atom.npy  float16 [total_atoms, n_layers, 3, C]
      Per-atom features: the l=0 channels, then the per-channel norms of the l=1 and l=2
      channels. Atoms of molecule i are rows offsets[i]:offsets[i + 1]. Feeds the learned
      pooling experiments.

  pooled.npy    float32 [n_molecules, n_layers, 3, C]
      The equivariant state averaged over atoms FIRST, then reduced to invariants. This is
      not the mean of per_atom: averaging l > 0 vectors before taking their norm lets
      opposing contributions cancel, which is a different (and also valid) descriptor.
      For l = 0 the two agree, so `layers='last', degrees=0` here reproduces exactly what
      models/atom_jepa.py feeds its readout today.

Layer slot j holds encoder block j + 1 for j < n_layers - 1; the final slot is 'last',
i.e. block 8 after the final norm, which is what AtomJEPA.forward returns by default.

    python scripts/cache_features.py --device mps --batch-size 16

Batch size matters on Apple GPUs: 16 ran at 358 ms/molecule here, 32 collapsed to 10 s.
Safe to interrupt and rerun; it resumes where it stopped, and it prints how much memory the
MPS allocator holds, because that figure growing without bound is the failure mode here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from atom_jepa import AtomJEPA

PROGRESS = 'progress.json'


def invariants(x):
    """[..., sphere, C] equivariant -> [..., 3, C] invariant: l=0, ||l=1||, ||l=2||."""
    return torch.cat([x[..., 0:1, :],
                      x[..., 1:4, :].norm(dim=-2, keepdim=True),
                      x[..., 4:9, :].norm(dim=-2, keepdim=True)], dim=-2)


def load_sample(path):
    with np.load(path) as d:
        return {'atomic_numbers': torch.tensor(d['z'], dtype=torch.long),
                'node_coordinates': torch.tensor(d['pos'], dtype=torch.float32)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, help='Default: <data>/features/atomjepa_<checkpoint>')
    p.add_argument('--checkpoint', default='molecules')
    p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--precision', default='fp32', choices=['fp32', 'bf16'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--limit', type=int, help='Smoke test: only the first N molecules')
    p.add_argument('--stop-after', type=int, metavar='N',
                   help='Process at most N molecules this run, then exit so the caller can '
                        'restart. Unlike --limit this does not change the cache itself, so '
                        'a loop over it is the way to get a fresh allocator every N '
                        'molecules without rebuilding anything.')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')

    conformers = args.data / 'conformers'
    if not conformers.is_dir():
        p.error(f'{conformers} not found; run prepare_data.py first')
    out = args.out or args.data / 'features' / f'atomjepa_{args.checkpoint}'
    out.mkdir(parents=True, exist_ok=True)

    keys = sorted(f.stem for f in conformers.glob('*.npz'))
    if args.limit:
        keys = keys[:args.limit]
    print(f'{len(keys)} molecules -> {out}', flush=True)

    print('Counting atoms', flush=True)
    natoms = np.array([len(np.load(conformers / f'{k}.npz')['z']) for k in keys], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    print(f'{offsets[-1]} atoms, {natoms.mean():.1f} per molecule', flush=True)

    model = AtomJEPA.from_pretrained(args.checkpoint, device=args.device,
                                     precision=args.precision, cuequivariance=False)
    model.eval()
    if model.lmax != 2:
        raise ValueError(f'invariants() assumes lmax=2, encoder has {model.lmax}')
    layers = list(range(1, model.num_layers + 1)) + ['last']
    channels, n_layers = model.embedding_dim, len(layers)

    meta = {'checkpoint': args.checkpoint, 'layers': [str(l) for l in layers],
            'n_layers': n_layers, 'degrees': ['l0', 'norm_l1', 'norm_l2'],
            'channels': channels, 'n_molecules': len(keys), 'precision': args.precision}
    meta_path = out / 'meta.json'
    done = 0
    if meta_path.exists() and (out / PROGRESS).exists():
        if json.loads(meta_path.read_text()) != meta:
            raise ValueError(f'{out} was built with different settings; use a new --out')
        done = json.loads((out / PROGRESS).read_text())['done']
        print(f'Resuming after {done} molecules', flush=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    np.save(out / 'keys.npy', np.array(keys))
    np.save(out / 'natoms.npy', natoms)
    np.save(out / 'offsets.npy', offsets)
    mode = 'r+' if done else 'w+'
    per_atom = open_memmap(out / 'per_atom.npy', mode=mode, dtype=np.float16,
                           shape=(int(offsets[-1]), n_layers, 3, channels))
    pooled = open_memmap(out / 'pooled.npy', mode=mode, dtype=np.float32,
                         shape=(len(keys), n_layers, 3, channels))

    stop = len(keys) if args.stop_after is None else min(len(keys), done + args.stop_after)
    for start in range(done, stop, args.batch_size):
        chunk = keys[start:start + args.batch_size]
        samples = [load_sample(conformers / f'{k}.npz') for k in chunk]
        with torch.no_grad():
            # [total_atoms_in_chunk, n_layers, sphere, C], still equivariant.
            x = model(model.collate(samples), layers=layers, degrees='all',
                      invariant=False, per_atom=True).float()
            counts = [len(s['atomic_numbers']) for s in samples]
            # Reduce on the device and transfer once per batch: doing it per molecule
            # costs one sync and one scattered memmap write each, and ran 2.6x slower.
            means = torch.stack([atoms.mean(dim=0) for atoms in x.split(counts)])
            atom_rows = invariants(x).half().cpu().numpy()
            molecule_rows = invariants(means).cpu().numpy()
        finished = start + len(chunk)
        per_atom[offsets[start]:offsets[finished]] = atom_rows
        pooled[start:finished] = molecule_rows
        (out / PROGRESS).write_text(json.dumps({'done': finished}))
        if args.device == 'mps':
            # Every batch has a different atom and edge count, so the MPS caching allocator
            # keeps adding buffer sizes it then never reuses. Left alone it grew to a 34 GB
            # footprint here and wedged the machine in swap after ~2000 molecules; releasing
            # once per batch costs a few milliseconds and keeps it flat. The figure printed
            # below is what makes that visible rather than a guess.
            torch.mps.empty_cache()
        if (start // args.batch_size) % 10 == 0 or finished == len(keys):
            # Flushing here bounds the dirty pages of a 3.8 GB mapping and makes the file
            # on disk agree with progress.json, so a resume never re-reads stale rows.
            per_atom.flush()
            pooled.flush()
            held = (f'  mps {torch.mps.driver_allocated_memory() / 1e9:5.2f} GB'
                    if args.device == 'mps' else '')
            print(f'{finished}/{len(keys)}{held}', flush=True)

    per_atom.flush()
    pooled.flush()
    if stop < len(keys):
        print(f'Stopped after {stop}/{len(keys)} as asked; rerun to continue', flush=True)
        return
    print(f'Done. per_atom {per_atom.shape} {per_atom.nbytes / 1e9:.2f} GB, '
          f'pooled {pooled.shape} {pooled.nbytes / 1e6:.0f} MB', flush=True)


if __name__ == '__main__':
    main()
