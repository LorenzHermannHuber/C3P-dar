"""Raw model outputs for the evaluation: one appendable HDF5 file per scene.

Written by generation.py (GPU), read by validation.py (CPU). ``<directory>/<scene_token>.h5`` with the
attrs ``kind`` ("ppp" | "radargen"), ``model``, ``step``, ``seed``, ``point_limit`` (+ ``mark_link`` for "ppp"):

    <frame_index:06d>/              attrs sample_token, is_key_frame
        fields  (3,H,W) float32     "ppp": raw [log-intensity, RCS, Doppler] outputs, attr seconds
        <draw>/maps (3,H,W) float32 "radargen": [point density (sums to 1), RCS, Doppler], attr seconds
        <draw>/points (N,4) float64 drawn cloud [x, y, RCS, Doppler]; written last, marks a complete draw

Entries are only added, never rewritten, so an interrupted generation resumes where it stopped.
"""

from pathlib import Path

import h5py

KINDS = ("ppp", "radargen")


def resolve(name, root):
    """Raw output directory of ``name`` (below ``root`` or a path) and its name for the metric tables.

    A run directory that holds a single checkpoint's outputs resolves to that checkpoint.
    """
    raw = Path(name) if Path(name).is_absolute() else root / name
    if raw.is_dir() and not any(raw.glob("*.h5")):
        checkpoints = sorted(p for p in raw.iterdir() if p.is_dir() and any(p.glob("*.h5")))
        if len(checkpoints) > 1:
            raise ValueError(f"{raw} holds several checkpoints, pick one: {[p.name for p in checkpoints]}")
        raw = checkpoints[0] if checkpoints else raw
    return raw, raw.relative_to(root) if raw.is_relative_to(root) else Path(raw.name)


class RawStore:
    """``mode="a"`` writes with the given file attrs, ``mode="r"`` takes them from the existing files."""

    def __init__(self, directory, mode="r", **attrs):
        self.directory, self.mode, self.files = Path(directory), mode, {}
        if mode == "r":
            paths = sorted(self.directory.glob("*.h5"))
            if not paths:
                raise FileNotFoundError(f"no raw outputs in {self.directory}")
            with h5py.File(paths[0], "r") as f:
                attrs = dict(f.attrs)
        elif attrs.get("kind") not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.attrs = attrs

    def _file(self, scene):
        if scene not in self.files:
            path = self.directory / f"{scene}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            f = self.files[scene] = h5py.File(path, self.mode)
            for key, value in self.attrs.items():
                stored = f.attrs.setdefault(key, value) if self.mode != "r" else f.attrs.get(key)
                if stored != value:
                    raise ValueError(f"{path}: {key} is {stored}, not {value}")
        return self.files[scene]

    @staticmethod
    def key(frame, name):
        return f"{int(frame):06d}/{name}"

    def has(self, scene, frame, name):
        if scene not in self.files and not (self.directory / f"{scene}.h5").exists():
            return False
        return self.key(frame, name) in self._file(scene)

    def draws(self, frames):
        """Number of draws complete for all ``frames`` ((scene, frame) pairs)."""
        count = 0
        while all(self.has(scene, frame, f"{count}/points") for scene, frame in frames):
            count += 1
        return count

    def write(self, scene, frame, name, data, frame_attrs, **attrs):
        f = self._file(scene)
        f.require_group(f"{int(frame):06d}").attrs.update(frame_attrs)
        chunks = dict(chunks=(1, *data.shape[1:]), compression="gzip", compression_opts=4, shuffle=True)
        dataset = f.create_dataset(self.key(frame, name), data=data, **(chunks if data.ndim == 3 else {}))
        dataset.attrs.update(attrs)

    def read(self, scene, frame, name):
        dataset = self._file(scene)[self.key(frame, name)]
        return dataset[()], dict(dataset.attrs)

    def flush(self):
        for f in self.files.values():
            f.flush()

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
