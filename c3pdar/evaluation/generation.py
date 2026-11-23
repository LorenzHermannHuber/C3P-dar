"""GPU part of the evaluation: predict the selected frames and draw their point clouds into a RawStore."""

import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from . import PROGRESS, radargen
from ..data import RadarGenBatch
from ..models.dit import NUM_STREAMS
from ..models.radargen import COORDINATE_RANGE, DOPPLER_RANGE, LATENT_CHANNELS, RCS_RANGE
from ..sampling import draw_generator, ppp_raster, sample_points


def frame_info(dataset, index):
    """Scene token, frame index and the frame attrs of the store."""
    frame = dataset.frames.iloc[index]
    return frame.scene_token, int(frame.frame_index), dict(sample_token=frame.sample_token,
                                                            is_key_frame=bool(frame.is_key_frame))


def write_draws(store, scene, frame, attrs, draws, seed, point_limit, fields=None):
    """Draw and store the missing clouds of one frame (``fields`` None: read them); returns their number."""
    missing = [d for d in range(draws) if not store.has(scene, frame, f"{d}/points")]
    if missing:
        fields = store.read(scene, frame, "fields")[0] if fields is None else fields
        raster = ppp_raster(fields, point_limit, store.attrs["mark_link"])
        for draw in missing:
            points = sample_points(raster.masses, raster.rcs, raster.doppler, point_limit,
                                   draw_generator(seed, scene, frame, draw))
            store.write(scene, frame, f"{draw}/points", points, attrs)
    return len(missing)


@torch.no_grad()
def generate_ppp(model, dataset, indices, store, draws, seed, device, batch_size=1, num_workers=4):
    """Store the raw fields (bf16 autocast) and ``draws`` clouds of every selected frame that lacks them.

    Draw d of a frame is sampled on the CPU from the float32 fields with draw_generator(seed, scene, frame, d),
    so it does not depend on batching, selection or resumption. New fields are drawn from directly, earlier
    ones are read back. Returns the numbers of new fields and draws.
    """
    todo = [i for i in indices if not store.has(*frame_info(dataset, i)[:2], "fields")]
    loader = DataLoader(Subset(dataset, todo), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=RadarGenBatch.collate)
    drawn = 0
    progress = tqdm(total=len(todo), desc=f"predict + draw {draws}x", unit="frame", **PROGRESS)
    for batch in loader:
        maps = batch.to(device, non_blocking=True).normalized_conditions().unbind(1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            fields = model(*maps)
        fields = fields.float().cpu().numpy()
        seconds = (time.perf_counter() - start) / len(fields)
        for index, field in zip(batch.index.tolist(), fields):
            scene, frame, attrs = frame_info(dataset, index)
            store.write(scene, frame, "fields", field, attrs, seconds=seconds)
            drawn += write_draws(store, scene, frame, attrs, draws, seed, dataset.point_limit, field)
        store.flush()
        progress.update(len(fields))
        progress.set_postfix(s_per_frame=f"{seconds:.3f}", refresh=False)
    progress.close()
    predicted = set(todo)
    rest = [i for i in indices if i not in predicted]
    for index in tqdm(rest, desc=f"draw {draws}x (stored)", unit="frame", **PROGRESS):
        drawn += write_draws(store, *frame_info(dataset, index), draws, seed, dataset.point_limit)
    store.flush()
    return len(todo), drawn


@torch.no_grad()
def generate_radargen(model, dataset, indices, store, draws, seed, device, batch_size=1, num_workers=4):
    """Store the maps and the cloud of ``draws`` RadarGen samples of every selected frame that lacks them.

    Draw d of a frame denoises noise drawn on the CPU with draw_generator(seed, scene, frame, d), so it does
    not depend on batching, selection or resumption. Its cloud is recovered from the stored maps by
    RadarGen's sparse deconvolution; the maps' ``seconds`` cover encoding, denoising and decoding.
    Returns the numbers of new maps and clouds.
    """
    if dataset.point_limit != COORDINATE_RANGE:
        raise ValueError(f"RadarGen covers +-{COORDINATE_RANGE} m, the dataset +-{dataset.point_limit} m")

    def complete(index):
        scene, frame, _ = frame_info(dataset, index)
        return all(store.has(scene, frame, f"{d}/points") for d in range(draws))

    todo = [i for i in indices if not complete(i)]
    loader = DataLoader(Subset(dataset, todo), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=RadarGenBatch.collate)
    sampled = recovered = 0
    progress = tqdm(total=len(todo), desc=f"RadarGen {draws}x", unit="frame", **PROGRESS)
    for batch in loader:
        frames = [frame_info(dataset, i) for i in batch.index.tolist()]
        maps = batch.to(device, non_blocking=True).normalized_conditions()
        conditions = None
        for draw in range(draws):
            new = [b for b, (scene, frame, _) in enumerate(frames) if not store.has(scene, frame, f"{draw}/maps")]
            if new:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                if conditions is None:
                    conditions = model.encode(maps)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    encoding = (time.perf_counter() - start) / len(frames)
                    start = time.perf_counter()
                noise = torch.stack([
                    torch.randn((NUM_STREAMS, LATENT_CHANNELS, *conditions.shape[-2:]),
                                generator=draw_generator(seed, scene, frame, draw))
                    for scene, frame, _ in (frames[b] for b in new)])
                images = model.decode(model.denoise(conditions[new], noise.to(device))).cpu().numpy()
                seconds = encoding + (time.perf_counter() - start) / len(new)
                for b, image in zip(new, images):
                    scene, frame, attrs = frames[b]
                    store.write(scene, frame, f"{draw}/maps", radargen.radar_maps(image, RCS_RANGE, DOPPLER_RANGE),
                                attrs, seconds=seconds)
                sampled += len(new)
            for scene, frame, attrs in frames:
                if not store.has(scene, frame, f"{draw}/points"):
                    radar_maps = store.read(scene, frame, f"{draw}/maps")[0]
                    store.write(scene, frame, f"{draw}/points",
                                radargen.recover_points(radar_maps, COORDINATE_RANGE, device), attrs)
                    recovered += 1
        store.flush()
        progress.update(len(frames))
    progress.close()
    return sampled, recovered
