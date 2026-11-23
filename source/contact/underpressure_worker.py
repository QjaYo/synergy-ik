"""Persistent process boundary for upstream's absolute util/data/anim imports."""

import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch


def predict(model, contacts_class, positions, fps):
    source_times = np.arange(len(positions), dtype=np.float64) / fps
    model_times = np.arange(int(np.floor(source_times[-1] * 100)) + 1) / 100
    flat = positions.reshape(len(positions), -1)
    sampled = np.stack([np.interp(model_times, source_times, col) for col in flat.T], axis=-1)
    sampled = torch.from_numpy(sampled.reshape(-1, 23, 3).astype(np.float32))[None]
    with torch.inference_mode():
        forces = model.vGRFs(sampled)
        contacts = contacts_class.from_forces(forces.clone())[0].numpy()
    indices = np.minimum(np.floor(source_times * 100 + 0.5).astype(int), len(model_times) - 1)
    return dict(contacts=contacts[indices].reshape(-1, 4), forces=forces[0].numpy(), model_times=model_times)


def main():
    upstream = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(upstream))
    import models
    from data import Contacts
    torch.set_num_threads(2)
    checkpoint = torch.load(upstream / "pretrained.tar", map_location="cpu", weights_only=True)
    model = models.DeepNetwork(state_dict=checkpoint["model"]).eval()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with np.load(request["source"], allow_pickle=False) as archive:
                positions, fps = archive["positions"], float(archive["fps"])
            if request["mode"] == "offline":
                result = predict(model, Contacts, positions, fps)
            elif request["mode"] == "prefix":
                # Re-run each prefix, including resampling and all temporal smoothing.
                # Slicing labels from a full-sequence prediction would leak future data.
                labels = [predict(model, Contacts, positions[:t+1], fps)["contacts"][-1]
                          for t in range(len(positions))]
                result = dict(contacts=np.asarray(labels, dtype=bool))
            else:
                raise ValueError(f"Unknown inference mode: {request['mode']}")
            np.savez(request["target"], **result)
            response = dict(ok=True)
        except Exception:
            response = dict(error=traceback.format_exc())
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
