"""Inspect configured contact detectors with the shared motion/contact viewer."""

import argparse
from functools import partial
import json
from pathlib import Path
from collections.abc import Iterable

from contact import FootContact, JointGroundContact, UnderPressureContact, load_contact
from contact.playlist import ROOT, add_viewer_arguments, select_entries, load_motion_descriptions, Entry
from contact.sequence import prepare_contact_clip, ContactClip
from contact.viewer import ContactViewer


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_viewer_arguments(parser)
    parser.add_argument("--contact-config", type=Path, action="append",
                        help="Detector JSON; repeat for synchronized comparison panels. Default: single_detection")
    args = parser.parse_args(argv)
    args.contact_config = args.contact_config or [ROOT / "preprocess/training/ankle_contact.json"]
    return args


def load_entry(
    entry: Entry,
    data_root: Path,
    detectors: Iterable[FootContact | JointGroundContact | UnderPressureContact],
) -> ContactClip:
    return prepare_contact_clip(data_root / entry.key, detectors)


def main() -> None:
    args = arguments()
    entries = select_entries(args)
    detectors = tuple(load_contact(path) for path in args.contact_config)
    loader = partial(load_entry, data_root=args.data_root, detectors=detectors)
    print(f"Playlist: {len(entries)} motions | Contact channels: {[d.width for d in detectors]}", flush=True)
    if args.check:
        for entry in entries:
            clip = loader(entry)
            print(json.dumps(dict(clip=entry.key, frames=clip.motion.num_frames(),
                                  methods=[track.summary() for track in clip.tracks])), flush=True)
        return
    viewer = ContactViewer(entries, loader, load_motion_descriptions(args.metadata),
                           speed=args.speed, show_mesh=args.mesh,
                           uniform_root=args.uniform_root, uniform_rig=args.uniform_rig,
                           height_band=.040 if isinstance(detectors[0], UnderPressureContact) else detectors[0].height_threshold,
                           footer="UnderPressure is an InterDigital product" if any(isinstance(d, UnderPressureContact) for d in detectors) else "")
    try:
        viewer.run()
    finally:
        viewer.shutdown()


if __name__ == "__main__":
    main()
