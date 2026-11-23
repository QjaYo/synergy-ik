"""Export fixed Uniform heel/forefoot offsets once; USD is not needed in training."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from contact import FootContact, SITE_JOINTS, SITE_LABELS


def calibrate(rig_path):
    from pxr import Usd, UsdGeom, UsdSkel

    rig_path = Path(rig_path)
    stage = Usd.Stage.Open(str(rig_path))
    if UsdGeom.GetStageUpAxis(stage) != "Y":
        raise ValueError("Expected the distributed Y-up Uniform rig")
    mesh = stage.GetPrimAtPath("/OUTPUT/c_geometry_grp/Mesh")
    binding = UsdSkel.BindingAPI(mesh)
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath("/OUTPUT/c_skeleton_grp/Root"))
    names = [str(n).split("/")[-1] for n in binding.GetJointsAttr().Get()]
    skeleton_names = [str(n).split("/")[-1] for n in skeleton.GetJointsAttr().Get()]
    bind = np.asarray(skeleton.GetBindTransformsAttr().Get())
    indices = np.asarray(binding.GetJointIndicesPrimvar().Get()).reshape(-1, binding.GetJointIndicesPrimvar().GetElementSize())
    weights = np.asarray(binding.GetJointWeightsPrimvar().Get()).reshape(indices.shape)
    raw = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
    points = (np.c_[raw, np.ones(len(raw))] @ np.asarray(binding.GetGeomBindTransformAttr().Get()))[:, :3]
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    offsets = {}
    for side in ("Left", "Right"):
        foot_names = [side + name for name in ("Foot", "ToeBase", "ToeEnd")]
        foot_weight = (weights * np.isin(indices, [names.index(name) for name in foot_names])).sum(axis=1)
        foot = points[foot_weight > 0.5]
        forward = bind[skeleton_names.index(side + "ToeBase"), 3, :3] - bind[skeleton_names.index(side + "Foot"), 3, :3]
        forward[1] = 0
        forward /= np.linalg.norm(forward)
        along = foot @ forward
        low, high = along.min(), along.max()
        regions = (("rear", "Foot", along <= low + 0.25 * (high - low)),
                   ("front", "ToeBase", along >= low + 0.60 * (high - low)))
        for label, joint, mask in regions:
            region = foot[mask]
            patch = region[region[:, 1] <= region[:, 1].min() + 0.002 / scale]
            local = np.linalg.solve(bind[skeleton_names.index(side + joint)].T,
                                    np.r_[patch.mean(axis=0), 1])[:3] * scale
            offsets[side + " " + label] = tuple(float(v) for v in local)
    return FootContact(tuple(offsets[label] for label in SITE_LABELS),
                       hashlib.sha256(rig_path.read_bytes()).hexdigest(), joints=SITE_JOINTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.rig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved {result.width} calibrated sole sites to {args.output}")


if __name__ == "__main__":
    main()
