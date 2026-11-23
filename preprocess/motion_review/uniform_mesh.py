"""Animate the distributed Uniform USD skin with original Bones-seed BVHs.

Uses the rig's classic linear blend skinning, without SOMA pose correctives.
Original files remain unchanged. Public positions/vertices are in meters.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "fairmotion"))
from fairmotion.data import bvh
from fairmotion.core.motion import Motion

DEFAULT_UNIFORM_ROOT = Path("D:/data/motion/bones-seed/soma_uniform/bvh")
DEFAULT_UNIFORM_RIG = Path("D:/data/motion/bones-seed/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd")


def compute_world_transforms(motion: Motion, frame: int) -> np.ndarray:
    """Column-vector FK; length units are those of the loaded Motion."""
    world = np.empty((len(motion.skel.joints), 4, 4))
    pose = motion.poses[frame]
    for j, joint in enumerate(motion.skel.joints):
        local = joint.xform_from_parent_joint @ pose.data[j]
        parent = joint.parent_joint
        world[j] = local if parent is None else world[motion.skel.get_index_joint(parent)] @ local
    return world


@dataclass
class MeshFrame:
    vertices: np.ndarray
    joint_positions: np.ndarray


class UniformMesh:
    """One read-only USD rig and one BVH; skin only the requested frame."""

    def __init__(self, bvh_path: Path, rig_path: Path = DEFAULT_UNIFORM_RIG) -> None:
        from pxr import Usd, UsdGeom, UsdSkel

        if not Path(bvh_path).is_file():
            raise FileNotFoundError(bvh_path)
        if not Path(rig_path).is_file():
            raise FileNotFoundError(rig_path)
        self.stage = Usd.Stage.Open(str(rig_path))
        if UsdGeom.GetStageUpAxis(self.stage) != "Y":
            raise ValueError("Expected Y-up Uniform rig")
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.stage)
        if not np.isclose(self.meters_per_unit, 0.01):
            raise ValueError("Expected centimeter Uniform rig")
        skel_prim = self.stage.GetPrimAtPath("/OUTPUT/c_skeleton_grp/Root")
        mesh_prim = self.stage.GetPrimAtPath("/OUTPUT/c_geometry_grp/Mesh")
        skel, mesh = UsdSkel.Skeleton(skel_prim), UsdGeom.Mesh(mesh_prim)
        if not skel or not mesh:
            raise ValueError("Expected the distributed soma_base_skel_minimal.usd structure")
        binding = UsdSkel.BindingAPI(mesh_prim)
        if binding.GetSkinningMethodAttr().Get() != "classicLinear":
            raise ValueError("Expected classicLinear skinning")
        if not np.allclose(np.asarray(UsdGeom.Xformable(skel_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())), np.eye(4)):
            raise ValueError("Expected identity skeleton stage transform")
        self.cache = UsdSkel.Cache()
        self.cache.Populate(UsdSkel.Root(self.stage.GetPrimAtPath("/OUTPUT")), Usd.PrimDefaultPredicate)
        self.skel_query = self.cache.GetSkelQuery(skel)
        self.skin_query = self.cache.GetSkinningQuery(mesh_prim)
        if list(self.skin_query.GetJointOrder()) != list(self.skel_query.GetJointOrder()):
            raise ValueError("Mesh and skeleton joint order differ")
        self.joint_names = tuple(str(name).split("/")[-1] for name in skel.GetJointsAttr().Get())
        self.bind_inverse = np.linalg.inv(np.asarray(skel.GetBindTransformsAttr().Get()))
        self.rest_points = mesh.GetPointsAttr().Get()
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        triangles = []
        start = 0
        for count in counts:
            if count < 3:
                raise ValueError("Invalid mesh face")
            face = indices[start:start + count]
            triangles.extend((face[0], face[k], face[k + 1]) for k in range(1, count - 1))
            start += count
        self.triangles = np.asarray(triangles, dtype=np.uint32)
        self.motion = bvh.load(str(bvh_path))
        if self.motion.num_frames() == 0:
            raise ValueError("Empty original BVH")
        if tuple(j.name for j in self.motion.skel.joints) != self.joint_names:
            raise ValueError("BVH does not match the Uniform rig joint order")
        self.parents = tuple(-1 if j.parent_joint is None else self.motion.skel.get_index_joint(j.parent_joint)
                             for j in self.motion.skel.joints)
        usd_parents = tuple(self.skel_query.GetTopology().GetParentIndices())
        if self.parents != usd_parents:
            raise ValueError("BVH and USD parent hierarchies differ")
        offsets = np.asarray([j.xform_from_parent_joint[:3, 3] for j in self.motion.skel.joints])
        usd_offsets = np.asarray(skel.GetRestTransformsAttr().Get())[:, 3, :3]
        if not np.allclose(offsets, usd_offsets, atol=2e-4, rtol=0):
            raise ValueError("BVH bone offsets differ from the Uniform rig")
        root = self.motion.skel.root_joint
        if root.name != "Root" or not np.allclose(root.xform_from_parent_joint, np.eye(4)):
            raise ValueError("Expected identity wrapper Root")
        if any(not np.allclose(p.data[0], np.eye(4), atol=1e-6) for p in self.motion.poses):
            raise ValueError("Animated wrapper Root is unsupported")
        # Bones-seed Hips position channels are absolute, not added to its OFFSET.
        self.motion.skel.get_joint("Hips").xform_from_parent_joint[:3, 3] = 0
        self.last_frame = -1
        self.last_result: MeshFrame | None = None

    def frame(self, index: int) -> MeshFrame:
        from pxr import Vt

        if not 0 <= index < self.motion.num_frames():
            raise IndexError(index)
        if index == self.last_frame:
            return self.last_result
        world = compute_world_transforms(self.motion, index)
        # USD uses row vectors; Fairmotion uses column vectors.
        skin_transforms = self.bind_inverse @ world.transpose(0, 2, 1)
        points = Vt.Vec3fArray(self.rest_points)
        transforms = Vt.Matrix4dArray.FromNumpy(np.ascontiguousarray(skin_transforms))
        if not self.skin_query.ComputeSkinnedPoints(transforms, points):
            raise RuntimeError("USD skinning failed")
        vertices = np.array(points, dtype=np.float32) * self.meters_per_unit
        if not np.isfinite(vertices).all():
            raise ValueError("Non-finite skinned surface")
        self.last_result = MeshFrame(vertices, world[:, :3, 3] * self.meters_per_unit)
        self.last_frame = index
        return self.last_result


def load_aligned_uniform_mesh(original_path: Path, rig_path: Path, canonical: Motion) -> UniformMesh:
    """Reject frame/timing or skeleton mismatches rather than hiding them by shifting."""
    mesh = UniformMesh(original_path, rig_path)
    if mesh.motion.num_frames() != canonical.num_frames() or not np.isclose(mesh.motion.fps, canonical.fps):
        raise ValueError("Original and canonical frame count/FPS differ")
    renamed = {"Spine1": "Spine", "Spine2": "Spine1", "Chest": "Spine2",
               "LeftLeg": "LeftUpLeg", "LeftShin": "LeftLeg",
               "RightLeg": "RightUpLeg", "RightShin": "RightLeg"}
    canonical_names = {j.name for j in canonical.skel.joints}
    pairs = [(i, renamed.get(name, name)) for i, name in enumerate(mesh.joint_names)
             if renamed.get(name, name) in canonical_names]
    max_error = 0.0
    for f in sorted({0, canonical.num_frames() // 2, canonical.num_frames() - 1}):
        raw_positions = mesh.frame(f).joint_positions
        for i, name in pairs:
            target = canonical.poses[f].get_transform(name, local=False)[:3, 3]
            max_error = max(max_error, float(np.linalg.norm(raw_positions[i] - target)))
    if max_error > 1e-4:
        raise ValueError(f"Original/canonical joint mismatch: {max_error * 1000:.4f} mm")
    mesh.alignment_error_m = max_error
    return mesh


def surface_colors(vertices: np.ndarray, triangles: np.ndarray,
                   height_threshold: float = 0.040) -> np.ndarray:
    """Height diagnostic: red below zero, yellow within the chosen band."""
    normals = np.zeros_like(vertices)
    face = vertices[triangles]
    face_normals = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.3, 0.7, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.abs(normals @ light)
    colors = np.tile(np.array([0.55, 0.72, 0.84], dtype=np.float32), (len(vertices), 1))
    colors[vertices[:, 1] <= height_threshold] = [0.95, 0.73, 0.15]
    colors[vertices[:, 1] < 0] = [0.94, 0.19, 0.18]
    return np.ascontiguousarray(colors * shade[:, None], dtype=np.float32)
