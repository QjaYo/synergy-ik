"""Original Uniform surface adapted to canonical IK poses for display only."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocess/motion_review"))
from uniform_mesh import DEFAULT_UNIFORM_ROOT, DEFAULT_UNIFORM_RIG, load_aligned_uniform_mesh, surface_colors


class IKSurface:
    def __init__(self, canonical, bvh_path, uniform_root=DEFAULT_UNIFORM_ROOT, rig_path=DEFAULT_UNIFORM_RIG):
        path = Path(bvh_path)
        original = Path(uniform_root) / path.parent.name / path.name
        self.mesh = load_aligned_uniform_mesh(original, rig_path, canonical)
        renamed = dict(Spine1="Spine", Spine2="Spine1", Chest="Spine2", LeftLeg="LeftUpLeg",
                       LeftShin="LeftLeg", RightLeg="RightUpLeg", RightShin="RightLeg")
        names = {j.name for j in canonical.skel.joints}
        self.mapping = {i: renamed.get(name, name) for i, name in enumerate(self.mesh.joint_names)
                        if renamed.get(name, name) in names}
        self.cache = {}

    def vertices(self, panel, pose, frame):
        previous = self.cache.get(panel)
        if previous is not None and previous[0] == frame and np.array_equal(previous[1], pose.data):
            return previous[2]
        if panel == 0:
            vertices = self.mesh.frame(frame).vertices
        else:
            from pxr import Vt
            world = np.empty((len(self.mesh.joint_names), 4, 4))
            for i, parent in enumerate(self.mesh.parents):
                if i in self.mapping:
                    world[i] = pose.get_transform(self.mapping[i], local=False).copy()
                    world[i, :3, 3] /= self.mesh.meters_per_unit
                else:
                    # Joints absent from the canonical pose retain their initial local pose.
                    joint = self.mesh.motion.skel.joints[i]
                    local = joint.xform_from_parent_joint @ self.mesh.motion.poses[0].data[i]
                    world[i] = local if parent == -1 else world[parent] @ local
            transforms = self.mesh.bind_inverse @ world.transpose(0, 2, 1)
            points = Vt.Vec3fArray(self.mesh.rest_points)
            if not self.mesh.skin_query.ComputeSkinnedPoints(Vt.Matrix4dArray.FromNumpy(np.ascontiguousarray(transforms)), points):
                raise RuntimeError("USD skinning failed for IK pose")
            vertices = np.asarray(points, dtype=np.float32) * self.mesh.meters_per_unit
        self.cache[panel] = (frame, np.array(pose.data, copy=True), vertices)
        return vertices

    def draw(self, panel, pose, frame, threshold):
        from OpenGL import GL
        vertices = self.vertices(panel, pose, frame)
        colors = surface_colors(vertices, self.mesh.triangles, threshold)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        try:
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, vertices)
            GL.glColorPointer(3, GL.GL_FLOAT, 0, colors)
            GL.glDrawElements(GL.GL_TRIANGLES, self.mesh.triangles.size, GL.GL_UNSIGNED_INT, self.mesh.triangles)
        finally:
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
