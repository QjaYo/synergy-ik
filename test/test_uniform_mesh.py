"""Check USD convention, independent FK alignment and rigid translation."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess/motion_review"))
from uniform_mesh import (DEFAULT_UNIFORM_ROOT, DEFAULT_UNIFORM_RIG, PROJECT_ROOT,
                          UniformMesh, compute_world_transforms, load_aligned_uniform_mesh)
sys.path.insert(0, str(PROJECT_ROOT / 'source'))
from util import load_canonicalized_bvh
from fairmotion.data import bvh


CLIP = Path('211117/Neutral_walk_forward_002__A057.bvh')


@unittest.skipUnless(DEFAULT_UNIFORM_RIG.is_file() and (DEFAULT_UNIFORM_ROOT / CLIP).is_file(),
                     'Local Bones-seed Uniform rig and validation clip required')
class UniformMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = UniformMesh(DEFAULT_UNIFORM_ROOT / CLIP)

    def test_base_bvh_matches_independent_usd_rest_pose(self):
        base = bvh.load(str(DEFAULT_UNIFORM_RIG.with_suffix('.bvh')))
        bvh_world = compute_world_transforms(base, 0)
        usd_world = np.asarray(self.mesh.skel_query.ComputeJointSkelTransforms()).transpose(0,2,1)
        np.testing.assert_allclose(bvh_world, usd_world, atol=2e-5, rtol=0)

    def test_skinning_reference_and_rigid_translation(self):
        from pxr import Vt
        points = Vt.Vec3fArray(self.mesh.rest_points)
        self.assertTrue(self.mesh.skin_query.ComputeSkinnedPoints(self.mesh.skel_query.ComputeSkinningTransforms(), points))
        base = bvh.load(str(DEFAULT_UNIFORM_RIG.with_suffix('.bvh')))
        saved_motion = self.mesh.motion
        try:
            self.mesh.motion = base
            self.mesh.last_frame = -1
            calculated = self.mesh.frame(0).vertices.copy()
            np.testing.assert_allclose(calculated, np.asarray(points)*0.01, atol=1e-6, rtol=0)
            base.poses[1].data = np.array(base.poses[0].data, copy=True)
            delta_cm = np.array([20, -30, 10])
            base.poses[1].data[1, :3, 3] += delta_cm
            shifted = self.mesh.frame(1).vertices
            np.testing.assert_allclose(shifted, calculated + delta_cm*0.01, atol=1e-6, rtol=0)
        finally:
            self.mesh.motion = saved_motion
            self.mesh.last_frame = -1

    @unittest.skipUnless((PROJECT_ROOT / 'data/canonical_bvh' / CLIP).is_file(),
                         'Canonical BVH validation clip required')
    def test_original_to_canonical_alignment(self):
        canonical = load_canonicalized_bvh(PROJECT_ROOT / 'data/canonical_bvh' / CLIP)
        mesh = load_aligned_uniform_mesh(DEFAULT_UNIFORM_ROOT / CLIP, DEFAULT_UNIFORM_RIG, canonical)
        self.assertLess(mesh.alignment_error_m, 2e-6)
        for frame in (0, 100, canonical.num_frames()-1):
            np.testing.assert_allclose(compute_world_transforms(mesh.motion, frame),
                np.array([mesh.motion.poses[frame].get_transform(name, local=False) for name in mesh.joint_names]), atol=1e-9)


if __name__ == '__main__':
    unittest.main()
