"""Zgodność obrazu oraz przepływ GPU renderer → CNN bez obrazów w historii CPU."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from camera import render_camera
from tensor_camera import TensorCamera, camera_pose
from episode import Episode, Action
from maps import generate_maze
from batching import run_group
from visual_agent import VisualAgent


class TensorCameraTests(unittest.TestCase):
    def test_matches_reference_on_maps_rotations_and_subtile_positions(self):
        for seed in (3, 7, 11):
            ep = Episode(generate_maze(9, 9, seed))
            camera = TensorCamera(ep.world)
            poses, references = [], []
            for heading in range(0, 360, 10):
                ep.player.heading = heading
                poses.append(camera_pose(ep.player))
                references.append(render_camera(ep.player, 80, 60))
            ep.player.x += .13
            ep.player.y -= .09
            poses.append(camera_pose(ep.player))
            references.append(render_camera(ep.player, 80, 60))
            actual = camera.render(poses).numpy().astype(int)
            diff = np.abs(actual - np.stack(references).astype(int))
            # Różnice zaokrągleń na krawędziach tekstur i trafieniach w narożniki.
            self.assertLess(diff.mean(), .15)
            self.assertLess(np.mean(diff.max(-1) > 3), .002)

    def test_walls_hide_terrain_and_boundary_views_are_finite(self):
        a, b = Episode(['S.#E']), Episode(['S.#W.E'])
        # Wspólna najbliższa ściana zasłania różne dalsze otoczenie.
        first = TensorCamera(a.world).render([camera_pose(a.player)])
        second = TensorCamera(b.world).render([camera_pose(b.player)])
        torch.testing.assert_close(first, second)
        ep = Episode(['S.E'])
        ep.player.x = 1.
        for heading in (0, 90, 180, 270):
            ep.player.heading = heading
            result = TensorCamera(ep.world).render([camera_pose(ep.player)])
            self.assertEqual(result.dtype, torch.uint8)
            self.assertEqual(tuple(result.shape), (1, 60, 80, 3))

    def test_group_and_bptt_use_poses_without_cpu_camera(self):
        agent = VisualAgent(device='cpu', hidden_size=32)
        with patch('camera.render_camera', side_effect=AssertionError('Legacy camera used')), \
                patch.object(agent, 'observe', side_effect=AssertionError('Individual rendering used')):
            result = run_group(agent, [Episode(['S.E'], max_time=.2) for _ in range(3)], [1, 2, 3], .7)
            trace = result['trajectory']
            self.assertNotIn('frames', trace)
            self.assertEqual(len(trace['poses']), len(trace['actions']) + 1)
            before = agent.network.encoder[0].weight.detach().clone()
            self.assertTrue(np.isfinite(agent.learn_episode(trace)))
            self.assertEqual(agent.updates, 1)
            self.assertFalse(torch.equal(before, agent.network.encoder[0].weight))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_tensor_stays_on_device_and_agrees_with_cpu(self):
        ep = Episode(generate_maze(9, 9, 7))
        poses = []
        for heading in range(0, 360, 10):
            ep.player.heading = heading
            poses.append(camera_pose(ep.player))
        actual = TensorCamera(ep.world, 'cuda').render(poses)
        self.assertEqual(actual.device.type, 'cuda')
        reference = TensorCamera(ep.world, 'cpu').render(poses)
        diff = (actual.cpu().int() - reference.int()).abs()
        self.assertLess(diff.float().mean().item(), .15)

if __name__ == '__main__':
    unittest.main()
