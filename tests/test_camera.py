import unittest
import numpy as np
from camera import render_camera
from episode import Episode, Action

class CameraTests(unittest.TestCase):
    def test_hidden_terrain_does_not_change_pixels(self):
        a = Episode(['#######', '#S.#.E#', '#######'])
        b = Episode(['#######', '#S.#WE#', '#######'])
        np.testing.assert_array_equal(render_camera(a.player), render_camera(b.player))

    def test_turn_changes_image_and_full_turn_restores_it(self):
        episode = Episode(['#######', '#S...E#', '#######'])
        original = render_camera(episode.player)
        episode.step(Action.RIGHT)
        self.assertFalse(np.array_equal(original, render_camera(episode.player)))
        for _ in range(35):
            episode.step(Action.RIGHT)
        np.testing.assert_array_equal(original, render_camera(episode.player))

    def test_rgb_is_independent_of_map_memory_and_does_not_mutate_episode(self):
        episode = Episode(['S...E'])
        first = render_camera(episode.player)
        episode.visited.clear()
        np.testing.assert_array_equal(first, render_camera(episode.player))
        self.assertEqual(first.shape, (120, 160, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(episode.steps, 0)

if __name__ == '__main__':
    unittest.main()
