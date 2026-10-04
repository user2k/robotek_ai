import unittest
from copy import deepcopy
from episode import Episode
from curriculum import default_plan, prepare_curriculum, level_options, validate_plan


class StartDirectionTests(unittest.TestCase):
    def test_fixed_directions_and_resets(self):
        for direction, heading in [('right', 0), ('left', 180), ('up', 270), ('down', 90)]:
            with self.subTest(direction=direction):
                episode = Episode(['S.E'], start_direction=direction)
                self.assertEqual(episode.player.heading, heading)
                episode.player.heading = 45
                episode.reset()
                self.assertEqual(episode.player.heading, heading)

    def test_random_is_reproducible_and_rerolls_on_reset(self):
        first = Episode(['S.E'], start_direction='random', start_seed=123)
        second = Episode(['S.E'], start_direction='random', start_seed=123)
        headings = []
        for _ in range(30):
            self.assertEqual(first.player.heading, second.player.heading)
            headings.append(first.player.heading)
            first.reset()
            second.reset()
        self.assertEqual(set(headings), {0, 90, 180, 270})
        self.assertGreater(len({Episode(['S.E'], start_direction='random', start_seed=i).player.heading for i in range(16)}), 1)

    def test_old_plan_migrates_without_reset_and_direction_is_passed(self):
        plan = default_plan()[:1]
        old = deepcopy(plan)
        del old[0]['start_direction']
        state = dict(plan=old, wins=[True], level=0)
        self.assertIs(prepare_curriculum(state, plan), state)
        plan[0]['start_direction'] = 'up'
        state = prepare_curriculum(state, plan)
        self.assertEqual(state['wins'], [])
        self.assertEqual(Episode(plan[0]['rows'], **level_options(plan[0])).player.heading, 270)
        plan[0]['start_direction'] = 'bad'
        with self.assertRaises(ValueError):
            validate_plan(plan)
        with self.assertRaises(ValueError):
            Episode(['SE'], start_direction='bad')

if __name__ == '__main__':
    unittest.main()
