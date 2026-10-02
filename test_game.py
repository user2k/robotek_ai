"""Metry, kołowy obrys i ciągłe akcje zamiast ruchu po kratkach."""
import math
import unittest
from episode import Episode, Action
from player import Player, segment_rectangle_distance_sq
from world import World


class MovementTests(unittest.TestCase):
    def test_per_level_meter_step_and_right_angle_turn(self):
        e = Episode(['S..', '..E', '...'], move_distance=1., turn_degrees=90.)
        self.assertAlmostEqual(e.step(Action.FORWARD).duration, 1.)
        self.assertEqual(e.player.cell, (1, 0))
        self.assertAlmostEqual(e.step(Action.RIGHT).duration, .5)
        self.assertEqual(e.player.heading, 90.)
        e.step(Action.FORWARD)
        self.assertEqual(e.player.cell, (1, 1))
        e.step(Action.LEFT)
        e.step(Action.FORWARD)
        self.assertTrue(e.player.won)
        e.reset()
        self.assertEqual(e.player.move_distance, 1.)
        self.assertEqual(e.player.turn_degrees, 90.)
        other = Episode(['S.E'])
        self.assertEqual(other.player.move_distance, .1)
        self.assertEqual(other.player.turn_degrees, 10.)

    def test_large_diagonal_step_cannot_cut_through_wall_corner(self):
        e = Episode(['S#.', '...', '..E'], move_distance=1., turn_degrees=45.)
        e.step(Action.RIGHT)
        self.assertFalse(e.step(Action.FORWARD).moved)
        self.assertEqual((e.player.x, e.player.y), (.5, .5))
        self.assertEqual(e.collisions, 1)

    def test_large_step_accounts_for_each_terrain_segment(self):
        e = Episode(['SBE'], move_distance=1., turn_degrees=90.)
        self.assertAlmostEqual(e.step(Action.FORWARD).duration, 1.5)
        self.assertAlmostEqual(e.player.x, 1.5)
        self.assertAlmostEqual(e.step(Action.BACKWARD).duration, 1.5)
        self.assertAlmostEqual(e.player.x, .5)
        for distance, angle in ((0, 90), (1, 0), (float('nan'), 10), (1, float('inf'))):
            with self.assertRaises(ValueError):
                Episode(['SE'], move_distance=distance, turn_degrees=angle)

    def test_exploration_is_in_reward_but_not_score(self):
        e = Episode(['S..E'])
        for _ in range(5):
            e.step(Action.FORWARD)
        self.assertAlmostEqual(e.score, -.5)
        self.assertAlmostEqual(e.total_reward, .2 - .005 * .5)
        self.assertAlmostEqual(sum(e.reward_totals.values()), e.total_reward)

    def test_staying_penalty_has_grace_and_grows_even_while_moving(self):
        e = Episode(['S..E'])
        for i in range(30):
            e.step(Action.FORWARD if i % 2 == 0 else Action.BACKWARD)
        self.assertAlmostEqual(e.reward_totals['staying'], 0)
        first = e.step(Action.FORWARD)
        penalty = e.reward_parts['staying']
        self.assertLess(penalty, 0)
        e.step(Action.BACKWARD)
        self.assertLess(e.reward_parts['staying'], penalty)
        self.assertEqual(e.reward_totals['new_tile'], 0)

    def test_rotations_penalized_and_return_does_not_reset_cell_time(self):
        e = Episode(['S..E'])
        for _ in range(72):
            e.step(Action.RIGHT)
        self.assertLess(e.reward_parts['staying'], 0)
        for _ in range(5):
            e.step(Action.FORWARD)
        old = e.cell_time[(0, 0)]
        e.step(Action.BACKWARD)
        e.step(Action.FORWARD)
        self.assertGreater(e.cell_time[(0, 0)], old)
        self.assertLess(e.reward_parts['staying'], 0)
        e.reset()
        self.assertEqual(e.cell_time, {})
        self.assertEqual(e.reward_totals, {})

    def test_exploration_budget_is_bounded_below_goal_bonus(self):
        e = Episode(['S' + '.' * 30 + 'E'])
        for _ in range(300):
            e.step(Action.FORWARD)
        self.assertAlmostEqual(e.reward_totals['new_tile'], 5.)
        self.assertEqual(e.reward_parts['new_tile'], 0.)

    def test_start_is_cell_center_and_diameter_is_20_cm(self):
        episode = Episode(['#####', '#S.E#', '#####'])
        self.assertEqual((episode.player.x, episode.player.y), (1.5, 1.5))
        self.assertEqual(episode.player.radius, .1)

    def test_forward_backward_equal_distance_and_cost(self):
        e = Episode(['S...E'])
        forward = e.step(Action.FORWARD)
        self.assertAlmostEqual(e.player.x, .6)
        backward = e.step(Action.BACKWARD)
        self.assertAlmostEqual(e.player.x, .5)
        self.assertAlmostEqual(forward.duration, .1)
        self.assertAlmostEqual(backward.reward, forward.reward)
        self.assertNotIn('revisit', e.reward_parts)
        self.assertEqual(e.player.heading, 0)

    def test_turn_10_degrees_and_diagonal_translation(self):
        e = Episode(['S..', '..E'])
        e.step(Action.RIGHT)
        self.assertEqual(e.player.heading, 10)
        e.step(Action.FORWARD)
        self.assertAlmostEqual(e.player.x, .5 + .1 * math.cos(math.radians(10)))
        self.assertAlmostEqual(e.player.y, .5 + .1 * math.sin(math.radians(10)))
        e.step(Action.LEFT)
        self.assertEqual(e.player.heading, 0)
        for _ in range(36):
            e.step(Action.LEFT)
        self.assertEqual(e.player.heading, 0)

    def test_wall_blocks_body_not_just_center_and_penalizes(self):
        e = Episode(['S#E'])
        for _ in range(4):
            self.assertTrue(e.step(Action.FORWARD).moved)
        self.assertAlmostEqual(e.player.x, .9)
        result = e.step(Action.FORWARD)
        self.assertFalse(result.moved)
        self.assertAlmostEqual(e.player.x, .9)
        self.assertEqual(e.reward_parts['collision'], -.5)
        self.assertEqual(e.collisions, 1)
        self.assertAlmostEqual(result.duration, .1)
        self.assertTrue(e.step(Action.BACKWARD).moved)
        self.assertEqual(e.reward_parts['collision'], 0)

    def test_repeated_collisions_grow_and_cap_even_with_turns(self):
        e = Episode(['S#E'])
        for _ in range(4):
            e.step(Action.FORWARD)
        for expected in (.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.):
            e.step(Action.FORWARD)
            self.assertEqual(e.reward_parts['collision'], -expected)
            e.step(Action.LEFT)
            self.assertEqual(e.reward_parts['collision'], 0)
            e.step(Action.RIGHT)
        self.assertEqual(e.collision_streak, 8)
        self.assertAlmostEqual(e.reward_totals['collision'], -10.75)
        self.assertAlmostEqual(e.total_reward, sum(e.reward_totals.values()))
        self.assertTrue(e.step(Action.BACKWARD).moved)
        self.assertEqual(e.collision_streak, 0)
        self.assertEqual(e.reward_parts['collision'], 0)
        e.step(Action.FORWARD)
        e.step(Action.FORWARD)
        self.assertEqual(e.reward_parts['collision'], -.5)
        e.reset()
        self.assertEqual(e.collision_streak, 0)
        self.assertEqual(e.collisions, 0)

    def test_timeout_does_not_count_unexecuted_collision(self):
        e = Episode(['S#E'], max_time=.45)
        for _ in range(4):
            e.step(Action.FORWARD)
        e.step(Action.FORWARD)
        self.assertTrue(e.timed_out)
        self.assertEqual(e.reward_parts['collision'], 0)
        self.assertEqual(e.collision_streak, 0)

    def test_edges_block_backwards_and_turning_near_wall_is_allowed(self):
        e = Episode(['S.E'])
        for _ in range(4):
            e.step(Action.BACKWARD)
        self.assertAlmostEqual(e.player.x, .1)
        self.assertFalse(e.step(Action.BACKWARD).moved)
        self.assertEqual(e.reward_parts['collision'], -.5)
        e.step(Action.LEFT)
        self.assertEqual(e.player.heading, 350)
        self.assertEqual(e.reward_parts['collision'], 0)

    def test_swept_circle_cannot_clip_corner_with_clear_endpoints(self):
        # Oba końce mają >10 cm od narożnika, środek odcinka <10 cm.
        world = World.from_text(['S..', '.#.', '..E'])
        p = Player(world, .9, .98)
        self.assertTrue(Player(world, .98, .9).can_travel(.98, .9))
        self.assertFalse(p.can_travel(.98, .9))
        self.assertAlmostEqual(segment_rectangle_distance_sq(.9, .98, .98, .9, 1, 1), .0072)

    def test_first_cell_bonus_once_no_bonus_each_10_cm(self):
        e = Episode(['S..E'])
        for _ in range(4):
            e.step(Action.FORWARD)
            self.assertEqual(e.reward_parts['new_tile'], 0)
        e.step(Action.FORWARD)
        self.assertEqual(e.reward_parts['new_tile'], .2)
        e.step(Action.BACKWARD)
        self.assertEqual(e.reward_parts['new_tile'], 0)
        e.step(Action.FORWARD)
        self.assertEqual(e.reward_parts['new_tile'], 0)

    def test_goal_at_deadline_and_no_movement_past_deadline(self):
        e = Episode(['SE'], max_time=.5)
        for _ in range(5):
            e.step(Action.FORWARD)
        self.assertTrue(e.player.won)
        late = Episode(['SE'], max_time=.49)
        for _ in range(5):
            late.step(Action.FORWARD)
        self.assertFalse(late.player.won)
        self.assertTrue(late.timed_out)
        self.assertAlmostEqual(late.player.x, .9)
        self.assertAlmostEqual(late.player.elapsed_time, .49)
        with self.assertRaises(RuntimeError):
            late.step(Action.FORWARD)

    def test_timeout_reset_and_invalid_setup(self):
        e = Episode(['S#E'], max_time=.3)
        while not e.done:
            e.step(Action.RIGHT)
        self.assertTrue(e.timed_out)
        e.reset()
        self.assertFalse(e.done)
        self.assertEqual(e.player.heading, 0)
        for limit in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                Episode(['SE'], max_time=limit)
        with self.assertRaises(ValueError):
            Episode(['S..'])

    def test_hazards_use_distance_not_tenfold_step_counts(self):
        e = Episode(['SWWW.E'])
        for _ in range(15):
            e.step(Action.FORWARD)
        self.assertAlmostEqual(e.player.water_distance, 1.)
        self.assertAlmostEqual(e.player.damage, 10.)
        before = e.player.water_distance
        e.step(Action.RIGHT)
        self.assertEqual(e.player.water_distance, before)
        e.step(Action.LEFT)
        while not e.done:
            e.step(Action.FORWARD)
        self.assertFalse(e.player.alive)
        self.assertAlmostEqual(e.player.damage, 30.)

    def test_time_integrates_terrain_on_both_sides_of_boundary(self):
        e = Episode(['SB.E'])
        e.player.x = .95
        result = e.step(Action.FORWARD)
        self.assertAlmostEqual(result.duration, .05 + .05 / .5)
        self.assertAlmostEqual(e.player.x, 1.05)

    def test_collision_does_not_apply_terrain_damage(self):
        e = Episode(['SW#E'])
        for _ in range(14):
            e.step(Action.FORWARD)
        before = e.player.damage
        self.assertFalse(e.step(Action.FORWARD).moved)
        self.assertEqual(e.player.damage, before)

if __name__ == '__main__':
    unittest.main()
