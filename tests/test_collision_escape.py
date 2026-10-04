"""Audyt kolizji: kontakt, narożniki, cofanie i niezależna geometria.

Uruchomienie: python -m unittest tests.test_collision_escape -v
Losowania mają stałe seedy; nie używają modeli ani plików treningowych.
"""
import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from batching import run_group
from episode import Action, Episode
from maps import generate_maze
from player import EPS, Player, segment_rectangle_distance_sq
from world import World


def oracle_distance(ax, ay, bx, by, left, top):
    """Niezależna minimalizacja wypukłej odległości punkt–prostokąt."""
    def distance(t):
        x, y = ax + t*(bx-ax), ay + t*(by-ay)
        dx = max(left-x, 0., x-left-1)
        dy = max(top-y, 0., y-top-1)
        return dx*dx+dy*dy
    lo, hi = 0., 1.
    for _ in range(70):
        a, b = (2*lo+hi)/3, (lo+2*hi)/3
        if distance(a) < distance(b):
            hi = b
        else:
            lo = a
    return min(distance(0), distance(1), distance((lo+hi)/2))


def clear_position(world, x, y, radius=.1):
    """Obrys punktu sprawdzany bez Player.can_travel."""
    if not radius-EPS <= x <= world.width-radius+EPS:
        return False
    if not radius-EPS <= y <= world.height-radius+EPS:
        return False
    for ty in range(world.height):
        for tx in range(world.width):
            if not world.tile_at(tx, ty).walkable:
                dx, dy = max(tx-x, 0., x-tx-1), max(ty-y, 0., y-ty-1)
                if dx*dx+dy*dy < radius*radius-EPS:
                    return False
    return True


class CollisionEscapeTests(unittest.TestCase):
    def test_wall_contacts_all_sides_reverse_after_twenty_collisions(self):
        rows = ['#######', '#.....#', '#.S...#', '#.....#', '#....E#', '#######']
        contacts = [(1.1, 2.5, 180), (5.9, 2.5, 0), (3.5, 1.1, 270), (3.5, 4.9, 90)]
        for distance in (.01, .1, .5, 1.):
            for x, y, heading in contacts:
                with self.subTest(distance=distance, contact=(x, y, heading)):
                    ep = Episode(rows, max_time=1000, move_distance=distance, turn_degrees=15)
                    ep.player = Player(ep.world, x, y, heading, move_distance=distance, turn_degrees=15)
                    for _ in range(20):
                        self.assertFalse(ep.step(Action.FORWARD).moved)
                        self.assertEqual((ep.player.x, ep.player.y), (x, y))
                    self.assertFalse(ep.done)
                    self.assertTrue(ep.step(Action.BACKWARD).moved)
                    self.assertEqual(ep.collision_streak, 0)
                    self.assertTrue(clear_position(ep.world, ep.player.x, ep.player.y))

    def test_outer_edges_and_corners_allow_retreat(self):
        world = World.from_text(['S.....', '......', '......', '.....E'])
        contacts = [(.1, 2., 180), (5.9, 2., 0), (3., .1, 270), (3., 3.9, 90),
                    (.1, .1, 225), (5.9, .1, 315), (.1, 3.9, 135), (5.9, 3.9, 45)]
        for distance in (.01, .1, .5, 1.):
            for x, y, heading in contacts:
                with self.subTest(distance=distance, contact=(x, y, heading)):
                    player = Player(world, x, y, heading, move_distance=distance)
                    self.assertTrue(player.movement_plan()[2])
                    self.assertFalse(player.movement_plan(backward=True)[2])
                    self.assertEqual((player.x, player.y), (x, y))

    def test_convex_wall_corner_tangent_and_retreat_at_every_angle(self):
        world = World.from_text(['S.....', '......', '..#...', '......', '.....E'])
        for degrees in range(0, 91):
            radians = math.radians(degrees)
            x, y = 2-.1*math.cos(radians), 2-.1*math.sin(radians)
            player = Player(world, x, y, degrees, move_distance=.1)
            with self.subTest(angle=degrees):
                self.assertTrue(player.can_travel(x, y))
                self.assertTrue(player.movement_plan()[2])
                self.assertFalse(player.movement_plan(backward=True)[2])

    def test_swept_corner_collision_with_clear_endpoints_is_reversible(self):
        world = World.from_text(['S..', '.#.', '..E'])
        player = Player(world, .9, .98, heading=315)
        self.assertTrue(clear_position(world, .98, .9))
        self.assertFalse(player.can_travel(.98, .9))
        self.assertEqual((player.x, player.y), (.9, .98))
        self.assertFalse(player.movement_plan(backward=True)[2])

    def test_turn_at_wall_never_changes_position_or_penetrates(self):
        for angle in (10., 15., 45., 90.):
            ep = Episode(['#####', '#S.E#', '#####'], max_time=10000, turn_degrees=angle)
            ep.player = Player(ep.world, 1.1, 1.1, 225, turn_degrees=angle)
            for _ in range(720):
                ep.step(Action.LEFT)
                self.assertEqual((ep.player.x, ep.player.y), (1.1, 1.1))
                self.assertTrue(clear_position(ep.world, ep.player.x, ep.player.y))
            self.assertFalse(ep.done)

    def test_cul_de_sac_can_reverse_entire_route(self):
        for distance in (.01, .1, .5, 1.):
            ep = Episode(['########', '#S....##', '######E#', '########'],
                         move_distance=distance, max_time=10000)
            successful = 0
            while ep.step(Action.FORWARD).moved:
                successful += 1
                self.assertLess(successful, 1000)
            for _ in range(10):
                self.assertFalse(ep.step(Action.FORWARD).moved)
            for _ in range(successful):
                self.assertTrue(ep.step(Action.BACKWARD).moved)
            self.assertAlmostEqual(ep.player.x, 1.5, places=9)
            self.assertAlmostEqual(ep.player.y, 1.5, places=9)

    def test_both_directions_blocked_can_be_orientation_not_physical_trap(self):
        ep = Episode(['#######', '#S...E#', '#######'], move_distance=.5, turn_degrees=15)
        ep.player.heading = 90
        self.assertFalse(ep.step(Action.FORWARD).moved)
        self.assertFalse(ep.step(Action.BACKWARD).moved)
        for _ in range(6):
            ep.step(Action.LEFT)
        self.assertTrue(ep.step(Action.FORWARD).moved)
        self.assertTrue(ep.step(Action.BACKWARD).moved)
        self.assertAlmostEqual(ep.player.x, 1.5)

    def test_terrain_boundaries_are_not_walls(self):
        for terrain in '.PBWF':
            for distance in (.01, .1, .5, 1.):
                ep = Episode(['S'+terrain+'..E'], move_distance=distance, max_time=1000)
                for _ in range(math.ceil(.5/distance)):
                    self.assertTrue(ep.step(Action.FORWARD).moved)
                self.assertTrue(ep.step(Action.BACKWARD).moved)
                self.assertEqual(ep.collisions, 0)

    def test_timeout_is_terminal_not_a_collision_lock(self):
        ep = Episode(['S#E'], max_time=.5)
        for _ in range(5):
            ep.step(Action.FORWARD)
        self.assertTrue(ep.timed_out)
        self.assertFalse(ep.player.movement_plan(backward=True)[2])
        with self.assertRaises(RuntimeError):
            ep.step(Action.BACKWARD)

    def test_distance_geometry_against_independent_oracle_6000_segments(self):
        rng = random.Random(4404)
        for index in range(6000):
            points = [rng.uniform(-2, 3) for _ in range(4)]
            actual = segment_rectangle_distance_sq(*points, 0, 0)
            expected = oracle_distance(*points, 0, 0)
            self.assertAlmostEqual(actual, expected, delta=1e-9, msg=f'{index}: {points}')
            reverse = segment_rectangle_distance_sq(*points[2:], *points[:2], 0, 0)
            self.assertAlmostEqual(actual, reverse, delta=1e-12)

    def test_random_walks_never_penetrate_and_each_move_can_reverse(self):
        rng = random.Random(4104)
        moved, blocked = 0, 0
        for seed in range(20):
            world = World.from_text(generate_maze(11, 11, seed))
            sx, sy = world.start_position()
            for distance in (.01, .1, .5, 1.):
                player = Player(world, sx+.5, sy+.5, move_distance=distance)
                for _ in range(500):
                    player.heading = rng.randrange(360)
                    before = player.x, player.y
                    backward = bool(rng.randrange(2))
                    x, y, collision, _, _ = player.movement_plan(backward)
                    if collision:
                        blocked += 1
                        self.assertEqual((player.x, player.y), before)
                        continue
                    moved += 1
                    self.assertTrue(clear_position(world, x, y), f'penetration: seed={seed}, {before} -> {(x,y)}')
                    player.x, player.y = x, y
                    rx, ry, reverse_blocked, _, _ = player.movement_plan(not backward)
                    self.assertFalse(reverse_blocked, f'reverse locked: seed={seed}, step={distance}, {before} -> {(x,y)}')
                    self.assertAlmostEqual(rx, before[0], places=9)
                    self.assertAlmostEqual(ry, before[1], places=9)
        self.assertGreater(moved, 10000)
        self.assertGreater(blocked, 1000)
        print(f'\n40 000 prób ruchu: {moved} dozwolonych i odwracalnych, {blocked} kolizji bez zmiany pozycji.')

    def test_batch_robots_reverse_after_repeated_wall_collisions(self):
        episodes = [Episode(['#####', '#S#E#', '#####'], max_time=10) for _ in range(8)]
        tick = 0
        def actions(states, previous, hidden, generators, epsilon):
            nonlocal tick
            tick += 1
            return [int(Action.FORWARD if tick <= 12 else Action.BACKWARD)]*len(previous), hidden
        agent = SimpleNamespace(device=torch.device('cpu'), hidden_size=4, env_steps=0, act_batch=actions)
        with patch('batching.TensorCamera') as camera:
            camera.return_value.render.side_effect = lambda poses: torch.zeros(len(poses), 60, 80, 3)
            result = run_group(agent, episodes, list(range(8)), 0)
        for ep, trace in zip(episodes, result['trajectories']):
            self.assertGreater(ep.collisions, 0)
            self.assertLess(ep.player.x, 1.9)
            self.assertAlmostEqual(trace['poses'][13][0], 1.8)
            self.assertTrue(clear_position(ep.world, ep.player.x, ep.player.y))

    def test_full_path_decision_against_independent_geometry_4000_paths(self):
        world = World.from_text(['S....', '.#.#.', '.....', '..#..', '....E'])
        walls = [(1, 1), (3, 1), (2, 3)]
        rng = random.Random(4004)
        checked = 0
        while checked < 4000:
            x, y = rng.uniform(.1, 4.9), rng.uniform(.1, 4.9)
            if not clear_position(world, x, y):
                continue
            player = Player(world, x, y)
            tx, ty = x+rng.uniform(-1, 1), y+rng.uniform(-1, 1)
            expected = (.1-EPS <= tx <= 4.9+EPS and .1-EPS <= ty <= 4.9+EPS and
                        all(oracle_distance(x, y, tx, ty, wx, wy) >= .01-EPS for wx, wy in walls))
            self.assertEqual(player.can_travel(tx, ty), expected, f'{(x,y)} -> {(tx,ty)}')
            checked += 1

    def test_long_wall_contact_roundoff_does_not_accumulate_overlap(self):
        ep = Episode(['#######', '#S...E#', '#######'], max_time=10000, turn_degrees=15)
        ep.player = Player(ep.world, 1.1, 1.1, 270, turn_degrees=15)
        for _ in range(1000):
            self.assertFalse(ep.step(Action.FORWARD).moved)
            self.assertTrue(ep.step(Action.BACKWARD).moved)
            self.assertTrue(ep.step(Action.FORWARD).moved)
            self.assertEqual((ep.player.x, ep.player.y), (1.1, 1.1))
        self.assertFalse(ep.done)
        self.assertTrue(ep.step(Action.BACKWARD).moved)


if __name__ == '__main__':
    unittest.main(verbosity=2)
