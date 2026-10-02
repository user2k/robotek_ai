from collections import deque
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from episode import Action, Episode
from maps import generate_maze
from math import atan2, degrees
from train import TEST_SEEDS, VALIDATION_SEEDS, latest_path, train, training_seed


def safe_route(rows):
    start = next((x, y) for y, row in enumerate(rows) for x, symbol in enumerate(row) if symbol == "S")
    end = next((x, y) for y, row in enumerate(rows) for x, symbol in enumerate(row) if symbol == "E")
    parents = {start: None}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pos = x + dx, y + dy
            if (0 <= pos[1] < len(rows) and 0 <= pos[0] < len(rows[0])
                    and rows[pos[1]][pos[0]] in ".PSE" and pos not in parents):
                parents[pos] = x, y
                queue.append(pos)
    if end not in parents:
        raise AssertionError("Brak bezpiecznej trasy")
    route = []
    pos = end
    while pos is not None:
        route.append(pos)
        pos = parents[pos]
    return list(reversed(route))


class MazeTests(unittest.TestCase):
    def test_many_mazes_can_be_completed_without_damage_within_budget(self):
        cases = [(15, 9, seed, 200) for seed in range(150)]
        cases += [(5, 5, 0, 2), (31, 21, 4, 200), (9, 9, 3, 20)]
        for width, height, seed, limit in cases:
            with self.subTest(width=width, height=height, seed=seed, limit=limit):
                rows = generate_maze(width, height, seed, limit)
                self.assertEqual(len(rows), height)
                self.assertTrue(all(len(row) == width for row in rows))
                self.assertEqual("".join(rows).count("S"), 1)
                self.assertEqual("".join(rows).count("E"), 1)
                self.assertTrue(all(c == "#" for c in rows[0] + rows[-1]))
                self.assertTrue(all(row[0] == row[-1] == "#" for row in rows))
                episode = Episode(rows, max_time=limit)
                route = safe_route(rows)
                for (old_x, old_y), (x, y) in zip(route, route[1:]):
                    heading = degrees(atan2(y - old_y, x - old_x)) % 360
                    difference = (heading - episode.player.heading + 180) % 360 - 180
                    for _ in range(round(abs(difference) / 10)):
                        episode.step(Action.RIGHT if difference >= 0 else Action.LEFT)
                    for _ in range(10):
                        if episode.done:
                            break
                        result = episode.step(Action.FORWARD)
                        self.assertTrue(result.moved)
                    if episode.done:
                        break
                self.assertTrue(episode.player.won)
                self.assertEqual(episode.player.damage, 0)
                self.assertLessEqual(episode.player.elapsed_time, limit)

    def test_repeatable_and_diverse(self):
        self.assertEqual(generate_maze(seed=42), generate_maze(seed=42))
        maps = [tuple(generate_maze(seed=seed)) for seed in range(30)]
        self.assertEqual(len(set(maps)), 30)
        self.assertTrue(set("BWF").issubset(set("".join(row for rows in maps for row in rows))))

    def test_bad_parameters(self):
        for width, height in ((4, 9), (15, 8), (3, 5), (5.5, 9)):
            with self.assertRaises(ValueError):
                generate_maze(width, height)
        for limit in (0, 1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                generate_maze(max_time=limit)

    def test_training_validation_and_test_seed_sets_are_disjoint(self):
        seeds = {training_seed(7, number) for number in range(1, 1001)}
        self.assertEqual(len(seeds), 1000)
        self.assertTrue(seeds.isdisjoint(VALIDATION_SEEDS))
        self.assertTrue(seeds.isdisjoint(TEST_SEEDS))
        self.assertTrue(set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS))


class ResumeTests(unittest.TestCase):
    def test_stopped_training_saves_without_any_automatic_evaluation(self):
        from threading import Event
        from visual_agent import VisualAgent
        stop = Event()
        stop.set()
        with tempfile.TemporaryDirectory() as directory, \
                patch('train.evaluate_suite', side_effect=AssertionError('Automatic validation')):
            path = Path(directory) / 'agent.pt'
            report = train(episodes=1, model_path=path, stop=stop)
            self.assertEqual(report['training_episodes'], 0)
            self.assertTrue(latest_path(path).exists())
            self.assertEqual(VisualAgent.load(latest_path(path)).episodes, 0)
            self.assertNotIn('test', report)
            self.assertNotIn('evaluation', report)

    def test_live_callback_reports_real_training_steps(self):
        summary = {"score": 0, "maps": 1, "wins": 0, "success_rate": 0,
                   "time": 200, "damage": 0, "results": []}
        frames = []

        def on_step(episode, metadata):
            frames.append((episode.steps, episode.done, dict(metadata)))

        with tempfile.TemporaryDirectory() as directory, patch("train.evaluate_suite", return_value=summary), \
                patch("train.Episode", side_effect=lambda rows: Episode(rows, max_time=0.3)):
            train(episodes=1, model_path=Path(directory) / "agent.pt", width=5, height=5,
                  on_step=on_step, batch_size=1)
        self.assertGreater(len(frames), 1)
        self.assertEqual([frame[0] for frame in frames], list(range(len(frames))))
        self.assertIsNone(frames[0][2]["action"])
        self.assertTrue(frames[-1][1])
        self.assertTrue(all(frame[2]["action"] in (0, 1, 2, 3) for frame in frames[1:]))
        self.assertEqual(len({frame[2]["map_seed"] for frame in frames}), 1)

    def test_continuation_uses_latest_and_keeps_history(self):
        summary = {"score": 0, "maps": 1, "wins": 0, "success_rate": 0,
                   "time": 200, "damage": 0, "results": []}
        with tempfile.TemporaryDirectory() as directory, patch("train.evaluate_suite", return_value=summary), \
                patch("train.Episode", side_effect=lambda rows: Episode(rows, max_time=0.3)):
            path = Path(directory) / "agent.pt"
            first = train(episodes=2, model_path=path, width=5, height=5)
            second = train(episodes=1, model_path=path, width=5, height=5)
            self.assertFalse(first["resumed"])
            self.assertTrue(second["resumed"])
            self.assertEqual(second["started_from_episode"], 2)
            self.assertEqual(second["total_episodes"], 3)
            from visual_agent import VisualAgent
            self.assertEqual(VisualAgent.load(latest_path(path)).episodes, 3)
            self.assertNotEqual(first["history"], second["history"])
            seeds = []
            for report in (first, second):
                with Path(report["history"]).open(encoding="utf-8") as file:
                    seeds.extend(row["map_seed"] for row in csv.DictReader(file))
            self.assertEqual(len(set(seeds)), 3)


if __name__ == "__main__":
    unittest.main()
