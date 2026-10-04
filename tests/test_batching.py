"""Batch polityki, niezależność pamięci i wybór jednego pełnego epizodu."""
import csv
from pathlib import Path
import random
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from batching import run_group, best_index, rollout_seed
from episode import Episode
from visual_agent import VisualAgent, START_ACTION
from train import train, latest_path
from resources import gpu_sample, format_resources


class BatchTests(unittest.TestCase):
    def test_batched_policy_matches_independent_hidden_states(self):
        agent = VisualAgent(device='cpu', hidden_size=32)
        rng = np.random.default_rng(5)
        states = [rng.integers(0, 256, (60, 80, 3), dtype=np.uint8) for _ in range(3)]
        hidden = torch.randn(1, 3, 32)
        previous = [START_ACTION, 1, 3]
        actions, batch_hidden = agent.act_batch(states, previous, hidden,
                                               [random.Random(i) for i in range(3)], 0.)
        for i in range(3):
            with torch.no_grad():
                values, expected = agent.network(torch.from_numpy(states[i])[None, None],
                                                  torch.tensor([[previous[i]]]), hidden[:, i:i+1])
            self.assertEqual(actions[i], int(values[0, 0].argmax()))
            torch.testing.assert_close(batch_hidden[:, i:i+1], expected)

    def test_group_packs_only_active_robots_and_does_not_update_weights(self):
        agent = VisualAgent(device='cpu', hidden_size=32)
        episodes = [Episode(['S.E'], max_time=t) for t in (.05, .12, .2)]
        sizes = []
        original = agent.act_batch
        def act(states, previous, hidden, generators, epsilon):
            sizes.append(len(states))
            return original(states, previous, hidden, generators, epsilon)
        with patch.object(agent, 'act_batch', side_effect=act):
            result = run_group(agent, episodes, [1, 2, 3], .7)
        self.assertEqual(sizes[0], 3)
        self.assertLess(sizes[-1], 3)
        self.assertTrue(all(ep.done for ep in episodes))
        self.assertEqual(agent.updates, 0)
        self.assertTrue(result['trajectory']['dones'][-1])
        self.assertFalse(any(result['trajectory']['dones'][:-1]))
        self.assertEqual(len(result['trajectory']['poses']), len(result['trajectory']['actions']) + 1)

    def test_winner_prefers_goal_then_reward_and_seeds_are_repeatable(self):
        episodes = [SimpleNamespace(player=SimpleNamespace(won=won), total_reward=reward, score=score, steps=10)
                    for won, reward, score in [(False, 20, 0), (True, 5, 900), (True, 6, 800)]]
        self.assertEqual(best_index(episodes), 2)
        seeds = [rollout_seed(7, 10, i) for i in range(5)]
        self.assertEqual(len(set(seeds)), 5)
        self.assertEqual(seeds, [rollout_seed(7, 10, i) for i in range(5)])
        self.assertNotEqual(seeds[0], rollout_seed(7, 11, 0))

    def test_one_optimizer_step_per_group_and_full_audit_per_robot(self):
        agent = VisualAgent(device='cpu', hidden_size=32)
        with tempfile.TemporaryDirectory() as directory, \
                patch('train.VisualAgent', wraps=VisualAgent) as factory, \
                patch('train.Episode', side_effect=lambda rows: Episode(rows, max_time=.2)), \
                patch.object(agent, 'learn_episode', wraps=agent.learn_episode) as learn, \
                patch.object(agent.optimizer, 'step', wraps=agent.optimizer.step) as step:
            factory.return_value = agent
            path = Path(directory) / 'batch.pt'
            report = train(episodes=2, batch_size=3, model_path=path, device='cpu')
            self.assertEqual(step.call_count, 2)
            self.assertEqual(learn.call_count, 2)
            self.assertEqual(agent.updates, 2)
            self.assertEqual(agent.rollouts, 6)
            self.assertEqual(len(agent.memory), 0)
            with Path(report['rollout_history']).open(encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            for offset in (0, 3):
                group = rows[offset:offset+3]
                self.assertEqual(len({r['map_seed'] for r in group}), 1)
                self.assertEqual(len({r['rollout_seed'] for r in group}), 3)
                self.assertEqual(sum(r['selected'] == 'True' for r in group), 1)
            restored = VisualAgent.load(latest_path(path), training=True, device='cpu')
            self.assertEqual(restored.rollouts, 6)
            self.assertEqual(restored.updates, 2)

    def test_stop_discards_incomplete_group_without_optimizer_step(self):
        stop = Event()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(VisualAgent, 'learn_episode', side_effect=AssertionError('Unexpected BPTT')):
            report = train(1, batch_size=3, model_path=Path(directory)/'stop.pt', device='cpu',
                           stop=stop, on_step=lambda *args: stop.set())
            self.assertEqual(report['training_episodes'], 0)
            self.assertEqual(report['training_rollouts'], 0)

    def test_out_of_memory_keeps_last_disk_checkpoint(self):
        agent = VisualAgent(device='cpu', hidden_size=32)
        original = agent.network.head[-1].bias.detach().clone()
        def fail(trajectory):
            with torch.no_grad():
                agent.network.head[-1].bias.add_(100)
            raise MemoryError()
        with tempfile.TemporaryDirectory() as directory, \
                patch('train.VisualAgent', wraps=VisualAgent) as factory, \
                patch('train.Episode', side_effect=lambda rows: Episode(rows, max_time=.05)), \
                patch.object(agent, 'learn_episode', side_effect=fail):
            factory.return_value = agent
            path = Path(directory) / 'oom.pt'
            report = train(1, model_path=path, batch_size=2, device='cpu')
            self.assertIsNotNone(report['error'])
            restored = VisualAgent.load(latest_path(path), device='cpu')
            torch.testing.assert_close(restored.network.head[-1].bias, original)
            self.assertEqual(restored.updates, 0)

    def test_live_step_counter_without_preview(self):
        agent = VisualAgent(device='cpu', hidden_size=32)
        messages = []
        episodes = [Episode(['S.E'], max_time=.1) for _ in range(2)]
        run_group(agent, episodes, [1, 2], .7, on_status=messages.append)
        self.assertIn('łącznie 0 ruchów', messages[0])
        self.assertTrue(any('łącznie 2 ruchów' in message for message in messages))
        self.assertIn('ukończono 2/2', messages[-1])
        self.assertIn(f"łącznie {sum(ep.steps for ep in episodes)} ruchów", messages[-1])

    def test_last_twenty_counts_groups_and_persists_not_individual_robots(self):
        agent = VisualAgent(device='cpu', hidden_size=32)
        metrics = Episode(['SE']).metrics()
        groups = []
        for number in range(21):
            results = [{**metrics, 'won': number > 0}, {**metrics, 'won': False}, {**metrics, 'won': False}]
            groups.append({'winner': 0, 'trajectory': {}, 'results': results, 'steps_per_second': 1.})
        progress = []
        with tempfile.TemporaryDirectory() as directory, \
                patch('train.VisualAgent', wraps=VisualAgent) as factory, \
                patch('train.run_group', side_effect=groups), \
                patch.object(agent, 'learn_episode', return_value=0.):
            factory.return_value = agent
            path = Path(directory) / 'groups.pt'
            train(21, model_path=path, batch_size=3, device='cpu', progress=progress.append)
            self.assertEqual(progress[1]['recent_group_wins'], 1)
            self.assertEqual(progress[1]['recent_group_count'], 2)
            self.assertEqual(progress[1]['success_rate'], .5)
            self.assertEqual(progress[19]['next_width'], 5)
            self.assertEqual(progress[20]['next_width'], 7)
            self.assertEqual(progress[20]['recent_group_wins'], 20)
            restored = VisualAgent.load(latest_path(path), device='cpu')
            self.assertEqual(restored.recent_group_wins, [True] * 20)

    def test_gpu_metrics_and_missing_metrics(self):
        with patch('resources.subprocess.run', return_value=SimpleNamespace(stdout='55, 4096, 12288')):
            sample = gpu_sample()
        text = format_resources(sample)
        self.assertIn('55%', text)
        self.assertIn('4.00/12.00', text)
        self.assertIn('brak odczytu', format_resources({}))
        with patch('resources.subprocess.run', side_effect=FileNotFoundError):
            self.assertEqual(gpu_sample(), {})

if __name__ == '__main__':
    unittest.main()
