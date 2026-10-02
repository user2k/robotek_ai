"""Podział rankingu, niezależne historie i jedna wspólna aktualizacja BPTT."""
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sampling import validate_sampling, sample_ranked
from visual_agent import VisualAgent
from episode import Episode


class SamplingTests(unittest.TestCase):
    def test_rounding_sampling_membership_and_reproducibility(self):
        self.assertEqual(validate_sampling(128, [10, 20, 70], [3, 2, 3]), [13, 26, 89])
        results = [dict(won=i > 110, reward=i, score=i, steps=128-i) for i in range(128)]
        selected, pools = sample_ranked(results, [10, 20, 70], [3, 2, 3], 7)
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual([sum(i in pool for i in selected) for pool in pools], [3, 2, 3])
        self.assertEqual(pools[0], list(range(127, 114, -1)))
        self.assertEqual(sample_ranked(results, [10, 20, 70], [3, 2, 3], 7), (selected, pools))
        self.assertNotEqual(sample_ranked(results, [10, 20, 70], [3, 2, 3], 8)[0], selected)
        self.assertEqual(sorted(sum(pools, [])), list(range(128)))

    def test_invalid_counts_and_empty_pools(self):
        for percentages, counts in (([10, 20, 60], [1, 1, 1]), ([10, 20, 70], [3, 2, 3]),
                                    ([10, 20, 70], [0, 0, 0]), ([10, 20, 70], [-1, 1, 1])):
            with self.subTest(percentages=percentages, counts=counts), self.assertRaises(ValueError):
                validate_sampling(5, percentages, counts)
        self.assertEqual(validate_sampling(5, [0, 0, 100], [0, 0, 2]), [0, 0, 5])

    def test_win_has_priority_over_reward(self):
        results = [dict(won=False, reward=5, score=5, steps=1), dict(won=True, reward=-5, score=-5, steps=5)]
        selected, _ = sample_ranked(results, [50, 50, 0], [1, 0, 0], 7)
        self.assertEqual(selected, [1])

    def test_multi_episode_gradient_is_mean_and_hidden_is_separate(self):
        agent = VisualAgent(seed=9, device='cpu', hidden_size=16)
        rng = np.random.default_rng(9)
        traces = [dict(frames=[rng.integers(0, 256, (60, 80, 3), dtype=np.uint8) for _ in range(n+1)],
                       actions=[i % 4 for i in range(n)], rewards=[reward]*n,
                       dones=[False]*(n-1)+[True]) for n, reward in ((2, 1.), (4, -1.))]
        gradients, losses = [], []
        for trace in traces:
            reference = VisualAgent(seed=9, device='cpu', hidden_size=16)
            with patch.object(reference.optimizer, 'step'), patch('visual_agent.nn.utils.clip_grad_norm_'):
                losses.append(reference.learn_episode(trace))
                gradients.append([p.grad.clone() for p in reference.network.parameters()])
        calls = []
        def capture(module, args):
            calls.append(args)
        hook = agent.network.register_forward_pre_hook(capture)
        try:
            with patch.object(agent.optimizer, 'step') as step, patch('visual_agent.nn.utils.clip_grad_norm_') as clip:
                loss = agent.learn_episodes(traces)
            step.assert_called_once()
            clip.assert_called_once()
        finally:
            hook.remove()
        self.assertAlmostEqual(loss, sum(losses)/2, places=6)
        self.assertEqual(agent.updates, 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(args) == 2 for args in calls))  # hidden defaults to None every time
        for parameter, first, second in zip(agent.network.parameters(), *gradients):
            torch.testing.assert_close(parameter.grad, (first + second)/2, atol=1e-6, rtol=1e-4)

    def test_training_logs_all_selected_and_updates_once(self):
        from train import train, latest_path
        from curriculum import default_plan
        agent = VisualAgent(device='cpu', hidden_size=16)
        selection = dict(percentages=[25, 25, 50], counts=[1, 1, 2])
        reports = []
        with tempfile.TemporaryDirectory() as folder, patch('train.VisualAgent', wraps=VisualAgent) as factory, \
                patch('train.Episode', side_effect=lambda rows, **kwargs: Episode(rows, max_time=.1)):
            factory.return_value = agent
            path = Path(folder)/'sampled.pt'
            report = train(1, model_path=path, device='cpu', batch_size=8, selection=selection,
                           plan=default_plan(), progress=reports.append)
            self.assertEqual(agent.updates, 1)
            self.assertEqual(reports[0]['bptt_count'], 4)
            with Path(report['rollout_history']).open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(row['selected'] == 'True' for row in rows), 4)
            self.assertEqual([sum(row['selected'] == 'True' and row['rank_group'] == str(i) for row in rows) for i in (1, 2, 3)], [1, 1, 2])
            self.assertEqual(VisualAgent.load(latest_path(path), device='cpu').training_selection, selection)
            self.assertEqual(report['selection'], selection)

    def test_reject_partial_episode_before_updating(self):
        agent = VisualAgent(device='cpu', hidden_size=16)
        with patch.object(agent.optimizer, 'step') as step:
            with self.assertRaises(ValueError):
                agent.learn_episodes([dict(dones=[False])])
            step.assert_not_called()

if __name__ == '__main__':
    unittest.main()
