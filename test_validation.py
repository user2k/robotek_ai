import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from episode import Episode
from train import train, run_validation, VALIDATION_SEEDS, training_seed, latest_path
from visual_agent import VisualAgent


class ValidationTests(unittest.TestCase):
    def test_eval_restores_memory_rng_mode_and_appends_same_csv(self):
        agent = VisualAgent(hidden_size=16, device='cpu')
        agent.episodes = agent.updates = 1000
        agent.hidden = torch.ones(1, 1, 16)
        agent.previous_action = 2
        hidden, rng = agent.hidden, agent.random.getstate()
        options = dict(max_time=.05, move_distance=1., turn_degrees=90., start_direction='random')
        real_episode = Episode
        with tempfile.TemporaryDirectory() as directory, \
                patch('train.generate_maze', return_value=['S.E']) as generator, \
                patch('train.Episode', side_effect=lambda rows, **kwargs: real_episode(rows, **kwargs)), \
                patch.object(agent, 'learn_episode', side_effect=AssertionError('Learning in eval')):
            path = Path(directory) / 'agent.pt'
            for _ in range(2):
                result = run_validation(agent, path, episode_options=options)
                self.assertEqual(result['maps'], 100)
            with path.with_name('agent.validation.csv').open(encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 200)
            self.assertEqual(rows[0]['episodes'], '1000')
            self.assertEqual(rows[0]['move_distance'], '1.0')
            self.assertEqual(rows[0]['turn_degrees'], '90.0')
            self.assertEqual(rows[0]['epsilon'], '0.0')
            self.assertEqual([r['map_seed'] for r in rows[:100]], [r['map_seed'] for r in rows[100:]])
            generator.assert_called_with(11, 11, VALIDATION_SEEDS[-1], max_time=.05)
        self.assertTrue(agent.network.training)
        self.assertIs(agent.hidden, hidden)
        self.assertEqual(agent.previous_action, 2)
        self.assertEqual(agent.random.getstate(), rng)
        self.assertEqual(agent.updates, 1000)
        self.assertTrue(set(VALIDATION_SEEDS).isdisjoint(training_seed(7, i) for i in range(2000)))

    def test_resume_runs_at_global_boundary_and_saves_config(self):
        agent = VisualAgent(hidden_size=16, device='cpu')
        agent.episodes = agent.updates = 999
        metrics = Episode(['SE']).metrics()
        group = dict(winner=0, trajectory={}, results=[metrics], steps_per_second=1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.pt'
            agent.save(latest_path(path))
            with patch('train.run_group', return_value=group), \
                    patch.object(VisualAgent, 'learn_episode', return_value=0.), \
                    patch('train.run_validation', return_value={}) as validate:
                train(2, model_path=path, batch_size=1, device='cpu')
            self.assertEqual(validate.call_count, 1)
            snapshots = list((Path(directory) / 'checkpoints').glob('*.pt'))
            self.assertEqual(len(snapshots), 1)
            checkpoint = torch.load(snapshots[0], weights_only=True)
            self.assertEqual(checkpoint['episodes'], 1000)
            self.assertEqual(checkpoint['training_config']['batch_size'], 1)
            self.assertEqual(checkpoint['training_config']['validation_size'], [11, 11])
            self.assertIn('episode_options', checkpoint['training_config'])


if __name__ == '__main__':
    unittest.main()
