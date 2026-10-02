"""Regresje wejścia wizualnego, pamięci rekurencyjnej i treningu sekwencji."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from visual_agent import VisualAgent, VisualQNetwork, START_ACTION
from episode import Episode, Action
from train import evaluate


class VisualAgentTests(unittest.TestCase):
    def test_four_actions_include_backward_and_start_is_separate(self):
        agent = VisualAgent(device='cpu')
        state = agent.observe(Episode(['S.E']))
        with torch.no_grad():
            agent.network.head[-1].weight.zero_()
            agent.network.head[-1].bias.copy_(torch.tensor([0., 0., 0., 10.]))
        self.assertEqual(agent.previous_action, 4)
        self.assertEqual(agent.act(state), Action.BACKWARD)
        self.assertEqual(agent.previous_action, 3)

    def test_checkpoint_records_current_motion_and_full_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'continuous.pt'
            VisualAgent(device='cpu').save(path)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint['version'], 6)
            self.assertEqual(checkpoint['movement'], (.1, .1, 10))
            self.assertEqual(checkpoint['camera'][2], 120.)
            self.assertEqual(checkpoint['actions'], ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'])
            resumed = VisualAgent.load(path, training=True, device='cpu')
            resumed.save(path)
            self.assertEqual(torch.load(path, weights_only=True)['bptt'], 'full_episode')

    def test_learning_uses_whole_life_longer_than_64_steps(self):
        agent = VisualAgent(device='cpu')
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        for step in range(100):
            agent.remember(frame, step % len(Action), 1.0, frame, step == 99)
        with patch.object(agent.network, 'forward', wraps=agent.network.forward) as forward:
            self.assertTrue(np.isfinite(agent.learn(batch_size=1)))
            self.assertEqual(forward.call_args.args[0].shape[1], 101)
            self.assertEqual(len(forward.call_args.args), 2)

    def test_unfinished_life_is_excluded_from_learning(self):
        agent = VisualAgent(device='cpu')
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        agent.remember(frame, 0, 1., frame, False)
        self.assertIsNone(agent.learn(batch_size=1))
        agent.remember(frame, 0, 1., frame, True)
        agent.remember(frame, 0, 1., frame, False)
        with patch.object(agent.network, 'forward', wraps=agent.network.forward) as forward:
            self.assertTrue(np.isfinite(agent.learn(batch_size=1)))
            self.assertEqual(forward.call_args.args[0].shape[1], 3)

    def test_training_resets_gru_before_each_episode(self):
        from train import train
        agent = VisualAgent(device='cpu')
        with torch.no_grad():
            agent.network.head[-1].weight.zero_()
            agent.network.head[-1].bias.copy_(torch.tensor([10., 0., 0., 0.]))
        original_act = agent.act_batch
        starts = []

        def act(states, previous, hidden, generators, epsilon):
            if all(action == START_ACTION for action in previous):
                starts.append((bool(torch.count_nonzero(hidden) == 0), len(previous)))
            return original_act(states, previous, hidden, generators, epsilon=0)

        with tempfile.TemporaryDirectory() as directory, \
                patch('train.VisualAgent', wraps=VisualAgent) as factory, \
                patch('train.generate_maze', return_value=['SE']), \
                patch('train.evaluate_suite', return_value={'score': 0}), \
                patch.object(agent, 'act_batch', side_effect=act):
            factory.return_value = agent
            train(episodes=2, model_path=Path(directory) / 'reset.pt', device='cpu', batch_size=5)
        self.assertEqual(starts, [(True, 5), (True, 5)])

    def make_replay(self, agent):
        rng = np.random.default_rng(2)
        for _ in range(8):
            state = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
            for step in range(8):
                next_state = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
                agent.remember(state, step % len(Action), 1.0, next_state, step == 7)
                state = next_state

    def test_cnn_and_gru_learn_and_checkpoint_resumes(self):
        agent = VisualAgent()
        self.make_replay(agent)
        before_cnn = agent.network.encoder[0].weight.detach().clone()
        before_gru = agent.network.gru.weight_ih_l0.detach().clone()
        loss = agent.learn(batch_size=2)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before_cnn, agent.network.encoder[0].weight))
        self.assertFalse(torch.equal(before_gru, agent.network.gru.weight_ih_l0))
        agent.curriculum = {"width": 7, "height": 7, "wins": [True, False, True]}
        agent.episodes = 9
        agent.env_steps = 64
        state = agent.memory[0]['frames'][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cnn.pt'
            agent.save(path)
            loaded = VisualAgent.load(path, training=True)
            self.assertEqual(loaded.episodes, 9)
            self.assertEqual(loaded.curriculum, agent.curriculum)
            self.assertEqual(loaded.updates, 1)
            self.assertEqual(loaded.env_steps, 64)
            self.assertEqual(loaded.memory_steps, 0)
            self.assertTrue(loaded.optimizer.state)
            self.assertEqual(agent.random.getstate(), loaded.random.getstate())
            self.assertEqual(agent.act(state), loaded.act(state))
            torch.testing.assert_close(agent.hidden, loaded.hidden)
            for a, b in zip(agent.target.parameters(), loaded.target.parameters()):
                torch.testing.assert_close(a, b)

    def test_exploration_updates_memory_and_reset_clears_it(self):
        agent = VisualAgent()
        state = agent.observe(Episode(['S.E']))
        agent.act(state, epsilon=1)
        first = agent.hidden.clone()
        self.assertEqual(tuple(first.shape), (1, 1, agent.hidden_size))
        agent.act(state, epsilon=1)
        self.assertFalse(torch.equal(first, agent.hidden))
        agent.reset_memory()
        self.assertIsNone(agent.hidden)
        self.assertEqual(agent.previous_action, START_ACTION)

    def test_final_output_gradient_reaches_first_frame(self):
        network = VisualQNetwork(hidden_size=32)
        # Deterministyczna ścieżka z dodatnimi aktywacjami i długą pamięcią GRU.
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.fill_(0.001)
            network.gru.bias_ih_l0[32:64].fill_(4.)
        frames = torch.full((1, 101, 60, 80, 3), 128., requires_grad=True)
        previous = torch.zeros((1, 101), dtype=torch.long)
        previous[0, 0] = START_ACTION
        values, _ = network(frames, previous)
        values[0, -1].sum().backward()
        self.assertGreater(frames.grad[0, 0].abs().sum().item(), 0)

    def test_replay_evicts_whole_episodes_and_preserves_boundaries(self):
        agent = VisualAgent(capacity=32)
        self.make_replay(agent)
        self.assertLessEqual(agent.memory_steps, 32)
        for episode in agent.memory:
            self.assertEqual(len(episode['frames']), len(episode['actions']) + 1)
            self.assertTrue(episode['dones'][-1])
            self.assertFalse(any(episode['dones'][:-1]))

    def test_observation_does_not_use_visited_cells(self):
        episode = Episode(['#######', '#S.#.E#', '#######'])
        agent = VisualAgent()
        first = agent.observe(episode)
        episode.visited.clear()
        torch.testing.assert_close(first, agent.observe(episode))
        self.assertEqual(first.shape, (60, 80, 3))

    def test_evaluation_resets_memory_for_each_map(self):
        agent = VisualAgent()
        with torch.no_grad():
            agent.network.head[-1].weight.zero_()
            agent.network.head[-1].bias.copy_(torch.tensor([10., 0., 0., 0.]))
        with patch.object(agent, 'reset_memory', wraps=agent.reset_memory) as reset:
            self.assertTrue(evaluate(agent, ['SE'])['won'])
            self.assertTrue(evaluate(agent, ['SE'])['won'])
            self.assertEqual(reset.call_count, 2)

    def test_legacy_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save({'version': 5, 'architecture': 'cnn_gru'}, path)
            with self.assertRaisesRegex(ValueError, '4 akcji'):
                VisualAgent.load(path)

if __name__ == '__main__':
    unittest.main()
