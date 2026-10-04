"""Kolejka ocen, wspólny podgląd i terminalna kara operatora."""
import json
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from batching import run_group
from episode import Action, Episode
import tests.test_live_gui as test_live_gui
from train import latest_path, run_validation, train
from visual_agent import VisualAgent


class InterventionTests(unittest.TestCase):
    def test_swat_is_exactly_minus_twenty_and_idempotent(self):
        episode = Episode(['S.E'])
        episode.step(Action.FORWARD)
        score, reward = episode.score, episode.total_reward
        self.assertTrue(episode.swat())
        self.assertTrue(episode.done)
        self.assertEqual(episode.score, score-20)
        self.assertEqual(episode.total_reward, reward-20)
        self.assertFalse(episode.swat())
        self.assertTrue(episode.player.alive)
        self.assertEqual(episode.outcome, 'Boża kara −20')

    def test_operator_penalty_reaches_terminal_bptt_transition(self):
        agent = VisualAgent(hidden_size=16, device='cpu')
        episodes = [Episode(['S..E'], max_time=.4) for _ in range(2)]
        requests = Queue()

        def publish(episode, metadata):
            if episode.steps == 1:
                requests.put((6, 0))  # przeterminowana grupa
                requests.put((7, 0))
                requests.put((7, 0))  # podwójne kliknięcie

        with patch.object(agent, 'act_batch', side_effect=lambda states, previous, hidden, rng, epsilon:
                          ([0]*len(previous), hidden)):
            result = run_group(agent, episodes, [1, 2], 0., on_step=publish,
                               metadata={'episode': 7}, interventions=requests)
        killed = result['trajectories'][0]
        self.assertTrue(episodes[0].swatted)
        self.assertFalse(episodes[1].swatted)
        self.assertEqual(len(killed['actions']), 1)
        self.assertTrue(killed['dones'][-1])
        self.assertFalse(any(killed['dones'][:-1]))
        self.assertAlmostEqual(killed['rewards'][-1], -20.0005)
        self.assertEqual(len(killed['poses']), len(killed['actions'])+1)
        agent.learn_episode(killed)
        self.assertEqual(agent.updates, 1)

    def test_manual_queue_and_periodic_validation_both_report_results(self):
        agent = VisualAgent(hidden_size=16, device='cpu')
        agent.episodes = agent.updates = 999
        group = dict(winner=0, trajectory={}, results=[Episode(['SE']).metrics()], steps_per_second=1.)
        requests = Queue()
        requests.put('manual')
        results = []
        with TemporaryDirectory() as directory:
            path = Path(directory)/'agent.pt'
            agent.save(latest_path(path))
            with patch('train.run_group', return_value=group), \
                    patch.object(VisualAgent, 'learn_episode', return_value=0.), \
                    patch('train.run_validation', return_value={'maps': 100}) as validate:
                train(1, model_path=path, batch_size=1, device='cpu',
                      validation_requests=requests, on_validation=results.append)
        self.assertEqual(validate.call_count, 2)
        self.assertNotIn('validation_kind', validate.call_args_list[0].kwargs)
        self.assertEqual(validate.call_args_list[1].kwargs['validation_kind'], 'periodic')
        self.assertEqual(len(results), 2)
        self.assertTrue(requests.empty())

    def test_validation_persists_quality_summary(self):
        agent = VisualAgent(hidden_size=16, device='cpu')
        with TemporaryDirectory() as directory, patch('train.generate_maze', return_value=['S.E']):
            path = Path(directory)/'agent.pt'
            result = run_validation(agent, path, episode_options={'max_time': .05}, validation_kind='periodic')
            summary = json.loads(path.with_name('agent.validation.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(summary['success_rate'], result['wins']/result['maps'])
        self.assertNotIn('q_values', summary)
        self.assertNotIn('results', summary)
        self.assertEqual(summary['kind'], 'periodic')

    def test_request_at_last_group_is_not_lost(self):
        requests = Queue()
        group = dict(winner=0, trajectory={}, results=[Episode(['SE']).metrics()], steps_per_second=1.)
        with TemporaryDirectory() as directory, patch('train.run_group', return_value=group), \
                patch.object(VisualAgent, 'learn_episode', return_value=0.), \
                patch('train.run_validation', return_value={}) as validate:
            train(1, model_path=Path(directory)/'agent.pt', batch_size=1, device='cpu',
                  validation_requests=requests, progress=lambda row: requests.put('manual'))
        validate.assert_called_once()
        self.assertTrue(requests.empty())

    def test_request_at_last_group_is_not_lost(self):
        requests = Queue()
        group = dict(winner=0, trajectory={}, results=[Episode(['SE']).metrics()], steps_per_second=1.)
        with TemporaryDirectory() as directory, patch('train.run_group', return_value=group), \
                patch.object(VisualAgent, 'learn_episode', return_value=0.), \
                patch('train.run_validation', return_value={}) as validate:
            train(1, model_path=Path(directory)/'agent.pt', batch_size=1, device='cpu',
                  validation_requests=requests, progress=lambda row: requests.put('manual'))
        validate.assert_called_once()
        self.assertTrue(requests.empty())


class DashboardTests(unittest.TestCase):
    metadata = test_live_gui.LiveGuiTests.metadata
    tearDown = test_live_gui.LiveGuiTests.tearDown

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model = patch('gui.MODEL_PATH', Path(directory.name)/'agent.pt')
        model.start()
        self.addCleanup(model.stop)
        test_live_gui.LiveGuiTests.setUp(self)

    def test_all_bots_are_detached_and_camera_can_switch(self):
        self.app.watch_training.set()
        self.app.preview_delay = 0
        bots = [Episode(['S.E']), Episode(['S.E'])]
        bots[1].step(Action.FORWARD)
        metadata = dict(self.metadata(), bots=bots, q_values=[[1., 2., 3., 4.], [5., 6., 7., 8.]])
        self.app.publish_training_frame(bots[0], metadata)
        frame, copied = self.app.preview_frames.get_nowait()
        bots[1].step(Action.FORWARD)
        self.app.show_training_frame(frame, copied)
        self.assertEqual(len(self.app.canvas.find_withtag('bot-0')), 1)
        self.assertEqual(len(self.app.canvas.find_withtag('bot-1')), 1)
        self.app.selected_bot.set('2')
        self.app.select_bot()
        self.assertAlmostEqual(self.app.player.x, .6)
        self.assertIn('brak walidacji', self.app.q_label['text'])

    def test_manual_validations_do_not_change_four_periodic_q_mean(self):
        for value in range(1, 6):
            self.app.add_validation(dict(maps=100, wins=value*10, score=value, episodes=value*1000,
                                         kind='periodic', q_values=[float(value)]*4))
        self.app.add_validation(dict(maps=100, wins=4, score=88., episodes=5001,
                                     kind='manual', q_values=[99.]*4))
        self.assertIn('4/4', self.app.q_average_label['text'])
        self.assertIn('35.0%', self.app.q_average_label['text'])
        self.assertIn('4.0%', self.app.q_label['text'])
        self.assertEqual(self.app.validation_log.item(self.app.validation_log.get_children()[-1], 'values')[3], '4.0%')
        self.assertEqual(len(self.app.validation_log.get_children()), 6)
        self.assertTrue(self.app.validation_graph.find_all())

    def test_click_queues_swat_for_current_group_and_selected_bot(self):
        self.app.watch_training.set()
        self.app.training_thread = SimpleNamespace(is_alive=lambda: True)
        self.app.training_bots = [Episode(['S.E']), Episode(['S.E'])]
        self.app.live_metadata = {'episode': 27}
        self.app.selected_bot.set('2')
        self.app.swatter.set(True)
        self.app.click_bot(SimpleNamespace(x=24, y=24))
        self.assertEqual(self.app.interventions.get_nowait(), (27, 1))
        self.app.stop_work()
        self.app.training_thread = None

    def test_validation_click_queues_during_training_and_stop_clears_queue(self):
        self.app.training_thread = SimpleNamespace(is_alive=lambda: True)
        self.app.work_kind = 'training'
        self.app.start_validation()
        self.app.start_validation()
        self.assertEqual(self.app.validation_requests.qsize(), 2)
        self.app.stop_work()
        self.assertTrue(self.app.validation_requests.empty())
        self.app.training_thread = None

    def test_history_reloads_scores_and_periodic_q(self):
        import gui
        entries = [dict(maps=100, wins=9, score=42., episodes=1000,
                        kind='periodic', q_values=[1., 2., 3., 4.]),
                   dict(maps=100, wins=10, score=43., episodes=1001, kind='manual')]
        gui.MODEL_PATH.with_name('agent.validation.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in entries)+'\n', encoding='utf-8')
        self.app.load_validation_history()
        self.assertEqual(len(self.app.validation_log.get_children()), 2)
        self.assertIn('1/4', self.app.q_average_label['text'])
        self.assertIn('9.0%', self.app.q_average_label['text'])
        self.assertIn('10.0%', self.app.q_label['text'])


if __name__ == '__main__':
    unittest.main()
