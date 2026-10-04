"""Sprawdzenie przekazania klatek treningu do głównego wątku GUI."""
from types import SimpleNamespace
from threading import Event
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from episode import Action, Episode
from gui import GameApp


class LiveGuiTests(unittest.TestCase):
    def setUp(self):
        from curriculum import default_plan
        plan_loader = patch('gui.load_plan', side_effect=default_plan)
        plan_loader.start()
        self.addCleanup(plan_loader.stop)
        settings_loader = patch('gui.load_settings', return_value=dict(enabled=False, percentages=[10, 20, 70], counts=[3, 2, 3]))
        settings_loader.start()
        self.addCleanup(settings_loader.stop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = GameApp(self.root)

    def tearDown(self):
        self.app.close()
        if self.app.training_thread is not None:
            self.app.training_thread.join(timeout=3)

    def test_keys_move_continuously_and_robot_is_drawn_to_scale(self):
        from visual_agent import observe_camera
        import numpy as np
        self.app.current_rows = ['S..E']
        self.app.reset()
        self.app.on_key(SimpleNamespace(keysym='w'))
        self.assertAlmostEqual(self.app.player.x, .6)
        self.app.on_key(SimpleNamespace(keysym='s'))
        self.assertAlmostEqual(self.app.player.x, .5)
        self.app.on_key(SimpleNamespace(keysym='d'))
        self.assertEqual(self.app.player.heading, 10)
        self.app.on_key(SimpleNamespace(keysym='a'))
        self.assertEqual(self.app.player.heading, 0)
        left, top, right, bottom = self.app.canvas.coords('robot')
        self.assertAlmostEqual(right - left, .2 * 48)
        self.assertAlmostEqual(bottom - top, .2 * 48)
        np.testing.assert_array_equal(self.app.camera_frame, observe_camera(self.app.episode))

    def test_selected_start_level_is_forwarded_once(self):
        with patch('train.train', return_value={'training_episodes': 0}) as train:
            self.app.start_level_choice.set('3')
            self.app.start_training()
            self.app.training_thread.join(timeout=2)
            self.assertEqual(train.call_args.kwargs['start_level'], 3)
            self.assertEqual(self.app.start_level_choice.get(), 'Kontynuuj')
            self.root.after_cancel(self.app.poll_callback)
            self.app.poll_training()
            self.assertEqual(str(self.app.start_level_input['state']), 'readonly')

    def test_sampling_settings_saved_and_forwarded(self):
        import tempfile
        from pathlib import Path
        from sampling_gui import load_settings
        self.app.batch_count.set('128')
        self.app.open_sampling()
        dialog = self.app.sampling_dialog
        dialog.enabled.set(True)
        with tempfile.TemporaryDirectory() as folder, patch('sampling_gui.SETTINGS_PATH', Path(folder)/'settings.json'):
            dialog.save()
            self.assertEqual(load_settings()['counts'], [3, 2, 3])
        with patch('train.train', return_value={'training_episodes': 0}) as train:
            self.app.start_training()
            self.app.training_thread.join(timeout=2)
            self.assertEqual(train.call_args.kwargs['selection'], dict(percentages=[10, 20, 70], counts=[3, 2, 3]))
            self.root.after_cancel(self.app.poll_callback)
            self.app.poll_training()
        self.app.batch_count.set('5')
        with patch('train.train') as train:
            self.app.start_training()
            train.assert_not_called()
            self.assertIn('nie można', self.app.training_status['text'])

    def test_preview_level_keeps_movement_for_manual_play(self):
        from curriculum import default_plan
        level = default_plan()[0]
        level.update(mode='custom', name='Duży krok', move_distance=1., turn_degrees=90.)
        self.app.preview_level(level)
        self.assertEqual(self.app.map_info['text'], 'Duży krok')
        self.app.play_action(Action.FORWARD)
        self.assertAlmostEqual(self.app.player.x, 2.5)
        self.app.play_action(Action.RIGHT)
        self.assertEqual(self.app.player.heading, 90.)
        self.app.reset()
        self.assertEqual(self.app.player.move_distance, 1.)
        self.assertIn('90°', self.app.movement_label['text'])
        self.app.show_demo()
        self.assertEqual(self.app.player.move_distance, .1)

    def test_stop_unlocks_training_and_allows_another_run(self):
        entered = Event()
        def fake_train(*args, stop, **kwargs):
            entered.set()
            stop.wait(timeout=2)
            return {'training_episodes': 0}
        with patch('train.train', side_effect=fake_train) as train:
            for _ in range(2):
                entered.clear()
                self.app.start_training()
                self.assertTrue(entered.wait(timeout=2))
                self.assertEqual(str(self.app.train_button['state']), 'disabled')
                self.app.stop_work()
                self.app.training_thread.join(timeout=2)
                self.root.after_cancel(self.app.poll_callback)
                self.app.poll_training()
                self.assertEqual(str(self.app.train_button['state']), 'normal')
                self.assertEqual(str(self.app.validation_button['state']), 'normal')
            self.assertEqual(train.call_count, 2)

    def test_manual_validation_result_unlocks_buttons(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.pt'
            path.touch()
            with patch('gui.MODEL_PATH', path), \
                    patch('visual_agent.VisualAgent.load'), \
                    patch('train.run_validation', return_value={'wins': 2, 'maps': 100, 'score': 5}) as evaluate:
                self.app.start_validation()
                self.app.training_thread.join(timeout=2)
                self.root.after_cancel(self.app.poll_callback)
                self.app.poll_training()
                evaluate.assert_called_once()
                self.assertIn('2/100', self.app.training_status['text'])
                self.assertEqual(str(self.app.train_button['state']), 'normal')
                self.assertEqual(str(self.app.watch_button['state']), 'normal')

    def test_disabled_preview_does_not_render_camera_or_resize_window(self):
        self.root.update_idletasks()
        size = (self.root.winfo_reqwidth(), self.root.winfo_reqheight())
        self.app.camera_enabled.set(False)
        with patch('gui.TensorCamera', side_effect=AssertionError('Unwanted preview')):
            self.app.draw()
        self.root.update_idletasks()
        self.assertEqual(size, (self.root.winfo_reqwidth(), self.root.winfo_reqheight()))
        self.assertIsNone(self.app.camera_frame)

    def metadata(self):
        return {"episode": 1, "session_episode": 1, "map_seed": 42,
                "epsilon": 0.8, "action": None, "reward": 0}

    def test_preview_is_detached_and_queue_is_bounded(self):
        self.app.watch_training.set()
        self.app.preview_delay = 0
        episode = Episode(["S.E"])
        for _ in range(5):
            self.app.publish_training_frame(episode, self.metadata())
        self.assertEqual(self.app.preview_frames.qsize(), 1)
        frame, metadata = self.app.preview_frames.get_nowait()
        episode.step(Action.FORWARD)
        self.assertEqual(frame.player.x, 0.5)
        self.assertEqual(frame.player.elapsed_time, 0)
        self.app.show_training_frame(frame, metadata)
        self.assertIs(self.app.episode, frame)
        self.assertIn("80%", self.app.message)
        self.app.stop_work()
        self.app.publish_training_frame(episode, self.metadata())
        self.assertTrue(self.app.preview_frames.empty())

    def test_watching_training_blocks_manual_input_and_stop_interrupts_delay(self):
        published = Event()
        stopped = Event()

        def fake_train(*args, on_step, stop, **kwargs):
            episode = Episode(["S.E"])
            published.set()
            on_step(episode, self.metadata())
            stopped.set()
            return {"training_episodes": 0, "test": {"wins": 0, "maps": 16, "score": -300}}

        self.app.preview_delay = 5
        with patch("train.train", side_effect=fake_train):
            self.app.start_watching()
            self.assertTrue(published.wait(timeout=3))
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and self.app.episode.rows != ["S.E"]:
                self.root.update()
                time.sleep(0.01)
            self.assertEqual(self.app.episode.rows, ["S.E"])
            episode = self.app.episode
            self.app.reset()
            self.app.new_maze()
            self.assertIs(self.app.episode, episode)
            self.app.stop_work()
            self.assertTrue(stopped.wait(timeout=1))
            self.app.training_thread.join(timeout=1)
            self.assertFalse(self.app.training_thread.is_alive())
            self.assertFalse(self.app.watch_training.is_set())

    def test_max_preview_never_waits_and_skips_intermediate_copies(self):
        self.app.watch_training.set()
        self.app.preview_max.set(True)
        self.app.set_preview_speed(10)
        self.assertEqual(self.app.preview_delay, 0)
        episode = Episode(["S.E"])
        with patch("gui.monotonic", return_value=100), \
                patch("gui.deepcopy", wraps=__import__("copy").deepcopy) as copy, \
                patch.object(self.app.training_stop, "wait") as wait:
            self.app.publish_training_frame(episode, self.metadata())
            episode.step(Action.FORWARD)
            self.app.publish_training_frame(episode, self.metadata())
            copy.assert_called_once()
            wait.assert_not_called()
        self.app.preview_max.set(False)
        self.app.set_preview_speed(20)
        self.assertEqual(self.app.preview_delay, 0.05)

    def test_finished_training_episode_has_no_pause(self):
        self.app.watch_training.set()
        episode = Episode(["SE"])
        for _ in range(5):
            episode.step(Action.FORWARD)
        with patch.object(self.app.training_stop, "wait") as wait:
            self.app.publish_training_frame(episode, self.metadata())
            wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
