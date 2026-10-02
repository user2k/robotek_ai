import tempfile
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from curriculum import *


class CurriculumTests(unittest.TestCase):
    def test_order_failure_resume_and_completion(self):
        plan = default_plan()[:3]
        plan[0].update(mode='custom', required=2)
        plan[1].update(required=2)
        plan[2].update(mode='custom', size=7, rows=blank_map(7), required=1)
        state = prepare_curriculum(None, plan)
        self.assertEqual(level_rows(state, 1), level_rows(state, 2))
        for won in (True, False, True):
            advance_level(state, won)
        self.assertEqual(state['level'], 0)
        self.assertIs(prepare_curriculum(state, plan), state)
        advance_level(state, True)
        self.assertEqual(state['level'], 1)
        self.assertEqual(state['wins'], [])
        advance_level(state, True)
        advance_level(state, True)
        self.assertEqual(state['width'], 7)
        advance_level(state, True)
        self.assertTrue(state['completed'])
        plan[0]['required'] = 3
        self.assertEqual(prepare_curriculum(state, plan)['level'], 0)

    def test_loss_resets_streak_and_requires_twenty_new_wins(self):
        state = prepare_curriculum(None, default_plan()[:2])
        for _ in range(19):
            advance_level(state, True)
        self.assertEqual(len(state['wins']), 19)
        advance_level(state, False)
        self.assertEqual(state['wins'], [])
        self.assertEqual(state['level'], 0)
        for _ in range(19):
            advance_level(state, True)
        self.assertEqual(state['level'], 0)
        advance_level(state, True)
        self.assertEqual(state['level'], 1)
        self.assertEqual(state['wins'], [])

    def test_old_window_migrates_to_trailing_streak(self):
        plan = default_plan()
        state = prepare_curriculum(None, plan)
        state['wins'] = [True, True, False, True]
        restored = prepare_curriculum(state, plan)
        self.assertEqual(restored['wins'], [True])
        advance_level(restored, False)
        self.assertEqual(restored['wins'], [])

    def test_explicit_start_resets_only_progress_and_resume_keeps_it(self):
        plan = default_plan()
        previous = prepare_curriculum(None, plan)
        previous.update(completed=True, wins=[True] * 20)
        state = prepare_curriculum(previous, plan, start_level=4)
        self.assertEqual(state['level'], 3)
        self.assertEqual(state['width'], plan[3]['size'])
        self.assertEqual(state['height'], plan[3]['size'])
        self.assertFalse(state['completed'])
        self.assertEqual(state['wins'], [])
        advance_level(state, True)
        self.assertIs(prepare_curriculum(state, plan), state)
        self.assertEqual(state['wins'], [True])
        for invalid in (0, 11, True, 1.5):
            with self.subTest(level=invalid), self.assertRaises(ValueError):
                prepare_curriculum(state, plan, start_level=invalid)

    def test_movement_migration_and_rename_preserve_progress(self):
        plan = default_plan()[:1]
        old = deepcopy(plan)
        for key in ('name', 'move_distance', 'turn_degrees'):
            del old[0][key]
        state = dict(plan=old, level=0, wins=[True], completed=False)
        self.assertIs(prepare_curriculum(state, plan), state)
        plan[0]['name'] = 'Nowa nazwa'
        self.assertIs(prepare_curriculum(state, plan), state)
        self.assertEqual(state['wins'], [True])
        plan[0]['move_distance'] = 1.
        self.assertEqual(prepare_curriculum(state, plan)['wins'], [])

    def test_validation_and_disk_roundtrip(self):
        plan = default_plan()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'levels.json'
            save_plan(plan, path)
            self.assertEqual(load_plan(path), plan)
        plan[0].update(mode='custom', rows=['S####', '#####', '#####', '#####', '####E'])
        with self.assertRaisesRegex(ValueError, 'blokują'):
            validate_plan(plan)

    def test_training_uses_plan_and_stops_at_end(self):
        from train import train, latest_path
        from visual_agent import VisualAgent
        from episode import Episode
        agent = VisualAgent(device='cpu', hidden_size=32)
        plan = default_plan()[:2]
        plan[0].update(mode='custom', required=1, max_time=35)
        plan[1].update(size=7, rows=blank_map(7), required=1, max_time=75, move_distance=1., turn_degrees=90., start_direction='up')
        observed = []
        limits = []
        movements = []
        headings = []
        def group(agent, episodes, seeds, epsilon, **kwargs):
            observed.append(episodes[0].rows)
            limits.append(episodes[0].max_time)
            headings.append(episodes[0].player.heading)
            movements.append((episodes[0].player.move_distance, episodes[0].player.turn_degrees))
            return dict(winner=0, trajectory={}, results=[dict(episodes[0].metrics(), won=True)], steps_per_second=1.)
        with tempfile.TemporaryDirectory() as folder, patch('train.VisualAgent', wraps=VisualAgent) as factory, patch('train.run_group', side_effect=group), patch.object(agent, 'learn_episode', return_value=0.):
            factory.return_value = agent
            path = Path(folder)/'agent.pt'
            result = train(10, model_path=path, batch_size=1, device='cpu', plan=plan)
            self.assertEqual(result['training_episodes'], 2)
            self.assertEqual(limits, [35, 75])
            self.assertEqual(headings, [0., 270.])
            self.assertEqual(movements, [(.1, 10.), (1., 90.)])
            self.assertEqual(observed[0], plan[0]['rows'])
            self.assertEqual(len(observed[1]), 7)
            self.assertTrue(VisualAgent.load(latest_path(path), device='cpu').curriculum['completed'])
            result = train(10, model_path=path, batch_size=1, device='cpu', plan=plan)
            self.assertEqual(result['training_episodes'], 0)
            with patch.object(VisualAgent, 'learn_episode', return_value=0.):
                result = train(10, model_path=path, batch_size=1, device='cpu', plan=plan, start_level=2)
            self.assertEqual(result['training_episodes'], 1)
            self.assertEqual(limits, [35, 75, 75])

    def test_old_plan_retains_progress_and_time_validation(self):
        old = default_plan()[:1]
        del old[0]['max_time']
        state = dict(plan=old, level=0, wins=[True], completed=False)
        self.assertIs(prepare_curriculum(state, default_plan()[:1]), state)
        self.assertEqual(state['wins'], [True])
        self.assertEqual(state['plan'][0]['max_time'], 200)
        for invalid in (0, -1, 1, float('nan'), float('inf'), True, '30'):
            plan = default_plan()[:1]
            plan[0]['max_time'] = invalid
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_random_generator_receives_limit_and_episode_times_out(self):
        from episode import Episode, Action
        plan = default_plan()[:1]
        plan[0]['max_time'] = 2
        state = prepare_curriculum(None, plan)
        with patch('curriculum.generate_maze', wraps=generate_maze) as generate:
            rows = level_rows(state, 10)
            generate.assert_called_once_with(5, 5, 10, max_time=2)
        episode = Episode(rows, max_time=2)
        while not episode.done:
            episode.step(Action.LEFT)
        self.assertTrue(episode.timed_out)
        self.assertEqual(episode.player.elapsed_time, 2)

    def test_editor_paint_switch_and_save(self):
        from map_designer import MapDesigner
        from gui import COLORS
        root = tk.Tk()
        root.withdraw()
        saved = []
        editor = MapDesigner(root, default_plan(), COLORS, saved.append)
        try:
            editor.mode.set('custom')
            editor.brush.set('S')
            editor.paint(SimpleNamespace(x=10, y=10))
            self.assertEqual(editor.plan[0]['rows'][0][0], 'S')
            self.assertEqual(sum(row.count('S') for row in editor.plan[0]['rows']), 1)
            editor.required.set('10')
            editor.max_time.set('45,5')
            editor.name.set('Pierwsze kroki')
            editor.move_distance.set('1')
            editor.turn_degrees.set('90')
            editor.start_direction.set('Losowy')
            editor.levels.selection_clear(0, 'end')
            editor.levels.selection_set(1)
            editor.select()
            self.assertEqual(editor.plan[0]['required'], 10)
            self.assertEqual(editor.plan[0]['max_time'], 45.5)
            self.assertEqual(editor.plan[0]['name'], 'Pierwsze kroki')
            self.assertEqual(editor.plan[0]['move_distance'], 1.)
            self.assertEqual(editor.plan[0]['turn_degrees'], 90.)
            self.assertEqual(editor.plan[0]['start_direction'], 'random')
            self.assertEqual(editor.plan[0]['mode'], 'custom')
            with patch('map_designer.save_plan') as save, patch('map_designer.messagebox.showinfo'):
                editor.save()
                save.assert_called_once()
                self.assertEqual(saved[0][0]['rows'][0][0], 'S')
        finally:
            editor.destroy()
            root.destroy()

if __name__ == '__main__':
    unittest.main()
