"""Przełączniki poziomu: nagrody, zapis formularza i walidacja modelu."""
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from unittest.mock import patch
from curriculum import default_plan, level_options, validate_plan, save_plan, load_plan
from episode import Episode, Action
from gui import COLORS
from map_designer import MapDesigner
from train import run_validation
from visual_agent import VisualAgent


class PenaltySwitchTests(unittest.TestCase):
    def test_collision_switch_keeps_block_and_time(self):
        for enabled in (True, False):
            ep = Episode(['S#E'], collision_penalty=enabled)
            for _ in range(4):
                ep.step(Action.FORWARD)
            result = ep.step(Action.FORWARD)
            self.assertFalse(result.moved)
            self.assertEqual(ep.collisions, 1)
            self.assertEqual(ep.reward_parts['collision'], -.01 if enabled else 0.)
            self.assertAlmostEqual(result.duration, .1)

    def test_turn_switch_both_directions_reset_and_unexecuted_action(self):
        for enabled in (True, False):
            ep = Episode(['S.E'], turn_penalty=enabled)
            for action in (Action.LEFT, Action.RIGHT):
                ep.step(action)
                self.assertEqual(ep.reward_parts['turn'], -.01 if enabled else 0.)
            ep.step(Action.FORWARD)
            self.assertEqual(ep.reward_parts['turn'], 0.)
            ep.reset()
            self.assertEqual(ep.turn_penalty, enabled)
        ep = Episode(['S.E'], max_time=.01, turn_penalty=True)
        ep.step(Action.RIGHT)
        self.assertEqual(ep.player.heading, 0)
        self.assertEqual(ep.reward_parts['turn'], 0.)

    def test_old_plan_defaults_and_invalid_switch_types(self):
        plan = default_plan()[:1]
        del plan[0]['collision_penalty']
        del plan[0]['turn_penalty']
        options = level_options(validate_plan(plan)[0])
        self.assertTrue(options['collision_penalty'])
        self.assertFalse(options['turn_penalty'])
        for key in ('collision_penalty', 'turn_penalty'):
            for invalid in ('false', 0, None):
                changed = dict(plan[0], **{key: invalid})
                with self.assertRaises(ValueError):
                    validate_plan([changed])

    def test_editor_switch_levels_and_disk_roundtrip(self):
        root = tk.Tk()
        root.withdraw()
        editor = MapDesigner(root, default_plan()[:2], COLORS, lambda plan: None)
        try:
            editor.collision_penalty.set(False)
            editor.turn_penalty.set(True)
            editor.levels.selection_clear(0, 'end')
            editor.levels.selection_set(1)
            editor.select()
            self.assertTrue(editor.collision_penalty.get())
            self.assertFalse(editor.turn_penalty.get())
            with TemporaryDirectory() as folder:
                path = Path(folder)/'plan.json'
                save_plan(editor.plan, path)
                level = load_plan(path)[0]
            self.assertFalse(level['collision_penalty'])
            self.assertTrue(level['turn_penalty'])
            self.assertTrue(Episode(level['rows'], **level_options(level)).turn_penalty)
        finally:
            editor.destroy()
            root.destroy()

    def test_validation_uses_saved_switches_and_records_them(self):
        agent = VisualAgent(hidden_size=16, device='cpu')
        options = dict(max_time=.01, collision_penalty=False, turn_penalty=True)
        agent.training_config['episode_options'] = options
        with TemporaryDirectory() as folder, patch('train.generate_maze', return_value=['S.E']):
            result = run_validation(agent, Path(folder)/'agent.pt')
        self.assertEqual(result['episode_options'], options)
        self.assertEqual(result['maps'], 100)
