"""Pełna kopia poziomu, niezależność mapy i kolejność planu."""
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from unittest.mock import patch
from curriculum import default_plan, save_plan, load_plan
from gui import COLORS
from map_designer import MapDesigner


class LevelOrderTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.editor = MapDesigner(self.root, default_plan()[:3], COLORS, lambda plan: None)

    def tearDown(self):
        self.editor.destroy()
        self.root.destroy()

    def test_duplicate_copies_entire_level_and_form_to_end_without_shared_data(self):
        editor = self.editor
        editor.name.set('Mój poziom')
        editor.pola.set('SE#.P')
        editor.turn_penalty.set(True)
        editor.collision_penalty.set(False)
        editor.plan[0]['extra'] = {'future_setting': [1, 2]}
        editor.duplicate()
        self.assertEqual(editor.index, 3)
        self.assertEqual(editor.plan[0], editor.plan[3])
        self.assertEqual(editor.plan[3]['name'], 'Mój poziom')
        self.assertTrue(editor.turn_penalty.get())
        editor.plan[3]['rows'][0] = '#####'
        editor.plan[3]['extra']['future_setting'].append(3)
        self.assertNotEqual(editor.plan[0]['rows'], editor.plan[3]['rows'])
        self.assertEqual(editor.plan[0]['extra']['future_setting'], [1, 2])

    def test_move_keeps_selected_level_form_and_saves_order(self):
        editor = self.editor
        editor.name.set('Przesuwany')
        editor.move_level(1)
        self.assertEqual(editor.index, 1)
        self.assertEqual(editor.plan[1]['name'], 'Przesuwany')
        self.assertEqual(editor.name.get(), 'Przesuwany')
        editor.move_level(1)
        self.assertEqual(editor.index, 2)
        self.assertEqual(str(editor.move_down_button['state']), 'disabled')
        editor.move_level(-1)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'plan.json'
            with patch('map_designer.save_plan', side_effect=lambda plan: save_plan(plan, path)), \
                    patch('map_designer.messagebox.showinfo'):
                editor.save()
            self.assertEqual([level['name'] for level in load_plan(path)],
                             ['Poziom 2', 'Przesuwany', 'Poziom 3'])

    def test_boundaries_and_invalid_form_do_not_reorder_or_duplicate(self):
        editor = self.editor
        before = [level['name'] for level in editor.plan]
        editor.move_level(-1)
        self.assertEqual(editor.index, 0)
        self.assertEqual(str(editor.move_up_button['state']), 'disabled')
        editor.required.set('invalid')
        with patch('map_designer.messagebox.showerror'):
            editor.duplicate()
            editor.move_level(1)
        self.assertEqual([level['name'] for level in editor.plan], before)
        self.assertEqual(editor.index, 0)
