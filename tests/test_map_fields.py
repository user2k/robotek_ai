"""Regresja: pole formularza musi dotrzeć do planu i wrócić z dysku."""
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from unittest.mock import patch

from curriculum import default_plan, load_plan, save_plan, validate_plan
from gui import COLORS
from map_designer import MapDesigner


class MapFieldsTests(unittest.TestCase):
    def test_editor_switch_save_reopen_preserves_each_level_fields(self):
        root = tk.Tk()
        root.withdraw()
        editor = MapDesigner(root, default_plan()[:2], COLORS, lambda plan: None)
        self.addCleanup(root.destroy)
        self.addCleanup(editor.destroy)
        editor.pola.set('SE.#')
        editor.levels.selection_clear(0, 'end')
        editor.levels.selection_set(1)
        editor.select()
        self.assertEqual(editor.plan[0]['pola'], 'SE.#')
        self.assertEqual(editor.pola.get(), 'SE#.')
        editor.pola.set('SE.PBWF#')
        with TemporaryDirectory() as folder:
            path = Path(folder)/'levels.json'
            with patch('map_designer.save_plan', side_effect=lambda plan: save_plan(plan, path)), \
                    patch('map_designer.messagebox.showinfo'):
                editor.save()
            restored = load_plan(path)
        self.assertEqual([level['pola'] for level in restored], ['SE.#', 'SE.PBWF#'])
        reopened = MapDesigner(root, restored, COLORS, lambda plan: None)
        try:
            self.assertEqual(reopened.pola.get(), 'SE.#')
            reopened.index = 1
            reopened.load()
            self.assertEqual(reopened.pola.get(), 'SE.PBWF#')
        finally:
            reopened.destroy()

    def test_invalid_editor_fields_do_not_replace_saved_level(self):
        root = tk.Tk()
        root.withdraw()
        editor = MapDesigner(root, default_plan()[:1], COLORS, lambda plan: None)
        try:
            for fields in ('S.', 'E.', 'SEX', '', 'SE.', 'SE'):
                editor.pola.set(fields)
                with patch('map_designer.messagebox.showerror') as error:
                    self.assertFalse(editor.commit())
                    error.assert_called_once()
                self.assertEqual(editor.plan[0]['pola'], 'SE#.')
        finally:
            editor.destroy()
            root.destroy()

    def test_old_plan_defaults_and_invalid_json_field_types(self):
        plan = default_plan()[:1]
        del plan[0]['pola']
        self.assertEqual(validate_plan(plan)[0]['pola'], 'SE#.')
        for fields in (None, 123, ['S', 'E'], {}, 'SE?'):
            plan[0]['pola'] = fields
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_plan(plan)
