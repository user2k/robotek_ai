"""Rozmiary 5–255, skalowanie i kliknięcia w zmniejszonym podglądzie."""
from collections import deque
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import patch
import torch

from curriculum import default_plan, validate_plan, blank_map
from maps import generate_maze, validate_size
from map_view import map_scale
from gui import GameApp, COLORS
from map_designer import MapDesigner
from episode import Episode
from tensor_camera import TensorCamera, camera_pose


class LargeMapTests(unittest.TestCase):
    def test_sizes_and_maximum_map_connectivity(self):
        for size in range(5, 256, 2):
            plan = default_plan()[:1]
            plan[0]['size'] = size
            self.assertEqual(validate_plan(plan)[0]['size'], size)
        for size in (3, 4, 254, 256, 257, 11.0, True, None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                validate_size(size)
        rows = generate_maze(255, 255, 42, pola='SE#.')
        self.assertEqual(len(rows), 255)
        self.assertEqual(len(rows[0]), 255)
        self.assertEqual(set(''.join(rows)), set('SE#.'))
        start = next((x,y) for y,row in enumerate(rows) for x,tile in enumerate(row) if tile=='S')
        pending, seen = deque([start]), {start}
        while pending:
            x,y = pending.popleft()
            for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0<=nx<255 and 0<=ny<255 and rows[ny][nx]!='#' and (nx,ny) not in seen:
                    seen.add((nx,ny))
                    pending.append((nx,ny))
        self.assertTrue(any(rows[y][x]=='E' for x,y in seen))

    def test_scaling_preserves_small_cells_and_bounds_large_maps(self):
        self.assertEqual(map_scale(5, 5, 720, 530), 48)
        for width,height in ((255,255),(255,5),(5,255),(31,41)):
            cell = map_scale(width, height, 720, 530)
            self.assertLessEqual(width*cell, 720+1e-9)
            self.assertLessEqual(height*cell, 530+1e-9)

    def test_large_gui_raster_cache_and_swat_coordinates(self):
        root = tk.Tk()
        root.withdraw()
        with patch('gui.load_plan', side_effect=default_plan), patch('gui.GameApp.load_validation_history'):
            app = GameApp(root)
        try:
            app.camera_enabled.set(False)
            app.current_rows = blank_map(255)
            app.reset()
            self.assertLess(app.cell, 3)
            self.assertEqual(len(app.canvas.find_withtag('terrain')), 3)
            terrain = app.canvas.find_withtag('terrain')
            app.draw()
            self.assertEqual(app.canvas.find_withtag('terrain'), terrain)
            app.training_bots = [Episode(app.current_rows)]
            app.live_metadata = {'episode': 1}
            app.training_thread = SimpleNamespace(is_alive=lambda: True)
            app.watch_training.set()
            app.swatter.set(True)
            app.click_bot(SimpleNamespace(x=1.5*app.cell, y=1.5*app.cell))
            self.assertEqual(app.interventions.get_nowait(), (1,0))
            root.update_idletasks()
            self.assertLess(root.winfo_reqwidth(), root.winfo_screenwidth())
            self.assertLess(root.winfo_reqheight()+40, root.winfo_screenheight())
        finally:
            app.training_thread = None
            app.close()

    def test_editor_accepts_typed_size_and_rejects_even_or_oversized(self):
        root = tk.Tk()
        root.withdraw()
        editor = MapDesigner(root, default_plan()[:1], COLORS, lambda plan: None)
        try:
            editor.size.set('255')
            with patch('map_designer.messagebox.askyesno', return_value=True):
                self.assertTrue(editor.resize_map())
            self.assertEqual(editor.plan[0]['size'], 255)
            editor.mode.set('custom')
            editor.draw()
            self.assertLess(len(editor.canvas.find_all()), 10)
            for value in ('256', '254', 'wat'):
                editor.size.set(value)
                with patch('map_designer.messagebox.showerror'):
                    self.assertFalse(editor.resize_map())
                self.assertEqual(editor.size.get(), '255')
        finally:
            editor.destroy()
            root.destroy()

    def test_large_camera_matches_unfiltered_ray_geometry(self):
        ep = Episode(generate_maze(255, 255, 42, pola='SE#.'))
        camera = TensorCamera(ep.world, 'cpu', width=20, height=15)
        actual = camera.render([camera_pose(ep.player)])
        # Duże ściany pozostają na mapie; test porównuje obcięcie poza zasięgiem
        # z pełnym raycastingiem przy zwiększonym promieniu filtra.
        with patch('tensor_camera.cos', return_value=.000001):
            expected = camera.render([camera_pose(ep.player)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
