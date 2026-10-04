"""Wybrana paleta: generator, curriculum, checkpoint i walidacja."""
from collections import deque
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from curriculum import default_plan, level_rows, prepare_curriculum
from episode import Episode
from maps import generate_maze, validate_fields
from train import run_validation, train, latest_path
from visual_agent import VisualAgent


class GeneratorFieldsTests(unittest.TestCase):
    def test_every_palette_uses_only_selected_terrain_and_preserves_connectivity(self):
        for count in range(6):
            for special in combinations('.PBWF', count):
                pola = 'SE#'+''.join(special)
                allowed = set(pola) | ({'.'} if not special else set())
                for seed in range(8):
                    rows = generate_maze(11, 11, seed, pola=pola)
                    flat = ''.join(rows)
                    self.assertTrue(set(flat) <= allowed, pola)
                    self.assertEqual(flat.count('S'), 1)
                    self.assertEqual(flat.count('E'), 1)
                    self.assertEqual(set(rows[0]+rows[-1]), {'#'})
                    start = next((x,y) for y,row in enumerate(rows) for x,tile in enumerate(row) if tile=='S')
                    seen, pending = {start}, deque([start])
                    while pending:
                        x,y = pending.popleft()
                        for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                            if 0<=nx<11 and 0<=ny<11 and rows[ny][nx]!='#' and (nx,ny) not in seen:
                                seen.add((nx,ny))
                                pending.append((nx,ny))
                    self.assertTrue(any(rows[y][x]=='E' for x,y in seen))

    def test_repeatability_plain_ground_and_wall_validation(self):
        for pola in ('SE#', 'SE#.', 'SE#.P', 'SE#.PBWF'):
            self.assertEqual(generate_maze(seed=42, pola=pola), generate_maze(seed=42, pola=pola))
        self.assertEqual(generate_maze(seed=42, pola='SE#'), generate_maze(seed=42, pola='SE#.'))
        for invalid in ('SE.', 'S#.', 'E#.', '', 'SE#X', None, 123):
            with self.subTest(pola=invalid), self.assertRaises(ValueError):
                validate_fields(invalid)

    def test_selected_hazards_are_not_removed_from_goal_route(self):
        rows = generate_maze(11, 11, 42, pola='SE#F')
        self.assertEqual(set(''.join(rows)), set('SE#F'))
        # Każdy korytarz poza START/END jest ogniem, także dojście do END.
        self.assertGreater(''.join(rows).count('F'), 10)

    def test_level_selection_and_legacy_checkpoint_plan(self):
        plan = default_plan()[:1]
        plan[0]['pola'] = 'SE#.P'
        state = prepare_curriculum(None, plan)
        self.assertTrue(set(''.join(level_rows(state, 42))) <= set('SE#.P'))
        old = dict(state, plan=[dict(plan[0], pola='SE')], wins=[True])
        fresh = prepare_curriculum(old, plan)
        self.assertEqual(fresh['wins'], [])
        self.assertEqual(fresh['plan'][0]['pola'], 'SE#.P')

    def test_training_saves_palette_and_validation_uses_saved_palette(self):
        plan = default_plan()[:1]
        plan[0]['pola'] = 'SE#.P'
        group = dict(winner=0, trajectory={}, results=[Episode(['SE']).metrics()], steps_per_second=1.)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'agent.pt'
            with patch('train.run_group', return_value=group), \
                    patch.object(VisualAgent, 'learn_episode', return_value=0.):
                train(1, model_path=path, batch_size=1, device='cpu', plan=plan)
            agent = VisualAgent.load(latest_path(path), device='cpu')
            self.assertEqual(agent.training_config['pola'], 'SE#.P')
            summary = dict(maps=1, wins=0, score=0., success_rate=0., results=[])
            with patch('train.evaluate_suite', return_value=summary) as evaluate:
                result = run_validation(agent, path)
            self.assertEqual(evaluate.call_args.kwargs['pola'], 'SE#.P')
            self.assertEqual(result['pola'], 'SE#.P')
