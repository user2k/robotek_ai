"""Plan poziomów i trwały postęp treningu."""
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from maps import generate_maze, validate_fields, DEFAULT_FIELDS, validate_size
from world import World
from player import validate_movement

from config.config import PLAN_PATH
SIZES = tuple(range(5, 256, 2))


def blank_map(size):
    rows = [['.' for _ in range(size)] for _ in range(size)]
    rows[1][1], rows[size - 2][size - 2] = 'S', 'E'
    return [''.join(row) for row in rows]


def default_plan():
    return [dict(name=f'Poziom {i + 1}', start_direction='right', move_distance=.1, turn_degrees=10., mode='random', size=SIZES[min(i // 3, 3)], required=20, max_time=200.0, pola=DEFAULT_FIELDS,
                 collision_penalty=True, turn_penalty=False,
                 rows=blank_map(SIZES[min(i // 3, 3)])) for i in range(10)]


def validate_plan(plan):
    if not isinstance(plan, list) or not plan:
        raise ValueError('Plan musi zawierać przynajmniej jeden poziom.')
    plan = deepcopy(plan)
    for i, level in enumerate(plan, 1):
        level.setdefault('name', f'Poziom {i}')


        try:
            level['pola'] = validate_fields(level.get('pola', DEFAULT_FIELDS))
        except ValueError as error:
            raise ValueError(f'Poziom {i}: {error}') from error

        if not isinstance(level['name'], str) or not 1 <= len(level['name'].strip()) <= 120:
            raise ValueError(f'Poziom {i}: nazwa/opis musi mieć 1–120 znaków.')
        level['name'] = level['name'].strip()
        level.setdefault('start_direction', 'right')
        if level['start_direction'] not in ('random', 'right', 'left', 'up', 'down'):
            raise ValueError(f'Poziom {i}: nieprawidłowy kierunek startowy.')
        level.setdefault('move_distance', .1)
        level.setdefault('turn_degrees', 10.)
        for key, default in (('collision_penalty', True), ('turn_penalty', False)):
            level.setdefault(key, default)
            if type(level[key]) is not bool:
                raise ValueError(f'Poziom {i}: {key} musi mieć wartość true albo false.')
        validate_movement(level['move_distance'], level['turn_degrees'])
        level.setdefault('max_time', 200.0)
        limit = level['max_time']
        if type(limit) not in (int, float) or not isfinite(limit) or not 2 <= limit <= 1000000:
            raise ValueError(f'Poziom {i}: limit czasu musi wynosić od 2 do 1 000 000 sekund.')
        if level.get('mode') not in ('random', 'custom'):
            raise ValueError(f'Poziom {i}: nieprawidłowy typ mapy.')
        try:
            validate_size(level.get('size'))
        except ValueError as error:
            raise ValueError(f'Poziom {i}: {error}') from error
        if type(level.get('required')) is not int or not 1 <= level['required'] <= 1000000:
            raise ValueError(f'Poziom {i}: podaj liczbę zaliczeń 1–1 000 000.')
        if level['mode'] == 'custom':
            rows = level.get('rows', [])
            size = level['size']
            if len(rows) != size or any(len(row) != size for row in rows):
                raise ValueError(f'Poziom {i}: mapa musi mieć rozmiar {size}×{size}.')
            world = World.from_text(rows)
            start = world.start_position()
            if sum(row.count('E') for row in rows) != 1:
                raise ValueError(f'Poziom {i}: wymagane dokładnie jedno pole END.')
            seen, pending = {start}, [start]
            while pending:
                x, y = pending.pop()
                for p in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                    if world.contains(*p) and world.tile_at(*p).walkable and p not in seen:
                        seen.add(p)
                        pending.append(p)
            if not any(rows[y][x] == 'E' for x, y in seen):
                raise ValueError(f'Poziom {i}: ściany blokują wszystkie drogi do END.')
    return deepcopy(plan)


def load_plan(path=PLAN_PATH):
    return validate_plan(json.loads(Path(path).read_text(encoding='utf-8'))) if Path(path).exists() else default_plan()


def save_plan(plan, path=PLAN_PATH):
    plan = validate_plan(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def prepare_curriculum(previous, plan, start_level=None):
    plan = validate_plan(plan)
    if start_level is not None and (type(start_level) is not int or not 1 <= start_level <= len(plan)):
        raise ValueError(f'Poziom startowy musi wynosić od 1 do {len(plan)}.')
    previous_plan = None
    if previous and previous.get('plan'):
        try:
            previous_plan = validate_plan(previous['plan'])
        except ValueError:
            pass  # Stary plan bez wymaganej ściany: zacznij nowy plan.
    if (start_level is None and previous_plan and [{k: v for k, v in level.items() if k != 'name'} for level in previous_plan] ==
            [{k: v for k, v in level.items() if k != 'name'} for level in plan]):
        previous['plan'] = plan
        trim_streak(previous['wins'])
        return previous
    index = 0 if start_level is None else start_level - 1
    return dict(plan=plan, level=index, wins=[], completed=False, unit='groups',
                width=plan[index]['size'], height=plan[index]['size'])


def level_options(level):
    return dict(max_time=level.get('max_time', 200.), move_distance=level.get('move_distance', .1),
                turn_degrees=level.get('turn_degrees', 10.), start_direction=level.get('start_direction', 'right'),
                collision_penalty=level.get('collision_penalty', True), turn_penalty=level.get('turn_penalty', False))


def level_rows(state, seed):
    level = state['plan'][state['level']]
    return (generate_maze(level['size'], level['size'], seed, max_time=level.get('max_time', 200.0),
                          pola=level.get('pola', DEFAULT_FIELDS))
            if level['mode'] == 'random' else list(level['rows']))


def trim_streak(wins):
    """Migruj stare okno wyników do serii po ostatniej porażce."""
    for index in range(len(wins) - 1, -1, -1):
        if not wins[index]:
            del wins[:index + 1]
            break


def record_streak(wins, won, required):
    trim_streak(wins)
    if won:
        wins.append(True)
        del wins[:-required]
    else:
        wins.clear()
    return len(wins) / required


def advance_level(state, won):
    if state['completed']:
        return 1.0
    level = state['plan'][state['level']]
    wins = state['wins']
    average = record_streak(wins, won, level['required'])
    if len(wins) == level['required'] and all(wins):
        if state['level'] == len(state['plan']) - 1:
            state['completed'] = True
        else:
            state['level'] += 1
            wins.clear()
            size = state['plan'][state['level']]['size']
            state['width'] = state['height'] = size
    return average
