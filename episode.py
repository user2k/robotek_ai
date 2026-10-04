"""Epizod robota o średnicy 20 cm; ruch ciągły na mapie pól 1×1 m."""
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
import random
from maps import DEMO_MAP
from player import Player, MOVE_DISTANCE, TURN_DEGREES, validate_movement
from world import World


START_HEADINGS = {'right': 0., 'left': 180., 'up': 270., 'down': 90.}

NEW_TILE_REWARD = 0.2
EXPLORATION_BUDGET = 5.0
CELL_GRACE_TIME = 3.0
COLLISION_BASE_COST = 0.5
COLLISION_REPEAT_COST = 0.25
COLLISION_MAX_COST = 2.0


def residence_cost(time):
    """Całka stawki: po 3 s wzrost 0.02/s aż do 0.1/s."""
    excess = max(0., time - CELL_GRACE_TIME)
    ramp = min(excess, 5.)
    return 0.01 * ramp * ramp + 0.1 * max(0., excess - 5.)


class Action(IntEnum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    BACKWARD = 3


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    duration: float
    moved: bool
    reason: str


class Episode:
    def __init__(self, rows=None, max_time=200.0, move_distance=MOVE_DISTANCE, turn_degrees=TURN_DEGREES, start_direction='right', start_seed=None):
        if not isfinite(max_time) or max_time <= 0:
            raise ValueError('Limit czasu musi być dodatni i skończony')
        validate_movement(move_distance, turn_degrees)
        if start_direction not in (*START_HEADINGS, 'random'):
            raise ValueError('Nieprawidłowy kierunek startowy robota')
        self.start_direction = start_direction
        self.start_random = random.Random(start_seed)
        self.move_distance, self.turn_degrees = move_distance, turn_degrees
        self.rows = list(DEMO_MAP if rows is None else rows)
        self.max_time = max_time
        self.reset()

    def reset(self):
        self.world = World.from_text(self.rows)
        sx, sy = self.world.start_position()
        heading = (self.start_random.choice(tuple(START_HEADINGS.values())) if self.start_direction == 'random'
                   else START_HEADINGS[self.start_direction])
        self.player = Player(self.world, sx + 0.5, sy + 0.5, heading=heading, move_distance=self.move_distance, turn_degrees=self.turn_degrees)
        if sum(row.count('E') for row in self.rows) != 1:
            raise ValueError('Epizod wymaga dokładnie jednego pola END')
        self.timed_out = False
        self.swatted = False
        self.steps = self.collisions = 0
        self.collision_streak = 0
        self.total_reward = 0.
        self.visited = {self.player.cell}
        self.reward_parts = {}
        self.reward_totals = {}
        self.cell_time = {}

    @property
    def done(self):
        return self.swatted or self.player.won or not self.player.alive or self.timed_out

    @property
    def score(self):
        p = self.player
        return 1000 * p.won - p.elapsed_time - 2 * p.damage - 250 * (not p.alive) - 100 * self.timed_out - 20 * self.swatted

    @property
    def outcome(self):
        if self.swatted:
            return 'Boża kara −20'
        if self.player.won:
            return 'Wygrana!'
        if not self.player.alive:
            return self.player.death_reason
        return 'Koniec czasu' if self.timed_out else 'W trakcie'

    def swat(self):
        """Terminalna kara operatora, bez dodatkowej kary za zwykłą śmierć."""
        if self.done:
            return False
        self.swatted = True
        self.reward_parts = {'swat': -20.}
        self.reward_totals['swat'] = -20.
        self.total_reward -= 20.
        return True

    def step(self, action):
        if self.done:
            raise RuntimeError('Epizod zakończony — wywołaj reset()')
        action = Action(action)
        p = self.player
        old_time, old_damage = p.elapsed_time, p.damage
        old_cell = p.cell
        moving = action in (Action.FORWARD, Action.BACKWARD)
        if moving:
            x, y, blocked, parts, duration = p.movement_plan(action is Action.BACKWARD)
        else:
            duration, blocked = p.turn_duration(), False
        moved, collision, new_tile = False, False, False
        self.steps += 1
        if old_time + duration > self.max_time + 1e-9:
            p.elapsed_time = self.max_time
            self.timed_out = True
        else:
            p.elapsed_time = min(self.max_time, old_time + duration)
            if moving:
                if blocked:
                    collision = True
                    self.collisions += 1
                else:
                    p.x, p.y = x, y
                    moved = True
                    for terrain, distance in parts:
                        p.apply_terrain(terrain, distance)
                    p.apply_terrain(self.world.tile_at(*p.cell).terrain, 0.)
                    new_tile = p.cell not in self.visited
                    self.visited.add(p.cell)
            else:
                p.heading = (p.heading + (p.turn_degrees if action is Action.RIGHT else -p.turn_degrees)) % 360
            if p.elapsed_time >= self.max_time - 1e-9 and not (p.won or not p.alive):
                self.timed_out = True
        if collision:
            self.collision_streak += 1
        elif moved:
            self.collision_streak = 0
        # Obrót pozwala szukać wyjścia, ale nie kasuje kolejnych uderzeń.
        collision_cost = (min(COLLISION_MAX_COST, COLLISION_BASE_COST +
                              COLLISION_REPEAT_COST * (self.collision_streak - 1))
                          if collision else 0.)
        elapsed = p.elapsed_time - old_time
        # Czas akcji przypisujemy polu, w którym akcja się rozpoczęła.
        previous_time = self.cell_time.get(old_cell, 0.)
        self.cell_time[old_cell] = previous_time + elapsed
        staying = residence_cost(previous_time + elapsed) - residence_cost(previous_time)
        exploration = min(NEW_TILE_REWARD, max(0., EXPLORATION_BUDGET - self.reward_totals.get('new_tile', 0.))) if new_tile else 0.
        self.reward_parts = {
            'time': -0.005 * (p.elapsed_time - old_time),
            'damage': -0.02 * (p.damage - old_damage),
            'goal': 10. if p.won else 0.,
            'death': -3. if not p.alive else 0.,
            'timeout': -1.5 if self.timed_out else 0.,
            'new_tile': exploration,
            #'staying': -staying,
            #'collision': -collision_cost,
        }
        for name, value in self.reward_parts.items():
            self.reward_totals[name] = self.reward_totals.get(name, 0.) + value
        reward = sum(self.reward_parts.values())
        self.total_reward += reward
        reason = self.outcome if self.done else ('Kolizja ze ścianą' if collision else '')
        return StepResult(reward, self.done, p.elapsed_time - old_time, moved, reason)

    def metrics(self):
        p = self.player
        return {'won': p.won, 'damage': p.damage, 'time': p.elapsed_time,
                'score': self.score, 'steps': self.steps, 'outcome': self.outcome,
                'reward': self.total_reward}
