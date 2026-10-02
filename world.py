"""Świat z kwadratowych pól. Współrzędne: x w prawo, y w dół."""

from dataclasses import dataclass
from enum import Enum


class Terrain(Enum):
    WALL = "#"
    GROUND = "."
    SWAMP = "B"
    FIRE = "F"
    WATER = "W"
    PAVEMENT = "P"
    START = "S"
    END = "E"


@dataclass(frozen=True)
class Tile:
    terrain: Terrain
    speed_multiplier: float
    walkable: bool


TILES = {
    Terrain.WALL: Tile(Terrain.WALL, 0.0, False),
    Terrain.GROUND: Tile(Terrain.GROUND, 1.0, True),
    Terrain.SWAMP: Tile(Terrain.SWAMP, 0.5, True),
    Terrain.FIRE: Tile(Terrain.FIRE, 1.0, True),
    Terrain.WATER: Tile(Terrain.WATER, 0.8, True),
    Terrain.PAVEMENT: Tile(Terrain.PAVEMENT, 1.2, True),
    Terrain.START: Tile(Terrain.START, 1.0, True),
    Terrain.END: Tile(Terrain.END, 1.0, True),
}


class World:
    def __init__(self, width: int, height: int, fill: Terrain = Terrain.GROUND):
        if width <= 0 or height <= 0:
            raise ValueError("Wymiary świata muszą być dodatnie")
        self.width = width
        self.height = height
        self._tiles = [[TILES[fill] for _ in range(width)] for _ in range(height)]

    def contains(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def start_position(self) -> tuple[int, int]:
        starts = [(x, y) for y in range(self.height) for x in range(self.width)
                  if self.tile_at(x, y).terrain is Terrain.START]
        if len(starts) != 1:
            raise ValueError("Mapa musi mieć dokładnie jedno pole START")
        return starts[0]

    def tile_at(self, x: int, y: int) -> Tile:
        if not self.contains(x, y):
            raise IndexError("Pole poza mapą")
        return self._tiles[y][x]

    def set_tile(self, x: int, y: int, terrain: Terrain) -> None:
        self.tile_at(x, y)
        self._tiles[y][x] = TILES[terrain]

    @classmethod
    def from_text(cls, rows: list[str]) -> "World":
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Mapa musi być niepustym prostokątem")
        world = cls(len(rows[0]), len(rows))
        for y, row in enumerate(rows):
            for x, symbol in enumerate(row):
                world.set_tile(x, y, Terrain(symbol))
        return world
