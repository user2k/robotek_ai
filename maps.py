"""Mapa demonstracyjna i losowe labirynty z bezpieczną drogą do celu."""

from collections import deque
from math import isfinite
import random

DEMO_MAP = [
    "################",
    "#S...BB...PP...#",
    "#..#.BB...PP...#",
    "#..#....#......#",
    "#..#WW..#..FF..#",
    "#.......#...W..#",
    "#..PPP.....BB..#",
    "#..........BB.E#",
    "################",
]


def generate_maze(width=15, height=9, seed=None, max_time=200.0):
    """Wycinaj korytarze DFS, po czym chroń jedną trasę START → END.

    Bezpieczna trasa zawiera wyłącznie ziemię i bruk. Jej długość jest
    ograniczona tak, aby nawet koszt obrotów zmieścił się w max_time.
    Seed odtwarza identyczną mapę. Generator nie ujawnia trasy agentowi.
    """
    if (not isinstance(width, int) or not isinstance(height, int)
            or width < 5 or height < 5 or width % 2 == 0 or height % 2 == 0):
        raise ValueError("Wymiary labiryntu muszą być nieparzyste i wynosić co najmniej 5")
    if not isfinite(max_time) or max_time < 2:
        raise ValueError("Limit czasu generatora musi być skończony i wynosić co najmniej 2")
    rng = random.Random(seed)
    grid = [["#"] * width for _ in range(height)]
    cells = [(x, y) for y in range(1, height - 1, 2) for x in range(1, width - 1, 2)]
    start = rng.choice(cells)
    grid[start[1]][start[0]] = "."
    stack = [start]
    while stack:
        x, y = stack[-1]
        neighbors = [(x + dx, y + dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                     if 0 < x + dx < width - 1 and 0 < y + dy < height - 1
                     and grid[y + dy][x + dx] == "#"]
        if not neighbors:
            stack.pop()
            continue
        nx, ny = rng.choice(neighbors)
        grid[(y + ny) // 2][(x + nx) // 2] = "."
        grid[ny][nx] = "."
        stack.append((nx, ny))
    # Kilka dodatkowych połączeń pozwala wybierać alternatywne drogi.
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] == "#" and (x + y) % 2 == 1 and rng.random() < 0.12:
                grid[y][x] = "."
    parents = {start: None}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pos = x + dx, y + dy
            if (0 < pos[0] < width - 1 and 0 < pos[1] < height - 1
                    and grid[pos[1]][pos[0]] != "#" and pos not in parents):
                parents[pos] = x, y
                distances[pos] = distances[x, y] + 1
                queue.append(pos)
    # 10 kroków po 10 cm ≤ 1 s, 18 obrotów po 10° ≤ 1 s: 2 s / pole.
    reachable = [pos for pos, distance in distances.items() if 0 < distance <= int(max_time / 2)]
    longest = max(distances[pos] for pos in reachable)
    end = rng.choice([pos for pos in reachable if distances[pos] >= max(1, longest * 0.75)])
    safe_path = set()
    pos = end
    while pos is not None:
        safe_path.add(pos)
        pos = parents[pos]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] == "#":
                continue
            if (x, y) in safe_path:
                grid[y][x] = rng.choices([".", "P"], weights=[8, 2])[0]
            else:
                grid[y][x] = rng.choices([".", "P", "B", "W", "F"], weights=[55, 15, 15, 8, 7])[0]
    grid[start[1]][start[0]] = "S"
    grid[end[1]][end[0]] = "E"
    return ["".join(row) for row in grid]
