"""Deterministyczna kamera RGB bez GUI i bez pamięci odkrytej mapy."""
from math import floor, tan, radians
import numpy as np
from world import Terrain

CAMERA_ANGLE = 120.0
CAMERA_DEPTH = 6.0

PALETTE = {
    Terrain.GROUND: (151, 126, 91), Terrain.SWAMP: (83, 108, 41),
    Terrain.FIRE: (244, 91, 26), Terrain.WATER: (36, 128, 205),
    Terrain.PAVEMENT: (153, 166, 180), Terrain.START: (40, 181, 131),
    Terrain.END: (185, 100, 240), Terrain.WALL: (103, 123, 145),
}


def render_camera(player, width=160, height=120, angle=CAMERA_ANGLE, max_depth=CAMERA_DEPTH):
    """Zwraca uint8 [H,W,3]. Kamera w środku robota, ściany zasłaniają teren.

    Stała kamera perspektywiczna; brak HUD, współrzędnych i zapamiętanych pól.
    Pozycja i kierunek wynikają z ciągłych współrzędnych robota.
    """
    if width < 2 or height < 2 or not 0 < angle < 180 or max_depth <= 0:
        raise ValueError("Nieprawidłowe parametry kamery")
    world = player.world
    px, py = player.x, player.y
    dx, dy = player.direction_vector
    plane = tan(radians(angle) / 2)
    focal = width / (2 * plane)
    horizon = height / 2
    camera_x = (2 * (np.arange(width) + 0.5) / width - 1) * plane
    rays_x, rays_y = dx - dy * camera_x, dy + dx * camera_x
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = (19, 25, 36)
    # Podłoga liczona jednocześnie dla wszystkich pikseli.
    rows = np.arange(int(horizon), height)
    distances = 0.5 * focal / (rows + 0.5 - horizon)
    xs = px + distances[:, None] * rays_x
    ys = py + distances[:, None] * rays_y
    tx, ty = np.floor(xs).astype(int), np.floor(ys).astype(int)
    valid = ((distances[:, None] <= max_depth) & (tx >= 0) & (tx < world.width)
             & (ty >= 0) & (ty < world.height))
    palette = np.array([[PALETTE[world.tile_at(x, y).terrain] for x in range(world.width)]
                        for y in range(world.height)], dtype=np.uint8)
    colors = palette[np.clip(ty, 0, world.height - 1), np.clip(tx, 0, world.width - 1)].copy()
    fx, fy = xs - tx, ys - ty
    ends = np.array([[world.tile_at(x, y).terrain is Terrain.END for x in range(world.width)]
                     for y in range(world.height)])
    end_pixels = ends[np.clip(ty, 0, world.height - 1), np.clip(tx, 0, world.width - 1)]
    checker = (np.floor(fx * 4).astype(int) + np.floor(fy * 4).astype(int)) % 2 == 1
    colors[end_pixels & checker] = (236, 204, 255)
    grid = np.where(np.minimum(fx, fy) < 0.045, 0.65, 1.0)
    shade = np.maximum(0.25, 1 - distances / (max_depth * 1.3))
    floor_pixels = (colors * (shade[:, None] * grid)[..., None]).astype(np.uint8)
    image[int(horizon):][valid] = floor_pixels[valid]
    for col in range(width):
        rx, ry = rays_x[col], rays_y[col]
        mx, my = floor(px), floor(py)
        delta_x = abs(1 / rx) if rx else float('inf')
        delta_y = abs(1 / ry) if ry else float('inf')
        sx, sy = (1 if rx >= 0 else -1), (1 if ry >= 0 else -1)
        side_x = (mx + 1 - px if rx >= 0 else px - mx) * delta_x
        side_y = (my + 1 - py if ry >= 0 else py - my) * delta_y
        while True:
            if side_x < side_y:
                distance = side_x
                side_x += delta_x
                mx += sx
                side = 0
            else:
                distance = side_y
                side_y += delta_y
                my += sy
                side = 1
            if distance > max_depth:
                break
            if not world.contains(mx, my) or not world.tile_at(mx, my).walkable:
                wall_height = focal / max(distance, 0.001)
                top = max(0, int(horizon - wall_height / 2))
                bottom = min(height, int(horizon + wall_height / 2) + 1)
                u = (py + distance * ry) % 1 if side == 0 else (px + distance * rx) % 1
                shade = max(0.25, 1 - distance / (max_depth * 1.3)) * (0.78 if side else 1)
                rows = np.arange(top, bottom)
                v = (rows + 0.5 - horizon) / wall_height + 0.5
                brick_rows = np.floor(v * 6).astype(int)
                mortar = ((v * 6 % 1 < 0.08)
                          | ((u * 3 + (brick_rows % 2) * 0.5) % 1 < 0.06))
                colors = np.where(mortar[:, None], (49, 60, 72), PALETTE[Terrain.WALL])
                image[top:bottom, col] = (colors * shade).astype(np.uint8)
                break
    return image


def camera_ppm(frame):
    """Bezstratne dane PPM dla Tkinter; sieć może używać bezpośrednio tablicy RGB."""
    height, width, _ = frame.shape
    return f"P6\n{width} {height}\n255\n".encode('ascii') + frame.tobytes()
