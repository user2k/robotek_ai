"""Deterministyczny raycasting PyTorch: cała paczka kamer na urządzeniu sieci."""
from math import tan, radians
import torch
from camera import PALETTE, CAMERA_ANGLE, CAMERA_DEPTH
from world import Terrain


def camera_pose(player):
    return (player.x, player.y, *player.direction_vector)


class TensorCamera:
    def __init__(self, world, device='cpu', width=80, height=60):
        self.device = torch.device(device)
        self.width, self.height = width, height
        self.map_width, self.map_height = world.width, world.height
        self.focal = width / (2 * tan(radians(CAMERA_ANGLE) / 2))
        self.horizon = height / 2
        dtype = torch.float64
        terrains = list(Terrain)
        grid = [[terrains.index(world.tile_at(x, y).terrain) for x in range(world.width)] for y in range(world.height)]
        self.grid = torch.tensor(grid, device=self.device)
        self.palette = torch.tensor([PALETTE[t] for t in terrains], device=self.device, dtype=dtype)
        self.end_id = terrains.index(Terrain.END)
        walls = [(x, y, x + 1, y + 1) for y in range(world.height) for x in range(world.width)
                 if not world.tile_at(x, y).walkable]
        walls += [(-1, -1, 0, world.height + 1), (world.width, -1, world.width + 1, world.height + 1),
                  (0, -1, world.width, 0), (0, world.height, world.width, world.height + 1)]
        self.walls = torch.tensor(walls, device=self.device, dtype=dtype)
        self.columns = (2 * (torch.arange(width, device=self.device, dtype=dtype) + .5) / width - 1) * tan(radians(CAMERA_ANGLE) / 2)
        self.rows = torch.arange(height, device=self.device, dtype=dtype)
        self.floor_distances = .5 * self.focal / (self.rows[int(self.horizon):] + .5 - self.horizon)

    @torch.no_grad()
    def render(self, poses):
        poses = torch.as_tensor(poses, device=self.device, dtype=torch.float64)
        # Bounded temporary ray/rectangle tensors even for full-episode reconstruction.
        return torch.cat([self._render(chunk) for chunk in poses.split(64)], dim=0)

    def _render(self, poses):
        b = poses.shape[0]
        px, py, dx, dy = [poses[:, i:i+1] for i in range(4)]
        rx, ry = dx - dy * self.columns, dy + dx * self.columns
        image = torch.tensor([19, 25, 36], device=self.device, dtype=torch.uint8).expand(b, self.height, self.width, 3).clone()
        xs = px[:, None] + self.floor_distances[None, :, None] * rx[:, None]
        ys = py[:, None] + self.floor_distances[None, :, None] * ry[:, None]
        tx, ty = xs.floor().long(), ys.floor().long()
        valid = ((self.floor_distances[None, :, None] <= CAMERA_DEPTH) & (tx >= 0) & (tx < self.map_width)
                 & (ty >= 0) & (ty < self.map_height))
        terrain = self.grid[ty.clamp(0, self.map_height-1), tx.clamp(0, self.map_width-1)]
        colors = self.palette[terrain]
        fx, fy = xs - tx, ys - ty
        checker = ((fx * 4).floor().long() + (fy * 4).floor().long()) % 2 == 1
        colors = torch.where(((terrain == self.end_id) & checker)[..., None],
                             torch.tensor([236, 204, 255], device=self.device), colors)
        grid = torch.where(torch.minimum(fx, fy) < .045, torch.full_like(fx, .65), torch.ones_like(fx))
        shade = (1 - self.floor_distances / (CAMERA_DEPTH * 1.3)).clamp(min=.25)
        floor_pixels = (colors * (shade[None, :, None] * grid)[..., None]).to(torch.uint8)
        image[:, int(self.horizon):] = torch.where(valid[..., None], floor_pixels, image[:, int(self.horizon):])
        # Przecięcie promieni ze wszystkimi prostokątami ścian; najbliższy zasłania resztę.
        near, far = [], []
        for origin, ray, low, high in ((px, rx, self.walls[:, 0], self.walls[:, 2]),
                                       (py, ry, self.walls[:, 1], self.walls[:, 3])):
            parallel = ray.abs() < 1e-15
            safe_ray = torch.where(parallel, torch.ones_like(ray), ray)
            a = (low - origin[..., None]) / safe_ray[..., None]
            z = (high - origin[..., None]) / safe_ray[..., None]
            inside = (origin[..., None] >= low) & (origin[..., None] <= high)
            near.append(torch.where(parallel[..., None], torch.where(inside, -torch.inf, torch.inf), torch.minimum(a, z)))
            far.append(torch.where(parallel[..., None], torch.where(inside, torch.inf, -torch.inf), torch.maximum(a, z)))
        entry, leave = torch.maximum(*near), torch.minimum(*far)
        hits = (entry >= 0) & (entry <= leave) & (entry <= CAMERA_DEPTH)
        distance, wall_id = torch.where(hits, entry, torch.inf).min(dim=-1)
        hit = distance.isfinite()
        distance = torch.where(hit, distance, torch.ones_like(distance))
        side = (near[0] <= near[1]).gather(-1, wall_id[..., None]).squeeze(-1)
        wall_height = self.focal / distance.clamp(min=.001)
        top = (self.horizon - wall_height / 2).trunc().clamp(min=0)
        bottom = (self.horizon + wall_height / 2).trunc().add(1).clamp(max=self.height)
        wall_pixels = hit[:, None] & (self.rows[None, :, None] >= top[:, None]) & (self.rows[None, :, None] < bottom[:, None])
        u = torch.where(side, px + distance * rx, py + distance * ry).remainder(1)
        v = (self.rows[None, :, None] + .5 - self.horizon) / wall_height[:, None] + .5
        mortar = ((v * 6).remainder(1) < .08) | ((u[:, None] * 3 + (v * 6).floor().remainder(2) * .5).remainder(1) < .06)
        colors = torch.where(mortar[..., None], torch.tensor([49, 60, 72], device=self.device),
                             torch.tensor(PALETTE[Terrain.WALL], device=self.device))
        shade = (1 - distance / (CAMERA_DEPTH * 1.3)).clamp(min=.25) * torch.where(side, torch.full_like(distance, .78), torch.ones_like(distance))
        wall_rgb = (colors * shade[:, None, :, None]).to(torch.uint8)
        return torch.where(wall_pixels[..., None], wall_rgb, image)
