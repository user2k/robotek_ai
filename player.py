"""Kołowy robot: metry, ruch 10 cm i obrót 10°, bez dyskretnego FOV."""
from math import cos, sin, radians, floor, hypot, isfinite
from world import Terrain

ROBOT_RADIUS = 0.10
MOVE_DISTANCE = 0.10
TURN_DEGREES = 10
EPS = 1e-10


def validate_movement(move_distance, turn_degrees):
    for value, low, high, label in ((move_distance, .01, 1., 'Długość kroku (m)'),
                                    (turn_degrees, 1., 180., 'Kąt obrotu (°)')):
        if type(value) not in (int, float) or not isfinite(value) or not low <= value <= high:
            raise ValueError(f'{label}: podaj wartość od {low:g} do {high:g}.')


def segment_rectangle_distance_sq(ax, ay, bx, by, left, top):
    """Dokładna odległość od odcinka ruchu środka do kwadratu 1×1 m."""
    lo, hi = 0., 1.
    for origin, delta, lower in ((ax, bx - ax, left), (ay, by - ay, top)):
        if abs(delta) < EPS:
            if not lower <= origin <= lower + 1:
                lo, hi = 1., 0.
                break
        else:
            first, last = sorted(((lower - origin) / delta, (lower + 1 - origin) / delta))
            lo, hi = max(lo, first), min(hi, last)
    if lo <= hi:
        return 0.
    def point_rect(x, y):
        return (x - min(max(x, left), left + 1)) ** 2 + (y - min(max(y, top), top + 1)) ** 2
    best = min(point_rect(ax, ay), point_rect(bx, by))
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    for x, y in ((left, top), (left + 1, top), (left, top + 1), (left + 1, top + 1)):
        t = min(1., max(0., ((x - ax) * dx + (y - ay) * dy) / length_sq)) if length_sq else 0.
        best = min(best, (x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2)
    return best


class Player:
    def __init__(self, world, x, y, heading=0, base_speed=1.0, move_distance=MOVE_DISTANCE, turn_degrees=TURN_DEGREES):
        """x,y to pozycja środka w metrach; 0°=prawo, 90°=dół."""
        if not all(isfinite(v) for v in (x, y, heading, base_speed)) or base_speed <= 0:
            raise ValueError('Pozycja i prędkość muszą być skończone; prędkość dodatnia')
        validate_movement(move_distance, turn_degrees)
        self.move_distance = float(move_distance)
        self.turn_degrees = float(turn_degrees)
        self.world = world
        self.x, self.y = float(x), float(y)
        self.heading = heading % 360
        self.radius = ROBOT_RADIUS
        self.base_speed = base_speed
        if not self.can_travel(x, y):
            raise ValueError('Obrys robota musi mieścić się na dostępnej powierzchni')
        self.alive, self.won = True, False
        self.elapsed_time = self.damage = 0.
        self.death_reason = None
        self.water_distance = self.fire_distance = 0.
        self.burning_distance = None

    @property
    def cell(self):
        return floor(self.x), floor(self.y)

    @property
    def direction_vector(self):
        angle = radians(self.heading)
        return cos(angle), sin(angle)

    @property
    def speed(self):
        return self.base_speed * self.world.tile_at(*self.cell).speed_multiplier

    def can_travel(self, x, y):
        """Kolizja całej przesuwanej tarczy; sam kontakt styczny jest dozwolony."""
        r = self.radius
        if min(self.x, x) < r - EPS or min(self.y, y) < r - EPS:
            return False
        if max(self.x, x) > self.world.width - r + EPS or max(self.y, y) > self.world.height - r + EPS:
            return False
        for ty in range(max(0, floor(min(self.y, y) - r)), min(self.world.height, floor(max(self.y, y) + r) + 1)):
            for tx in range(max(0, floor(min(self.x, x) - r)), min(self.world.width, floor(max(self.x, x) + r) + 1)):
                if (not self.world.tile_at(tx, ty).walkable and
                    segment_rectangle_distance_sq(self.x, self.y, x, y, tx, ty) < r * r - EPS):
                    return False
        return True

    def movement_plan(self, backward=False):
        dx, dy = self.direction_vector
        sign = -1 if backward else 1
        x, y = round(self.x + sign * self.move_distance * dx, 12), round(self.y + sign * self.move_distance * dy, 12)
        if not self.can_travel(x, y):
            return x, y, True, [], self.move_distance / self.speed
        # Podział kroku na części w poszczególnych terenach.
        dx, dy = x - self.x, y - self.y
        cuts = {0., 1.}
        for origin, delta in ((self.x, dx), (self.y, dy)):
            if abs(delta) > EPS:
                for edge in range(floor(min(origin, origin + delta)) + 1, floor(max(origin, origin + delta)) + 1):
                    t = (edge - origin) / delta
                    if EPS < t < 1 - EPS:
                        cuts.add(t)
        cuts = sorted(cuts)
        parts = []
        duration = 0.
        for a, b in zip(cuts, cuts[1:]):
            mid = (a + b) / 2
            cell = floor(self.x + mid * dx), floor(self.y + mid * dy)
            tile = self.world.tile_at(*cell)
            distance = (b - a) * hypot(dx, dy)
            duration += distance / (self.base_speed * tile.speed_multiplier)
            parts.append((tile.terrain, distance))
        return x, y, False, parts, duration

    def turn_duration(self):
        return (0.5 * self.turn_degrees / 90) / self.speed

    def apply_terrain(self, terrain, distance):
        """Zagrożenia zależą od metrów drogi, nie liczby mikrokroków."""
        if not self.alive:
            return
        if terrain is Terrain.WATER:
            self.water_distance += distance
            self.fire_distance = 0.
            self.burning_distance = None
            self.damage += 10 * distance
            if self.water_distance >= 3 - EPS:
                self.alive, self.death_reason = False, 'Utonięcie'
        elif terrain is Terrain.FIRE:
            self.fire_distance += distance
            self.water_distance = 0.
            self.burning_distance = 0.
            self.damage += 25 * distance
            if self.fire_distance >= 2 - EPS:
                self.alive, self.death_reason = False, 'Spłonięcie w ogniu'
        else:
            self.water_distance = self.fire_distance = 0.
            if self.burning_distance is not None:
                self.burning_distance += distance
                self.damage += 10 * distance
                if self.burning_distance >= 3 - EPS:
                    self.alive, self.death_reason = False, 'Spłonięcie po opuszczeniu ognia'
        if terrain is Terrain.END and self.alive:
            self.won = True
