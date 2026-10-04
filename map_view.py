"""Skala podglądu i raster terenu; współrzędne fizyki pozostają w metrach."""
def map_scale(width, height, max_width, max_height, preferred=48):
    return min(preferred, max_width / width, max_height / height)


def terrain_ppm(rows, cell, colors):
    width, height = max(1, int(len(rows[0])*cell)), max(1, int(len(rows)*cell))
    palette = {symbol: bytes.fromhex(color.lstrip('#')) for symbol, color in colors.items()}
    columns = [min(len(rows[0])-1, int(x/cell)) for x in range(width)]
    scanlines = {}
    pixels = []
    for y in range(height):
        index = min(len(rows)-1, int(y/cell))
        if index not in scanlines:
            scanlines[index] = b''.join(palette[rows[index][x]] for x in columns)
        pixels.append(scanlines[index])
    return f'P6\n{width} {height}\n255\n'.encode('ascii') + b''.join(pixels)
