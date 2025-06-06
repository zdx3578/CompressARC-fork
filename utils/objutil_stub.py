import numpy as np

# Minimal implementation of all_pureobjects_from_grid used when the external
# objutil package is not available. It extracts 4-connected components for each
# color (excluding background) and returns them as sets of (color,(r,c)).

def all_pureobjects_from_grid(param, pair_id, in_or_out, grid, shape, background_color=None):
    h, w = shape
    grid = np.array(grid)
    visited = np.zeros_like(grid, dtype=bool)
    objects = []

    for i in range(h):
        for j in range(w):
            color = grid[i, j]
            if background_color is not None and color == background_color:
                continue
            if visited[i, j]:
                continue
            # BFS to gather component
            stack = [(i, j)]
            visited[i, j] = True
            pixels = []
            while stack:
                x, y = stack.pop()
                pixels.append((color, (x, y)))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if not visited[nx, ny] and grid[nx, ny] == color:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
            if pixels:
                objects.append(frozenset(pixels))
    return objects
