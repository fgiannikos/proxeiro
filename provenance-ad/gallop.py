# "Living painting": warp the painting with a thin-plate spline driven by a few
# control points moved along a gallop cycle; background points stay pinned.
import numpy as np, sys
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates, zoom

src = np.asarray(Image.open('painting.jpg').convert('RGB').resize((768, 768), Image.LANCZOS)).astype(np.float32)
S = 768 / 1024
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TAU = 2 * np.pi

def controls(ph):
    s = lambda k=0.0: np.sin(TAU * ph + k)
    c = lambda k=0.0: np.cos(TAU * ph + k)
    bob = -14 * s()
    pts = [
        # (x, y, dx, dy) in 1024 px space
        (515, 300, 0, bob + 6 * s(0.6)),            # head
        (515, 470, 0, bob + 4 * s(0.4)),            # tongue / jaw
        (440, 240, 0, bob + 6 * s(0.6)),            # eye L
        (600, 235, 0, bob + 6 * s(0.6)),            # eye R
        (200, 370, -8 * c(1.2), bob - 26 * s(1.2)), # ear tip L
        (330, 255, 0, bob + 3 * s(0.6)),            # ear root L
        (815, 370, 8 * c(1.2), bob - 26 * s(1.2)),  # ear tip R
        (700, 235, 0, bob + 3 * s(0.6)),            # ear root R
        (500, 600, 0, bob),                         # chest
        (590, 940, 14 * c(), bob - 24 * max(0, s())),     # front paw (lifts)
        (370, 760, -12 * c(np.pi), bob - 26 * max(0, s(np.pi))),  # other foreleg
        (470, 880, 10 * c(np.pi / 2), bob + 14 * s(np.pi / 2)),   # hind
        (215, 495, 6 * c(2.0), bob + 18 * s(2.0)),  # tail
    ]
    pins = [(x, y, 0, 0) for x, y in [
        (0, 0), (512, 0), (1023, 0), (0, 512), (1023, 512), (0, 1023), (512, 1023), (1023, 1023),
        (120, 120), (900, 110), (110, 620), (930, 640), (160, 940), (880, 940), (300, 1010), (780, 1010),
        (512, 60), (90, 330), (950, 330), (760, 700), (250, 700),
        (420, 1023), (590, 1023), (700, 1023), (180, 1023), (880, 1023)]]
    return np.array(pts + pins, dtype=np.float64)

G = 96
gy, gx = np.mgrid[0:G, 0:G] * (1023 / (G - 1))
grid = np.stack([gx.ravel(), gy.ravel()], 1)
for i in range(N):
    P = controls(i / N)
    rbf = RBFInterpolator(P[:, :2], P[:, 2:], kernel='thin_plate_spline', smoothing=1.0)
    d = rbf(grid).reshape(G, G, 2)
    dx = zoom(d[..., 0], 768 / G, order=1) * S
    dy = zoom(d[..., 1], 768 / G, order=1) * S
    yy, xx = np.mgrid[0:768, 0:768].astype(np.float32)
    sx, sy = xx - dx, yy - dy          # backward map (small displacements)
    out = np.stack([map_coordinates(src[..., ch], [sy, sx], order=1, mode='nearest') for ch in range(3)], -1)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(f'memory-frames/f{i:02d}.jpg', quality=90)
print('frames', N)
