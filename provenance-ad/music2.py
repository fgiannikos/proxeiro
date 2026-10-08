# Score v2 for the 16 s v6 cut: paw steps on a wooden floor, then a happy
# ukulele groove (C-G-Am-F, 120 bpm) whose downbeat lands on the painting reveal.
import numpy as np, wave
from scipy.signal import fftconvolve, butter, sosfilt

SR, DUR = 44100, 16.0
N = int(SR * DUR)
rng = np.random.default_rng(11)
mix = np.zeros((N, 2))
hz = lambda n: 440.0 * 2 ** ((n - 69) / 12)

def put(sig, t0, pan=0.0):
    i = int(SR * t0)
    if i >= N or i < 0: return
    sig = sig[: N - i]
    mix[i:i + len(sig), 0] += sig * np.sqrt(0.5 * (1 - pan))
    mix[i:i + len(sig), 1] += sig * np.sqrt(0.5 * (1 + pan))

def band(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], 'bandpass', fs=SR, output='sos'), x)

# ── paw steps on wood
def paw(t0, vel, pan):
    L = int(SR * 0.09); t = np.arange(L) / SR
    thump = np.sin(2 * np.pi * (140 - 400 * t) * t) * np.exp(-t * 70)
    pad = band(rng.normal(0, 1, L), 700, 2600) * np.exp(-t * 90) * 0.5
    L2 = int(SR * 0.012); nail = band(rng.normal(0, 1, L2), 4500, 9000) * np.exp(-np.arange(L2) / SR * 500) * 0.35
    put((thump + pad) * vel, t0, pan); put(nail * vel, t0 + 0.004, pan)

for i in range(7):                           # one print every 0.27 s from the door
    s = 2.6 + i * 0.27 + 0.05
    pan = -0.65 + 0.55 * i / 6
    paw(s, 0.55, pan)
    paw(s + 0.07 + rng.uniform(-0.01, 0.01), 0.3, pan + 0.05)   # the other paw

# ── instruments
def uke(n, t0, vel=0.25, length=1.3, pan=0.0):
    """Karplus-Strong plucked string, block-vectorised."""
    P = int(SR / hz(n)); L = int(SR * length)
    y = np.zeros(L + P + 1); y[:P] = rng.uniform(-1, 1, P)
    for a in range(P, L, P):
        b = min(a + P, L)
        y[a:b] = 0.4985 * (y[a - P:b - P] + y[a - P + 1:b - P + 1])
    put(band(y[:L], 120, 6000) * vel, t0, pan)

def bass(n, t0, vel=0.5, length=0.45):
    L = int(SR * length); t = np.arange(L) / SR
    sig = (np.sin(2 * np.pi * hz(n) * t) + 0.25 * np.sin(4 * np.pi * hz(n) * t)) * np.exp(-t * 4) * np.minimum(1, t / 0.005)
    put(sig * vel, t0, 0.0)

def bell(n, t0, vel=0.18, pan=0.0):
    L = int(SR * 1.4); t = np.arange(L) / SR
    sig = (np.sin(2 * np.pi * hz(n) * t) + 0.35 * np.sin(2 * np.pi * hz(n) * 2.76 * t) * np.exp(-t * 6)) * np.exp(-t * 3.5)
    put(sig * vel, t0, pan)

def shaker(t0, vel=0.05):
    L = int(SR * 0.05); t = np.arange(L) / SR
    put(band(rng.normal(0, 1, L), 5000, 12000) * np.sin(np.pi * t / t[-1]) * vel, t0, 0.35)

def clap(t0, vel=0.22):
    sig = np.zeros(int(SR * 0.16))
    for k, d in enumerate((0, 0.008, 0.017)):            # a clap is a few quick bursts
        L = int(SR * (0.14 - d)); t = np.arange(L) / SR
        sig[int(SR * d):int(SR * d) + L] += band(rng.normal(0, 1, L), 900, 4000) * np.exp(-t * (60 if k < 2 else 25))
    put(sig * vel, t0, -0.15)

def boing(t0, vel=0.2):
    L = int(SR * 0.16); t = np.arange(L) / SR
    f = 380 + 900 * t / t[-1]
    put(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18) * vel, t0, 0.25)

# ── groove
BEAT = 0.5; START = 4.5                      # bars at 4.5, 6.5, 8.5, 10.5, 12.5, 14.5
CH = {'C': ([60, 64, 67, 72], 48), 'G': ([59, 62, 67, 71], 43), 'Am': ([60, 64, 69, 72], 45), 'F': ([60, 65, 69, 72], 41)}
BARS = ['C', 'G', 'Am', 'F', 'C', 'C']
STRUM = [(0, 1, 0.30), (1, -1, 0.16), (1.5, 1, 0.22), (2.5, -1, 0.16), (3, 1, 0.24), (3.5, -1, 0.16)]  # (beat, dir, vel)
MELODY = [  # (bar, beat, note)
    (0, 0.0, 76), (0, 1.0, 79), (0, 2.5, 84), (1, 0.0, 83), (1, 1.5, 79), (1, 3.0, 74),
    (2, 0.0, 76), (2, 1.0, 81), (2, 2.0, 84), (3, 0.0, 81), (3, 1.5, 77), (3, 2.5, 79), (3, 3.0, 81), (3, 3.5, 83),
    (4, 0.0, 84), (4, 1.0, 88), (4, 2.0, 91), (5, 0.0, 84),
]
for b, name in enumerate(BARS):
    t_bar = START + b * 4 * BEAT
    voicing, root = CH[name]
    memory = 8.5 <= t_bar < 12.5              # lighter while the dog remembers
    last = b == len(BARS) - 1
    for beat, d, v in (STRUM[:1] if last else STRUM):
        notes = voicing if d > 0 else voicing[::-1]
        for k, n in enumerate(notes):
            uke(n, t_bar + beat * BEAT + k * 0.011 + rng.normal(0, 0.003), v * (0.8 if memory else 1.0), 1.8 if last else 1.2, 0.2)
    for beat in ((0,) if last else (0, 2)):
        bass(root, t_bar + beat * BEAT, 0.55 if not memory else 0.4, 1.6 if last else 0.45)
        if not last: bass(root + 7, t_bar + (beat + 1) * BEAT, 0.35 if not memory else 0.25)
    if not last:
        for e in range(8):
            shaker(t_bar + e * BEAT / 2, 0.06 if e % 2 else 0.035)
        if not memory:
            clap(t_bar + 1 * BEAT); clap(t_bar + 3 * BEAT)
for b, beat, n in MELODY:
    bell(n, START + (b * 4 + beat) * BEAT, 0.16, -0.25)

for t0 in (7.05, 7.55): boing(t0)            # the hops
boing(12.35, 0.18)
for n in (84, 88, 91, 96):                   # sparkle on the painting reveal
    bell(n, 12.5 + 0.03 * (n - 84) / 4, 0.12, 0.2)
clap(12.5, 0.3)

# ── room, fade, master
ir_t = np.arange(int(SR * 1.2)) / SR
ir = rng.normal(0, 1, (len(ir_t), 2)) * np.exp(-ir_t * 4.5)[:, None]; ir[0] = 0
wet = np.stack([fftconvolve(mix[:, c], ir[:, c])[:N] for c in range(2)], 1)
out = mix + 0.15 * wet * (np.max(np.abs(mix)) / np.max(np.abs(wet)))
t = np.arange(N) / SR
out *= np.clip((DUR - t) / 1.4, 0, 1)[:, None]
out = np.tanh(out / np.max(np.abs(out)) * 1.3) * 0.9
with wave.open('music2-v6.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype(np.int16).tobytes())
print('ok')
