# Sound effects only, for the walking-cat episode (music is added separately):
# lamp switch, light paw taps and collar bell on the walk, a purr, the camera shutter.
import numpy as np, wave
from scipy.signal import fftconvolve, butter, sosfilt

SR, DUR = 44100, 16.0
N = int(SR * DUR)
rng = np.random.default_rng(47)
mix = np.zeros((N, 2))

def put(sig, t0, pan=0.0):
    i = int(round(SR * t0))
    if i >= N or i < 0: return
    sig = sig[: N - i]
    mix[i:i + len(sig), 0] += sig * np.sqrt(0.5 * (1 - pan))
    mix[i:i + len(sig), 1] += sig * np.sqrt(0.5 * (1 + pan))

def filt(x, kind, f):
    return sosfilt(butter(2, f, kind, fs=SR, output='sos'), x)

def click(t0, vel=0.45, pan=0.8):
    """Table-lamp switch: a short plastic tick and its release."""
    for dt, v in [(0, 1.0), (0.035, 0.5)]:
        L = int(SR * 0.03); t = np.arange(L) / SR
        c = filt(rng.normal(0, 1, L), 'bandpass', [1800, 6000]) * np.exp(-t * 220) + 0.3 * np.sin(2 * np.pi * 2300 * t) * np.exp(-t * 150)
        put(c * vel * v, t0 + dt, pan)

def swish(t0, length=0.3, vel=0.12, pan=0.6):
    L = int(SR * length); t = np.arange(L) / SR
    env = np.sin(np.pi * t / t[-1]) ** 3
    put(filt(rng.normal(0, 1, L), 'bandpass', [500, 2500]) * env * vel, t0, pan)

def bell(t0, vel=0.12, pan=0.5):
    """Tiny collar bell: inharmonic partials, a few quick strikes."""
    L = int(SR * 0.7); t = np.arange(L) / SR
    tone = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t * d)
               for f, a, d in [(3950, 1.0, 7), (5720, 0.6, 10), (8310, 0.35, 15), (10450, 0.2, 22)])
    for k in range(4):
        put(tone * vel * (0.9 ** k) * rng.uniform(0.5, 1.0), t0 + k * rng.uniform(0.03, 0.06), pan)

def land(t0, vel=0.35, pan=0.3):
    L = int(SR * 0.12); t = np.arange(L) / SR
    thump = np.sin(2 * np.pi * (120 - 300 * t) * t) * np.exp(-t * 55)
    pad = filt(rng.normal(0, 1, L), 'bandpass', [400, 1800]) * np.exp(-t * 70) * 0.5
    put((thump + pad) * vel, t0, pan); put((thump + pad) * vel * 0.6, t0 + 0.06, pan + 0.05)

def purr(t0, length, vel=0.09, pan=0.3):
    """Low rumble pulsing at ~26 Hz, louder on the out-breath."""
    L = int(SR * length); t = np.arange(L) / SR
    rumble = filt(rng.normal(0, 1, L), 'lowpass', 380)
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 26 * t) ** 2
    breath = 0.55 + 0.45 * np.sin(2 * np.pi * t / 1.15) ** 2
    fade = np.clip(t / 0.4, 0, 1) * np.clip((t[-1] - t) / 0.5, 0, 1)
    put(rumble * pulse * breath * fade * vel / np.max(np.abs(rumble)) * 3, t0, pan)

def tap(t0, vel, pan):
    L = int(SR * 0.06); t = np.arange(L) / SR
    put((np.sin(2 * np.pi * (170 - 500 * t) * t) * np.exp(-t * 90) + filt(rng.normal(0, 1, L), 'bandpass', [900, 3000]) * np.exp(-t * 120) * 0.4) * vel, t0, pan)

def shutter(t0, vel=0.4, pan=0.0):
    """Camera shutter: two mechanical clicks with a short whirr between."""
    for dt, v, band in [(0, 1.0, [2500, 9000]), (0.07, 0.7, [1500, 6000])]:
        L = int(SR * 0.025); t = np.arange(L) / SR
        put(filt(rng.normal(0, 1, L), 'bandpass', band) * np.exp(-t * 260) * vel * v, t0 + dt, pan)
    L = int(SR * 0.06); t = np.arange(L) / SR
    put(filt(rng.normal(0, 1, L), 'bandpass', [600, 1800]) * np.sin(np.pi * t / t[-1]) * vel * 0.25, t0 + 0.01, pan)

click(2.48)                                     # lamp on
# walk, as in the scene: x = 200 + 490 * (1 - cos(pi u)) / 2, u over 3.2-6.5 s; a pair lands every 36 px
W0, W1, X0, X1 = 3.2, 6.5, 200, 690
ts = np.arange(W0, W1 + 0.3, 0.002)
d = (X1 - X0) * (1 - np.cos(np.pi * np.clip((ts - W0) / (W1 - W0), 0, 1))) / 2
steps = ts[1:][np.diff(np.floor(d / 36)) > 0]
for k, t0 in enumerate(steps):
    pan = -0.6 + 0.9 * (k / max(1, len(steps) - 1))
    tap(t0, 0.16, pan); tap(t0 + 0.03, 0.09, pan + 0.05)
    if k % 2 == 0: bell(t0 + 0.01, 0.025, pan)
swish(6.8, 0.25, 0.06, 0.3)                     # head turns to the camera
purr(7.1, 0.9, 0.06)                            # posing
shutter(9.15)                                    # the photo -> the painting
bell(12.88, 0.06, 0.3); purr(12.9, 1.6, 0.05)   # slow blink under her portrait

# small room ambience, master
ir_t = np.arange(int(SR * 0.6)) / SR
ir = rng.normal(0, 1, (len(ir_t), 2)) * np.exp(-ir_t * 9)[:, None]; ir[0] = 0
wet = np.stack([fftconvolve(mix[:, c], ir[:, c])[:N] for c in range(2)], 1)
out = mix + 0.08 * wet * (np.max(np.abs(mix)) / np.max(np.abs(wet)))
out = out / np.max(np.abs(out)) * 0.85
with wave.open('sfx-cat2.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype(np.int16).tobytes())
print('ok')
