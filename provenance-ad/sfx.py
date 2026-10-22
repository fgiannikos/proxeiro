# Sound effects only, for the v9 cut (music is added separately):
# door opens, paws pad in, door closes, and the dog's hops.
import numpy as np, wave
from scipy.signal import fftconvolve, butter, sosfilt

SR, DUR = 44100, 16.0
N = int(SR * DUR)
rng = np.random.default_rng(31)
mix = np.zeros((N, 2))

def put(sig, t0, pan=0.0):
    i = int(round(SR * t0))
    if i >= N or i < 0: return
    sig = sig[: N - i]
    mix[i:i + len(sig), 0] += sig * np.sqrt(0.5 * (1 - pan))
    mix[i:i + len(sig), 1] += sig * np.sqrt(0.5 * (1 + pan))

def filt(x, kind, f):
    return sosfilt(butter(2, f, kind, fs=SR, output='sos'), x)

def latch(t0, vel=0.5, pan=-0.6):
    L = int(SR * 0.05); t = np.arange(L) / SR
    c = filt(rng.normal(0, 1, L), 'bandpass', [2000, 7000]) * np.exp(-t * 160) + 0.4 * np.sin(2 * np.pi * 2900 * t) * np.exp(-t * 90)
    put(c * vel, t0, pan); put(c * vel * 0.6, t0 + 0.045, pan)

def swing(t0, length=0.55, vel=0.35, pan=-0.55):
    L = int(SR * length); t = np.arange(L) / SR
    put(filt(rng.normal(0, 1, L), 'lowpass', 900) * np.sin(np.pi * t / t[-1]) ** 2 * vel, t0, pan)

def creak(t0, length=0.5, vel=0.07, pan=-0.6):
    L = int(SR * length); t = np.arange(L) / SR
    phase = np.cumsum(70 - 25 * t / t[-1]) / SR
    pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    grain = filt(rng.normal(0, 1, 400), 'bandpass', [600, 1500]) * np.exp(-np.arange(400) / 60)
    put(fftconvolve(pulses, grain)[:L] * np.sin(np.pi * t / t[-1]) * vel, t0, pan)

def thud(t0, vel=0.5, pan=-0.6):
    L = int(SR * 0.18); t = np.arange(L) / SR
    put((np.sin(2 * np.pi * (95 - 120 * t) * t) * np.exp(-t * 28) + filt(rng.normal(0, 1, L), 'lowpass', 500) * np.exp(-t * 40) * 0.5) * vel, t0, pan)

def paw(t0, vel, pan):
    L = int(SR * 0.09); t = np.arange(L) / SR
    thump = np.sin(2 * np.pi * (140 - 400 * t) * t) * np.exp(-t * 70)
    pad = filt(rng.normal(0, 1, L), 'bandpass', [700, 2600]) * np.exp(-t * 90) * 0.5
    L2 = int(SR * 0.012); nail = filt(rng.normal(0, 1, L2), 'bandpass', [4500, 9000]) * np.exp(-np.arange(L2) / SR * 500) * 0.35
    put((thump + pad) * vel, t0, pan); put(nail * vel, t0 + 0.004, pan)

def spring(t0, vel=0.16, pan=0.0):
    """Soft, playful take-off: a short rising tone with a little wobble."""
    L = int(SR * 0.22); t = np.arange(L) / SR
    f = 330 + 520 * (t / t[-1]) ** 0.7 + 18 * np.sin(2 * np.pi * 28 * t)
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) * np.exp(-t * 6)
    put(sig * vel, t0, pan)

# door opens
latch(2.35); swing(2.41); creak(2.45)
# paws from the door towards the middle (one print every 0.27 s)
for i in range(7):
    s = 2.95 + i * 0.27 + 0.05
    pan = -0.65 + 0.55 * i / 6
    paw(s, 0.5, pan); paw(s + 0.07 + rng.uniform(-0.01, 0.01), 0.28, pan + 0.05)
# door closes behind the dog
swing(4.72, 0.45, 0.3); creak(4.75, 0.35, 0.05); thud(5.12, 0.45); latch(5.14, 0.4)
# hops: (take-off, duration) -> spring on take-off, two paws on landing
for t0, d in [(7.05, 0.42), (7.55, 0.38), (12.35, 0.46)]:
    spring(t0 - 0.02)
    paw(t0 + d, 0.55, 0.0); paw(t0 + d + 0.05, 0.35, 0.08)

# small room ambience, master
ir_t = np.arange(int(SR * 0.6)) / SR
ir = rng.normal(0, 1, (len(ir_t), 2)) * np.exp(-ir_t * 9)[:, None]; ir[0] = 0
wet = np.stack([fftconvolve(mix[:, c], ir[:, c])[:N] for c in range(2)], 1)
out = mix + 0.08 * wet * (np.max(np.abs(mix)) / np.max(np.abs(wet)))
out = out / np.max(np.abs(out)) * 0.85
with wave.open('sfx-v9.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype(np.int16).tobytes())
print('ok')
