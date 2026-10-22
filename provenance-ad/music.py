# Synthesised score for the 16 s v6 cut: soft piano, a dreamy pad and a
# music-box line, timed to the picture (hops, memory, painting reveal).
import numpy as np, wave
from scipy.signal import fftconvolve

SR, DUR = 44100, 16.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
mix = np.zeros((N, 2))

def hz(n):  # MIDI note -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)

def piano(n, t0, vel=0.5, length=2.4, pan=0.0):
    f, L = hz(n), int(SR * length)
    t = np.arange(L) / SR
    tone = sum(a * np.sin(2 * np.pi * f * k * np.sqrt(1 + 0.0004 * k * k) * t) * np.exp(-t * (1.6 + 0.9 * k))
               for k, a in [(1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (5, 0.06)])
    tone *= np.minimum(1, t / 0.004) * vel
    put(tone, t0, pan)

def musicbox(n, t0, vel=0.25, pan=0.0):
    f, L = hz(n), int(SR * 1.6)
    t = np.arange(L) / SR
    tone = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 4.02 * t) * np.exp(-t * 9)) * np.exp(-t * 3.2)
    put(tone * np.minimum(1, t / 0.002) * vel, t0, pan)

def pad(notes, t0, t1, vel=0.06):
    L = int(SR * (t1 - t0)); t = np.arange(L) / SR
    env = np.minimum(1, t / 1.2) * np.minimum(1, (t1 - t0 - t) / 1.2)
    sig = np.zeros(L)
    for n in notes:
        for det in (-0.12, 0.0, 0.12):
            sig += np.sin(2 * np.pi * hz(n + det) * t + rng.uniform(0, 6.28)) * (1 + 0.15 * np.sin(2 * np.pi * 0.3 * t))
    put(sig * env * vel / len(notes), t0, 0.0, width=0.5)

def put(sig, t0, pan, width=0.0):
    i = int(SR * t0)
    if i >= N: return
    sig = sig[: N - i]
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    mix[i:i + len(sig), 0] += sig * l
    mix[i:i + len(sig), 1] += sig * r * (1 - width) + np.roll(sig, 220) * r * width

BEAT = 60 / 96                     # 96 bpm, eighth = BEAT / 2
CHORDS = [                         # (start, end, root, voicing)
    (0.0, 2.5, 48, [60, 64, 67, 71]),    # Cmaj7
    (2.5, 5.0, 45, [57, 60, 64, 67]),    # Am7
    (5.0, 7.5, 41, [57, 60, 64, 65]),    # Fmaj7
    (7.5, 10.0, 40, [59, 62, 64, 67]),   # Em7  (the dog starts remembering)
    (10.0, 11.25, 41, [57, 60, 65, 69]), # F
    (11.25, 12.5, 43, [59, 62, 67, 72]), # Gsus → G (the cloud swells)
    (12.5, 16.0, 36, [55, 62, 64, 67]),  # Cadd9 (the painting)
]
for c0, c1, root, v in CHORDS:
    piano(root, c0, 0.30, 3.2, -0.2)                       # soft bass
    patt = [v[0], v[2], v[3], v[1], v[2], v[3], v[1], v[2]]
    k, tt = 0, c0
    while tt < c1 - 0.05 and tt < 15.0:
        vel = (0.20 if k % 4 == 0 else 0.13) * (0.75 if c0 >= 7.5 and c1 <= 10.0 else 1.0)
        piano(patt[k % len(patt)], tt + rng.normal(0, 0.006), vel, 2.0, 0.25 * np.sin(k))
        k += 1; tt += BEAT / 2

# Hops and ball: little plucks
for t0, n in [(7.05, 79), (7.55, 84), (12.35, 84)]:
    musicbox(n, t0, 0.22, 0.3)
# Memory: dreamy pad + music-box melody
pad([52, 59, 62, 67], 7.9, 12.6, 0.07)
for t0, n in [(8.7, 76), (9.25, 79), (9.8, 83), (10.4, 81), (11.0, 79), (11.4, 81), (11.8, 83), (12.15, 86)]:
    musicbox(n, t0, 0.20, -0.2)
# Reveal: chime chord on the painting, then a warm sustained Cadd9
for n in (72, 76, 79, 84):
    musicbox(n, 12.5 + 0.02 * (n - 72) / 4, 0.18, 0.1)
pad([48, 55, 62, 64], 12.4, 16.0, 0.06)
piano(48, 12.5, 0.32, 3.5); piano(55, 12.5, 0.2, 3.5)

# Room reverb (exponentially decaying noise), gentle master fade and limiting
ir_t = np.arange(int(SR * 1.8)) / SR
ir = rng.normal(0, 1, (len(ir_t), 2)) * np.exp(-ir_t * 3.2)[:, None]; ir[0] = 0
wet = np.stack([fftconvolve(mix[:, c], ir[:, c])[:N] for c in range(2)], 1)
out = mix + 0.06 * wet / np.max(np.abs(wet)) * np.max(np.abs(mix)) * 4
t = np.arange(N) / SR
out *= np.minimum(1, t / 0.3)[:, None] * np.clip((DUR - t) / 1.6, 0, 1)[:, None]
out = np.tanh(out / np.max(np.abs(out)) * 1.2) * 0.89
pcm = (out * 32767).astype(np.int16)
with wave.open('music-v6.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
rms = np.sqrt(np.mean(out ** 2)); print(f'peak {np.max(np.abs(out)):.3f}  rms {20*np.log10(rms):.1f} dBFS')
