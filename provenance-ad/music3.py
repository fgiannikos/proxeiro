# Score v3 for the v8 cut: a door opens, paws pad in, then a premium but
# joyful cue (warm piano, strings, pizzicato, celesta; D major, 90 bpm) whose
# fourth bar lands exactly on the painting reveal at 12.5 s.
import numpy as np, wave
from scipy.signal import fftconvolve, butter, sosfilt

SR, DUR = 44100, 16.0
N = int(SR * DUR)
rng = np.random.default_rng(23)
mix = np.zeros((N, 2))
hz = lambda n: 440.0 * 2 ** ((n - 69) / 12)

def put(sig, t0, pan=0.0, width=0.0):
    i = int(round(SR * t0))
    if i >= N or i < 0: return
    sig = sig[: N - i]
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    mix[i:i + len(sig), 0] += sig * l
    other = np.concatenate([np.zeros(300), sig])[: len(sig)] if width else sig
    mix[i:i + len(sig), 1] += (sig * (1 - width) + other * width) * r

def filt(x, kind, f):
    return sosfilt(butter(2, f, kind, fs=SR, output='sos'), x)

# ── door: latch, soft swing, gentle wood creak
def door(t0):
    L = int(SR * 0.05); t = np.arange(L) / SR
    click = filt(rng.normal(0, 1, L), 'bandpass', [2000, 7000]) * np.exp(-t * 160) + 0.4 * np.sin(2 * np.pi * 2900 * t) * np.exp(-t * 90)
    put(click * 0.5, t0, -0.6); put(click * 0.3, t0 + 0.045, -0.6)
    L = int(SR * 0.6); t = np.arange(L) / SR
    swing = filt(rng.normal(0, 1, L), 'lowpass', 900) * np.sin(np.pi * t / t[-1]) ** 2 * 0.35
    put(swing, t0 + 0.06, -0.55, 0.3)
    # stick-slip creak: a train of tiny pulses whose rate glides 70 -> 45 Hz
    rate = 70 - 25 * t / t[-1]
    phase = np.cumsum(rate) / SR
    pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    creak = fftconvolve(pulses, filt(rng.normal(0, 1, 400), 'bandpass', [600, 1500]) * np.exp(-np.arange(400) / 60))[:L]
    put(creak * np.sin(np.pi * t / t[-1]) * 0.07, t0 + 0.1, -0.6)

def paw(t0, vel, pan):
    L = int(SR * 0.09); t = np.arange(L) / SR
    thump = np.sin(2 * np.pi * (140 - 400 * t) * t) * np.exp(-t * 70)
    pad = filt(rng.normal(0, 1, L), 'bandpass', [700, 2600]) * np.exp(-t * 90) * 0.5
    L2 = int(SR * 0.012); nail = filt(rng.normal(0, 1, L2), 'bandpass', [4500, 9000]) * np.exp(-np.arange(L2) / SR * 500) * 0.35
    put((thump + pad) * vel, t0, pan); put(nail * vel, t0 + 0.004, pan)

door(2.35)
for i in range(7):
    s = 2.95 + i * 0.27 + 0.05
    pan = -0.65 + 0.55 * i / 6
    paw(s, 0.5, pan); paw(s + 0.07 + rng.uniform(-0.01, 0.01), 0.28, pan + 0.05)

# ── instruments
def piano(n, t0, vel=0.2, length=2.6, pan=0.0):
    f, L = hz(n), int(SR * length); t = np.arange(L) / SR
    tone = sum(a * np.sin(2 * np.pi * f * k * np.sqrt(1 + 0.0003 * k * k) * t + 0.3 * k) * np.exp(-t * (0.9 + 0.7 * k))
               for k, a in [(1, 1.0), (2, 0.5), (3, 0.28), (4, 0.15), (5, 0.08), (6, 0.04)])
    ham = filt(rng.normal(0, 1, int(SR * 0.01)), 'bandpass', [1500, 5000]) * 0.03
    tone[: len(ham)] += ham
    put(tone * np.minimum(1, t / 0.003) * vel, t0, pan, 0.25)

def strings(notes, t0, t1, vel=0.05, swell=False):
    L = int(SR * (t1 - t0)); t = np.arange(L) / SR
    env = np.minimum(1, t / 0.9) * np.minimum(1, (t1 - t0 - t) / 0.8)
    if swell: env *= 0.5 + 0.5 * (t / t[-1]) ** 2
    sig = np.zeros(L)
    for n in notes:
        for det in (-0.08, 0.0, 0.09):
            f = hz(n + det) * (1 + 0.003 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6)))   # vibrato
            ph = 2 * np.pi * np.cumsum(f) / SR
            sig += sum(np.sin(k * ph) / k for k in range(1, 10))
    sig = filt(sig, 'lowpass', 2600) * env * vel / len(notes)
    put(sig, t0, 0.0, 0.6)

def pizz(n, t0, vel=0.18, pan=0.2):
    P = int(SR / hz(n)); L = int(SR * 0.6)
    y = np.zeros(L + P + 1); y[:P] = filt(rng.uniform(-1, 1, P + 64), 'lowpass', 2500)[64:]
    for a in range(P, L, P):
        b = min(a + P, L); y[a:b] = 0.493 * (y[a - P:b - P] + y[a - P + 1:b - P + 1])
    put(y[:L] * np.exp(-np.arange(L) / SR * 5) * vel, t0, pan)

def celesta(n, t0, vel=0.12, pan=-0.2):
    L = int(SR * 1.6); t = np.arange(L) / SR
    sig = (np.sin(2 * np.pi * hz(n) * t) + 0.2 * np.sin(2 * np.pi * hz(n) * 3 * t) * np.exp(-t * 8)
           + 0.12 * np.sin(2 * np.pi * hz(n) * 4.1 * t) * np.exp(-t * 14)) * np.exp(-t * 2.8)
    put(sig * np.minimum(1, t / 0.002) * vel, t0, pan, 0.3)

def shimmer(t0, length=1.2, vel=0.05):
    L = int(SR * length); t = np.arange(L) / SR
    put(filt(rng.normal(0, 1, L), 'highpass', 7000) * (t / t[-1]) ** 2 * np.exp(-np.maximum(0, t - length * 0.8) * 12) * vel, t0 - length * 0.85, 0.0, 0.8)

# ── harmony: half-bar chords, 90 bpm; bars at 4.5, 7.167, 9.833, 12.5, 15.167
BEAT = 60 / 90; HALF = 2 * BEAT; START = 4.5
CHORDS = [  # (voicing for the arpeggio, bass)
    ([62, 66, 69, 73, 76], 38),   # Dmaj9
    ([61, 64, 69, 73, 76], 37),   # A/C#
    ([62, 66, 69, 71, 74], 35),   # Bm7      (hops land here)
    ([62, 67, 71, 74, 78], 31),   # Gmaj7    (the memory begins)
    ([62, 66, 67, 71, 74], 40),   # Em9
    ([62, 64, 67, 69, 73], 33),   # A7sus4 -> builds to the reveal
    ([62, 66, 69, 74, 76], 38),   # Dadd9    (12.5 s: the painting)
    ([62, 66, 69, 74, 76], 38),
]
ARP = [0, 2, 4, 3, 1, 3, 2, 4]    # broken-chord pattern in eighths
for c, (voicing, root) in enumerate(CHORDS):
    t0 = START + c * HALF
    if t0 >= DUR: break
    memory = 7.9 <= t0 < 12.4
    reveal = c >= 6
    piano(root, t0, 0.24 if not memory else 0.18, 3.0, -0.25)
    piano(root + 12, t0, 0.12, 2.4, -0.2)
    for e in range(4):                                          # eighths over the half bar
        n = voicing[ARP[(c * 4 + e) % len(ARP)]]
        vel = (0.13 if e == 0 else 0.095) * (0.8 if memory else 1.0)
        piano(n, t0 + e * BEAT / 2 + rng.normal(0, 0.004), vel, 2.2, 0.15 * np.sin(c + e))
    if not memory and not reveal:
        pizz(root + 24, t0 + BEAT, 0.16); pizz(root + 31, t0 + BEAT * 1.5, 0.12)
    strings([voicing[0] - 12, voicing[2] - 12, voicing[3]], t0, t0 + HALF + 0.6,
            0.05 if not memory else 0.07, swell=(c == 5))

# celesta melody: joyful leaps, then a rising run into the reveal
MEL = [(4.5, 78), (5.17, 81), (5.83, 85), (6.5, 83), (7.05, 86), (7.55, 90),          # (hops on 7.05 / 7.55)
       (8.7, 83), (9.5, 81), (10.2, 79), (10.9, 81), (11.5, 83), (11.85, 85), (12.1, 86), (12.3, 88)]
for t0, n in MEL:
    celesta(n, t0, 0.11 if t0 < 12 else 0.13)
for k, n in enumerate((86, 90, 93, 98)):                        # sparkle on the painting
    celesta(n, 12.5 + 0.045 * k, 0.1, 0.25)
shimmer(12.5, 1.1, 0.06)
strings([50, 57, 62, 66], 12.5, 16.0, 0.08)

# ── hall reverb, fade, master
ir_t = np.arange(int(SR * 2.4)) / SR
ir = filt(rng.normal(0, 1, len(ir_t)), 'lowpass', 6000)[:, None] * np.exp(-ir_t * 2.4)[:, None] * rng.choice([1, -1], (1, 2))
ir[:int(SR * 0.02)] = 0
wet = np.stack([fftconvolve(mix[:, c], np.roll(ir[:, 0], 37 * c))[:N] for c in range(2)], 1)
out = mix + 0.22 * wet * (np.max(np.abs(mix)) / np.max(np.abs(wet)))
t = np.arange(N) / SR
out *= np.clip((DUR - t) / 1.2, 0, 1)[:, None]
out = np.tanh(out / np.max(np.abs(out)) * 1.15) * 0.9
with wave.open('music3-v8.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype(np.int16).tobytes())
print('ok')
