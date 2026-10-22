# Mix the client's track under the v9 effects: the music sits low under the
# door / paws / hops, then rises into the lead from 8 s; its drop (17.1 s in
# the track) is aligned to the rise. Fades out with the end card.
import numpy as np, wave, subprocess, re

SR, DUR = 44100, 16.0
OFFSET = 17.1 - 8.0                     # track time = video time + OFFSET
LOW_DB, RISE = -16.0, (7.7, 8.2)

def read(path):
    w = wave.open(path); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32768
    assert w.getframerate() == SR
    return x

N = int(SR * DUR)
music = read('music-user.wav')[int(SR * OFFSET):int(SR * OFFSET) + N]
sfx = read('sfx-v9.wav')[:N]
t = np.arange(N) / SR
low = 10 ** (LOW_DB / 20)
u = np.clip((t - RISE[0]) / (RISE[1] - RISE[0]), 0, 1); u = u * u * (3 - 2 * u)
env = low + (1 - low) * u
env *= np.clip(t / 0.6, 0, 1)                    # gentle fade-in
env *= np.clip((DUR - t) / 1.4, 0, 1)            # fade-out with the end card
out = (music * env[:, None] * 10 ** (-6 / 20) + sfx * 10 ** (1.5 / 20)) * 0.85   # headroom before the final limiter
with wave.open('mix-v9-user.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
print('peak', np.max(np.abs(out)).round(3))
