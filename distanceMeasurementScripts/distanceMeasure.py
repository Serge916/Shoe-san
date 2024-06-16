import numpy as np

H = 100

f = [234.9, 234.9, 221.4, 222.75, 226.8, 245.7, 259.2]
h = [87, 58, 41, 33, 28, 26, 24]
pieces = [2, 3, 4, 5, 6, 7, 8]
d = [135 * piece for piece in pieces]

foc = []
for i in range(len(h)):
    foc.append(h[i] * d[i] / H)

f = np.array(f)
f_avg = np.mean(f[0:-2])

# f_avg = 235.09285714285716

dist = []
for i in range(len(h)):
    dist.append(f_avg * H / h[i])
# d = f * H / h

dist = np.array(dist)
d = np.array(d)

error = dist - d
print(error)
