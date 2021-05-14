import json
import numpy as np
import matplotlib.pyplot as plt


timings = {}
bs = [16, 32, 64, 128]
for i in [16, 32, 64, 128]:
    with open(f'timings_{i}.json') as f:
        timings[i] = json.load(f)

cats = timings[bs[0]].keys()
catego = {}
for c in cats:
    catego[c] = []
    for i in range(len(bs)):
        catego[c].append(timings[bs[i]][c]) 


catego_keys = list(catego.keys())

for i, c in enumerate(catego_keys):
    rest = catego_keys[i+1:]
    rest = np.array([catego[r] for r in rest]).sum(axis=0)
    plt.bar(np.arange(len(bs)), catego[c], width=1, label=c, bottom=rest)

plt.xticks(np.arange(len(bs)), bs)
plt.ylabel("prop time")
plt.xlabel("batch size")
plt.legend(loc="lower right")

plt.savefig('plottsss.png')

