from collections import Counter
import sklearn.metrics as metrics
import numpy as np

def coverage(weak_labs, k=1):
    n = 0
    for wl in weak_labs:
        x = [y for y in wl if y != -1]
        if len(x) >= k:
            n += 1
    return float(n) / len(weak_labs)


def pre_f1(weak_labs, labs):
    yhat, y = [], []
    for wl, l in zip(weak_labs, labs):
        x = [y for y in wl if y != -1]
        if len(x) == 0:
            continue
        pred = Counter(x).most_common(1)[0][0]
        yhat.append(pred)
        y.append(l)

    return metrics.precision_score(y, yhat, average='macro'), metrics.f1_score(y, yhat, average='macro')


