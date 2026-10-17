import numpy as np
import math

def ideal_lowpass(cutoff,numtaps):
    center=(numtaps-1)//2
    h=np.zeros(numtaps)

    for n in range(numtaps):
        d=n-center

        if d==0:
            h[n]=cutoff
        else:
            h[n]=math.sin(math.pi*cutoff*d)/(math.pi*d)
    return h

def hamming_window(numtaps):
    window=np.zeros(numtaps)

    for n in range(numtaps):
        window[n]=0.54-0.46*math.cos(2*math.pi*n/(numtaps-1))
    return window

def filtered_sample(x,h,up,q):
    total=0.0

    for k in range(len(x)):
        tap_index=q-k*up

        if 0<=tap_index<len(h):
            total+=x[k]*h[tap_index]
    return total

def resample_reference(x,h,up,down):
    center=(len(h)-1)//2
    output_length=math.ceil(len(x)*up/down)

    out_put=[]

    for m in range(output_length):
        q=center+down*m
        value=filtered_sample(x,h,up,q)
        out_put.append(value)

    return np.asarray(out_put)

def design_resample_filter(up,down,numtaps):
    cutoff = 1.0 / max(up, down)

    h_ideal = ideal_lowpass(cutoff, numtaps)
    window = hamming_window(numtaps)

    h_windowed = h_ideal * window
    h_base = h_windowed / np.sum(h_windowed)

    return h_base * up

def decimate(x, M):
    M = validate_factor(M, "M")
    up = 1
    down = M
    numtaps = 20 * M + 1

    h = design_resample_filter(up, down, numtaps)
    return resample_reference(x, h, up, down)


def interpolate(x, L):
    L = validate_factor(L, "L")
    up = L
    down = 1
    numtaps = 20 * L + 1

    h = design_resample_filter(up, down, numtaps)
    return resample_reference(x, h, up, down)

def validate_factor(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} 必须是正整数")

    return int(value)
