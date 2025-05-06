import matplotlib.pyplot as plt
from cycler import cycler


mm=1/25.4

label_ftsz = 6
label_pad = 1
tick_ftsz = 5
tick_pad = 1

lw = 0.5

def set_color_cycle():
    return


def sys_color(system):
    systems = ['TaskSwitch', 'SP-50', 'SP-75', 'UM+MPS', 'Sirius', 'Infer-Only']
    for i, s in enumerate(systems):
        if s == system:
            return f'C{i}'
    raise ValueError(f"Unknown system: {system}")


