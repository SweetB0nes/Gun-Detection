import numpy as np
import matplotlib.pyplot as plt 

def draw_box(cords, label):
    # print(coords)
    # print(label)
    # return None
    x = np.array((coords[0], coords[2]))
    y = np.array((coords[1], coords[3]))
    color = 'g'
    if label == 0:
        color = 'r'

    plt.plot(x.mean(), y.mean(), '*' + color)

    plt.plot([x[0], x[0]], [y[0], y[1]], color)
    plt.plot([x[1], x[1]], [y[0], y[1]], color)
    plt.plot([x[0], x[1]], [y[0], y[0]], color)
    plt.plot([x[0], x[1]], [y[1], y[1]], color)