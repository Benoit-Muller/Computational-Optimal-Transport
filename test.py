import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider
"""
N = 100

X = np.linspace(0, 20, N)
Y = np.linspace(0, 20, N)
x, y = np.meshgrid(X, Y)
z = np.sin(x) + np.sin(y)


fig = plt.figure()

ax1 = fig.add_axes([0, 0, 1, 0.8], projection = '3d')
ax2 = fig.add_axes([0.1, 0.85, 0.8, 0.1])

s = Slider(ax = ax2, label = 'value', valmin = 0, valmax = 5, valinit = 2)

def update(val):
    value = s.val
    ax1.cla()
    ax1.plot_surface(x, y, z + value, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    ax1.set_zlim(-2, 7)

s.on_changed(update)
update(0)

plt.show()
"""
plt.figure()
x = np.linspace(0,1,100)
plt.plot(x,x**2)
plt.show()