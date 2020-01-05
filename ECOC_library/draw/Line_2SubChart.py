# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/4/2 8:35
# file: Line_2SubChart.py
# description:

import matplotlib.pyplot as plt
import numpy as np

# create some data to use for the plot
dt = 0.001
t = np.arange(0.0, 10.0, dt)
r = np.exp(-t[:1000] / 0.05)  # impulse response
x = np.random.randn(len(t))
s = np.convolve(x, r)[:len(x)] * dt  # colored noise

# the main axes is subplot(111) by default
plt.plot(t, s)
plt.axis([0, 1, 1.1 * np.amin(s), 2 * np.amax(s)])#x_start.x_end,y_start,y_end
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Gaussian colored noise')


# this is an inset axes over the main axes
data = np.random.randint(1, 5, [3, 4])
index = np.arange(data.shape[1])
color_index = ['r', 'g', 'b']
a = plt.axes([.65, .6, .2, .2], axisbg='y')#only y axis
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)#hist
# plt.bar(1, [13,3,2,55,2,51,4,23], width=2, color='r')
for i in range(data.shape[0]):
    plt.bar(index + i * .25 + .1, data[i], width=.25, color=color_index[i], alpha=.5)
# plt.axis([40, 160, 0, 0.03])
plt.title('Probability')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.grid(True)
plt.xticks()
plt.yticks([])

# this is another inset axes over the main axes
a = plt.axes([0.2, 0.6, .2, .2], axisbg='y')
plt.plot(t[:len(r)], r)
plt.title('Impulse response')
plt.xlim(0, 0.2)
plt.xticks([])
plt.yticks([])

plt.show()