import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(np.sin(np.arange(200)/(5*np.pi)))

zoom_factory(ax)
ph = panhandler(fig, button=2)

clicks = clicker(
   ax,
   ["positive", "negative"],
   markers=["o", "x"]
)

data = \
np.array([[ 5.83720666e+00, -5.73988654e-01],
         [ 2.46956149e+01, -1.41575199e-02],
         [ 5.20403030e+01,  5.70227612e-01],
         [ 8.55139728e+01,  7.56837990e-01],
         [ 1.30302686e+02,  3.73795635e-01],
         [ 1.69433877e+02, -2.40054293e-01],
         [ 2.01493167e+02, -5.05237462e-01]])

plt.plot(data[:,0],data[:,1],c='r')


plt.show()