from skeleton_parts import *
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from IPython.display import HTML

# Function plot_3D_skeletons takes as inputs a np.array xyz_vec of shape [NUM_JOINTS, 3] (equivalent to one video frame)
# and returns an animated 3D plot of the skeleton in that frame

def plot_3D_skeleton(xyz_vec):
  # Obtain the x_vec, y_vec and z_vec
  x_vec = xyz_vec[:,0]
  y_vec = xyz_vec[:,1]
  z_vec = xyz_vec[:,2]

  # Plot the Skeleton
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  def init():
    for part in skeleton_parts:
      ax.plot([x_vec[i] for i in part], [y_vec[i] for i in part], [z_vec[i] for i in part] )
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    return fig,

  def animate(i):
      ax.view_init(elev=0, azim=3.6*i)
      return fig,

  # Animate
  ani = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=100, interval=100, blit=True)
  # Display the final result
  return HTML(ani.to_html5_video())