import numpy
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

import skeletalModel
import pose2Dto3D
    

def backpropagationBasedFiltering_v2(
  lines0_values, # initial (logarithm of) bones lenghts
  rootsx0_values, # head position
  rootsy0_values, 
  rootsz0_values,
  anglesx0_values, # angles of limbs
  anglesy0_values,
  anglesz0_values,   
  tarx_values, # target   
  tary_values,
  w_values, # weights of estimated points (likelihood of estimation)   
  structure,
  dtype,
  learningRate=0.1,
  nCycles=1000,
  regulatorRates=[0.001, 0.1],
):

  T = rootsx0_values.shape[0]
  nBones, nPoints = skeletalModel.structureStats(structure)
  nLimbs = len(structure)

  # vector of (logarithm of) bones length
  #   shape: (nLines,)
  lines = tf.Variable(lines0_values, dtype=dtype) 
  # x cooordinates of head
  #   shape: (T, 1)
  rootsx = tf.Variable(rootsx0_values, dtype=dtype)
  # y cooordinates of head
  rootsy = tf.Variable(rootsy0_values, dtype=dtype) 
  # z cooordinates of head
  rootsz = tf.Variable(rootsz0_values, dtype=dtype)
  # x coordinate of angles 
  #   shape: (T, nLimbs)
  anglesx = tf.Variable(anglesx0_values, dtype=dtype)
  # y coordinate of angles 
  anglesy = tf.Variable(anglesy0_values, dtype=dtype)
  # z coordinate of angles 
  anglesz = tf.Variable(anglesz0_values, dtype=dtype)   

  # target
  #   shape: (T, nPoints)
  ##tarx = tf.compat.v1.placeholder(dtype=dtype)
  ##tary = tf.compat.v1.placeholder(dtype=dtype)
  # likelihood from previous pose estimator
  #   shape: (T, nPoints)
  ##w = tf.compat.v1.placeholder(dtype=dtype)
  
  # resultant coordinates. It's a list for now. It will be concatenated into a matrix later
  #   shape: (T, nPoints)
  x = [None for i in range(nPoints)]
  y = [None for i in range(nPoints)]
  z = [None for i in range(nPoints)]

  # head first
  x[0] = rootsx
  y[0] = rootsy
  z[0] = rootsz

  # now other limbs
  i = 0
  # for numerical stability of angles normalization
  epsilon = 1e-10
  for a, b, l, _ in structure:
    # limb length
    L = tf.exp(lines[l])
    # angle
    Ax = anglesx[0:T, i:(i + 1)]
    Ay = anglesy[0:T, i:(i + 1)]
    Az = anglesz[0:T, i:(i + 1)]
    # angle norm
    normA = tf.sqrt(tf.square(Ax) + tf.square(Ay) + tf.square(Az)) + epsilon
    # new joint position
    x[b] = x[a] + L * Ax / normA
    y[b] = y[a] + L * Ay / normA
    z[b] = z[a] + L * Az / normA
    i = i + 1

  # making a matrix from the list
  x = tf.Variable(tf.concat(x, axis=1))
  y = tf.Variable(tf.concat(y, axis=1))
  z = tf.Variable(tf.concat(z, axis=1))

  @tf.function()
  def loss():
    return tf.math.reduce_sum(input_tensor=w_values * tf.square(x - tarx_values) + w_values * tf.square(y - tary_values)) / (T * nPoints) \
         + tf.reduce_sum(input_tensor=tf.square(x[0:(T - 1), 0:nPoints] - x[1:T, 0:nPoints]) \
                                                    + tf.square(y[0:(T - 1), 0:nPoints] - y[1:T, 0:nPoints]) \
                                                    + tf.square(z[0:(T - 1), 0:nPoints] - z[1:T, 0:nPoints])) / ((T - 1) * nPoints) \
         + tf.reduce_sum(input_tensor=tf.exp(lines))  # ?¿?¿ right now that's a constant!

  # the backpropagation
  opt = tf.keras.optimizers.SGD(learning_rate=learningRate)
  for iCycle in range(nCycles):
    opt.minimize(loss, var_list=[x, y, z])
    if (iCycle+1) == nCycles:
      print("iCycle = %3d, loss = %e" % (iCycle, loss()), flush=True)

  # returning final coordinates
  return x.numpy(), y.numpy(), z.numpy()


# retrieves the bone length
# the length of a bone is calculated as the average across all frames and clips
def get_bone_length(kp_3d, structure, dtype="float32"):
  lines = numpy.zeros((len(structure), ), dtype=dtype)
  Ls = {}
  for i in range(len(kp_3d)):
    Yx = kp_3d[i][0:kp_3d[i].shape[0], 0:(kp_3d[i].shape[1]):3]
    Yy = kp_3d[i][0:kp_3d[i].shape[0], 1:(kp_3d[i].shape[1]):3]
    Yz = kp_3d[i][0:kp_3d[i].shape[0], 2:(kp_3d[i].shape[1]):3]
    T = kp_3d[i].shape[0]
    for iBone in range(len(structure)):
      a, b, _, _ = structure[iBone]
      line = iBone
      if not line in Ls:
        Ls[line] = []
      for t in range(T):
        ax = Yx[t, a]
        ay = Yy[t, a]
        az = Yz[t, a]
        bx = Yx[t, b]
        by = Yy[t, b]
        bz = Yz[t, b]
        L = pose2Dto3D.norm([ax - bx, ay - by, az - bz])
        Ls[line].append(L)
  for i in range(len(structure)):
    #lines[i] = pose2Dto3D.perc(Ls[i], 0.5)
    lines[i] = numpy.mean(Ls[i])
  return lines


if __name__ == "__main__":
  # debug - don't run it
  
  #
  #             (0)
  #              |
  #              |
  #              0
  #              |
  #              |
  #     (2)--1--(1)--1--(3)
  #
  structure = (
    (0, 1, 0),
    (1, 2, 1),
    (1, 3, 1),
  )

  T = 3
  nBones, nPoints = skeletalModel.structureStats(structure)
  nLimbs = len(structure)
  
  dtype = "float32"

  lines0_values = numpy.zeros((nBones, ), dtype=dtype) 
  rootsx0_values = numpy.ones((T, 1), dtype=dtype)
  rootsy0_values = numpy.ones((T, 1), dtype=dtype) 
  rootsz0_values = numpy.ones((T, 1), dtype=dtype)
  anglesx0_values = numpy.ones((T, nLimbs), dtype=dtype)
  anglesy0_values = numpy.ones((T, nLimbs), dtype=dtype)
  anglesz0_values = numpy.ones((T, nLimbs), dtype=dtype)   
  
  w_values = numpy.ones((T, nPoints), dtype=dtype)
  
  tarx_values = numpy.ones((T, nPoints), dtype=dtype)   
  tary_values = numpy.ones((T, nPoints), dtype=dtype)   

  x_values, y_values, z_values = backpropagationBasedFiltering(
    lines0_values, 
    rootsx0_values,
    rootsy0_values, 
    rootsz0_values,
    anglesx0_values,
    anglesy0_values,
    anglesz0_values,
    tarx_values,   
    tary_values,
    w_values,   
    structure,
    dtype,
  )
