# This function gives our structure of skeletal model


def getSkeletalModelStructure():
  # Definition of skeleton model structure:
  #   The structure is an n-tuple of:
  #
  #   (index of a start point, index of an end point, index of a bone) 
  #
  #   E.g., this simple skeletal model
  #
  #             (0)
  #              |
  #              |
  #              0
  #              |
  #              |
  #     (2)--1--(1)--1--(3)
  #      |               |
  #      |               |
  #      2               2
  #      |               |
  #      |               |
  #     (4)             (5)
  #
  #   has this structure:
  #
  #   (
  #     (0, 1, 0),
  #     (1, 2, 1),
  #     (1, 3, 1),
  #     (2, 4, 2),
  #     (3, 5, 2),
  #   )
  #
  #  Warning 1: The structure has to be a tree.  
  #
  #  Warning 2: The order isn't random. The order is from a root to lists.
  #
  #  (J, E, L, B)
  #  J: joint of the bone   E: end-joint of the bone    B: "before"/previous joint of the bone (serves as reference point)
  return ( 
    # head
    (0, 1, 0, -1),

    # right shoulder
    (1, 2, 1, 0),

    # right arm
    (2, 3, 2, 1),
    (3, 4, 3, 2),

    # left shoulder
    (1, 5, 1, 0),

    # left arm
    (5, 6, 2, 1),
    (6, 7, 3, 5),
  
    # right hand - wrist
    (4, 8, 4, 3),

    # right hand - 1st finger
    (8, 9, 5, 4),
    (9, 10, 6, 8),
    (10, 11, 7, 9),
    (11, 12, 8, 10),
    
    # right hand - 2nd finger
    (8, 13, 9, 4),
    (13, 14, 10, 8),
    (14, 15, 11, 13),
    (15, 16, 12, 14),

    # right hand - 3rd finger
    (8, 17, 13, 4),
    (17, 18, 14, 8),
    (18, 19, 15, 17),
    (19, 20, 16, 18),
 
    # right hand - 4th finger
    (8, 21, 17, 4),
    (21, 22, 18, 8),
    (22, 23, 19, 21),
    (23, 24, 20, 22),

    # right hand - 5th finger
    (8, 25, 21, 4),
    (25, 26, 22, 8),
    (26, 27, 23, 25),
    (27, 28, 24, 26),
  
    # left hand - wrist
    (7, 29, 4, 6),
  
    # left hand - 1st finger
    (29, 30, 5, 7), 
    (30, 31, 6, 29),
    (31, 32, 7, 30),
    (32, 33, 8, 31),

    # left hand - 2nd finger
    (29, 34, 9, 7),
    (34, 35, 10, 29),
    (35, 36, 11, 34),
    (36, 37, 12, 35),

    # left hand - 3rd finger
    (29, 38, 13, 7),
    (38, 39, 14, 29),
    (39, 40, 15, 38),
    (40, 41, 16, 39),

    # left hand - 4th finger
    (29, 42, 17, 7),
    (42, 43, 18, 29),
    (43, 44, 19, 42),
    (44, 45, 20, 43),

    # left hand - 5th finger
    (29, 46, 21, 7),
    (46, 47, 22, 29),
    (47, 48, 23, 46),
    (48, 49, 24, 47), 

  )


# Computing number of joints and limbs
def structureStats(structure):
  ps = {}
  ls = {}
  for a, b, l, _ in structure:
    ps[a] = "gotcha"
    ps[b] = "gotcha"
    ls[l] = "gotcha"
  return len(ls), len(ps)
