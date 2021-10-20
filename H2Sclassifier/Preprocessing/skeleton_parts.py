# Define the dictionary keypoint to bodypart
keypoint_to_bodypart = {
   0 : 'Neck',
   1 : 'Nose',
   2 : 'MidHip',
   3 : 'LShoulder',
   4 : 'LElbow',
   5 : 'LHand',
   6 : 'LHip',
   7 : 'LKnee',
   8 : 'LAnkle',
   9 : 'RShoulder',
   10 : 'RElbow',
   11 : 'RHand',
   12 : 'RHip',
   13 : 'RKnee',
   14 : 'RAnkle',
   15 : 'LEye',
   16 : 'LEar',
   17 : 'REye',
   18 : 'REar',
   19 : 'LBigToe',
   20 : 'LSmallToe',
   21 : 'LHeel',
   22 : 'RBigToe',
   23 : 'RSmallToe',
   24 : 'RHeel'
}

# Define
bodypart_to_keypoint = {}
for key in keypoint_to_bodypart:
  bodypart_to_keypoint[keypoint_to_bodypart[key]] = key

# Define the parts of the skeleton that are joint
RightArm = [bodypart_to_keypoint[key] for key in ['Neck', 'RShoulder', 'RElbow', 'RHand']]
LeftArm = [bodypart_to_keypoint[key] for key in ['Neck', 'LShoulder', 'LElbow', 'LHand']]
Column = [bodypart_to_keypoint[key] for key in ['Nose', 'Neck', 'MidHip']]
RightLeg = [bodypart_to_keypoint[key] for key in ['MidHip', 'RHip', 'RKnee', 'RAnkle']]
LeftLeg = [bodypart_to_keypoint[key] for key in ['MidHip', 'LHip', 'LKnee', 'LAnkle']]
RightFace = [bodypart_to_keypoint[key] for key in ['Nose', 'REye', 'REar']]
LeftFace = [bodypart_to_keypoint[key] for key in ['Nose', 'LEye', 'LEar']]
RightFoot = [bodypart_to_keypoint[key] for key in ['RAnkle', 'RHeel', 'RBigToe', 'RSmallToe']]
LeftFoot = [bodypart_to_keypoint[key] for key in ['LAnkle', 'LHeel', 'LBigToe', 'LSmallToe']]

skeleton_parts = [RightArm, LeftArm, Column, RightLeg, LeftLeg, RightFace, LeftFace, RightFoot, LeftFoot]