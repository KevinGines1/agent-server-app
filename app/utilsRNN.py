import cv2 as cv
import json
import math
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

# * define some constants
MAX_JOINT_ANGLES_LEN = 325
NUM_JOINT_ANGLE_PAIRS = 12
classes = ['CORRECT', 'WRONG']
model = tf.keras.models.load_model('./app/gruModel') # model built with GRU & Masking
# model = tf.keras.models.load_model('./gruModel') # model built with GRU & Masking


def logError(error, errorsList): # * fxn that logs the errors to a text file, for debugging purposes
  file = open('errors.txt', 'a')
  formattedError = str(error)+'\n'
  if(error not in errorsList):
    file.write(formattedError)
  file.close()

def getVector(point1, point2): # * fxn that creates a vector out of two points
  returnValue = ()
  try:
    returnValue = (point2[0]-point1[0], point2[1]-point1[1])
  except Exception as e:
    # print('error in getting vector: ', e)
    # logError(e, errors)
    returnValue = (None, None)
  return returnValue

def getAngleBetweenTwoVectors(vector1, vector2): # * fxn that gets the angle between two vectors
  x1 = vector1[0]
  y1 = vector1[1]
  x2 = vector2[0]
  y2 = vector2[1]
  returnValue = None
  try:
    radicand1 = (x1**2)+(y1**2)
    radicand2 = (x2**2)+(y2**2)
    magnitudes = np.sqrt([radicand1, radicand2])
    dotProd = (x1*x2)+(y1*y2)
    cosAngle = dotProd/(magnitudes[0]*magnitudes[1])
    returnValue = math.acos(cosAngle) # this is  in radians
  except Exception as e:
    # print('error in getting angle: ', e)
    # logError(e, errors)
    returnValue = None
  
  return returnValue

# * RNN FXNS

def loadVideoPoints(vidSrc, frameCount): # * fxn that loads a video file and extracts the joint_pair_angles in each frame

  # * initialize needed values
  net = cv.dnn.readNetFromTensorflow('graph_opt.pb') # load the weights for human pose estimation

  inWidth = 368
  inHeight = 368
  threshold = 0.1

  BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

  BODY_PART_LABEL = [key for key, value in BODY_PARTS.items()]

  POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

  JOINT_PAIRS = [ # this will be used for getting the joint angle pairs
    ['NeckRShoulder', 'NeckNose'],
    ['NeckRShoulder', 'NeckRHip'],
    ['NeckRShoulder', 'RShoulderRElbow'],
    ['NeckLShoulder', 'NeckNose'],
    ['NeckLShoulder', 'NeckLHip'],
    ['NeckLShoulder', 'LShoulderLElbow'],
    ['RShoulderRElbow', 'RElbowRWrist'],
    ['LShoulderLElbow', 'LElbowLWrist'],
    ['NeckRHip', 'RHipRKnee'],
    ['NeckLHip', 'LHipLKnee'],
    ['RHipRKnee', 'RKneeRAnkle'],
    ['LHipLKnee', 'LKneeLAnkle'],
  ]

  # * video processing
  videoCap = cv.VideoCapture(vidSrc)
  hasFrame, frame = videoCap.read()
  joint_angle_pairs_all_frames = list()
  joint_angle_pairs_all_frames_dict = {
    'NeckRShoulder-NeckNose': list(),
    'NeckRShoulder-NeckRHip': list(),
    'NeckRShoulder-RShoulderRElbow': list(),
    'NeckLShoulder-NeckNose': list(),
    'NeckLShoulder-NeckLHip': list(),
    'NeckLShoulder-LShoulderLElbow': list(),
    'RShoulderRElbow-RElbowRWrist': list(),
    'LShoulderLElbow-LElbowLWrist': list(),
    'NeckRHip-RHipRKnee': list(),
    'NeckLHip-LHipLKnee': list(),
    'RHipRKnee-RKneeRAnkle': list(),
    'LHipLKnee-LKneeLAnkle': list()
  }
  # while cv.waitKey(1) < 0:
  while hasFrame:
    # print('hasFrame: ', hasFrame)
    try:
      frame = cv.resize(frame, (800,800))
    except Exception as e:
      print('number of frames: ', frameCount)
      print('Exception: ', e, '\n', frame)
      break

    if not hasFrame:
      cv.waitKey()
      break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    #create blob
    blob = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    net.setInput(blob)

    out = net.forward()
    out = out[:, :19, :, :] # get the first 19 elements

    points = [] # store list of points
    vectors = {} # dictionary of vectors e.g. {'neckRightShoulder': (x,y)}

    for i in range(len(BODY_PARTS)): # generate the heatmap
      heatMap = out[0, i, :, :]
      #find (local) global maxima
      _, conf, _, point = cv.minMaxLoc(heatMap)
      x = (frameWidth*point[0]) / out.shape[3]
      y = (frameHeight*point[1]) / out.shape[2]

      # add point if confidence is higher than threshold
      if conf>threshold: 
        points.append((int(x), int(y)))
      else: 
        points.append(None)
    
    # get the connected body parts
    for pair in POSE_PAIRS:
      partFrom = pair[0]
      partTo = pair[1]

      idFrom = BODY_PARTS[partFrom]
      idTo = BODY_PARTS[partTo]
    
      # ? get the vectors for each body part
      # the keys are the merged name for each joint pair (refer to JOINT_PAIRS list)
      vectors[BODY_PART_LABEL[idFrom]+BODY_PART_LABEL[idTo]] = getVector(points[idFrom], points[idTo])

  # ? generate the joint pair angles of a frame
    joint_pair_angles = {}
    frameCount += 1
    for pair in JOINT_PAIRS:
      pairName = pair[0] + '-' + pair[1]
      closeAngle = getAngleBetweenTwoVectors(vectors[pair[0]], vectors[pair[1]])
      joint_pair_angles[pairName] = closeAngle
      joint_angle_pairs_all_frames_dict[pairName].append(closeAngle)
    joint_angle_pairs_all_frames.append(joint_pair_angles)
    hasFrame, frame = videoCap.read()
  return joint_angle_pairs_all_frames_dict

def replaceNan(array):
  index = 0
  for value in array: 
    if math.isnan(value): 
      array[index] = 0.0
    index += 1
  return array

def predict(dataset):
  # convert the string back to json format
  # get all the values for each joint angle pair, (this was done in loadVideo part already)
  jointAnglePairsPerFrame = json.loads(dataset)
  flattenedArray = []
  index = 0 
  for jointAnglePair in jointAnglePairsPerFrame:
    # convert to np.array()
    temp = np.array(jointAnglePairsPerFrame[jointAnglePair])
    # convert Nones to -1
    noNones = np.where(temp == None, 0.0, temp)
    # convert nans to -1
    noNans = replaceNan(noNones)
    # get the padWidth
    padWidth = abs(len(noNans)-MAX_JOINT_ANGLES_LEN)
    # pad with 0s
    paddedValues = np.pad(noNans, (0, padWidth), 'constant', constant_values=(0))
    # combine the joint angles
    flattenedArray = np.concatenate((flattenedArray, paddedValues), axis=None)
  # reshape to [1, 12, 325]
  flattenedArray = flattenedArray.tolist()
  predictMe = tf.reshape(flattenedArray, (1, NUM_JOINT_ANGLE_PAIRS, MAX_JOINT_ANGLES_LEN))
  # predict
  prediction = model.predict(predictMe)
  prediction = classes[np.argmax(prediction)]
  # prediction = 'CORRECT'
  return prediction

errors = []
