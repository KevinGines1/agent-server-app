from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
from app.utilsRNN import loadVideoPoints, predict
from app.utilsCNN import sequence_prediction

app = FastAPI()

class Data(BaseModel):
  videoUrl: str

class JointAnglePairs(BaseModel):
  jointAnglePairs: str

@app.get('/')
def root():
  return {'Hello': 'World!'}

@app.get('/index')
def hello_world():
  return 'Hello World!'

@app.post('/rnn/getJointAnglePairs')
def getJointAnglePairs(data: Data):
  frameCount = 0
  print('url: ', data.videoUrl)
  # * gather the joint pair angles for each frame in the video
  joint_pair_angles = loadVideoPoints(data.videoUrl, frameCount)
  print('frameCount: ', frameCount, '\njointPairAngles: \n', joint_pair_angles)
  return json.dumps(joint_pair_angles)
  

@app.post('/rnn/classify')
def classifyRNN(data: JointAnglePairs):
  # * use the ML model to classify either correct or wrong
  # print('classifyRNN: ', data.jointAnglePairs)
  prediction = predict(data.jointAnglePairs)
  return prediction

@app.post('/cnn/classify')
def classifyCNN(data: Data):
  print('cnn-url: ', data.videoUrl)
  print('predicting sequence...')
  prediction = sequence_prediction(data.videoUrl)
  return prediction

if __name__ == '__main__':
  uvicorn.run(app, port=8080, host='0.0.0.0')
