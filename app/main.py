from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from uvicorn import run
from utilsRNN import loadVideoPoints, predict
from utilsCNN import sequence_prediction
import os

app = FastAPI()

origins = ['*']
methods = ['*']
headers = ['*']

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

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
  port = int(os.environ.get('PORT', 5000))
  run(app, port=port, host='0.0.0.0', timeout_keep_alive=300)
