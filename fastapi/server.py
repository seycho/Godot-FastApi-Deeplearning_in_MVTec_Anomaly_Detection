from torchvision.transforms import Compose, Normalize, ToTensor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel

from src.dataloader import GetMVTecData
from src.model import FastFlow

import numpy as np
import logging, torch, json, time, copy, cv2


configJson = json.load(open("config.json", "r"))
rootLogger = logging.getLogger()
logging.basicConfig(filename=configJson["log_path"], encoding="utf-8", level=logging.DEBUG)
rootLogger.info("start")

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ReqFmtInit(BaseModel):
    req : str
    httpURL : str

class ReqFmtChangeSubject(BaseModel):
    req : str
    subject : str

class ReqFmtResetModel(BaseModel):
    req : str
    learningRate : float

def SetImageTrain():
    global imgTrain
    imgTrain = cv2.imread(dataDic[targetSubject]["train"][np.random.randint(lengthTest)], 1)
    imgTrain = cv2.resize(imgTrain, dsize=(imgSize, imgSize))
    return None

def SetImageTest():
    global imgTest
    imgTest = cv2.imread(dataDic[targetSubject]["test"][np.random.randint(lengthTest)], 1)
    imgTest = cv2.resize(imgTest, dsize=(imgSize, imgSize))
    return None

def RunModelTrain():
    timeSta = time.time()
    modelFF.train()
    data = preprocess(imgTrain)
    optimizer.zero_grad()
    output = modelFF(data.unsqueeze(0).to(device))
    loss = output["loss"]
    loss.backward()
    optimizer.step()
    rootLogger.info("train time %.4f"%(time.time() - timeSta))
    return None

def RunModelTest():
    timeSta = time.time()
    modelFF.eval()
    with torch.no_grad():
        data = preprocess(imgTest)
        output = modelFF(data.unsqueeze(0).to(device))
    loss = output["loss"].item()
    mask = output["anomaly_map"].detach().cpu().numpy()[0][0]
    if (mask.min() == 0) and (mask.max() == 0):
        mask = np.ones((imgSize, imgSize))
    else:
        mask -= mask.min()
        mask /= mask.max()
    rootLogger.info("test time %.4f"%(time.time() - timeSta))
    return loss, mask

@app.post("/init")
async def Init(msg : ReqFmtInit):
    rootLogger.info("/init")

    msgDic = msg.dict()
    rootLogger.info(msgDic)
    return JSONResponse(json.dumps(msgDic))

@app.get("/train/next")
async def TrainNext():
    rootLogger.info("/train/next")
    RunModelTrain()
    loss, mask = RunModelTest()
    SetImageTrain()

    msgDic = {}
    msgDic["req"] = "TrainModel"
    msgDic["size"] = [imgSize, imgSize]
    msgDic["loss"] = loss
    msgDic["matTrain"] = imgTrain.tolist()
    msgDic["mskTest"] = mask.tolist()
    return JSONResponse(json.dumps(msgDic))

@app.post("/change/lr")
async def TrainReset(msg : ReqFmtResetModel):
    rootLogger.info("/change/lr")
    optimizer.param_groups[0]['lr'] = msg.learningRate

    msgDic = msg.dict()
    return JSONResponse(json.dumps(msgDic))

@app.post("/reset/model")
async def TrainReset(msg : ReqFmtResetModel):
    rootLogger.info("/reset/model")
    global modelFF, optimizer
    modelFF.load_state_dict(parametersBak)
    optimizer = torch.optim.Adam(params=modelFF.parameters(), lr=msg.learningRate)

    msgDic = msg.dict()
    return JSONResponse(json.dumps(msgDic))

@app.post("/subject/change")
async def SubjectChange(msg : ReqFmtChangeSubject):
    rootLogger.info("/subject/change")
    global targetSubject, lengthTrain, lengthTest
    targetSubject = msg.subject
    lengthTrain = len(dataDic[targetSubject]["train"])
    lengthTest = len(dataDic[targetSubject]["test"])
    SetImageTrain()
    SetImageTest()

    msgDic = msg.dict()
    msgDic["size"] = [imgSize, imgSize]
    msgDic["matTrain"] = imgTrain.tolist()
    msgDic["matTest"] = imgTest.tolist()
    return JSONResponse(json.dumps(msgDic))

@app.get("/test/change")
async def TestChange():
    rootLogger.info("/test/change")
    SetImageTest()
    loss, mask = RunModelTest()

    msgDic = {}
    msgDic["req"] = "TestChange"
    msgDic["size"] = [imgSize, imgSize]
    msgDic["loss"] = loss
    msgDic["matTrain"] = imgTrain.tolist()
    msgDic["matTest"] = imgTest.tolist()
    msgDic["mskTest"] = mask.tolist()
    return JSONResponse(json.dumps(msgDic))


dataDic = GetMVTecData(configJson["root_path"])

imgSize = 224
modelFF = FastFlow((imgSize, imgSize), pretrained=True)
parametersBak = modelFF.state_dict().copy()
try:
    device = torch.device(configJson["device"])
except:
    device = torch.device("cpu")
    rootLogger.warning("device %s is not available"%configJson["device"])
modelFF.to(device)
optimizer = torch.optim.Adam(params=modelFF.parameters(), lr=0.0001)
preprocess = Compose([
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

targetSubject = "cable"
lengthTrain = len(dataDic[targetSubject]["train"])
lengthTest = len(dataDic[targetSubject]["test"])

imgTrain = np.zeros((imgSize, imgSize), dtype=np.uint8)
imgTest = np.zeros((imgSize, imgSize), dtype=np.uint8)
SetImageTrain()
SetImageTest()
