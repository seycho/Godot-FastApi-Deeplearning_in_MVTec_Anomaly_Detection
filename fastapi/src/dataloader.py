import os

def GetMVTecData(pathRootMVTec):
    dataDic = {}
    for subName1 in os.listdir(pathRootMVTec):
        dataDic[subName1] = {}
        dataDic[subName1]["train"] = []
        dataDic[subName1]["test"] = []
        pathSub1MVTec = os.path.join(pathRootMVTec, subName1)
        for subName2 in os.listdir(pathSub1MVTec):
            pathSub2MVTec = os.path.join(pathSub1MVTec, subName2)
            if subName2 == "train":
                for fileName in os.listdir(os.path.join(pathSub2MVTec, "good")):
                    dataDic[subName1]["train"].append(os.path.join(pathSub2MVTec, "good", fileName))
            if subName2 == "test":
                for subName3 in os.listdir(pathSub2MVTec):
                    if subName3 != "good":
                        for fileName in os.listdir(os.path.join(pathSub2MVTec, subName3)):
                            dataDic[subName1]["test"].append(os.path.join(pathSub2MVTec, subName3, fileName))
    return dataDic
