import json
import glob
import shutil


def getRBObjectID(data):
    for i in data['network_frames']['frames'][0]['updated_actors']:
        if 'attribute' in i.keys():
            if 'RigidBody' in i['attribute'].keys():
                return i['object_id']

def getBallActorID(data, RBObjectID, currentFrame, currentID):
    uac = data['network_frames']['frames'][currentFrame]['updated_actors']
    for i in range(len(uac)):
        if uac[i]['object_id'] == RBObjectID:
            acID = uac[i]['actor_id']
            for j in range(len(uac)):
                if uac[j]['actor_id'] == acID:
                    if 'attribute' in uac[j].keys():           
                        if 'ActiveActor' in uac[j]['attribute'].keys():
                            if uac[j]['attribute']['ActiveActor']['actor'] == 0:
                                return acID
    return currentID   
replay_folder = 'C:/Users/corma/OneDrive/Desktop/drknownjson/*'

replayList = glob.glob(replay_folder)

replayNumber = 0

ballPosList = [0]

ballTrackedCount = 0

for replay in replayList:
    print(replay)
    f = open(replay)
    data = json.load(f)
    ballActorID = 0
    initialBallPosListCount = len(ballPosList)
    ballAssignedFrameCount = 0
    RBObjectID = getRBObjectID(data)
    BallActorID = getBallActorID(data, RBObjectID, 0, 0)
    print(BallActorID)




    
    
