import json
import glob
import shutil
import pickle
import numpy as np
from scipy import interpolate
from scipy.spatial import distance

def getFrames():
    timestep = 0.1
    frameList = []
    
    j = 0
    count = 0
    while j < 1000:
        j += timestep
        frameList.append(j)
        count += 1
    print(count)    
    return frameList

def smallest_of_three(a, b, c):
    """
    Returns the smallest of three integer variables.
    """
    if a <= b and a <= c:
        return a
    elif b <= a and b <= c:
        return b
    else:
        return c

def biggest_of_three(a, b, c):
    """
    Returns the biggest of three integer variables.
    """
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c
    
def getBallPaths(ballList, ballVelList, player1List, player2List, minBallPathLength):
    listOfBallPaths = []
    ballPath = []
    length = smallest_of_three(len(player1List),len(player2List),len(ballList))
    for i in range(length):
        p1Dist = distance.euclidean(player1List[i],ballList[i])
        p2Dist = distance.euclidean(player2List[i],ballList[i])

        if p1Dist < 300 or p2Dist < 300:
            if len(ballPath) > minBallPathLength:
                listOfBallPaths.append(ballPath)
            ballPath = []
        else:
            bp = ballList[i]
            bp.extend(ballVelList[i])
            ballPath.append(bp)
    return listOfBallPaths        

def smooth_positions(listToSmooth, listToSmoothFrames, frameList):
    #Snip frame list
    for i in range(len(frameList) - 1):
        if frameList[i] >= listToSmoothFrames[0]:
            frameList = frameList[i:]
            break
        
    for i in range(len(frameList) - 1):
        if frameList[i] >= listToSmoothFrames[len(listToSmoothFrames) - 1]:
            frameList = frameList[:i]
            break
            
    # Convert the input lists to numpy arrays    
    positions = np.array(listToSmooth)
    times = np.array(listToSmoothFrames)
    new_times = np.array(frameList)
    if not all(len(pos) == 3 for pos in listToSmooth):
        raise ValueError("All tuples in listToSmooth must have length 3")
    # Create interpolation functions for x, y, and z positions
    f_x = interpolate.interp1d(times, positions[:,0],'linear')
    f_y = interpolate.interp1d(times, positions[:,1],'linear')
    f_z = interpolate.interp1d(times, positions[:,2],'linear')

    # Evaluate interpolation functions at new time values
    new_positions = np.zeros((len(new_times), 3))
    new_positions[:,0] = f_x(new_times)
    new_positions[:,1] = f_y(new_times)
    new_positions[:,2] = f_z(new_times)

    return new_positions.tolist()


def smooth_rotations(listToSmooth, listToSmoothFrames, frameList):
    #Snip frame list
    for i in range(len(frameList) - 1):
        if frameList[i] >= listToSmoothFrames[0]:
            frameList = frameList[i:]
            break
        
    for i in range(len(frameList) - 1):
        if frameList[i] >= listToSmoothFrames[len(listToSmoothFrames) - 1]:
            frameList = frameList[:i]
            break
            
    # Convert the input lists to numpy arrays    
    positions = np.array(listToSmooth)
    times = np.array(listToSmoothFrames)
    new_times = np.array(frameList)
    
    # Create interpolation functions for x, y, and z positions
    f_x = interpolate.interp1d(times, positions[:,0],'linear')
    f_y = interpolate.interp1d(times, positions[:,1],'linear')
    f_z = interpolate.interp1d(times, positions[:,2],'linear')
    f_w = interpolate.interp1d(times, positions[:,3],'linear')
    

    # Evaluate interpolation functions at new time values
    new_positions = np.zeros((len(new_times), 4))
    new_positions[:,0] = f_x(new_times)
    new_positions[:,1] = f_y(new_times)
    new_positions[:,2] = f_z(new_times)
    new_positions[:,3] = f_w(new_times)

    

    return new_positions.tolist()

replay_folder = 'C:/Users/corma/OneDrive/Desktop/drknownjson/*'

replayList = glob.glob(replay_folder)

ballPosList = []
ballVelList = []
player1PosList = []
player2PosList = []
player1RotList = []
player2RotList = []
player1VelList = []
player2VelList = []

ballPosTimeList = []
player1PosTimeList = []
player2PosTimeList = []

hitFrames = []

for replay in replayList:
    ballPosL = []
    ballVelL = []
    p1PosL = []
    p2PosL = []
    p1RotL = []
    p2RotL = []
    p1VelL = []
    p2VelL = []
    ballPosTL = []
    p1PosTL = []
    p2PosTL = []
    f = open(replay)
    data = json.load(f)
    foundBall = False
    ballActorID = -1
    player1ActorIDList = []
    player2ActorIDList = []
    objID = -1
    uac = data['network_frames']['frames'][0]['updated_actors']
    for ac in uac:
        if 'attribute' in ac.keys():
            if 'RigidBody' in ac['attribute'].keys():
                objID = ac['object_id']
                break
    for frame in data['network_frames']['frames']:
        for nac in frame['new_actors']:
            if 'initial_trajectory' in nac.keys():
                if nac['initial_trajectory']['location'] != None:
                    loc = nac['initial_trajectory']['location']
                    if loc['x'] == 0 and loc['y'] == 0 and (loc['z'] > 90):
                        ballActorID = nac['actor_id']
                    elif (abs(loc['x']) == 2048 and abs(loc['y']) == 2560) or (abs(loc['x']) == 256 and abs(loc['y']) == 3840) or (abs(loc['x']) == 0 and abs(loc['y']) == 4608):
                        if loc['y'] > 0:
                            player1ActorIDList.append(nac['actor_id'])
                        else:
                            player2ActorIDList.append(nac['actor_id'])
        for uac in frame['updated_actors']:
            if uac['actor_id'] == 55:
                hitFrames.append(frame['time'])
            if uac['actor_id'] == ballActorID:
                if uac['object_id'] == objID:
                    loc = uac['attribute']['RigidBody']['location']
                    vel = uac['attribute']['RigidBody']['linear_velocity']
                    ballPosL.append((loc['x'],loc['y'],loc['z']))
                    try:
                        ballVelL.append((vel['x'],vel['y'],vel['z']))
                    except:
                        try:
                            ballVelL.append(ballVelL[-1])
                        except:
                            ballVelL.append((0,0,0))
                    ballPosTL.append(frame['time'])
            if uac['actor_id'] in player1ActorIDList:
                if uac['object_id'] == objID:
                    loc = uac['attribute']['RigidBody']['location']
                    rot = uac['attribute']['RigidBody']['rotation']
                    vel = uac['attribute']['RigidBody']['linear_velocity']
                    
                    p1PosL.append((loc['x'],loc['y'],loc['z']))
                    p1RotL.append((rot['x'],rot['y'],rot['z'],rot['w']))
                    try:
                        p1VelL.append((vel['x'],vel['y'],vel['z']))
                    except:
                        p1VelL.append((0,0,0))
                    
                    p1PosTL.append(frame['time'])
                    player1ActorIDList = [uac['actor_id']]
                    
            if uac['actor_id'] in player2ActorIDList:
                if uac['object_id'] == objID:
                    loc = uac['attribute']['RigidBody']['location']
                    rot = uac['attribute']['RigidBody']['rotation']
                    vel = uac['attribute']['RigidBody']['linear_velocity']

                    p2PosL.append((loc['x'],loc['y'],loc['z']))
                    p2RotL.append((rot['x'],rot['y'],rot['z'],rot['w']))
                    try:
                        p2VelL.append((vel['x'],vel['y'],vel['z']))
                    except:
                        p2VelL.append((0,0,0))
                    p2PosTL.append(frame['time'])
                    player2ActorIDList = [uac['actor_id']]

    #make sure all are same size
    smallest = smallest_of_three(p1PosTL[len(p1PosTL) - 1],p2PosTL[len(p2PosTL) - 1],ballPosTL[len(ballPosTL) - 1])
    p1Good = False
    p2Good = False
    ballGood = False
    while(not(p1Good and p2Good and ballGood)):
        if p1PosTL[len(p1PosTL) - 1] > smallest:
            p1PosTL.pop()
            p1PosL.pop()
            p1VelL.pop()
            p1RotL.pop()
        else:
            p1Good = True
        if p2PosTL[len(p2PosTL) - 1] > smallest:
            p2PosTL.pop()
            p2PosL.pop()
            p2VelL.pop()
            p2RotL.pop()
        else:
            p2Good = True
        if ballPosTL[len(ballPosTL) - 1] > smallest:
            ballPosTL.pop()
            ballPosL.pop()
            ballVelL.pop()
        else:
            ballGood = True

    biggest = biggest_of_three(p1PosTL[0],p2PosTL[0],ballPosTL[0])
    p1Good = False
    p2Good = False
    ballGood = False
    while(not(p1Good and p2Good and ballGood)):
        if p1PosTL[0] < biggest:
            p1PosTL.pop(0)
            p1PosL.pop(0)
            p1RotL.pop(0)
            p1VelL.pop(0)
        else:
            p1Good = True
        if p2PosTL[0] < biggest:
            p2PosTL.pop(0)
            p2PosL.pop(0)
            p2RotL.pop(0)
            p2VelL.pop(0)
        else:
            p2Good = True
        if ballPosTL[0] < biggest:
            ballPosTL.pop(0)
            ballPosL.pop(0)
            ballVelL.pop(0)
        else:
            ballGood = True

    print(p1PosTL[len(p1PosTL) - 1])
    print(ballPosTL[len(ballPosTL) - 1])
    print(p2PosTL[len(p2PosTL) - 1])
    #smooth the data
    frames = getFrames()
    k = 0
    while frames[k] < ballPosTL[0]:
        k+=1
    frames = frames[k:]
    ballPosList.append(smooth_positions(ballPosL,ballPosTL,frames))
    ballVelList.append(smooth_positions(ballVelL,ballPosTL,frames))
    
    player1PosList.append(smooth_positions(p1PosL,p1PosTL,frames))
    player2PosList.append(smooth_positions(p2PosL, p2PosTL, frames))
    player1VelList.append(smooth_positions(p1VelL,p1PosTL,frames))
    player2VelList.append(smooth_positions(p2VelL,p2PosTL,frames))
    player1RotList.append(smooth_rotations(p1RotL,p1PosTL,frames))
    player2RotList.append(smooth_rotations(p2RotL, p2PosTL, frames))


print(len(ballPosList[0]))
print(len(player1PosList[0]))
print(len(player2PosList[0]))

#make sure all same size

for i in range(len(ballPosList)):
    smallest = smallest_of_three(len(ballPosList[i]), len(player1PosList[i]), len(player2PosList[i]))
    p1Good = False
    p2Good = False
    ballGood = False
    while(not(p1Good and p2Good and ballGood)):
        if len(player1PosList[i]) > smallest:
            player1PosList[i].pop()
            player1RotList[i].pop()
            player1VelList[i].pop()
        else:
            p1Good = True
        if len(player2PosList[i]) > smallest:
            player2PosList[i].pop()
            player2RotList[i].pop()
            player2VelList[i].pop()
        else:
            p2Good = True
        if len(ballPosList[i]) > smallest:
            ballPosList[i].pop()
            ballVelList[i].pop()
        else:
            ballGood = True

print(len(ballPosList[0]))
print(len(ballVelList[0]))
print(len(player1PosList[0]))
print(len(player1VelList[0]))
print(len(player2PosList[0]))
print(len(player2VelList[0]))



#cutting data so there is only play time data

for j in range(len(ballPosList)):
    i = 0
    while not(distance.euclidean(ballPosList[j][i],player1PosList[j][i]) < 300 or distance.euclidean(ballPosList[j][i],player2PosList[j][i]) < 300):
                del(ballPosList[j][i])
                del(player1PosList[j][i])
                del(player2PosList[j][i])
                del(player1RotList[j][i])
                del(player2RotList[j][i])
    i = 0
    while i < len(ballPosList[j]):
        if abs(ballPosList[j][i][1]) > 5050:
            while not(ballPosList[j][i][0] == 0 and ballPosList[j][i][1] == 0) and i < len(ballPosList[j]) - 1:
                del(ballPosList[j][i])
                del(player1PosList[j][i])
                del(player2PosList[j][i])
                del(player1RotList[j][i])
                del(player2RotList[j][i])
            while not(distance.euclidean(ballPosList[j][i],player1PosList[j][i]) < 300 or distance.euclidean(ballPosList[j][i],player2PosList[j][i]) < 300) and i < len(ballPosList[j]) - 1:
                del(ballPosList[j][i])
                del(player1PosList[j][i])
                del(player2PosList[j][i])
                del(player1RotList[j][i])
                del(player2RotList[j][i])
        i += 1
                  
ballPaths = []
for i in range(len(ballPosList)):
    ballPath = getBallPaths(ballPosList[i],ballVelList[i],player1PosList[i],player2PosList[i],15)
    ballPaths.extend(ballPath)    
for i in ballPaths:
    print(i)
    print()

with open("bp", "wb") as fp:   #Pickling
    pickle.dump(ballPosList, fp)
    
with open("bv", "wb") as fp:   #Pickling
    pickle.dump(ballVelList, fp)

with open("p1p", "wb") as fp:   #Pickling
    pickle.dump(player1PosList, fp)

with open("p1v", "wb") as fp:   #Pickling
    pickle.dump(player1VelList, fp)

with open("p2p", "wb") as fp:   #Pickling
    pickle.dump(player2PosList, fp)

with open("p2v", "wb") as fp:   #Pickling
    pickle.dump(player2VelList, fp)

with open("bp2", "wb") as fp:   #Pickling
    pickle.dump(ballPaths, fp)
    
file = open("newBallPos.txt", "w")
for i in ballPaths:
    for j in i:
        file.write(str(j[0]) + " " + str(j[1]) + " " + str(j[2]) + "\n")
file.close()
    
file = open("ballPos.txt", "w")
for j in ballPosList:
    for i in j:
        file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n")
file.close()
 
file = open("player1Pos.txt", "w")
for j in player1PosList:
    for i in j:
        file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n")
file.close()

file = open("player1Rot.txt", "w")
for j in player1RotList:
    for i in j:
        file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + "\n")
file.close()

file = open("player2Pos.txt", "w")
for j in player2PosList:
    for i in j:
        file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n")
file.close()

file = open("player2Rot.txt", "w")
for j in player2RotList:
    for i in j:
        file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + "\n")
file.close()

file = open("ballPosTime.txt", "w")
for i in ballPosTimeList:
    file.write(str(i) + "\n")
file.close()
          

        
