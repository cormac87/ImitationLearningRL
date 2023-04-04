import json
import glob
import shutil

replay_folder = 'C:/Users/corma/OneDrive/Desktop/drknownjson/*'

replayList = glob.glob(replay_folder)

ballPos = []
player1Time = []
player2Time = []
ballTime = []
player1Pos = []
player2Pos = []
player1Rot = []
player2Rot = []
player1Vel = []
player2Vel = []
player1Boost = []
player2Boost = []

counttt = 0
for replay in replayList:
    print(counttt)
    f = open(replay)

    data = json.load(f)

    player1Name = ""
    player2Name = ""
    if not('PlayerStats' in data['properties'].keys()):
        counttt += 1
        continue
    if len(data['properties']['PlayerStats']) < 2:
        counttt += 1
        continue
    
    if data['properties']['PlayerStats'][0]['OnlineID'] == '76561198202084876':
        player1Name = data['properties']['PlayerStats'][0]['Name']
        player2Name = data['properties']['PlayerStats'][1]['Name']
    else:
        player2Name = data['properties']['PlayerStats'][0]['Name']
        player1Name = data['properties']['PlayerStats'][1]['Name']

    print("p1 = ", player1Name)
    print("opponent = ", player2Name)

    ballActorID = 0
    player1ActorID = 0
    player2ActorID = 0
    player1BoostActorID = 0
    player2BoostActorID = 0
    BoostAct = []
    BoostActOwner = []


    ballPos.append((0,0,0))
    player1Time.append(0)
    player2Time.append(0)
    ballTime.append(0)
    player1Pos.append((0,0,0))
    player2Pos.append((0,0,0))
    player1Rot.append((0,0,0,0))
    player2Rot.append((0,0,0,0))
    player1Vel.append((0,0,0))
    player2Vel.append((0,0,0))
    player1Boost.append(85)
    player2Boost.append(85)

    replayCorrupted = False
    for frame in data['network_frames']['frames']:
        tempBoostActList = []
        tempBoostActOwnerList = []
        ballPosAdded = False
        player1PosAdded = False
        player2PosAdded = False
        player1BoostAdded = False
        player2BoostAdded = False
        time = frame['time']
        for nac in frame['new_actors']:
            if nac['object_id'] == 68:
                ballActorID = nac['actor_id']
        ocount = 0
        for uac in frame['updated_actors']:
            if uac['object_id'] == 25:
                if 'ActiveActor' not in uac['attribute'].keys():
                    replayCorrupted = True
                    break
                else:
                    ocount += 1
        if replayCorrupted or ocount < 1:
            break
                    
                    
        for uac in frame['updated_actors']:
            if replayCorrupted:
                break
            if uac['object_id'] == 25:
                if 'ActiveActor' not in uac['attribute'].keys():
                    replayCorrupted = True
                    break
                actorNameID = uac['attribute']['ActiveActor']['actor']
                for i in frame['updated_actors']:
                    if i['actor_id'] == actorNameID and i['object_id'] == 129:
                        if 'String' not in i['attribute'].keys():
                            replayCorrupted = True
                            break
                        playerName = i['attribute']['String']
                        if player1Name == playerName:
                            player1ActorID = uac['actor_id']
                        else:
                            player2ActorID = uac['actor_id']
            elif uac['object_id'] == 46:
                if uac['actor_id'] == ballActorID:
                    ballPos.append((uac['attribute']['RigidBody']['location']['x'],uac['attribute']['RigidBody']['location']['y'],uac['attribute']['RigidBody']['location']['z']))
                    ballPosAdded = True
                    ballTime.append(time)
                elif uac['actor_id'] == player1ActorID:
                    player1PosAdded = True
                    player1Time.append(time)
                    player1Rot.append((uac['attribute']['RigidBody']['rotation']['x'],uac['attribute']['RigidBody']['rotation']['y'],uac['attribute']['RigidBody']['rotation']['z'],uac['attribute']['RigidBody']['rotation']['w']))
                    player1Pos.append((uac['attribute']['RigidBody']['location']['x'],uac['attribute']['RigidBody']['location']['y'],uac['attribute']['RigidBody']['location']['z']))
                    if uac['attribute']['RigidBody']['linear_velocity'] != None:
                        player1Vel.append((uac['attribute']['RigidBody']['linear_velocity']['x'],uac['attribute']['RigidBody']['linear_velocity']['y'],uac['attribute']['RigidBody']['linear_velocity']['z']))
                elif uac['actor_id'] == player2ActorID:
                    player2PosAdded = True
                    player2Time.append(time)
                    player2Rot.append((uac['attribute']['RigidBody']['rotation']['x'],uac['attribute']['RigidBody']['rotation']['y'],uac['attribute']['RigidBody']['rotation']['z'],uac['attribute']['RigidBody']['rotation']['w']))
                    player2Pos.append((uac['attribute']['RigidBody']['location']['x'],uac['attribute']['RigidBody']['location']['y'],uac['attribute']['RigidBody']['location']['z']))
                    if uac['attribute']['RigidBody']['linear_velocity'] != None:
                        player2Vel.append((uac['attribute']['RigidBody']['linear_velocity']['x'],uac['attribute']['RigidBody']['linear_velocity']['y'],uac['attribute']['RigidBody']['linear_velocity']['z']))

            elif uac['object_id'] == 247:
                if uac['actor_id'] == player1BoostActorID:
                    player1Boost.append(uac['attribute']['Byte'])
                    player1BoostAdded = True
                elif uac['actor_id'] == player2BoostActorID:
                    player2Boost.append(uac['attribute']['Byte'])
                    player2BoostAdded = True
                else:
                    boostAct = uac['actor_id']
                    tempBoostActList.append(boostAct)
               #     for i in frame['updated_actors']:
               #         if i['actor_id'] == boostAct and i['object_id'] == 234:
               #             tempBoostActOwnerList.append(i['attribute']['ActiveActor']['actor'])

        if replayCorrupted:
            break
        for i in range(0,len(tempBoostActOwnerList)):
            if tempBoostActOwnerList[i] == player1ActorID:
                player1BoostActorID = tempBoostActList[i]
            elif tempBoostActOwnerList[i] == player2ActorID:
                player2BoostActorID = tempBoostActList[i]
        if ballPosAdded == False:
            ballPos.append(ballPos[len(ballPos) - 1])
            ballTime.append(0)
        if player1PosAdded == False:
            player1Pos.append(player1Pos[len(player1Pos) - 1])
            player1Rot.append(player1Rot[len(player1Rot) - 1])
            player1Vel.append(player1Vel[len(player1Vel) - 1])
            player1Time.append(0)
        if player2PosAdded == False:
            player2Pos.append(player2Pos[len(player2Pos) - 1])
            player2Rot.append(player2Rot[len(player2Rot) - 1])
            player2Vel.append(player2Vel[len(player2Vel) - 1])
            player2Time.append(0)
        if player1BoostAdded == False:
            player1Boost.append(player1Boost[len(player1Boost) - 1])
        if player2BoostAdded == False:
            player2Boost.append(player2Boost[len(player2Boost) - 1])

    jsonString = json.dumps((ballPos,player1Pos,player2Pos))
    counttt += 1



file = open("ballPos.txt", "w")
for i in ballPos:
    file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n")
file.close()

file = open("player1Pos.txt", "w")
for i in player1Pos:
    file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n")
file.close()

file = open("player1Rot.txt", "w")
for i in player1Rot:
    file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + "\n")
file.close()

file = open("player2Pos.txt", "w")
for i in player2Pos:
    file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n")
file.close()

file = open("player2Rot.txt", "w")
for i in player2Rot:
    file.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + "\n")
file.close()

file = open("player1Boost.txt","w")
for i in player1Boost:
    file.write(str(i) + "\n")

file = open("player2Boost.txt","w")
for i in player2Boost:
    file.write(str(i) + "\n")

file = open("player1Time.txt","w")
for i in player1Time:
    file.write(str(i) + "\n")

file = open("player2Time.txt","w")
for i in player2Time:
    file.write(str(i) + "\n")

file = open("player1Vel.txt","w")
for i in player1Vel:
    file.write(str(i) + "\n")

file = open("player2Vel.txt","w")
for i in player2Vel:
    file.write(str(i) + "\n")
