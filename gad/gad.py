# -*- coding: utf-8 -*-
import time
import cv2
import math
import argparse
import pyttsx3
import random

engine = pyttsx3.init()

list_of_compliments = ['Поздравляем с 8 Марта! Желаем в этот весенний и сказочный день услышать много восхитительных комплиментов, подарить море улыбок и исполнить хотя бы одну, но самую заветную мечту! Главное не забывать, что даже смелые и сумасшедшие желания обязательно сбываются там, где начинается счастье!',
'Сегодня ты не просто женщина, ты - королева! И мне досталась великая честь озвучить это поздравление. В день Восьмого марта я хочу пожелать, чтобы твой мир был сказкой, такой же яркой как солнечные лучики и такой же свежей как первые весенние деньки.',
'Прекрасной Даме, хранительнице очага, доброй волшебнице! Пусть в женский праздник сбудутся все самые лучшие пожелания! Пусть весеннее солнце с каждым днём светит всё ярче и греет всё жарче, наполняя Вашу жизнь светом и теплом, а родные и близкие радуют успехами.',
'Очаровательная, милая, прекрасная, манящая, добрая, нежная, искренняя, лучезарная, мечтательная, элегантная, превосходная, неотразимая, чудесная, неповторимая, в этот день и всегда ты заслуживаешь в свой адрес слышать только такие комплименты, только добрые слова и пожелания любви, тепла и счастья.',
'Ты обворожительна!',
'Ты будто сошла со страниц красивой сказки!',
'Твои глаза – два бездонных океана, в которых я готов утонуть прямо в эту минуту!',
'В тебе столько нежности, это меня привлекает и делает тебя невероятно женственной!',
'Когда ты рядом, я забываю о всех своих проблемах и причина тому эта невероятная легкость!',
'Ни в коем случае ничего не меняй в себе, ты просто божественна!',
'Твои черты списаны с самых лучших картин художников!',
'Ты свежа и красива, как розовый бутон!',
'Ты такая уточенная, такая изысканная и красивая, я не могу тобой не восхищаться!',
'Не могу подобрать слов, олицетворяющих твою красоту, ведь в словаре их просто не хватает!',
'Ты не воровка? Ведь совершенно бесстыдно ты украла все мои мысли и сердце!',
'У тебя космическая, невероятная и просто сказочная красота!',
'Спасибо, что позволяешь быть рядом с тобой и наслаждаться твоей красотой!',
'Если бы ты участвовала в конкурсах красоты, они бы потеряли свою значимость, ведь победа всегда была бы твоей!',
'Даже самый большой и глубокий океан не сравниться с величиной твоей красоты!',
'Твоя кожа такая белоснежная, губы такие мягкие, глаза такие яркие, что я не могу отвести от тебя глаз!',
'Можно хранить твою фотографию на прикроватном столике, чтобы любоваться прекраснейшими чертами круглосуточно?',
'Даже слепому не нужны очки, чтобы разглядеть твою красоту!',
'Невероятно! Твоя красота дарована Богом, ты знала о том, что ты его любимица?',
]



def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        # print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        # print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        # cv2.imshow("Detecting age and gender", resultImg)
        if gender == "Female":
            print("WOMAN")
            engine.say(list_of_compliments[random.randint(0, len(list_of_compliments))-1])
            engine.runAndWait()
    time.sleep(1)

