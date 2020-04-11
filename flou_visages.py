# -*- coding: utf-8 -*-

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
assert not face_cascade.empty(), 'Error: cascade not found!'

mask_blur = cv2.imread('mask_blur.png')
assert mask_blur is not None, 'Error: blurring mask not found!'

def blur_faces(frame):#déclaration pour plusieurs visages
    global mask_blur
    #déclaration de blur_face(frame)
    nlig,ncol=frame.shape[:2]
    if(nlig,ncol)>(240,320):
        scale=int(min(ncol/320,nlig/240))
        frame_process=cv2.resize(frame,dsize=(0,0),fx=1/scale,fy=1/scale)
    else:
        scale=1  
        frame_process=frame
        
    frame_process = cv2.cvtColor(frame_process, cv2.COLOR_BGR2GRAY)
    # à la fin de frame, on a ajouté 'process' pour acceler sa vitesse à état normarle
    faces = face_cascade.detectMultiScale(frame_process, scaleFactor=1.3, minNeighbors=10)

    for (x,y,w,h) in faces[:2]:
        xr=int(x*scale)
        yr=int(y*scale)
        wr=int(w*scale)
        hr=int(h*scale)
        #pour simplifier la langage, nous les avons écrit par la façons plus simple
        # ils sont utilisés à créer un rectancle à attraper les visages qui sont correspond
        # à la bonne vitesse.
        mask_blur=cv2.resize(mask_blur,dsize=(wr,hr)) #nous pouvons changé sa taille de mask.
        mask_blur=mask_blur/255.0   #nous pouvons chagé la valeur à changer la concentration de mask.
        
#        cv2.rectangle(frame,(x*scale,y*scale),((x+w)*scale,(y+h)*scale),(0,0,0),2)
        #Ici, 'cv2' qui correspond à dessiner un rectangle de attraper les visages. 
        #cv2.circle(frame,(x*scale,y*scale), 63, (0,0,255), -1)
        # on voit que la fonctions desus est sur créer un circle pour attraper les vissages.
        subimg=frame[yr:(yr+hr),xr:(xr+hr)]
        #subimg[:]=255-subimg
        #Après Avoir changé, nous avons des rectangles qui sont dans la même vitesse de floutage de la vidéo
        subimg[:]=cv2.blur(subimg,(45,45))
        #subimg[:] symbolise d'avoir une fonction suivant. cv2 est le rectangle.

    return frame
    
def process_video():
    cap = cv2.VideoCapture('geii2_video_test1.mp4')
    #cap = cv2.VideoCapture(0) #caméra
    assert cap.isOpened(), 'Cannot open video'
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #fps = 30   #caméra

    k = 0
    while(True):
        k += 1
        if k >= num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            k = 0

        ret, frame = cap.read()
        if ret != True:
            print('Erreur: flux video indisponible')
            break

        key = cv2.waitKey(1000//fps) & 0xFF
        if key == ord('q'):
            break

        frame = blur_faces(frame)
        cv2.imshow('Video', frame)

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    process_video()
