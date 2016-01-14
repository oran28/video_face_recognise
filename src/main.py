# -*- coding: utf-8 -*-



import glob
import time
import sys
from faceProc import FaceProc
from person import PersonData
import os
#from threading import Thread
from multiprocessing import Process, Queue, freeze_support

OPENCV_PATH=os.environ['OPENCV_PYTHON_PATH'] # opencv/build/python/2.7/x64
sys.path.append(OPENCV_PATH)

from PIL import ImageDraw, Image, ImageFont
from cv2 import VideoCapture, waitKey, imshow, destroyAllWindows

import numpy as NP


PROJ_PATH = str(os.path.split(os.path.split(__file__)[0])[0])
data_folder = PROJ_PATH + '/images'


font = ImageFont.truetype('c:/windows/fonts/Arial.ttf',72, encoding="unic")


def identify(img, all_people,frec):
    test = frec.processImg(img)
    
    all_guess = []

    for res in test:

        scores = {x : all_people[x].score(res['desc']) for x in all_people}
        guess = sorted(scores.keys(), key=lambda x:scores[x])[-1]
        scores = sorted(scores.values())[::-1][:3]
        print guess, scores
        if scores[0]>max(0.4, scores[1]+0.05):
            all_guess.append((guess,res['rect']))

    return all_guess


def load_person_db(frec):
    all_people = {}
    for person_dir in glob.glob(data_folder + '/*'):
        person_name = os.path.split(person_dir)[-1]
        if person_name == 'test':
            continue
    
        print 'building db for %s' % person_name
        all_people[person_name] = PersonData(person_name, person_dir, frec, font) 

    return all_people    


def draw_name(draw, person, name, name_counter, color):
    text_y = 360+name_counter*60
    text_center_x = person['rect'].center().x
    text_x = text_center_x - person['name_size'][0]/2
    draw.text((text_x,text_y),name, font=font,fill=color)
#        box = NP.array([res['rect'].left(), res['rect'].top(), 
#               res['rect'].right(), res['rect'].bottom()])*f
#        draw.rectangle(box)                


def video_loop(aframes_queue,person_queue):
    vc = VideoCapture(0)
    rval, frame = vc.read()
    people = {}
    colors = ((0,0,255),(255,255,0))
    while True:
        rval, frame = vc.read()
        if frame is None:
            c = waitKey(10)
            continue
        aframe = NP.asarray(frame[:,:])
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        
        while not person_queue.empty():
            name,rect,name_size = person_queue.get()
            people[name] = {'rect' : rect, 'name_size' : name_size, 
                            'time_found' : time.time()}

        name_counter = 0        
        for name in people.keys():
            if name_counter < 2:
                draw_name(draw, people[name], name, name_counter, colors[name_counter])
            name_counter += 1
            
            if time.time()>people[name]['time_found']+2:
                # stop displaying after 2 seconds
                people.pop(name)
                
        frame2 = NP.array(im)
        imshow('frame',frame2)


        if aframes_queue.empty():
            aframes_queue.put(aframe)
        c = waitKey(1)
        if c == 27: # exit on ESC
            break
    
    vc.release()
    destroyAllWindows()


def loop():
    guess = None
    counter = 0
    tload = time.time()
    

    aframes_queue = Queue()
    person_queue = Queue()
    print 'infinite loop'
    frec = FaceProc(PROJ_PATH)
    all_people = load_person_db(frec)

    video_thread = Process(target=video_loop, 
                           args=(aframes_queue,person_queue))
    video_thread.daemon = True
    video_thread.start()

    while True:
        tnew = time.time()
        video_thread.join(0.01)
        if not aframes_queue.empty():
            aframe = aframes_queue.get()
        
            all_guess = identify(aframe, all_people,frec)
            for newguess in all_guess:
                print all_people[newguess[0]].name
                all_people[newguess[0]].play_audio()
                person_queue.put((all_people[newguess[0]].name, newguess[1],
                                  all_people[newguess[0]].name_size))
#            if guess and guess == newguess:
#                counter += 1
#                if counter > 1:
#                    person_queue.put(all_people[guess].name)
#            else:
#                guess  = newguess
#                counter = 0         
        
        if not video_thread.is_alive():
            break


        if tnew - tload > 30:
            all_people = load_person_db(frec)
            tload = tnew




if __name__ == '__main__':
    freeze_support()
    loop()
