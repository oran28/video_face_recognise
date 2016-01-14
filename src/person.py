# -*- coding: utf-8 -*-

import glob
import numpy as NP
import os
import pickle
import codecs

FFMPEG = 'F:/ffmpeg-20151011-git-f05ff05-win64-static/bin/ffplay'

class PersonData(object):
    def __init__(self, name, folder, face_proc, font):
        self.name = name
        if os.path.exists('%s/name.txt' % folder):
            self.name = codecs.open('%s/name.txt' % folder,encoding='utf-8').read()[-1:0:-1]
            
        self.sound = None
        if os.path.exists('%s/sound.mp3' % folder):
            self.sound = '%s/sound.mp3' % folder
        self.name_size = font.getsize(self.name)
        self.load_db(folder,face_proc)
        
    def load_db(self,folder, face_proc):
        self.desc = []
        all_imgs = glob.glob('%s/*.jpg' % folder)
        for f in all_imgs:
            desc_file = f + '.desc'
            if os.path.exists(desc_file):
                d = pickle.load(file(desc_file,'rb'))
                if len(d) > 0:
                    self.desc.append(d)
            else:
                face_data = face_proc.processImg(f)
                if len(face_data) == 0:
                    continue
                if len(face_data) > 1:
                    face_data = sorted(face_data, key=lambda x:x['rect'].width())[::-1]
                    if face_data[0]['rect'].width()<2*face_data[1]['rect'].width():
                        print 'image %s has %d faces' % (f, len(face_data))
                        pickle.dump([], file(desc_file,'wb'))
                        continue
                pickle.dump(face_data[0]['desc'], file(desc_file,'wb'))
                self.desc.append(face_data[0]['desc'])
    
    def score(self,desc):
        if len(desc) == 0:
            return 0
        scores = sorted([NP.dot(x,desc)/(sum(x**2)*sum(desc**2))**0.5 for x in self.desc])[::-1]
        scores = scores[:min(3,len(scores))]
        return NP.mean(scores)
        
    def play_audio(self):
        if self.sound:
            os.system('%s -autoexit -nodisp -i %s' % (FFMPEG, self.sound))
        
    
            
        