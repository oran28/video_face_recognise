# -*- coding: utf-8 -*-

import sys,os
DLIB_PATH = os.environ['DLIB_PATH']

sys.path.append(DLIB_PATH)

import dlib
import os
import numpy as NP
from math import atan, sin, cos, pi

from skimage import io
from scipy.misc import imresize,imrotate


#win = dlib.image_window()

import caffe
caffe.set_mode_cpu()



def rgb2gray(rgb):
    return NP.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def transform(x, y, ang, s0, s1):

    x0 = x - s0[1]/2;
    y0 = y - s0[0]/2;
    xx = x0*cos(ang) - y0*sin(ang) + s1[1]/2
    yy = x0*sin(ang) + y0*cos(ang) + s1[0]/2
    return xx,yy

class FaceProc(object):
    def __init__(self,PROJ_PATH):
        caffe_model_path = PROJ_PATH + '/models/face_desc_maxout/'
        dlib_model_shape = PROJ_PATH + '/models/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_model_shape)
        self.desc_name = 'desc'

        self.net = caffe.Net( caffe_model_path + 'LightenedCNN_A_deploy.prototxt',
                              caffe_model_path + 'LightenedCNN_A.caffemodel',
                              caffe.TEST )
                              
        self.outputs = {'eltwise6': {'k':self.desc_name}}


    def processImg(self,img0):
        if isinstance(img0,basestring):
            img0 = io.imread(img0)
	
        s = img0.shape
        maxSize = 1280.0
        f = max(s[0]/maxSize, s[1]/maxSize, s[2]/maxSize,1.0)
        img = imresize(img0, 1/f)
        dets = self.detector(img, 1)

        
        res = []
        for i, d in enumerate(dets):


            f5pt = self.get5Points(img,d)

            aligned0, a_res, rect_res = self.align(img, f5pt)

            aligned = NP.array([rgb2gray(aligned0)/255])
            in_ = self.net.inputs[0]
            out_ = 'blobs'

            out = self.net.forward_all( **{in_ :aligned, out_:self.outputs.keys()} )
            attr = self.get_net_output(out) 

            # we have faces!
            desc = attr.pop(self.desc_name) # store desc outside attributes
            res.append({'desc' : desc, 'rect' : d, 'aligned' : aligned[0]})            
 
        return res


    def get5Points(self,img,d):
        shape = self.predictor(img,d)
        res = {}
        sem = {'leftEye': [36,37,38,39,40,41], 'rightEye': [42,43,44,45,46,47], 'nose' : [31,32,33,34,35], 'leftMouth': [48,50,59,60], 'rightMouth': [53,54,55, 64]}
        for nm, idxs in sem.iteritems() : 
            x = 0
            y = 0
            for i in idxs :
                p = shape.part(i)
                x += p.x
                y += p.y
            res[nm] = NP.array( [float(x)/len(idxs), float(y)/len(idxs)], dtype='f4') 
        return NP.array([res['leftEye'], res['rightEye'], res['nose'], res['leftMouth'], res['rightMouth']])

    def align(self, img, f5pt):
        ec_mc_y = 48.0
        crop_size = 128
        ec_y = 40
        
        ang_tan = (f5pt[0,1]-f5pt[1,1])/(f5pt[0,0]-f5pt[1,0])
        ang = atan(ang_tan) / pi * 180
        img_rot = imrotate(img, ang, 'bicubic')


        # eye center
        x = (f5pt[0,0]+f5pt[1,0])/2;
        y = (f5pt[0,1]+f5pt[1,1])/2;

        ang = -ang/180*pi;
        xx,yy = transform(x,y,ang,img.shape,img_rot.shape)

        eyec = NP.array([xx, yy]).astype(long)

        x = (f5pt[3,0]+f5pt[4,0])/2;
        y = (f5pt[3,1]+f5pt[4,1])/2;
        [xx, yy] = transform(x, y, ang, img.shape, img_rot.shape)
        mouthc = NP.array([xx, yy]).astype(long)

        resize_scale = ec_mc_y/(mouthc[1]-eyec[1])

        img_resize = imresize(img_rot, resize_scale);


        eyec2 = eyec*resize_scale 
        eyec2 = NP.round(eyec2);
        img_crop = NP.zeros((crop_size, crop_size,3),dtype='uint8')
        # crop_y = eyec2(2) -floor(crop_size/3.0);
        crop_y = eyec2[1] - ec_y
        crop_y_end = crop_y + crop_size
        crop_x = eyec2[0]-int(crop_size/2)
        crop_x_end = crop_x + crop_size
        
        
        box = NP.array([crop_x, crop_x_end, crop_y, crop_y_end])
        box[box<0] = 0
        box[1] = min(box[1], img_resize.shape[1])
        box[3] = min(box[3], img_resize.shape[0])
        try:
            img_crop[(box[2]-crop_y):(box[3]-crop_y), (box[0]-crop_x):(box[1]-crop_x),:] = \
                 img_resize[box[2]:box[3],box[0]:box[1],:]
        except:
            print box, crop_y, crop_x, img_resize.shape
            raise

        return img_crop,img_resize,dlib.rectangle(int(box[0]), int(box[2]),int(box[1]),int(box[3]))          
    
     

    def get_net_output( self, prediction ):
        """
        Get top classes within a prediction vector 
        """
        ret = {}
        for k,v in prediction.iteritems():
            v = v.squeeze()
            if any( NP.isnan(v) ) :
                v[NP.isnan(v)] = 0
            if k in self.outputs:
                ret[self.outputs[k]['k']] = v
        return ret   


if __name__ == '__main__':
    PROJ_PATH = str(os.path.split(os.path.split(__file__)[0])[0])
    f =FaceProc(PROJ_PATH)
    f.processImg('F:/project/images/test/IMG_2628.JPg')
