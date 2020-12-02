import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import time
import csv
from skimage import morphology
import glob
from shutil import copyfile

# convert matlab model to onnx and then to tensorflow
# ------------------------------------------------------------------------------
# # in matlab
# mode_name = 'Retrain1_alexnet_DSTvST_same_minBS100_maxEp50_learnRT0.0001_2'
# modeldir = '/Users/eexna/Work/NERC Volcano/Results/'
# load([modeldir, modelname, '.mat']);
# exportONNXNetwork(netFineTune, [modeldir, modelname, '.onnx']);
# ------------------------------------------------------------------------------

# Define flags
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "/gws/nopw/j04/nceo_geohazards_vol1/projects/LiCS/volc-proc/detection/positives/", "path to wrapped interf")
tf.flags.DEFINE_string("out_dir", "/gws/nopw/j04/nceo_geohazards_vol1/projects/LiCS/volc-proc/detection/retrain/", "path to output of detection")
tf.flags.DEFINE_string("model_name", "Retrain1_alexnet_DSTvST_same_minBS100_maxEp50_learnRT0.0001_2.pd", "path to output of detection")
tf.flags.DEFINE_integer("start_frame", 0, "First frame to process")
tf.flags.DEFINE_integer("skip_frames", 20, "Process every skip_frames frame")
tf.flags.DEFINE_bool("output_all", False, "Write the prob and rgb for all imaages, including those P < 0.4")

DATA_DIR="/gws/nopw/j04/nceo_geohazards_vol1/projects/LiCS/volc-proc/current/"
OUT_DIR ="/gws/nopw/j04/nceo_geohazards_vol1/projects/LiCS/volc-proc/detection/positives_new/"
POSITIVE_DIR="/gws/nopw/j04/nceo_geohazards_vol1/projects/LiCS/volc-proc/detection/positives/"

start = time.time()
# model, input and output locations
model_name = FLAGS.model_name # "Retrain1_alexnet_DSTvST_same_minBS100_maxEp50_learnRT0.0001_2.pd"
wrappedinterf_path = FLAGS.data_dir
checkpoint_path = "tfmodels/"
rgbprobmap_path = FLAGS.out_dir + model_name[:-3] 
probmap_path = rgbprobmap_path + "/probMap/"
start_frame = FLAGS.start_frame
skip_frames = FLAGS.skip_frames
output_all = FLAGS.output_all

if not os.path.exists(rgbprobmap_path + '/'):
    os.mkdir(rgbprobmap_path + '/')


if not os.path.exists(probmap_path):
    os.mkdir(probmap_path)

# # read onnx model
# ------------------------------------------------------------------------------
# in Matlab
# load('D:\VolcanicUnrest\LiCS\model_retrain\RetrainLiCS.mat','netFineTune')
# exportONNXNetwork(netFineTune,'D:\VolcanicUnrest\LiCS\model_retrain\RetrainLiCS.onnx');
# in Python
# import onnx
# from onnx_tf.backend import prepare
# onnx_model = onnx.load(checkpoint_path + model_name + ".onnx")  # load onnx model
# tf_rep = prepare(onnx_model)  # prepare tf representation
# tf_rep.export_graph(checkpoint_path + model_name + ".pd") 
# ------------------------------------------------------------------------------

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
image_dir = wrappedinterf_path # os.path.join(current_dir, wrappedinterf_path)

#get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('diff_pha.tif')]
print(image_dir + ' ' + str(len(img_files)))
# create subfolders
if not output_all:
    for k in np.arange(0.4,1,0.1):
        if not os.path.exists(rgbprobmap_path + '/' + '%.1f' % k + '/'):
            os.mkdir(rgbprobmap_path + '/' + '%.1f' % k + '/')

endt = time.time()
# print("time elapsed:" + str(endt - start))

##############################################################################################
start = time.time()
# crop input size
overlapRatio = 1./4.
hpatch = 227
wpatch = 227
hgap = int(float(hpatch)*overlapRatio)
wgap = int(float(wpatch)*overlapRatio)

# Create weight for each patch
a = norm(hpatch/2, hpatch/6).pdf(np.arange(hpatch))
b = norm(wpatch/2, wpatch/6).pdf(np.arange(wpatch))
wmap = np.matmul(a[np.newaxis].T,b[np.newaxis])
wmap = wmap/wmap.sum()

# Predict each InSAR image
with tf.compat.v1.Session() as sess:

    with tf.io.gfile.GFile(checkpoint_path + model_name, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    x = sess.graph.get_tensor_by_name('data:0')
    out = sess.graph.get_tensor_by_name('softmax:0')

    path = os.path.normpath(wrappedinterf_path)
    folderlist = path.split(os.sep)
    savename = folderlist[-3] + '_' + folderlist[-2] + '_prob' + str(start_frame) + '.csv'
    file=open(os.path.join(rgbprobmap_path, savename), 'w', newline='')
    writer = csv.writer(file)
    # 
    for i, f in enumerate(img_files[start_frame::skip_frames]): 
        print(i)
        image = cv2.imread(f, -1)
        # mask for nan pixels
        mask  = image == 0
        seedmask = morphology.disk(5)
        mask  = morphology.binary_closing(~mask, seedmask)
        # convert to grayscale
        image = (image + np.pi)/(2*np.pi)*255
        image = np.dstack((image, image, image))
        
        # Subtract the ImageNet mean
        img = image - imagenet_mean
        
        # Run through each overlaping patch
        himg = img.shape[0]
        wimg = img.shape[1]
        weightMap = np.zeros((himg,wimg),np.float32) + 0.00001
        probMap = np.zeros((himg,wimg),np.float32)
        for starty in np.concatenate((np.arange( 0, himg-hpatch, hgap),np.array([himg-hpatch])),axis=0):
            for startx in np.concatenate((np.arange( 0, wimg-wpatch, wgap),np.array([wimg-wpatch])),axis=0):
                crop_img = img[starty:starty+hpatch, startx:startx+wpatch]
                curmask = mask[starty:starty+hpatch, startx:startx+wpatch]

                weightMap[starty:starty+hpatch, startx:startx+wpatch] += wmap
                
                testimg = crop_img + imagenet_mean
                testimg[testimg!=0.] = 1.
                
                if ((testimg.sum()/hpatch/wpatch/3) > 0.5) and ((curmask.sum()/hpatch/wpatch) > 0.25):
                    
                    # Reshape as needed to feed into model
                    crop_img = np.transpose(crop_img, (2, 0, 1))
                    crop_img = crop_img.reshape((1,3, 227,227))
                    
                    # Run the session and calculate the class probability
                    probs = sess.run(out, feed_dict={x: crop_img})
                    
                    # Put in prob map
                    # if (starty>0) and (startx>0):
                    if np.isnan(probs[0,0]):
                        probs[0,0] = 0.95
                        
                    probMap[starty:starty+hpatch, startx:startx+wpatch] += probs[0,0]*wmap*(testimg.sum()/hpatch/wpatch/3) 
                    
                    # Show class name and prob in the title
                    # probtext = "Class: " + "%d" %np.argmax(probs) + ", probability: %.4f" %probs[0,1]
                    # print(probtext)
                    
        # Normalised weight
        probMap /= weightMap
        
        # record max prob
        filename = os.path.basename(f)
        writer.writerow([filename[:len(filename[1])-5],probMap.max()])

        # Overlay probmap on interferogram
        if (probMap.max() > 0.4) or output_all:
            im_scale = image/255.
            im_scale[:,:,2] = im_scale[:,:,2]*(1-probMap) + probMap
            im_scale[:,:,1] = im_scale[:,:,1]*(1-probMap) + probMap
            # Draw contour of high prob
            psbound = np.logical_and(probMap>0.5,probMap<0.525)
            im_scale[:,:,2] -= psbound
            im_scale[:,:,1] = im_scale[:,:,1]*(1-psbound) + 0.5*psbound
            im_scale[:,:,0] = im_scale[:,:,0]*(1-psbound) + 0.75*psbound
            psbound = np.logical_and(probMap>0.8,probMap<0.825)
            im_scale[:,:,0] -= psbound
            im_scale[:,:,1] += psbound
            im_scale[:,:,2] -= psbound
            # Cap values
            im_scale[im_scale<0] = 0.
            im_scale[im_scale>1] = 1.
            
            # Save image result
            if not output_all:
                subfolder = np.floor(probMap.max()*10)/10
                cv2.imwrite(rgbprobmap_path + '/' + '%.1f' % subfolder  + '/' +  "rgbprobMap_" + filename[:len(filename[1])-4] + "jpg", im_scale*255.)
            else:
                cv2.imwrite(rgbprobmap_path + '/' +  "rgbprobMap_" + filename[:len(filename[1])-4] + "jpg", im_scale*255.)

            if probMap.max() > 0.4: # 0.79:
                subname = f.split("/")
                # find if alread positve
                interfname = folderlist[-3] + '_' + folderlist[-2] + '_' + filename
                text_files = glob.glob(os.path.join(POSITIVE_DIR,interfname))
                if len(text_files) <= 0:
                    copyfile(f, os.path.join(OUT_DIR, interfname))

            cv2.imwrite(probmap_path  +  "probMap_" + filename[:len(filename[1])-4] + "jpg", probMap*255.)
            print(f)
        
        # Create visualise prob map with interferogram
        # fig.add_subplot(1,5,i+1)
        # plt.imshow(probMap + image[:,:,0]/255)
        # plt.title("Max prob of deform: %.4f" %probMap.max())
        # plt.axis('off')

endt = time.time()
print("time elapsed:" + str(endt - start))


