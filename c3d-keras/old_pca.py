#pca.py
import argparse
 
ap = argparse.ArgumentParser()
ap.add_argument('-p','--path', required=True,help="path of the video file")
args = vars(ap.parse_args())
path_vid = args['path']
import h5py
import numpy as np
file = './models/PCA_activitynet_v1-3.hdf5'
f = h5py.File(file, 'r')
# List all groups
#print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
# Get the data
data = list(f[a_group_key])
##print(list(f['data'].keys()))

d1 = f['data']
#list(d1)

np.array((d1['U'])).shape
mean = d1['x_mean']
U = np.array(d1['U'])*mean
PCAs = np.array(d1['U'])[:,0:500]
PCAs = PCAs #+ mean[:,0:500]
#exit()
import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json
import os
import cv2
import matplotlib.pyplot as plt
import c3d_model
import sys
import keras.backend as K
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
dim_ordering = K.image_dim_ordering()
#print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(dim_ordering)
backend = dim_ordering

def diagnose(data, verbose=True, label='input', plots=False, backend='tf'):
    # Convolution3D?
    if data.ndim > 2:
        if backend == 'th':
            data = np.transpose(data, (1, 2, 3, 0))
        #else:
        #    data = np.transpose(data, (0, 2, 1, 3))
        min_num_spatial_axes = 10
        max_outputs_to_show = 3
        ndim = data.ndim
        #print "[Info] {}.ndim={}".format(label, ndim)
        #print "[Info] {}.shape={}".format(label, data.shape)
        for d in range(ndim):
            num_this_dim = data.shape[d]
            if num_this_dim >= min_num_spatial_axes: # check for spatial axes
                # just first, center, last indices
                range_this_dim = [0, num_this_dim/2, num_this_dim - 1]
            else:
                # sweep all indices for non-spatial axes
                range_this_dim = range(num_this_dim)
            for i in range_this_dim:
                new_dim = tuple([d] + range(d) + range(d + 1, ndim))
                sliced = np.transpose(data, new_dim)[i, ...]
                #print("[Info] {}, dim:{} {}-th slice: ""(min, max, mean, std)=({}, {}, {}, {})".format(label,d, i,np.min(sliced),np.max(sliced),np.mean(sliced),np.std(sliced)))
        if plots:
            # assume (l, h, w, c)-shaped input
            if data.ndim != 4:
                #print("[Error] data (shape={}) is not 4-dim. Check data".format(data.shape))
                return
            l, h, w, c = data.shape
            if l >= min_num_spatial_axes or                 h < min_num_spatial_axes or                 w < min_num_spatial_axes:
                #print("[Error] data (shape={}) does not look like in (l,h,w,c) ""format. Do reshape/transpose.".format(data.shape))
                return
            nrows = int(np.ceil(np.sqrt(data.shape[0])))
            # BGR
            if c == 3:
                for i in range(l):
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                    im = np.squeeze(data[i, ...]).astype(np.float32)
                    im = im[:, :, ::-1] # BGR to RGB
                    # force it to range [0,1]
                    im_min, im_max = im.min(), im.max()
                    if im_max > im_min:
                        im_std = (im - im_min) / (im_max - im_min)
                    else:
                        #print "[Warning] image is constant!"
                        im_std = np.zeros_like(im)
                    plt.imshow(im_std)
                    plt.axis('off')
                    plt.title("{}: t={}".format(label, i))
                plt.show()
                #plt.waitforbuttonpress()
            else:
                for j in range(min(c, max_outputs_to_show)):
                    for i in range(l):
                        mng = plt.get_current_fig_manager()
                        mng.resize(*mng.window.maxsize())
                        plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                        im = np.squeeze(data[i, ...]).astype(np.float32)
                        im = im[:, :, j]
                        # force it to range [0,1]
                        im_min, im_max = im.min(), im.max()
                        if im_max > im_min:
                            im_std = (im - im_min) / (im_max - im_min)
                        else:
                            #print "[Warning] image is constant!"
                            im_std = np.zeros_like(im)
                        plt.imshow(im_std)
                        plt.axis('off')
                        plt.title("{}: o={}, t={}".format(label, j, i))
                    plt.show()
                    #plt.waitforbuttonpress()
    elif data.ndim == 1:
        #print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
                      #label,
                      #np.min(data),
                      #np.max(data),
                      #np.mean(data),
                      #np.std(data)))
        #print("[Info] data[:10]={}".format(data[:10]))

    	return



show_images = False
diagnose_plots = False
model_dir = '/home/supunK/GIT/c3d-keras/models'
global backend

# override backend if provided as an input arg
'''
if len(sys.argv) > 1:
    if 'tf' in sys.argv[1].lower():
        backend = 'tf'
    else:
        backend = 'th'
'''
backend = 'tf'
#print "[Info] Using backend={}".format(backend)


if backend == 'th':
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_th.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_th.json')
else:
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

#print("[Info] Reading model architecture...")
model = model_from_json(open(model_json_filename, 'r').read())
#model = c3d_model.get_model(backend=backend)

print('test')
# visualize model
model_img_filename = os.path.join(model_dir, 'c3d_model.png')
if not os.path.exists(model_img_filename):
    from keras.utils import plot_model
    plot_model(model, to_file=model_img_filename)

#print("[Info] Loading model weights...")
model.load_weights(model_weight_filename)
#print("[Info] Loading model weights -- DONE!")
model.compile(loss='mean_squared_error', optimizer='sgd')

print('test')
cap = cv2.VideoCapture(path_vid)

fps = cap.get(cv2.CAP_PROP_FPS)
vid = []
while True:
    ret, img = cap.read()
    if not ret:
        break
    vid.append(cv2.resize(img, (171, 128)))
vid = np.array(vid, dtype=np.float32)
#plt.imshow(vid[2000]/256)
#plt.show()
# sample 16-frame clip
#start_frame = 100
#start_frame = 2000
start = 0
end = vid.shape[0]
print('total number of frames are %d'%end)
#vid_feat = [] 
#count = 1

print('test')
vid_feat = np.zeros((1,500)) 
mean_cube = np.load('models/train01_16_128_171_mean.npy')
mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
sys.path.insert(0,'/home/supunK/GIT/c3d-keras/models' )
vid_array = []
if(end/10.0>1):
	period = int(end/10)
else:
	period = 1

period = 16

print('test')
#for start_frame in range(start,end,64):
for start_frame in range(start,end,period):
	print('frame no: %d'%start_frame)
	X = vid[start_frame:(start_frame + 16), :, :, :]
	#if(count==3):
	#	break
	#else:
	#	count+=1
	if(X.shape[0]<16):
		continue
	# subtract mean
	#print(mean_cube.shape)
	#diagnose(mean_cube, verbose=True, label='Mean cube', plots=show_images)
	X -= mean_cube
	#diagnose(X, verbose=True, label='Mean-subtracted X', plots=show_images)
	
	# center crop
	X = X[:, 8:120, 30:142, :] # (l, h, w, c)
	#diagnose(X, verbose=True, label='Center-cropped X', plots=show_images)
	
	if backend == 'th':
	    X = np.transpose(X, (3, 0, 1, 2)) # input_shape = (3,16,112,112)
	else:
	    pass                              # input_shape = (16,112,112,3)
	
	# get activations for intermediate layers if needed
	inspect_layers = [
	#    'fc6',
	    'fc7',
	    ]
	for layer in inspect_layers:
	    int_model = c3d_model.get_int_model(model=model, layer=layer, backend=backend)
	    int_output = int_model.predict_on_batch(np.array([X]))
	    int_output = int_output[0, ...]
	    #print "[Debug] at layer={}: output.shape={}".format(layer, int_output.shape)
	    diagnose(int_output,
	             verbose=True,
	             label='{} activation'.format(layer),
	             plots=diagnose_plots,
	             backend=backend)
	
	
	new_out = int_output.reshape(1,-1)
	
	reduced = np.dot(new_out,PCAs)
	#vid_feat = [vid_feat,[reduced]]
	#print(vid_feat)
	vid_array.append(reduced)

save = np.array(vid_array).reshape(-1,500)
#save = np.array(vid_feat)
print(save.shape)
#np.save('/home/supunK/GIT/feats.npy', save) 
np.save('/mnt/data/video-tagger-backend/video_attributes/feats.npy', save) 
#np.save('/home/supunK/GIT/fps.npy', fps) 
np.save('/mnt/data/video-tagger-backend/video_attributes/fps.npy', fps) 
#np.save('/home/supunK/GIT/period.npy', period) 
#np.save('/home/supunK/GIT/period.npy', period) 
##print(reduced.shape)

