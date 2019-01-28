import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p','--path', required=True,help="path of the video file")
ap.add_argument('-v','--video', required=True,help="video name")
ap.add_argument('-f','--features', required=True,help="feature path")
args = vars(ap.parse_args())
path_vid = args['path']
vid = args['video'].split('.')[0]
f_path = args['features']

import pickle
import os
exists = os.path.isfile('%s/%s.srt'%(path_vid,vid))
if exists:
	print("Not Executing")
	exit()

"""
Main function for testing
"""

import numpy as np
import json
import h5py
from opt import *
from data_provider import *
from model import *

import tensorflow as tf
import sys
#import time as T
from time import strftime,gmtime

sys.path.insert(0, './densevid_eval_master')
sys.path.insert(0, './densevid_eval_master/coco_caption')
#from evaluator import *
#from evaluator_efficient import *
from evaluator_old import *
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

cap_count = 3
dick = []

reload(sys)
sys.setdefaultencoding('utf-8')

def getKey(item):
    return item['score']

"""
For joint ranking: combine proposal scores and caption confidence
"""
def getJointKey(item):
    return 10.*item['proposal_score'] + item['sentence_confidence']

"""
Generate batch data and corresponding mask data for the input
"""
def process_batch_data(batch_data, max_length):

    dim = batch_data[0].shape[1]

    out_batch_data = np.zeros(shape=(len(batch_data), max_length, dim), dtype='float32')
    out_batch_data_mask = np.zeros(shape=(len(batch_data), max_length), dtype='int32')

    for i, data in enumerate(batch_data):
        effective_len = min(max_length, data.shape[0])
        out_batch_data[i, :effective_len, :] = data[:effective_len]
        out_batch_data_mask[i, :effective_len] = 1

    out_batch_data = np.asarray(out_batch_data, dtype='float32')
    out_batch_data_mask = np.asarray(out_batch_data_mask, dtype='int32')

    return out_batch_data, out_batch_data_mask

def time_f(time):
	return strftime('%M:%S',gmtime(time))

def test(options):
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])[1:-1]
    sess = tf.InteractiveSession(config=sess_config)

    # build model
    #print('Building model ...')
    model = CaptionModel(options)
    
    proposal_inputs, proposal_outputs = model.build_proposal_inference(reuse=False)
    caption_inputs, caption_outputs = model.build_caption_greedy_inference(reuse=False)

    # print variable names
    #for v in tf.trainable_variables():
       # print(v.name)
       # print(v.get_shape())

    print('Restoring model from %s'%options['init_from'])
    saver = tf.train.Saver()
    saver.restore(sess, options['init_from'])

    word2ix = options['vocab']
    ix2word = {ix:word for word,ix in word2ix.items()}
    
    print('Start to predict ...')
    
    t0 = time.time()

    # output json data for evaluation
    #out_data = {}
    #out_data['version'] = 'VERSION 1.0'
    #out_data['external_data'] = {'used':False, 'details': ''}
    #out_data['results'] = {}
    #results = {}

    count = 0

    split = 'val'

    test_ids = json.load(open('dataset/ActivityNet_Captions/%s_ids.json'%split, 'r'))
    #features = h5py.File('dataset/ActivityNet/features/c3d/sub_activitynet_v1-3_stride_64frame.c3d.hdf5', 'r')
    #features = h5py.File('/mnt/data/c3d_features/feature.c3d.hdf5', 'r')
    #features = h5py.File('/mnt/data/c3d_features/stride_64frame/sub_activitynet_v1-3_stride_64frame.c3d.hdf5', 'r')
    stride = 8
    c3d_resolution = 16
    #frame_rates = json.load(open('dataset/ActivityNet_Captions/video_fps.json'))
    anchor_data = open('dataset/ActivityNet_Captions/preprocess/anchors/anchors.txt', 'r').readlines()
    anchors = [float(line.strip()) for line in anchor_data]

    #print('Will evaluate %d videos from %s set ...'%(len(test_ids), split))
    #vid = 'v__86X1xtj67w'
    #vid = 'v_0rDLcTmgzGQ'
    #for vid in test_ids:
    print('\nProcessed %s'%(vid))

    # video feature sequence 
    #video_feat_fw = features[vid]['c3d_shorter_features'].value
    #video_feat_fw = np.load('%s/%s.npy'%(f_path,vid)) 
    
    with open('%s/%s.pkl'%(f_path,vid), 'rb') as f:
    	load_dick = pickle.load(f)
    video_feat_fw = load_dick['features']
    fps = load_dick['fps']
    
    video_feat_bw = np.flip(video_feat_fw, axis=0)
    video_feat_fw = np.expand_dims(video_feat_fw, axis=0)
    video_feat_bw = np.expand_dims(video_feat_bw, axis=0)


    proposal_score_fw, proposal_score_bw, rnn_outputs_fw, rnn_outputs_bw = sess.run([proposal_outputs['proposal_score_fw'], proposal_outputs['proposal_score_bw'], proposal_outputs['rnn_outputs_fw'], proposal_outputs['rnn_outputs_bw']], feed_dict={proposal_inputs['video_feat_fw']:video_feat_fw, proposal_inputs['video_feat_bw']:video_feat_bw})
    
    
    feat_len = video_feat_fw[0].shape[0]
    #fps = frame_rates[vid]
    #duration = feat_len*c3d_resolution*stride / fps
    duration = feat_len*stride / fps


    '''calculate final score by summarizing forward score and backward score
    '''
    proposal_score = np.zeros((feat_len, options['num_anchors']))
    proposal_infos = []
    maxx = -1000 
    result = 0
    
    for i in range(feat_len):
        pre_start = -1.
        for j in range(options['num_anchors']):
            forward_score = proposal_score_fw[i,j]
            # calculate time stamp
            end = (float(i+1)/feat_len)*duration
            start = end-anchors[j]
            start = max(0., start)

            if start == pre_start:
                continue

            # backward
            end_bw = duration - start
            i_bw = min(int(round((end_bw/duration)*feat_len)-1), feat_len-1)
            i_bw = max(i_bw, 0)
            backward_score = proposal_score_bw[i_bw,j]

            proposal_score[i,j] = forward_score*backward_score
            
            hidden_feat = np.concatenate([rnn_outputs_fw[i], rnn_outputs_bw[i_bw]], axis=-1)
            
            proposal_feats = video_feat_fw[0][feat_len-1-i_bw:i+1]
            proposal_infos.append({'timestamp':[start, end], 'score': proposal_score[i,j], 'event_hidden_feats': hidden_feat, 'proposal_feats': proposal_feats})
                        
            pre_start = start
    
    # add the largest proposal
    hidden_feat = np.concatenate([rnn_outputs_fw[feat_len-1], rnn_outputs_bw[feat_len-1]], axis=-1)
    
    proposal_feats = video_feat_fw[0]
    proposal_infos.append({'timestamp':[0., duration], 'score': 1., 'event_hidden_feats': hidden_feat, 'proposal_feats': proposal_feats})


    proposal_infos = sorted(proposal_infos, key=getKey, reverse=True)
    proposal_infos = proposal_infos[:options['max_proposal_num']]

    #print('Number of proposals: %d'%len(proposal_infos))

    # 
    event_hidden_feats = [item['event_hidden_feats'] for item in proposal_infos]
    proposal_feats = [item['proposal_feats'] for item in proposal_infos]

    event_hidden_feats = np.array(event_hidden_feats, dtype='float32')
    proposal_feats, _ = process_batch_data(proposal_feats, options['max_proposal_len'])

    # word ids
    word_ids, sentence_confidences = sess.run([caption_outputs['word_ids'], caption_outputs['sentence_confidences']], feed_dict={caption_inputs['event_hidden_feats']: event_hidden_feats, caption_inputs['proposal_feats']: proposal_feats})
    
    sentences = [[ix2word[i] for i in ids] for ids in word_ids]
    sentences = [sentence[1:] for sentence in sentences]
    
    # remove <END> word
    out_sentences = []
    for sentence in sentences:
        if '<END>' in sentence:
            sentence = sentence[:sentence.index('<END>')]
        out_sentences.append(' '.join(sentence))

    
    print('Output sentences: ')
    #for out_sentence in out_sentences:
        #print(out_sentence.encode('utf-8'))


    result = [{'timestamp': proposal['timestamp'], 'proposal_score': float(proposal['score']), 'sentence': out_sentences[i], 'sentence_confidence': float(sentence_confidences[i])} for i, proposal in enumerate(proposal_infos)]

    # jointly ranking by proposal score and sentence confidence
    result = sorted(result, key=getJointKey, reverse=True)
    #print('Total confidence %f : %s '%(result[0]['sentence_confidence'],result[0]['sentence']))        
    #print(result[0]['timestamp'])        
    #results[vid] = result

    #count += 1
    #Check Time
    cap_file = open('%s/%s.srt'%(path_vid,vid), 'a')
    time_file =  open('%s/%s.tme'%(path_vid,vid), 'a')
    #act_file =  open('%s/%s.act'%(path_vid,vid), 'a')
    for i,e in enumerate(result):
        #time_file.write("%s---->%s"%(time_f(R['timestamp'][0]),time_f(R['timestamp'][1])))
	
        time_file.write('%s--->%s\n\n'%(time_f(e['timestamp'][0]),time_f(e['timestamp'][1])))
        cap_file.write('%s\n\n'%e['sentence'])
        #act_file.write('%s\n\n'%e['sentence_confidence'])
        if(i>cap_count-2):
        	break
    cap_file.close()
    time_file.close()
    print('Total running time: %f seconds.'%(time.time()-t0))

if __name__ == '__main__':

    options = default_options()
    test(options)
    
