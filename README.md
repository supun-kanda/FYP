# FYP
The project is about Large scale activity recognition

c3d-keras is from 
https://github.com/axon-research/c3d-keras.git

DenseVideoCaptioning is from
https://github.com/JaywongWang/DenseVideoCaptioning.git


You can find the missing files there.

#BEFORE YOU DO ANYTHING

In the demo.sh the 2 different conda environments got activated which are 'c3d' and 'dense'. You can install them using the saved files which are located. This will install all the dependancies automatically so that you don't have to install them one by one.

FYP/c3d-keras/c3d.yml

FYP/DenseVideoCaptioning/dense.yml



you can install them using the below command

conda env create -f environment.yml (change accordingly to c3d.yml and dense.yml)


The results will be saved in the video_path as an srt file. Only the top 3 scored senteces will be printed. But there is an issue with c3d feature extraction which will give wrong output weights as it turns out the generating captions are false sometimes and I coudn't figure out why. So anyone is welcome for a pull request.

Sports 1m Model fc7 layer (4096D) is used.We have used ActivityNet Dataset. So we have reduced the dimension to 500D using the given PCA eigen vectors. The PCA metrice files are available on ActivityNet site

http://activity-net.org/challenges/2016/download.html

PCA weights: http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/PCA_activitynet_v1-3.hdf5


I don't think the trained model for DenseVideoCaptioning model is available online. So you will have to train the model to test the videos on it. The instructions are well given in the previously mentioned DenseVideoCaptionin link
