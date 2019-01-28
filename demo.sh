file=$1 #give video file name as a argument (example.mp4).. don't give the path. Just the filename
root_path='/mnt/data/video-tagger-backend' #the path you execute.. in case at the end it will be directed to where it started
feat='/mnt/data/video-tagger-backend/features' #feature extraction happening in c3d-keras. Extract it to this path so that it can be read by DenseVideoCaptioning path
vid_path='/mnt/data/video-tagger-backend/uploads' #the absolute path of video file (dont include file name or last backslash)

cd /home/supunK/FYP/c3d-keras # give the cloned absolute path
source activate c3d
python demo.py -p $vid_path/$file -v $file -f $feat

cd /home/supunK/FYP/DenseVideoCaptioning # give the cloned absolute path
source activate dense
python demo.py -v $file -p $vid_path -f $feat 

source deactivate
cd $root_path
echo 'Done processing $file'
