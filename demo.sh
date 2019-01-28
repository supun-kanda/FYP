file=$1
root_path='/mnt/data/video-tagger-backend'
feat='/mnt/data/video-tagger-backend/features'
vid_path='/mnt/data/video-tagger-backend/uploads'

cd /home/supunK/FYP/c3d-keras
source activate c3d
python demo.py -p $vid_path/$file -v $file -f $feat

cd /home/supunK/FYP/DenseVideoCaptioning
source activate dense
python demo.py -v $file -p $vid_path -f $feat 

source deactivate
cd $root_path
echo 'Done processing $file'
