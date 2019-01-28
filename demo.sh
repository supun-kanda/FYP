file=$1
feat='/mnt/data/video-tagger-backend/features'
cd /home/supunK/GIT/c3d-keras
source activate c3d
python demo.py -p /mnt/data/video-tagger-backend/uploads/$file -v $file -f $feat

cd /home/supunK/GIT/DenseVideoCaptioning
source activate dense
python demo.py -v $file -p /mnt/data/video-tagger-backend/uploads -f $feat 

source deactivate
cd /mnt/data/video-tagger-backend
echo 'Done processing $file'
