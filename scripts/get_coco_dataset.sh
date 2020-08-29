#!/bin/bash

# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

# Download Images
<<<<<<< HEAD
wget -c https://pjreddie.com/media/files/train2014.zip
wget -c https://pjreddie.com/media/files/val2014.zip
=======
#very slow downloading
#wget -c https://pjreddie.com/media/files/train2014.zip
#wget -c https://pjreddie.com/media/files/val2014.zip
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip

cd ..

# Download COCO Metadata
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

