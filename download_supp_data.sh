#!/bin/bash

# URL of the zip file to download
url_data="https://figshare.com/ndownloader/files/45074818"
url_pretrained="https://figshare.com/ndownloader/files/45074824"

# Directory where you want to save the downloaded zip file
download_dir="./"

# Download the zip file
wget "$url_data" -P "$download_dir"
wget "$url_pretrained" -P "$download_dir"

# unzip the two files
# url_data is called "datasets.zip"
unzip "$download_dir/datasets.zip" -d "$download_dir"
unzip "$download_dir/pre-trained_models.zip" -d "$download_dir"

# remove the zip files
rm "$download_dir/datasets.zip"
rm "$download_dir/pre-trained_models.zip"
# create new folders
mkdir -p "$download_dir/ppDL_models/dmasif"
mkdir -p "$download_dir/ppDL_models/PInet"
mkdir -p "$download_dir/ppDL_models/glinter"

mkdir -p "$download_dir/dmasif/models"
mkdir -p "$download_dir/PInet/old/seg"
# write for loop for i 1..5
for i in {1..5}
do
  mkdir -p "$download_dir/glinter/ckpts/CONTACT/train$i"
done

# move the pre-trained models to the correct directory for PPDL
mv "$download_dir/pre-trained_models/dmasif/CONTACT" "$download_dir/ppDL_models/dmasif/"
mv "$download_dir/pre-trained_models/dmasif/EPPIC" "$download_dir/ppDL_models/dmasif/"
mv "$download_dir/pre-trained_models/dmasif/PISA" "$download_dir/ppDL_models/dmasif/"

mv "$download_dir/pre-trained_models/PInet/CONTACT" "$download_dir/ppDL_models/PInet/"
mv "$download_dir/pre-trained_models/PInet/EPPIC" "$download_dir/ppDL_models/PInet/"
mv "$download_dir/pre-trained_models/PInet/PISA" "$download_dir/ppDL_models/PInet/"

mv "$download_dir/pre-trained_models/glinter/CONTACT/fold*.pth" "$download_dir/ppDL_models/glinter/"
mv "$download_dir/pre-trained_models/glinter/EPPIC/fold*.pth" "$download_dir/ppDL_models/glinter/"
mv "$download_dir/pre-trained_models/glinter/PISA/fold*.pth" "$download_dir/ppDL_models/glinter/"

# move the pre-trained models to the correct directory for the 3 baseline methods
mv "$download_dir/pre-trained_models/dmasif/*_CONTACT" "$download_dir/dmasif/models/"
mv "$download_dir/pre-trained_models/dmasif/*_EPPIC" "$download_dir/dmasif/models/"
mv "$download_dir/pre-trained_models/dmasif/*_PISA" "$download_dir/dmasif/models/"

mv "$download_dir/pre-trained_models/PInet/*_CONTACT*.pth" "$download_dir/PInet/old/seg/"
mv "$download_dir/pre-trained_models/PInet/*_EPPIC*.pth" "$download_dir/PInet/old/seg/"
mv "$download_dir/pre-trained_models/PInet/*_PISA*.pth" "$download_dir/PInet/old/seg/"

# move the pre-trained models to the correct directory for glinter
#write a for loop for i 1..5
for j in {1..5}
do
  mv "$download_dir/pre-trained_models/glinter/CONTACT/model_"$j"_best.pt" "$download_dir/glinter/ckpts/CONTACT/train$j/"
  mv "$download_dir/pre-trained_models/glinter/EPPIC/model_"$j"_best.pt" "$download_dir/glinter/ckpts/EPPIC/train$j/"
    mv "$download_dir/pre-trained_models/glinter/PISA/model_"$j"_best.pt" "$download_dir/glinter/ckpts/PISA/train$j/"
done

# remove the pre-trained models folder and datasets folder
rm -r "$download_dir/pre-trained_models"
rm -r "$download_dir/datasets"
