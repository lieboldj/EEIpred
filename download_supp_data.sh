#!/bin/bash

# URL of the zip file to download
url_data="https://figshare.com/ndownloader/files/45074818"
url_pretrained="https://figshare.com/ndownloader/files/51580901"

# Directory to download the zip file into best here, same download_sup_data.sh
download_dir="."

# Download the zip file into the download directory
wget -O "$download_dir/datasets.zip" "$url_data"
wget -O "$download_dir/pre-trained_models.zip" "$url_pretrained"

# unzip the two files
# url_data is called "datasets.zip"
unzip "$download_dir/datasets.zip" -d "$download_dir"
unzip "$download_dir/pre-trained_models.zip" -d "$download_dir"

# remove the zip files
rm "$download_dir/datasets.zip"
rm "$download_dir/pre-trained_models.zip"

# move datasets/CONTACT, datasets/EPPIC, datasets/PISA to data_collection/cv_splits
mv "$download_dir/datasets/CONTACT" "$download_dir/data_collection/cv_splits/"
mv "$download_dir/datasets/EPPIC" "$download_dir/data_collection/cv_splits/"
mv "$download_dir/datasets/PISA" "$download_dir/data_collection/cv_splits/"

# create new folders
mkdir -p "$download_dir/ppDL_models/dmasif"
mkdir -p "$download_dir/ppDL_models/PInet"
mkdir -p "$download_dir/ppDL_models/glinter/CONTACT"
mkdir -p "$download_dir/ppDL_models/glinter/EPPIC"
mkdir -p "$download_dir/ppDL_models/glinter/PISA"
mkdir -p "$download_dir/ppDL_models/ProteinMAE"

mkdir -p "$download_dir/dmasif/models"
mkdir -p "$download_dir/PInet/old/seg"
# write for loop for i 1..5
for i in {1..5}
do
  mkdir -p "$download_dir/glinter/ckpts/CONTACT/train$i"
  mkdir -p "$download_dir/glinter/ckpts/EPPIC/train$i"
  mkdir -p "$download_dir/glinter/ckpts/PISA/train$i"
done

# move the pre-trained models to the correct directory for PPDL
mv "$download_dir/pre-trained_models/dmasif/CONTACT" "$download_dir/ppDL_models/dmasif/"
mv "$download_dir/pre-trained_models/dmasif/EPPIC" "$download_dir/ppDL_models/dmasif/"
mv "$download_dir/pre-trained_models/dmasif/PISA" "$download_dir/ppDL_models/dmasif/"

mv "$download_dir/pre-trained_models/PInet/CONTACT" "$download_dir/ppDL_models/PInet/"
mv "$download_dir/pre-trained_models/PInet/EPPIC" "$download_dir/ppDL_models/PInet/"
mv "$download_dir/pre-trained_models/PInet/PISA" "$download_dir/ppDL_models/PInet/"

mv "$download_dir/pre-trained_models/ProteinMAE/CONTACT" "$download_dir/ppDL_models/ProteinMAE/"
mv "$download_dir/pre-trained_models/ProteinMAE/EPPIC" "$download_dir/ppDL_models/ProteinMAE/"
mv "$download_dir/pre-trained_models/ProteinMAE/PISA" "$download_dir/ppDL_models/ProteinMAE/"

for i in {1..5}
do
  mv "$download_dir/pre-trained_models/glinter/CONTACT/fold"$i".pth" "$download_dir/ppDL_models/glinter/CONTACT/"
  mv "$download_dir/pre-trained_models/glinter/EPPIC/fold"$i".pth" "$download_dir/ppDL_models/glinter/EPPIC/"
  mv "$download_dir/pre-trained_models/glinter/PISA/fold"$i".pth" "$download_dir/ppDL_models/glinter/PISA/"
done

# move the pre-trained models to the correct directory for the 3 baseline methods
for i in {1..5}
do
  mv "$download_dir/pre-trained_models/dmasif/dMaSIF_search_3layer_12_"$i"_CONTACT" "$download_dir/dmasif/models/"
  mv "$download_dir/pre-trained_models/dmasif/dMaSIF_search_3layer_12_"$i"_EPPIC" "$download_dir/dmasif/models/"
  mv "$download_dir/pre-trained_models/dmasif/dMaSIF_search_3layer_12_"$i"_PISA" "$download_dir/dmasif/models/"
done
for i in {0..4}
do
  mv "$download_dir/pre-trained_models/PInet/seg_model__CONTACT_"$i"_best.pth" "$download_dir/PInet/old/seg/"
  mv "$download_dir/pre-trained_models/PInet/seg_model__EPPIC_"$i"_best.pth" "$download_dir/PInet/old/seg/"
  mv "$download_dir/pre-trained_models/PInet/seg_model__PISA_"$i"_best.pth" "$download_dir/PInet/old/seg/"
done

# move the pre-trained models to the correct directory for glinter
#write a for loop for i 1..5
for j in {1..5}
do
  mv "$download_dir/pre-trained_models/glinter/CONTACT/model_"$j"_best.pt" "$download_dir/glinter/ckpts/CONTACT/train$j/"
  mv "$download_dir/pre-trained_models/glinter/EPPIC/model_"$j"_best.pt" "$download_dir/glinter/ckpts/EPPIC/train$j/"
  mv "$download_dir/pre-trained_models/glinter/PISA/model_"$j"_best.pt" "$download_dir/glinter/ckpts/PISA/train$j/"
done

mv $download_dir/pre-trained_models/ProteinMAE/Transformer_search_batch32_pre_* $download_dir/ProteinMAE/search/models/

# remove the pre-trained models folder and datasets folder
rm -r "$download_dir/pre-trained_models"
rm -r "$download_dir/datasets"
