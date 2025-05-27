#!/bin/bash

# URL of the zip file to download
url_data="https://figshare.com/ndownloader/files/45074818"
url_pretrained="https://figshare.com/ndownloader/files/51580901"
url_data_for_plots="added later"

# Directory to download the zip file into best here, same download_sup_data.sh
download_dir="."

# Download the zip file into the download directory
wget -O "$download_dir/datasets.zip" "$url_data"
wget -O "$download_dir/pre-trained.zip" "$url_pretrained"
wget -O "$download_dir/data_for_plots.zip" "$url_data_for_plots"

# unzip the two files
# url_data is called "datasets.zip"
unzip "$download_dir/datasets.zip" -d "$download_dir"
unzip "$download_dir/pre-trained.zip" -d "$download_dir"
unzip "$download_dir/data_for_plots.zip" -d "$download_dir"

# remove the zip files
rm "$download_dir/datasets.zip"
rm "$download_dir/pre-trained.zip"
rm "$download_dir/data_for_plots.zip"

# move the data_for_plots to analyse_results
mv "$download_dir/data_for_plots/"* "$download_dir/analyse_results/"

# move dataset/CLUST_CONTACT, dataset/CLUST_EPPIC, dataset/CLUST_PISA to data_collection/cv_splits
mv "$download_dir/dataset/CLUST_CONTACT" "$download_dir/data_collection/cv_splits/"
mv "$download_dir/dataset/CLUST_EPPIC" "$download_dir/data_collection/cv_splits/"
mv "$download_dir/dataset/CLUST_PISA" "$download_dir/data_collection/cv_splits/"
mv "$download_dir/dataset/BioGRID" "$download_dir/data_collection/cv_splits/"

# create new folders
mkdir -p "$download_dir/ppDL_models/dMaSIF"
mkdir -p "$download_dir/ppDL_models/PInet"
mkdir -p "$download_dir/ppDL_models/GLINTER/CLUST_CONTACT"
mkdir -p "$download_dir/ppDL_models/GLINTER/CLUST_EPPIC"
mkdir -p "$download_dir/ppDL_models/GLINTER/CLUST_PISA"
mkdir -p "$download_dir/ppDL_models/ProteinMAE"

mkdir -p "$download_dir/dmasif/models"
mkdir -p "$download_dir/PInet/old/seg"
# write for loop for i 1..5
for i in {1..5}
do
  mkdir -p "$download_dir/glinter/ckpts/CLUST_CONTACT/train$i"
  mkdir -p "$download_dir/glinter/ckpts/CLUST_EPPIC/train$i"
  mkdir -p "$download_dir/glinter/ckpts/CLUST_PISA/train$i"
done

# move the pre-trained models to the correct directory for PPDL
mv "$download_dir/pre-trained/dMaSIF/PPDL/CLUST_CONTACT" "$download_dir/ppDL_models/dMaSIF/"
mv "$download_dir/pre-trained/dMaSIF/PPDL/CLUST_EPPIC" "$download_dir/ppDL_models/dMaSIF/"
mv "$download_dir/pre-trained/dMaSIF/PPDL/CLUST_PISA" "$download_dir/ppDL_models/dMaSIF/"

mv "$download_dir/pre-trained/PInet/PPDL/CLUST_CONTACT" "$download_dir/ppDL_models/PInet/"
mv "$download_dir/pre-trained/PInet/PPDL/CLUST_EPPIC" "$download_dir/ppDL_models/PInet/"
mv "$download_dir/pre-trained/PInet/PPDL/CLUST_PISA" "$download_dir/ppDL_models/PInet/"

mv "$download_dir/pre-trained/ProteinMAE/PPDL/CLUST_CONTACT" "$download_dir/ppDL_models/ProteinMAE/"
mv "$download_dir/pre-trained/ProteinMAE/PPDL/CLUST_EPPIC" "$download_dir/ppDL_models/ProteinMAE/"
mv "$download_dir/pre-trained/ProteinMAE/PPDL/CLUST_PISA" "$download_dir/ppDL_models/ProteinMAE/"
mv "$download_dir/pre-trained/ProteinMAE/PPDL/BioGRID" "$download_dir/ppDL_models/ProteinMAE/"


for i in {1..5}
do
  mv "$download_dir/pre-trained/GLINTER/PPDL/CLUST_CONTACT/fold"$i".pth" "$download_dir/ppDL_models/GLINTER/CLUST_CONTACT/"
  mv "$download_dir/pre-trained/GLINTER/PPDL/CLUST_EPPIC/fold"$i".pth" "$download_dir/ppDL_models/GLINTER/CLUST_EPPIC/"
  mv "$download_dir/pre-trained/GLINTER/PPDL/CLUST_PISA/fold"$i".pth" "$download_dir/ppDL_models/GLINTER/CLUST_PISA/"
done

# move the pre-trained models to the correct directory for the 3 baseline methods
for i in {1..5}
do
  mv "$download_dir/pre-trained/dMaSIF/dMaSIF_search_3layer_12_"$i"_CLUST_CONTACT" "$download_dir/dmasif/models/"
  mv "$download_dir/pre-trained/dMaSIF/dMaSIF_search_3layer_12_"$i"_CLUST_EPPIC" "$download_dir/dmasif/models/"
  mv "$download_dir/pre-trained/dMaSIF/dMaSIF_search_3layer_12_"$i"_CLUST_PISA" "$download_dir/dmasif/models/"
done
for i in {0..4}
do
  mv "$download_dir/pre-trained/PInet/seg_model__CLUST_CONTACT_"$i"_best.pth" "$download_dir/PInet/old/seg/"
  mv "$download_dir/pre-trained/PInet/seg_model__CLUST_EPPIC_"$i"_best.pth" "$download_dir/PInet/old/seg/"
  mv "$download_dir/pre-trained/PInet/seg_model__CLUST_PISA_"$i"_best.pth" "$download_dir/PInet/old/seg/"
done

# move the pre-trained models to the correct directory for glinter
#write a for loop for i 1..5
for j in {1..5}
do
  mv "$download_dir/pre-trained/GLINTER/CLUST_CONTACT/model_"$j"_best.pt" "$download_dir/glinter/ckpts/CLUST_CONTACT/train$j/"
  mv "$download_dir/pre-trained/GLINTER/CLUST_EPPIC/model_"$j"_best.pt" "$download_dir/glinter/ckpts/CLUST_EPPIC/train$j/"
  mv "$download_dir/pre-trained/GLINTER/CLUST_PISA/model_"$j"_best.pt" "$download_dir/glinter/ckpts/CLUST_PISA/train$j/"
done

mv $download_dir/pre-trained/ProteinMAE/TS_32_* $download_dir/ProteinMAE/search/models/

# remove the pre-trained models folder and datasets folder
rm -r "$download_dir/pre-trained"
rm -r "$download_dir/dataset"
