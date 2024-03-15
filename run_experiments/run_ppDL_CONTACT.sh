echo "Running ppDL CONTACT training"
python ppDL.py -mth glinter -md train -d CONTACT -f 1,2,3,4,5 -c 1
echo "Running ppDL CONTACT test"
python ppDL.py -mth glinter -md test -d CONTACT -em test_set -f 1,2,3,4,5 -c 1
echo "Running ppDL CONTACT train"
python ppDL.py -mth glinter -md test -d CONTACT -em train_set -f 1,2,3,4,5 -c 1

#python ppDL.py -mth PInet -md train -d CONTACT -f 1,2,3,4,5 -c 0
#python ppDL.py -mth PInet -md test -d CONTACT -em test_set -f 1,2,3,4,5 -c 0
#python ppDL.py -mth PInet -md test -d CONTACT -em train_set -f 1,2,3,4,5 -c 0
