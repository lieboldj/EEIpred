
for i in {1..1}
do
    echo "Training fold $i"
    python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold $i --training 1
done
