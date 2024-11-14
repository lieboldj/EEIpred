
for i in {1..5}
do
    python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold $i --training 1
done 

for i in {1..5}
do
    python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold $i
done