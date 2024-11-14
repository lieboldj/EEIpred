import os
mode = "test"
dataset = "PISA"
for fold in range(1,6):
    with open(f"/cosybio/project/EEIP/EEIP/data_collection/cv_splits/{dataset}/{mode}{fold}.txt", "r") as f:
        train = f.readlines()
    with open(f"/cosybio/project/EEIP/EEIP/data_collection/cv_splits/{dataset}/{mode}_info{fold}.txt", "r") as f:
        train_info = f.readlines()
    new_train = []
    new_info = []
    for i in range(len(train)):
        pair = train[i].split("_")
        infos = train_info[i].split("\t")
        if not os.path.exists(f"/cosybio/project/EEIP/EEIP/glinter/examples/PDB/{pair[0]}_{pair[1]}:{pair[0]}_{pair[2][:-1]}/"):
            if os.path.exists(f"/cosybio/project/EEIP/EEIP/glinter/examples/PDB/{pair[0]}_{pair[2][:-1]}:{pair[0]}_{pair[1]}/"):
                # copy the folder and change the name
                new_name = pair[0] + "_" + pair[2][:-1] + "_" + pair[1] + "\n"
                new_train.append(new_name)
                new_info_name = infos[1][:-1]+ "\t" + infos[0] + "\n"
                new_info.append(new_info_name)
                #print(pair[0], pair[1], pair[2])

            else: 
                print("ERROR", pair[0], pair[1], pair[2])
        else:
            new_train.append(train[i])
            new_info.append(train_info[i])

    print(len(new_train), new_train)
    with open(f"/cosybio/project/EEIP/EEIP/data_collection/cv_splits/{dataset}/{mode}{fold}_glinter.txt", "w") as f:
        f.writelines(new_train)
    with open(f"/cosybio/project/EEIP/EEIP/data_collection/cv_splits/{dataset}/{mode}_info{fold}_glinter.txt", "w") as f:
        f.writelines(new_info)
