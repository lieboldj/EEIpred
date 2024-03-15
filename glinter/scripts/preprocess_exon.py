#%%
import os
import shutil
from tqdm import tqdm
# %%
# change path to your txt file with protein pairs
all_unprocessed_pairs = "../../data_collection/all_unprocessed_pairs.txt" 
with open (all_unprocessed_pairs, "r") as f:
    lines = f.read().splitlines()
# %%
def find_and_copy_folder(folder_name, directory='.'):
    for root, dirs, files in os.walk('.'):
        if folder_name in dirs:
            source_folder = os.path.join(root, folder_name)
            shutil.copytree(source_folder, directory+folder_name)
            print(f"Folder '{folder_name}' copied to current directory")
        break
def remove_empty_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist")
        return
    
    if not os.path.isdir(folder_path):
        print(f"'{folder_path}' is not a directory")
        return
    
    if not os.listdir(folder_path):
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' removed successfully")
    else:
        print(f"Folder '{folder_path}' is not empty")
#%%
counter = 0
for line in tqdm(lines):

    if os.path.exists(f"examples/PDB/{line[:5]+ line[12]}:{line[:5]+ line[21]}"):
        if os.path.exists(f"examples/PDB/{line[:5]+ line[12]}:{line[:5]+ line[21]}/"\
                          +f"{line[:5]+ line[12]}:{line[:5]+ line[21]}.pkl"):
            counter += 1
            
        #remove_empty_folder(f"examples/PDB/{line[:5]+ line[12]}:{line[:5]+ line[21]}/")
            print(f"examples/PDB/{line[:5]+ line[12]}:{line[:5]+ line[21]}")
        continue
    else:
        os.mkdir(f"examples/PDB/{line[:5]+ line[12]}:{line[:5]+ line[21]}")

        find_and_copy_folder(line[:5]+ line[12], f"examples/PDB/{line[:5]+ line[12]}:{line[:5]+ line[21]}/")
        if not line[:5]+ line[12] == line[:5]+ line[21]:  
            find_and_copy_folder(line[:5]+ line[21], f"examples/PDB/{line[:5]+ line[12]}:{line[:5]+ line[21]}/")
    
    os.system(f"bash scripts/build_hetero.sh examples/PDB/{line[:5]+ line[12]}.pdb examples/PDB/{line[:5]+ line[21]}.pdb examples/PDB/")

# %%