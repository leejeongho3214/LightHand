import json
import pickle

with open("../../../../data1/ArmoHand/training/revision_data.json", "r") as f:
    json_file = json.load(f)
    
with open("../../../../data1/ArmoHand/training/revision_data.pkl", "wb") as f:
    pickle.dump(json_file, f, protocol=pickle.HIGHEST_PROTOCOL)
    
print()
    
    

    