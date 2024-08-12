from datasets import load_dataset
import json
a={"version": 1.0, "data": [{"id": "Hobbies and Crafts_C48_Q5","syntatic_triplets": [[3, "nsubj", 1], [3, "advmod", 2]]}]}
# a={"version": 1.0, "data": [{"id": "Hobbies and Crafts_C48_Q5","syntatic_triplets": [[3, 1], [3, 2]]}]}

b=json.dumps(a)
print(b)
c=json.loads(b)
print(c)
# print(load_dataset('json',data_files=b))
data_files = {'train': "test.json"}
raw_datasets = load_dataset('json', field="data",data_files=data_files)
print(raw_datasets)