import pickle
import json
import torch
import pdb
import numpy as np

with open('./hf_dataset_training.pickle', 'rb') as f:
	train = pickle.load(f)
with open('./hf_dataset_testing.pickle', 'rb') as f:
	test = pickle.load(f)
with open('./hf_dataset_validation.pickle', 'rb') as f:
	val = pickle.load(f)

train_list = []
train1, train0, test1, test0 = 0,0,0,0
for i in range(len(train[0])):
    temp = {}
    temp['data'] = [list(map(int, item)) for item in train[0][i] if len(item) > 0]
    temp['label'] = int(train[1][i])
    train_list.append(temp)

#val_list = []
#for i in range(len(val[0])):
#    temp = {}
#    temp['data'] = [list(map(int, item)) for item in val[0][i] if len(item) > 0]
#    temp['label'] = int(val[1][i])
#    val_list.append(temp)

test_list = []
for i in range(len(test[0])):
    temp = {}
    temp['data'] = [list(map(int, item)) for item in test[0][i] if len(item) > 0]
    temp['label'] = int(test[1][i])
    test_list.append(temp)

total_list = train_list + test_list
train_num = int(len(total_list) * 0.95)
test_num = int(len(total_list) * 0.05)
print(len(total_list))
train_list = total_list[:train_num]
test_list = total_list[train_num:train_num+test_num]

for item in train_list:
    if item['label'] == 1:
        train1 += 1
    elif item['label'] == 0:
        train0 += 1
with open('./test_data_0.05.json', 'r') as f:
    test_list = json.load(f)
for item in test_list:
    if item['label'] == 1:
        test1 += 1
    elif item['label'] == 0:
        test0 += 1
pdb.set_trace()
print(len(train_list), train1, train0)
print(len(test_list), test1, test0)
with open('./train_data_0.95.json', 'w') as f:
	json.dump(train_list, f)
#with open('./val_data.json', 'w') as f:
#	json.dump(val_list, f)
with open('./test_data_0.05.json', 'w') as f:
	json.dump(test_list, f)
'''
with open('./train_data.json','r') as f:
    train = json.load(f)
with open('./val_data.json','r') as f:
    val = json.load(f)
with open('./test_data.json','r') as f:
    test = json.load(f)

min = 100
max = 100
for item in test:
    for li in item['data']:
        for index in li:
            if index < min:
                min = index
            if index > max:
                max = index
print('min: ', min, 'max: ', max)

rank = 10
embedding = torch.load('./PretrainedEmbedding.4')['word_embeddings.weight']
emb1 = embedding.unsqueeze(1)
emb2 = embedding.unsqueeze(0)
rep = emb1.size(0)
embed1 = torch.tile(emb1, (1, rep, 1))
embed2 = torch.tile(emb2, (rep, 1, 1))
distance = -torch.linalg.norm(embed2 - embed1, dim=-1)

rank_result = {}
for i in range(rep):
    code = distance[i]
    ids = torch.sort(code, descending=True)[1][1:rank+1]
    rank_result[i+1] = (ids + 1).tolist()

with open('./' + str(rank) + '_code_index.json', 'w') as f:
    json.dump(rank_result, f)


with open('./train_data.json', 'r') as f:
    train = json.load(f)

visit20 = 0
visit40 = 0
visit60 = 0
visit80 = 0
visit_above = 0

code10 = 0
code20 = 0
code30 = 0
code40 = 0
code50 = 0
print(len(train))
for data in train:
    visit = data['data']
    len_visit = len(visit)
    if len_visit < 20:
        visit20 += 1
    elif len_visit < 40:
        visit40 += 1
    elif len_visit < 60:
        visit60 += 1
    elif len_visit < 80:
        visit80 += 1
    else:
        visit_above += 1
    
    max_code = 0
    for vi in visit:
        if len(vi) > max_code:
            max_code = len(vi)
    if max_code < 10:
        code10 += 1
    elif max_code < 20:
        code20 += 1
    elif max_code < 30:
        code30 += 1
    elif max_code < 40:
        code40 += 1
    else:
        code50 += 1
print('Visit <20: ', visit20, '<40: ', visit40, '<60: ', visit60, '<80: ', visit80, '>80: ', visit_above)
print('Code <10: ', code10, '<20: ', code20, '<30: ', code30, '<40: ', code40, '<50: ', code50)

torch.set_printoptions(precision=2, sci_mode=False)
with open('./train_data_random.json', 'r') as f:
    data = json.load(f)

result = torch.zeros((len(data), 7)) + 1e-5
for item in data:
    visit = item['data']
    label = item['label']
    
    t = []
    for v in visit:
        t.extend(v)
    s = list(set(t))
    for code in s:
        result[code-1, label] += 1

result[:, 2] = result[:, 0] / 3863.0
result[:, 3] = result[:, 1] / 1816.0
result[:, 4] = result[:, 0] / (result[:, 0] + result[:, 1] + 0.0)
result[:, 5] = result[:, 1] / (result[:, 0] + result[:, 1] + 0.0)
result[:, 6] = result[:, 5] / result[:, 4]

rank = torch.argsort(result[:, -1], descending=True) + 1
print(rank[:100])

with open('dis.txt', 'a+') as f:
    for item in result:
        f.write(str(item)+'\n')
#print(result)
'''
'''
sub_25 = []
sub_10 = []

count0, count1 = 0, 0
for item in data:
    if item['label'] == 0 and count0 < 966:
        count0 += 1
        sub_25.append(item)
    elif item['label'] == 1 and count1 < 454:
        count1 += 1
        sub_25.append(item)
    
count0, count1 = 0, 0
for item in data:
    if item['label'] == 0 and count0 < 387:
        count0 += 1
        sub_10.append(item)
    elif item['label'] == 1 and count1 < 182:
        count1 += 1
        sub_10.append(item)   
        
print(len(sub_25), len(sub_10))
with open('./train_data_1400.json', 'w') as f:
    json.dump(sub_25, f)
with open('./train_data_500.json', 'w') as f:
    json.dump(sub_10, f)
'''