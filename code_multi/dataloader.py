import torch
import os
import pdb
import json
import numpy as numpy

import torch.utils.data as data

def multi_hot_to_ori(args, new_data):
    data = new_data.squeeze(0)
    
    if args.dataset == 'MALWARE':
        out = [(torch.nonzero(data-1e-5)+1).squeeze(1).tolist()]
    elif args.dataset == 'IPS':
        out = (torch.nonzero(data.permute(1,0)-1e-5)[:,1]+1).tolist()
    else:
        out = []
        for item in data:
            temp = (torch.nonzero(item-1e-5)+1).squeeze(1).tolist()#-1e-5
            if len(temp) == 0:
                break
            else:
                out.append(temp)
    return out
    
class Dataset(data.Dataset):

    def __init__(self, args, divid, additional_data=None, train_index=None, modi_id=None, labels=None, visual=None):
        self.dataset = args.dataset

        if visual is not None:
            self.train_data_path = os.path.join('../data', self.dataset, 'step_poisoned_data', visual + '.json')
        else:
            self.train_data_path = os.path.join('../data', self.dataset, 'train_data_0.95.json') 
            ###train_data_random.json / poisoned_data_100in200.json self.dataset, error_proto_poisoned_data_100in200.json
        self.val_data_path = os.path.join('../data', self.dataset, 'val_data.json')
        self.test_data_path = os.path.join('../data', self.dataset, 'test_data_0.05.json')

        if divid == 'train':
            print('Load train data from: ', self.train_data_path)
            with open(self.train_data_path, 'r') as f:
                self.data = json.load(f)

            if additional_data is not None: #after getting poisoned data, craft new dataset
                if args.dataset_craft == 'aggregate': #add new point to dataset
                    for i, item in enumerate(additional_data):
                        temp = {}
                        temp['label'] = int(labels[i])
                        if args.ntk:
                            temp['data'] = item
                        else:
                            temp['data'] = multi_hot_to_ori(args, item)
                        self.data.append(temp)
                elif args.dataset_craft == 'replace': #modify current data point
                    #self.data = additional_data
                    for i, item in enumerate(additional_data):
                        if args.ntk:
                            self.data[int(modi_id[i])]['data'] = item
                        else:
                            self.data[modi_id[i]]['data'] = multi_hot_to_ori(args, item)
                elif args.dataset_craft == 'add':
                    for i, item in enumerate(additional_data):
                        temp = {}
                        temp['label'] = int(labels[i])
                        if args.ntk:
                            temp['data'] = item
                        else:
                            temp['data'] = multi_hot_to_ori(args, item)
                        self.data.append(temp)
                        
                with open('../data/' + self.dataset + '/step_poisoned_data/' + args.new_dataset_name + '.json', 'w') as new_f:
                    json.dump(self.data, new_f)
                    print('Poisoned dataset generated: ', args.new_dataset_name)
                print('Modify data in dataloader successfully!')

            if train_index is not None:
                self.data = [self.data[i] for i in train_index]

        elif divid == 'test':
            print('Load test data from: ', self.test_data_path)
            with open(self.test_data_path, 'r') as f:
                self.data = json.load(f)

        elif divid == 'val':
            with open(self.val_data_path, 'r') as f:
                self.data = json.load(f)
        
        self.ids = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        visit = data['data']
        label = data['label']
        
        dim_one = len(visit)
        
        dim_two = 1
        '''
        for item in visit:
            if len(item) > dim_two:
                dim_two = len(item)
        
        multi_hot_visit = torch.zeros((dim_one, 20))
        for i, item in enumerate(visit):
            for code in item:
                multi_hot_visit[i, code-1] = 1
        ''' 
        #return dim_one, dim_two, visit, label
        return dim_one, visit, label

    def __len__(self):
        return self.ids

def collate_fn_ehr(data):
    dim_ones, visits, labels = zip(*data)
    
    batch_size = len(visits)
    if batch_size == 1:
        max_dim_one = 20#max(dim_ones) 80
    else:
        max_dim_one = 20 #max(dim_ones)
    #max_dim_two = max(dim_twos)

    tensor_visits = torch.zeros((batch_size, max_dim_one, 4130)) + 1e-5
    for i in range(batch_size):
        for j, visit in enumerate(visits[i]):
            if j == max_dim_one:
                break
            l = len(visit)
            for k, record in enumerate(visit):
                tensor_visits[i,j,record-1] = float((l - 1e-5 * (4130 - l)) / l) #1.0

    tensor_labels = torch.LongTensor(labels)

    return (tensor_visits, tensor_labels)
    
def collate_fn_malware(data):
    dim_ones, visits, labels = zip(*data)
    
    batch_size = len(visits)
    
    tensor_visits = torch.zeros((batch_size, 5000)) + 1e-5
    for i in range(batch_size):
        for j, record in enumerate(visits[i][0]):
            #if j > 200: #400
            #    break
            l = len(visits[i][0])
            tensor_visits[i,record-1] = float((l - 1e-5 * (5000 - l)) / l) #1.0

    tensor_labels = torch.LongTensor(labels)

    return (tensor_visits, tensor_labels)

def collate_fn_IPS(data):
    dim_ones, visits, labels = zip(*data)
    
    batch_size = len(visits)
    
    tensor_visits = torch.zeros((batch_size, 20, 1103)) + 1e-5
    for i in range(batch_size):
        for j, record in enumerate(visits[i]):
            try:
                tensor_visits[i, j, record-1] = float((1 - 1e-5 * (1103 - 1)) / 1) #1.0
            except:
                print(visits[i])
    tensor_labels = torch.LongTensor(labels)

    return (tensor_visits.transpose(1,2), tensor_labels)
    
def get_loader(args, divid, additional_data=None, train_index=None, modi_id=None, labels=None, batch_size=0, shuffle_bool=None, visual=None):
    
    data = Dataset(args, divid, additional_data, train_index, modi_id, labels, visual)

    if divid == 'test':
        bs = args.eval_size
    else:
        bs = args.batch_size
        
    if args.dataset == 'MALWARE':
        collate_fn = collate_fn_malware
    elif args.dataset == 'IPS':
        collate_fn = collate_fn_IPS
    elif args.dataset == 'EHR':
        collate_fn = collate_fn_ehr
        
    if batch_size == 0:
        if shuffle_bool is not None:
            data_loader = torch.utils.data.DataLoader(dataset = data,
                                                  batch_size = bs,
                                                  shuffle = shuffle_bool,
                                                  num_workers = args.num_workers,
                                                  collate_fn = collate_fn)
        else:
            data_loader = torch.utils.data.DataLoader(dataset = data,
                                                  batch_size = bs,
                                                  shuffle = args.train_shuffle,
                                                  num_workers = args.num_workers,
                                                  collate_fn = collate_fn)
    else:
        if shuffle_bool is not None:
            data_loader = torch.utils.data.DataLoader(dataset = data,
                                              batch_size = batch_size,
                                              shuffle = shuffle_bool,
                                              num_workers = args.num_workers,
                                              collate_fn = collate_fn)
        else:
            data_loader = torch.utils.data.DataLoader(dataset = data,
                                                  batch_size = batch_size,
                                                  shuffle = args.train_shuffle,
                                                  num_workers = args.num_workers,
                                                  collate_fn = collate_fn)

    return data_loader
