import torch
import pdb
import os
import sys
from copy import deepcopy
import numpy as np

import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_Model(nn.Module):

    def __init__(self, args):
        super(LSTM_Model, self).__init__()
        self.args = args
        self.vocab_size = args.code_range
        self.code_size = args.code_size
        self.hidden_size = args.hidden_size
        self.embedding_path = os.path.join('../data', args.dataset, args.embedding_name)

        self.emb_weight = torch.load(self.embedding_path)['word_embeddings.weight']
        self.code_embed = nn.Embedding(self.vocab_size+1, self.code_size, padding_idx=0)
        
        if self.args.multi_hot:
            self.fc = nn.Linear(20, 32)
        elif self.args.selfatt:
            self.fc = nn.Linear(70, 1)
        #self.fc = nn.Linear(20, 32)
        
        #self.code_embed_weight = nn.Parameter(self.emb_weight).cuda()
        #self.code_embed.weight.data = self.emb_weight
        #self.emb_hidden = nn.Linear(self.code_size, self.hidden_size)

        self.GRU = nn.GRU(self.code_size, self.hidden_size, 1, batch_first=True) #dropout=0.5
        self.mlp = nn.Linear(self.hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid() #can be included in BCEWithLogitsLoss function
        
        self.init_weights()
    
    def init_weights(self):
        init.xavier_uniform_(self.mlp.weight)
        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, visits):
        #load data
        #visits (batch_size, max_num1, max_num2)
        #labels (batch_size)
        batch_size = visits.size(0)
        
        #prepare data
        if self.args.multi_hot: #Visits: (batch_size, #visits, #codes)
            embed_codes_ = self.code_embed(visits).permute(0,1,3,2) #->(batch_size, #visits, 70, #codes)
            embed_codes = self.fc(embed_codes_) #->(batch_size, #visits, 70, 32)
            embedding = torch.sum(embed_codes, dim=3) #-> (batch_size, #visits, 70)
        elif self.args.selfatt:
            batch_size, max_num1 = visits.size(0), visits.size(1)
            #x = self.temp_x.expand(batch_size, max_num1, 4130)
            temp_x = torch.LongTensor(list(range(self.vocab_size))).repeat(batch_size, max_num1, 1).cuda()
            x = self.code_embed(temp_x) #x: (batch_size, max_num1, 4130, 70)
            weight_of_x = visits.unsqueeze(3) #(batch_size, max_num1, 4130, 1)
            embedding0 = torch.relu(x * weight_of_x) #(batch_size, max_num1, 4130, 70)
            
            #mean_embed = torch.mean(embedding0, dim=2, keepdim=True)
            #self.x_weight = Variable(F.softmax(self.fc(torch.relu(embedding0+mean_embed)).permute(0,1,3,2), dim=3), requires_grad=True) #(batch_size, max_num1, 1, 4130)
            self.x_weight = F.softmax(self.fc(torch.relu(embedding0)).permute(0,1,3,2), dim=3) #batch_size, max_num1, 1, 4130
            '''
            non_zero = (visits != 0) #(batch_size, max_num1, max_num2)
            divide_num_ = torch.sum(non_zero, axis=2) #(batch_size, max_num1)
            num2s = torch.ones_like(divide_num_) * visits.size(2)
            divide_num = torch.where(divide_num_ == 0, num2s, divide_num_)
            
            embedding0 = self.code_embed(visits) #(batch_size, max_num1, max_num2, code_size)
            embedding1 = torch.sum(embedding0.permute(3,0,1,2) * non_zero, axis=3) / divide_num #(code_size, batch_size, max_num1)
            mean_embed = embedding1.permute(1,2,0).unsqueeze(2)#(batch_size, max_num1, 1, code_size)
            
            weight = F.softmax(self.fc(torch.tanh(embedding0 + mean_embed)).permute(0,1,3,2), dim=3) #(batch_size, max_num1, 1, max_num2)
            '''
            embedding = torch.bmm(self.x_weight.view(-1, 1, self.x_weight.size(3)), embedding0.view(-1, embedding0.size(2), embedding0.size(3))) \
                        .view(batch_size, max_num1, 1, self.code_size).squeeze(2)
        else:
            non_zero = (visits != 0) #(batch_size, max_num1, max_num2)
            divide_num_ = torch.sum(non_zero, axis=2) #(batch_size, max_num1)
            num2s = torch.ones_like(divide_num_) * visits.size(2)
            divide_num = torch.where(divide_num_ == 0, num2s, divide_num_)
            embedding0 = self.code_embed(visits).permute(3,0,1,2) #(code_size, batch_size, max_num1, max_num2)
            embedding1 = torch.sum(embedding0 * non_zero, axis=3) / divide_num #(code_size, batch_size, max_num1)
            embedding = embedding1.permute(1,2,0)#(batch_size, max_num1, code_size)
            #embedding = self.emb_hidden(embedding_)
      
        if torch.cuda.is_available():
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        #pdb.set_trace()
        #rnn network
        hidden, _ = self.GRU(embedding, h_0)
        output = hidden[:, -1, :]
        logits = self.sigmoid(self.mlp(output)).squeeze(dim=-1) ###
        
        #if torch.isnan(torch.nn.functional.softmax(logits, dim=1)[0][0]):
        #    pdb.set_trace()
        return logits, output ###
        
class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.num_classes = 1
        self.num_filters = 128
        self.filter_sizes = [3]
        self.vocab_size = args.code_range
        self.code_size = args.code_size
        '''
        embedding = config['training']['embedding_name']
        self.embedding_pretrained = torch.tensor(
            np.load('./data/' + config['training']['dataset'] + '_' + embedding).astype('float32'))
        self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        '''
        self.fc1 = nn.Linear(4130, 64)
        self.code_embed = nn.Embedding(self.vocab_size + 1, self.code_size, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.code_size, self.num_filters, (k, 64)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.num_filters*len(self.filter_sizes), self.num_filters, bias=False)
        self.fc3 = nn.Linear(self.num_filters, self.num_classes, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=-1)
        
    def conv_and_pool(self, x, conv):
        x_ = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x_, x_.size(2)).squeeze(2)
        return x

    def forward(self, visits):
        batch_size, max_num1 = visits.size(0), visits.size(1)
        temp_x = torch.LongTensor(list(range(4130))).repeat(batch_size, max_num1, 1).cuda()
        x = self.code_embed(temp_x) #x: (batch_size, max_num1, 4130, 70)
        weight_of_x = visits.unsqueeze(3) #weight_of_x: (batch_size, max_num1, 4130, 1)
        emb_ = torch.relu(x * weight_of_x)
        
        emb = self.fc1(emb_.permute(0,3,1,2)) #(batch_size, 70, #visit, #code) -> (batch_size, 70, #visit, 64)
        out = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out1 = self.fc2(out)
        logit = self.sigmoid(self.fc3(out1)).squeeze(dim=-1)
        return logit, out1

class FNN_Model(nn.Module):
    def __init__(self, args):
        super(FNN_Model, self).__init__()
        self.args = args
        self.vocab_size = args.code_range
        self.code_size = args.code_size
        self.hidden_size = args.hidden_size
        self.code_embed = nn.Embedding(self.vocab_size + 1, self.code_size, padding_idx=0)
        
        self.fc0 = nn.Linear(70, 1)
        self.fc1 = nn.Linear(4130, 1)
        self.fc2 = nn.Linear(20, 8)###
        self.mlp = nn.Linear(8, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, visits): #visits: (batch_size, maxone, 4130)

        batch_size, max_num1 = visits.size(0), visits.size(1)
        temp_x = torch.LongTensor(list(range(4130))).repeat(batch_size, max_num1, 1).cuda()
        x = self.code_embed(temp_x) #x: (batch_size, max_num1, 4130, 70)
        weight_of_x = visits.unsqueeze(3) #weight_of_x: (batch_size, max_num1, 4130, 1)
        embedding = torch.relu(x * weight_of_x) #batch_size, max_num1, 4130, 70
        
        embed = F.relu(self.fc0(embedding)).squeeze(-1)
        out1 = F.relu(self.fc1(embed)).squeeze(-1)
        emb = F.relu(self.fc2(out1))
        logits = self.sigmoid(self.mlp(emb)).squeeze(-1)
        return logits, emb

class FNN_malware(nn.Module):
    def __init__(self, args):
        super(FNN_malware, self).__init__()
        self.args = args
        self.vocab_size = args.code_range
        self.code_size = args.code_size
        
        self.fc1 = nn.Linear(70, 1)
        self.fc2 = nn.Linear(5000, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.mlp = nn.Linear(64, 1)
        
        self.code_embed = nn.Embedding(self.vocab_size+1, self.code_size, padding_idx=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visits): #visits: (batch_size, 5000)
        batch_size = visits.size(0)
        temp_x = torch.LongTensor(list(range(self.vocab_size))).repeat(batch_size, 1).cuda()
        x = self.code_embed(temp_x) #x: (batch_size, 5000, 70)
        weight_of_x = visits.unsqueeze(2) #(batch_size, 5000, 1)
        embedding0 = F.relu(x * weight_of_x) #(batch_size, 5000, 70)
            
        out1 = F.relu(self.fc1(embedding0)).squeeze(-1)
        out2 = F.relu(self.fc2(out1))
        emb = F.relu(self.fc3(out2))
        logits = self.sigmoid(self.mlp(emb)).squeeze(-1)
        return logits, emb

class FNN_IPS(nn.Module):
    def __init__(self, args):
        super(FNN_IPS, self).__init__()
        self.args = args
        self.vocab_size = args.code_range
        self.code_size = args.code_size
        
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(self.code_size * 10, 70)
        self.mlp = nn.Linear(70, 3)
        
        self.code_embed = nn.Embedding(self.vocab_size+1, self.code_size, padding_idx=0)

    def forward(self, visits): #visits: (batch_size, 1103, 20)
        batch_size = visits.size(0)
        temp_x = torch.LongTensor(list(range(self.vocab_size))).reshape(1, 1, self.vocab_size).cuda() #(1, 1, 1103)
        x = self.code_embed(temp_x) #x: (1, 1, 1103, 70)
        weight_of_x = visits.transpose(1,2).unsqueeze(3) #(batch_size, 20, 1103, 1)
        embedding0 = (x * weight_of_x).sum(dim=2) #(batch_size, 20, 70)
            
        out1 = F.relu(self.fc1(embedding0.transpose(1,2))) #batch_size, 70, 10
        emb = F.relu(self.fc2(out1.transpose(1,2).reshape(out1.size(0), -1)))# batch_size, 700 -> batch_size, 70
        logits = F.softmax(self.mlp(emb), dim=-1)
        return logits, emb
    '''
    def input_handle(self, person):
        t_diagnosis_codes = self.pad_matrix(person)
        model_input = deepcopy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = Variable(torch.LongTensor(model_input))
        return model_input.transpose(0, 1), torch.tensor(t_diagnosis_codes).transpose(0, 1)
    
    def pad_matrix(self, seq_diagnosis_codes): # make multi-hot vector

        batch_size = seq_diagnosis_codes.size(0)
        dim_one = seq_diagnosis_codes.size(1)
        n_diagnosis_codes = 4130
        # maxlen = np.max(lengths)
    
        batch_diagnosis_codes = torch.zeros((batch_size, dim_one, n_diagnosis_codes)).cuda()
    
        for i, sample in enumerate(seq_diagnosis_codes):
            for j, visit in enumerate(sample): 
                for code in visit:
                    if code != 0:
                        batch_diagnosis_codes[i,j,code-1] = 1.0
    
        return batch_diagnosis_codes
    '''