import torch
import time
import datetime
#import jax.numpy as np
import copy
import pdb
import sys
import json
import logging
from collections import Counter
from copy import deepcopy
from itertools import combinations
from torch.autograd import Variable

from pathlib import Path
import torch.nn.functional as F
from attack_tools import *
from base_lines import *
#from NTK import neural_tangent_kernel
from dataloader import get_loader
    
def poison_sample_crafting(args, config, model, train_loader, test_loader, train_num, target=None, changable_index=None):
    indexes = []
    logit_list = [0.0, 0.0, 0.0, 0.0, 0.0]#[0.5, 0.6, 0.7, 0.8, 0.9]
    atonce_count = 0.0
    success_count = 0
    base_class_id = 0
    start = time.time()
    
    if args.ntk:
        worst_block_id, modi_point, modi_label, cluster_ids = neural_tangent_kernel(args, model, target)
        '''
        conclusion = torch.zeros(len(modi_point))
        for i, d in enumerate(modi_point):
            temp = tuple([1, d, modi_label[i]])
            data, _ = train_loader.collate_fn([temp])
            with torch.no_grad():
                logit, _ = model(data.cuda())
            conclusion[i] = logit
        print(conclusion)
        '''
        return worst_block_id, modi_point, modi_label, cluster_ids
            
    #calculate gradients of test data
    #grad_test1, grad_test0, grad_test_sum, index1, index0 = calc_grad_test(args, test_loader, model)
    proto = get_prototype(args)

    gn = int(train_num / args.max_code_step) #group number
    code_steps = torch.zeros(train_num).int()
    for s in range(args.max_code_step):
        if s != args.max_code_step-1:
            code_steps[s*gn:s*gn+gn] = s + 1
        else:
            code_steps[s*gn:] = s + 1
    
    #Get the ids of base training points
    if args.pyramid_logit:
        num_row = args.random_num
        train_sample_ids = torch.IntTensor(range(train_num)).repeat(num_row*len(logit_list), 1).T.reshape(-1)
        logit_pyramid_ = torch.Tensor(logit_list).repeat(num_row, 1).T.reshape(-1) 
        logit_pyramid = logit_pyramid_.repeat(train_num, 1).reshape(-1)    #[0, 0.1, 0.2, 0.3, 0.4]
        print(logit_pyramid)
        #train_sample_ids = list(range(train_num)) ###
        #logit_pyramid = torch.Tensor([0.0]).repeat(train_num)
    else:
        train_sample_ids = list(range(train_num))
    
    #Iterate each chosed sample, using inner loop function to get its best feature change plan
    
    if args.attack_method == 'witch':
        indexes = witch(args, model, train_loader, test_loader, target)
    else:
        for index, i in enumerate(train_sample_ids):
            #Get the ith sample
            #start = time.time()
            max_num1, ori_sample, label = train_loader.dataset[i]
            
            if args.pyramid_logit:
                target_logit = logit_pyramid[index]
            else:
                target_logit = -1

            #Inner loop: get score to choose features for each sample
            if args.pyramid_code:
                code_step = code_steps[i]
            else:
                code_step = args.code_step
            
            if args.attack_method == 'atonce':
                code_S_t, len_indices = Inner_loop(args, code_step, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target)
                atonce_count += len_indices
                if i == train_num-1:
                    print('Average number of changed codes at once: ', atonce_count/train_num)
            
            elif args.attack_method == 'gradattack' or args.attack_method == 'frogs':
                code_S_t, sc = Inner_loop(args, code_step, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target, target_logit)
                #success_count += sc
                #if index == len(train_sample_ids) - 1:
                #    print('Gradattack success count: ', success_count, ' / ', train_num*len(logit_list)*args.random_num)
            else: #OMPGS
                selected_codes, new_code, map_subset = Inner_loop(args, code_step, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target)

                #store the result for the last step
                if len(selected_codes) == 0:
                    code_S_t = [new_code]
                else:
                    code_S_t = selected_codes[map_subset[new_code]]
                    code_S_t.append(new_code)

            #store the result for each sample
            indexes.append(code_S_t)
            #end = time.time()
            #print('Time: ', end-start)
            display_progress("One train sample's all changes processed: ", index, train_num) #train_num*len(logit_list)*args.random_num / train_num
            torch.cuda.empty_cache()

    #Outer loop: get influence or alignment score to choose sample based on their modification
    scores = Outer_loop(args, config, indexes, model, proto, train_loader, test_loader, target)
    
    end = time.time()
    print('Time: ', (end-start))
    #print('Average alignment score decrease: ', torch.Tensor(scores).mean())

    #Conclude the results
    great_id, great_index, modi_point, modi_label, conclusion = conclude_results(args, scores, indexes, train_loader, model, train_sample_ids)

    if args.BFGS:
        return great_id, great_index, modi_point, modi_label, H_inv
    else:
        return great_id, great_index, modi_point, modi_label, conclusion

def Inner_loop(args, code_step, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target, target_logit=-1):

    if args.attack_method == 'atonce':
        code_S_t = []
        indices = get_indices(args, max_num1, ori_sample, label, model, proto, train_loader)#[:args.num_codes_once]
        print(indices[:30] % args.code_range + 1)
        sample_train = deepcopy(ori_sample)
        for indice in indices:
            visit_id = int(indice // args.code_range)
            code_id = int(indice % args.code_range) + 1
            code_S_t.append((visit_id, code_id))
        return code_S_t, len(indices)

    elif args.attack_method == 'gradattack':
        code_S_t, success_count = gradattack(args, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target, target_logit)
        return code_S_t, success_count

    elif args.attack_method == 'ompgs':
        selected_codes, new_code, map_subset = ompgsattack(args, code_step, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target, target_logit)
        return selected_codes, new_code, map_subset
    
    elif args.attack_method == 'frogs':
        code_S_t = frogs(args, max_num1, ori_sample, label, model, train_loader, test_loader, target)
        return code_S_t, 0

def Outer_loop(args, config, indexes, model, proto, train_loader, test_loader, target):
    scores = []
    sensi_scores = []
    return torch.zeros(len(indexes))
    '''
    with torch.no_grad():
        _, emb_target = model(target[0].cuda())
    
    if args.pyramid_logit:
        scores = torch.zeros(len(indexes))
        return scores

    if args.BFGS:
        iter_num = args.iter_num
        H_inv = BFGS(1024, model, train_loader, iter_num)
        grad_z_vec = torch.mean(grad_test, dim=0).view(-1).unsqueeze(0)#1x1024

    for i in range(len(indexes)):
        max_num1, sample_origin, label = train_loader.dataset[i]
        z_origin, t_train = train_loader.collate_fn([train_loader.dataset[i]])
        
        sample_temp = deepcopy(sample_origin)
        for j in range(len(indexes[i])):
            if indexes[i][j][1] in sample_temp[indexes[i][j][0]]:
                sample_temp[indexes[i][j][0]].remove(indexes[i][j][1])
            else:
                sample_temp[indexes[i][j][0]].append(indexes[i][j][1])
        temp = tuple([max_num1, sample_temp, label])
        z_train, _ = train_loader.collate_fn([temp])
        
        #get training algorithm sensitivity
        #grad_ori = grad_z(z_origin, t_train, model)[0]
        #grad_modi = grad_z(z_train, t_train, model)[0]
        #sensi_score = F.cosine_similarity(grad_ori.view(1, -1), grad_modi.view(1, -1))
        #sensi_scores.append(sensi_score)
        
        if args.BFGS:
            grad_train_ = grad_z_my(z_train, t_train, model)
            grad_train = grad_train_.view(-1).unsqueeze(1) #1024x1
            s_train = torch.mm(H_inv, grad_train)
            score = -torch.mm(grad_z_vec, s_train) / len(train_loader.dataset)
            
        elif args.outer_method == 'error':
            with torch.no_grad():
                logit, _ = model(z_train.cuda())
            if t_train == 0:
                score = 0.0 - logit
            else:
                score = logit - 1.0
                
        elif args.outer_method == 'dist':
            score = calc_distance_score(proto, model, z_train, label, emb_target)
                
        elif args.outer_method == 'align':
            score = calc_alignment_score(args, model, target, z_train, z_origin, t_train)
            
        elif args.outer_method == 'influence':
            score = - calc_influence_single(args, model, z_origin, z_train, torch.LongTensor([label]), train_loader, test_loader,
                        grad_test, gpu=config['gpu'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])

        #if args.abs_inf:
        #    scores.append(abs(score))
        #else:
        scores.append(score)
        torch.cuda.empty_cache()

    return scores
    '''
def conclude_results(args, scores, indexes, train_loader, model, train_sample_ids=None):
    #great_id = torch.argmax(torch.Tensor(scores)) ###
    if args.pyramid_logit:
        great_id = torch.Tensor(range(len(indexes))).int()
    else:
        great_id = torch.Tensor(range(len(indexes))).int()
        #great_id = torch.topk(torch.Tensor(scores), k = args.num_samples_allowed, largest=False)[1] #k largest (largest=False)
        
    torch.set_printoptions(precision=3)
    print('Candidates scores:\n', torch.Tensor(scores), '\nMean score: ', torch.Tensor(scores).mean())
    great_index = []
    great_point = []
    great_label = []
    conclusion = torch.zeros((4, len(great_id))) - 1
    for ind, id in enumerate(great_id):
        index = indexes[id]
        if train_sample_ids is not None:
            id = train_sample_ids[ind]

        modi_point, modi_label = train_loader.collate_fn([train_loader.dataset[id]])
        with torch.no_grad():
            modi_error, emb_before = model(modi_point.cuda())

        #test error before and after modification
        #conclusion[0, ind] = modi_error[0]
        conclusion[3, ind] = modi_label.float()
        #print('Before modified, points error: ', modi_error[0], ' points label: ', int(modi_label))
        max_num1, sample_origin, label = train_loader.dataset[id]
        sample_temp = deepcopy(sample_origin)

        for i in range(len(index)):
            if args.dataset == 'IPS':
                sample_temp[index[i][0]] = index[i][1]
            else:
                try:
                    if index[i][1] in sample_temp[index[i][0]]:
                        sample_temp[index[i][0]].remove(index[i][1])
                    else:
                        sample_temp[index[i][0]].append(index[i][1])
                except:
                    pdb.set_trace()

        temp = tuple([max_num1, sample_temp, label])
        modi_point, _ = train_loader.collate_fn([temp])
        #with torch.no_grad():
        #    modi_error, emb_after = model(modi_point.cuda())
        #conclusion[1, ind] = modi_error[0]
        #print('After modified, points error: ', modi_error[0], ' points label: ', int(modi_label))

        #store
        great_index.append(index)
        great_point.append(modi_point)
        great_label.append(modi_label)

        modi_point, modi_error = 0, 0
        torch.cuda.empty_cache()

    print('Finish conclude.')
    return great_id, great_index, great_point, great_label, conclusion

#only for visualization
'''
if i == 26 or i == 39 or i == 43 or i == 84 or i == 57:
    t = tuple([max_num1, ori_sample, label])
    z_train, _ = train_loader.collate_fn([t])
    _, vec = model(z_train.cuda())
    save_name = str(i) + '_0.pt'
    torch.save(vec, save_name)
     
    temp_sample = deepcopy(ori_sample)
    for j, code in enumerate(code_S_t):
        if code[1] in temp_sample[code[0]]:
            temp_sample[code[0]].remove(code[1])
        else:
            temp_sample[code[0]].append(code[1])
        temp = tuple([max_num1, temp_sample, label])
        z_train, _ = train_loader.collate_fn([temp])
        _, vec = model(z_train.cuda())
        
        save_name = str(i) + '_' + str(j+1) + '.pt'
        torch.save(vec, save_name)
'''