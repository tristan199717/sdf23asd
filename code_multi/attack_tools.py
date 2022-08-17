import sys
import torch
import json
import pdb
import torch.nn as nn
import numpy as np
from torch.autograd import grad, Variable
import torch.nn.functional as F
from copy import deepcopy
from itertools import combinations
from dataloader import get_loader
from pytorch_influence_functions.utils import display_progress
from torch.autograd.functional import vhp

def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'seed': 42,
        'gpu': 0,
        'dataset': 'CIFAR10',
        'num_classes': 2,
        'test_sample_num': 1,
        'test_start_index': True,
        'recursion_depth': 100, #200
        'r_averaging': 1,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }
    return config

def get_indices(args, max_num1, ori_sample, label, model, proto, train_loader, target):
    '''
    The first two steps in OMPGS algorithm.
    Calculate the gradient based on pre-defined loss and then select topk changable features based
    on the absolute value of their respective gradients
    '''
    model.train()
    #proto0, proto1, cov_inv_0, cov_inv_1 = proto

    temp = tuple([max_num1, ori_sample, label])
    z_origin, z_train = train_loader.collate_fn([temp])
    weight_of_embed_codes = Variable(z_origin.data, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=False):###
        logit, _ = model(weight_of_embed_codes.cuda())
    #logit = model(z_origin.cuda())
    #model.x_weight.retain_grad()

    #get loss
    criterion = nn.BCELoss()
    '''
    #loss1
    cross_entropy = nn.CrossEntropyLoss()
    if z_train == 0:
        out = torch.cat((logit, 1-logit)).unsqueeze(0)
    elif z_train == 1:
        out = torch.cat((1-logit, logit)).unsqueeze(0)

    loss1 = cross_entropy(out, (1-z_train).long().cuda())
    
    loss2 = criterion(logit, (1-z_train).float().cuda())
    
    if args.prototype_method == 'distance':
        loss3 = 0
        
        cov_inv_1 = (cov_inv_1 - cov_inv_1.mean()) / cov_inv_1.std()
        cov_inv_0 = (cov_inv_0 - cov_inv_0.mean()) / cov_inv_0.std()
        if label == 0:
            diff = emb.squeeze(0) - proto1
            loss3 = torch.mm(torch.mm(diff.unsqueeze(0), cov_inv_1), diff.unsqueeze(1))
        elif label == 1:
            diff = emb.squeeze(0) - proto0
            loss3 = torch.mm(torch.mm(diff.unsqueeze(0), cov_inv_0), diff.unsqueeze(1))
        
    elif args.prototype_method == 'predict':
        proto_model = Proto_Model()
        
        loss2 = 0
    '''
    loss4 = 0
    if args.dataset == 'IPS':
        poison_loss = criterion(logit, F.one_hot(z_train, num_classes=args.class_num).float().cuda())
    else:
        poison_loss = criterion(logit, z_train.float().cuda())
    poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)
    for i, tt in enumerate(target):
        if i == 0:
            grad_test = grad_z(args, tt[0], tt[1], model)
        else:
            grad_test += grad_z(args, tt[0], tt[1], model)
    '''
    g1 = torch.cat(tuple([x.reshape(1,-1) for x in poison_grad]), 1)#poison_grad[-3].reshape(1,-1)
    g2 = torch.cat(tuple([x.reshape(1,-1) for x in grad_test]), 1)#grad_test[-3].reshape(1,-1)
    loss4 = F.cosine_similarity(g1, g2)

    '''
    for i in range(len(poison_grad)):
        g1, g2 = grad_test[i], poison_grad[i]
        if args.alignment_calc == 'cosine':
            if len(g1.size()) == 1:
                loss4 += F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).sum()
            else:
                loss4 += F.cosine_similarity(g1, g2).sum()
        elif args.alignment_calc == 'dot':
            loss4 += torch.mm(g1.reshape(1, -1), g2.reshape(-1, 1))
    #print('loss1: ', loss1, 'loss2: ', loss2)
    #loss = args.lamb1 * loss1 + args.lamb2 * loss2 + (1 - args.lamb1 - args.lamb2) * loss3
    loss = 1 - loss4
    loss.backward(retain_graph=False)
        
    gradients = weight_of_embed_codes.grad.data #(1, max_num1, 4130) / (1, 5000)
    
    #after getting the gradients, 
    #for 'atonce': change them anyway
    #for others: rank the gradients based on their absolute values and select top INDICE_K ones
    #if args.attack_method == 'atonce':
    # else:

    #valid_indice = torch.nonzero(((weight_of_embed_codes[0,:len(ori_sample),:]-0.5)*gradients[0,:len(ori_sample),:] < 0).reshape(-1))
    #sorted_indice = np.argsort(-abs(gradients[0,:len(ori_sample),:]).reshape(-1))
    #indices = np.intersect1d(sorted_indice, valid_indice)
    #if len(indices) > args.indice_k:
    #    indices = indices[:args.indice_k]
    if args.dataset == 'MALWARE':
        indices = np.argsort(-np.reshape(abs(gradients[0,:]), (-1)))[:args.indice_k]
    elif args.dataset == 'IPS':
        indices = np.argsort(-np.reshape(abs(gradients[0].permute(1,0)), (-1)))[:args.indice_k]
    else:
        indices = np.argsort(-np.reshape(abs(gradients[0,:len(ori_sample),:]), (-1)))[:args.indice_k]

    torch.cuda.empty_cache()
    return indices

def gradattack(args, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target, target_logit):
    config = get_default_config()

    code_S_t = []
    break_flag = 0
    temp = tuple([max_num1, ori_sample, label])
    z_origin, _ = train_loader.collate_fn([temp])
    sample0 = deepcopy(ori_sample)
    step_size = args.num_codes_once // args.gradattack_group_size
    for i, tt in enumerate(target):
        with torch.no_grad():
            _, emb_t = model(tt[0].cuda())
        if i == 0:
            emb_target = emb_t
        else:
            emb_target += emb_t

    for step in range(step_size):
        samples = []
        selected_codes = []
        S_t = []
        min_score = 1000000.0
        best_com = None

        indices_total = get_indices(args, max_num1, sample0, label, model, proto, train_loader, target)
        if len(indices_total) == 0:
            print('Stop in advance.')
            break

        #indices = torch.Tensor(np.random.choice(indices_total, args.gradattack_group_size, replace=False)).int()
        for item in indices_total:
            visit_id = int(item // args.code_range)
            code_id = int(item % args.code_range) + 1
            S_t.append((visit_id, code_id))

        for num in range(len(S_t)+1):
            comb_set = combinations(S_t, num)
            for s in comb_set:
                selected_codes.append(list(s))

        for c in selected_codes: #e.g [[C], [A,C], [B,C], [A,B,C]]
            temp_sample = deepcopy(sample0)
            for code in c:
                if args.dataset == 'IPS':
                    temp_sample[code[0]] = code[1]
                else:
                    try:
                        if code[1] in temp_sample[code[0]]:
                            temp_sample[code[0]].remove(code[1])
                        else:
                            temp_sample[code[0]].append(code[1])
                    except:
                        pdb.set_trace()
            samples.append(temp_sample)

        for j in range(len(samples)):
            sample_train = deepcopy(samples[j])

            temp = tuple([max_num1, sample_train, label])
            z_train, _ = train_loader.collate_fn([temp])
            
            #If it is not under pyramid logit setting, each kind of modification is compared based
            #on their scores, which can be calculated with different methods (there are 4 in total)
            if args.inner_method == 'align':
                score = - calc_alignment_score(args, model, target, z_train, z_origin, torch.LongTensor([label]))
                min_score, best_com = gradattack_scores(j, best_com, step, score, min_score, code_S_t, selected_codes)

            elif args.inner_method == 'dist' or args.inner_method == 'jac_dist':
                score = calc_distance_score(args, proto, model, z_train, label, emb_target, target)
                min_score, best_com = gradattack_scores(j, best_com, step, score, min_score, code_S_t, selected_codes)
            elif args.inner_method == 'error':
                with torch.no_grad():
                    logit, _ = model(z_train.cuda())
                if label == 0:
                    score = 0.0 - logit
                else:
                    score = logit - 1.0
                min_score, best_com = gradattack_scores(j, best_com, step, score, min_score, code_S_t, selected_codes)

            elif args.inner_method == 'influence':
                score = - calc_influence_single(args, model, z_origin, z_train, torch.LongTensor([label]), train_loader, test_loader,
                            grad_test, gpu=config['gpu'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])    
                min_score, best_com = gradattack_scores(j, best_com, step, score, min_score, visit_id, code_id, code_S_t, selected_codes)
            torch.cuda.empty_cache()

        if best_com is None:
            best_com = selected_codes[j]
        if len(best_com) == 0:
            break
        code_S_t.extend(best_com)
        if break_flag == 1:
            return code_S_t, 1

        if args.dataset == 'IPS':
            sample0[code[0]] = code[1]
        else:
            for code in best_com:
                if code[1] in sample0[code[0]]:
                    sample0[code[0]].remove(code[1])
                else:
                    sample0[code[0]].append(code[1])
    #print('out_logit: ', out_logit, 'target_logit: ', target_logit)
    #print(code_S_t)
    if len(code_S_t) == 0:
        print('sample failed')
    return code_S_t, 0

def ompgsattack(args, code_step, max_num1, ori_sample, label, model, proto, train_loader, test_loader, target, target_logit):
    config = get_default_config()

    S_t = set()
    out_code = []
    first_time_flag = 1
    temp = tuple([max_num1, ori_sample, label])
    z_origin, _ = train_loader.collate_fn([temp])

    for step in range(code_step):
        flip_num = int(pow(2, len(S_t)))
        samples = []
        selected_codes = []
        map_subset = {}
        
        #select code
        if first_time_flag: #first time
            flip_num = 1
            samples.append(ori_sample)
            first_time_flag = 0
        else:
            for num in range(len(S_t)+1): #get every combination and enter into selected_codes
                comb_set = combinations(list(S_t), num)
                for s in comb_set:
                    selected_codes.append(list(s))
            if selected_codes: #after the second time
                for c in selected_codes:
                    c.append(new_code)
            else: #in the second time, the selected_codes is still []
                selected_codes.append([new_code])

            for c in selected_codes: #e.g [[C], [A,C], [B,C], [A,B,C]]
                temp_sample = deepcopy(ori_sample)
                for code in c:
                    try:
                        if code[1] in temp_sample[code[0]]:
                            temp_sample[code[0]].remove(code[1])
                        else:
                            temp_sample[code[0]].append(code[1])
                    except:
                        pdb.set_trace()
                samples.append(temp_sample)

            S_t.add(new_code)

        #get scores for all changes from one data
        min_score = 100000.0
        max_score = float('-inf')
        new_code = ()
        for j in range(flip_num):
            #get indices for each original sample
            indices = get_indices(args, max_num1, samples[j], label, model, proto, train_loader)
            c_num = len(indices) #(z_origin!=0).int().sum() #* (args.range_num - 1)

            #compare error score
            for k in range(c_num):
                indice = indices[k]
                visit_id = int(indice // args.code_range)
                code_id = int(indice % args.code_range) + 1
                sample_train = deepcopy(samples[j])
                
                if code_id in sample_train[visit_id]:
                    sample_train[visit_id].remove(code_id)
                else:
                    sample_train[visit_id].append(code_id)

                temp = tuple([max_num1, sample_train, label])
                z_train, _ = train_loader.collate_fn([temp])

                if args.inner_method == 'align':

                    score = - calc_alignment_score(args, model, target, z_train, z_origin, torch.LongTensor([label]))
                    min_score, new_code, map_subset = ompgs_scores(j, score, new_code, map_subset, step, min_score, visit_id, code_id, out_code)

                elif args.inner_method == 'dist':
                    score = calc_distance_score(proto, model, z_train, label, target)
                    min_score, new_code, map_subset = ompgs_scores(j, score, new_code, map_subset, step, min_score, visit_id, code_id, out_code)

                elif args.inner_method == 'error':
                    with torch.no_grad():
                        logit, _ = model(z_train.cuda())
                    if label == 0:
                        score = 0.0 - logit
                    else:
                        score = logit - 1.0
                    min_score, new_code, map_subset = ompgs_scores(j, score, new_code, map_subset, step, min_score, visit_id, code_id, out_code)

                elif args.inner_method == 'influence':
                    score = - calc_influence_single(args, model, z_origin, z_train, torch.LongTensor([label]), train_loader, test_loader,
                            grad_test, gpu=config['gpu'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])
                    min_score, new_code, map_subset = ompgs_scores(j, score, new_code, map_subset, step, min_score, visit_id, code_id, out_code)
                
                torch.cuda.empty_cache()
        
        out_code.append(new_code)
    print('Out code: ', out_code)
    print('Original_sample: ', ori_sample)
    return selected_codes, new_code, map_subset

def select_target(args, model, test_loader):
    flag = 0
    ids = [110,135,148,171,178]
    target_list = []
    print('Target ids: ', ids, 'Target class id: ', args.target_class_id)
    for id in ids:
        
        count = 0
        for i in range(len(test_loader.dataset)):
            z_test, t_test = test_loader.collate_fn([test_loader.dataset[i]])

            if t_test == args.target_class_id:
                if count == id:
                    target = (z_test, t_test)
                    with torch.no_grad():
                        logit, _ = model(z_test.cuda())
                    print('Target logit: ', logit)
                    break
                else:
                    count += 1
                
            torch.cuda.empty_cache()
        target_list.append(target)
        '''
        while flag == 0:
            
            if args.target_class_id == 1:
                target_id = np.random.randint(args.test1_num)
                if target_id in target_ids:
                    print('fail', target_id)
                    break
                #target_id = 298
            elif args.target_class_id == 0:
                target_id = np.random.randint(args.test0_num)
                #target_id = 71
                if target_id in target_ids:
                    print('fail', target_id)
                    break
            elif args.target_class_id == 2:
                target_id = np.random.randint(args.test2_num)
                #target_id = 128
            print('Target id: ', target_id, 'Target class id: ', args.target_class_id)
            count = 0
            for i in range(len(test_loader.dataset)):
                z_test, t_test = test_loader.collate_fn([test_loader.dataset[i]])

                if t_test == args.target_class_id:
                    if count == target_id:
                        target = (z_test, t_test)
                        with torch.no_grad():
                            logit, _ = model(z_test.cuda())
                            print('Target logit: ', logit)
                            break
                    else:
                        count += 1
                    
                torch.cuda.empty_cache()

            if args.dataset == 'IPS':
                #if torch.argmax(logit, dim=-1) == args.target_class_id: ###
                if logit[0,0] > 0.9:
                    flag = 1
            else:
                #if int(logit > 0.5) == args.target_class_id:
                if logit > 0.9 and logit < 0.96:
                    flag = 1
        pdb.set_trace()
        '''
    print('Done selecting target')
    
    return target_list

def get_prototype(args):
    clean = torch.load(args.clean_embedding_path)

    clean_samples = clean[:, 0:-1]
    clean_labels = clean[:, -1].long()
    
    data_0 = clean_samples[clean_labels==0,:]
    data_1 = clean_samples[clean_labels==1,:]
            
    prototype0 = torch.mean(data_0, dim=0)
    prototype1 = torch.mean(data_1, dim=0)
    
    cov_inv_0 = np.linalg.pinv(np.cov(F.normalize(data_0).numpy().T))
    cov_inv_1 = np.linalg.pinv(np.cov(F.normalize(data_1).numpy().T))
    
    if torch.cuda.is_available():
        prototype0, prototype1 = prototype0.cuda(), prototype1.cuda()
        cov_inv_0, cov_inv_1 = torch.Tensor(cov_inv_0).cuda(), torch.Tensor(cov_inv_1).cuda()

    return prototype0, prototype1, cov_inv_0, cov_inv_1

def gradattack_scores(j, best_com, step, score, min_score, code_S_t, selected_codes):
    if score < min_score:
        min_score = score
        best_com = selected_codes[j]

    return min_score, best_com

def ompgs_scores(j, score, new_code, map_subset, step, min_score, visit_id, code_id, out_code):
    if step == 0 and score < min_score:
        min_score = score
        new_code = (visit_id, code_id)
        map_subset[new_code] = j
    elif score < min_score and (visit_id, code_id) not in out_code:
        min_score = score
        new_code = (visit_id, code_id)
        map_subset[new_code] = j

    return min_score, new_code, map_subset

def calc_distance_score(args, proto, model, z_train, label, emb_target, target):
    #proto0, proto1, cov_inv_0, cov_inv_1 = proto
    score = 0.0
    if args.inner_method == 'dist':
        with torch.no_grad():
            _, emb = model(z_train.cuda())
        score = torch.sqrt(((emb - emb_target)**2).sum())
    elif args.inner_method == 'jac_dist':
        z_hot = 1 -((z_train - 1e-5) == 0).int()
        target_hots = []
        for tt in target:
            target_hot = 1 - ((tt[0] - 1e-5) == 0).int()
            target_hots.append(target_hot)
        for t_hot in target_hots:
            score += - (z_hot & t_hot).sum() / (z_hot | t_hot).sum().float()

    #emb = (emb - emb.mean()) / emb.std()
    #proto1, proto0 = (proto1 - proto1.mean()) / proto1.std(), (proto0 - proto0.mean()) / proto0.std()
    #grad_test_sum = (grad_test_sum.mean()) / grad_test_sum.std()
    #cov_inv_1 = (cov_inv_1 - cov_inv_1.mean()) / cov_inv_1.std()
    #cov_inv_0 = (cov_inv_0 - cov_inv_0.mean()) / cov_inv_0.std()
    #diff = emb.squeeze(0) - grad_test_sum
    '''
    diff = (emb - emb_target).squeeze(0)
    if target[1] == 1:
        #score = diff.pow(2).sum().sqrt()
        score = torch.mm(torch.mm(diff.unsqueeze(0), cov_inv_1), diff.unsqueeze(1))
    elif label == 1:
        #score = diff.pow(2).sum().sqrt()
        score = torch.mm(torch.mm(diff.unsqueeze(0), cov_inv_0), diff.unsqueeze(1))
    '''
    return score

def calc_alignment_score(args, model, target, z_train, z_origin, t_train):
    
    for i, tt in enumerate(target):
        if i == 0:
            grad_test = grad_z(args, tt[0], tt[1], model)
        else:
            grad_test += grad_z(args, tt[0], tt[1], model)
    grad_train_old = grad_z(args, z_origin, t_train, model)#[0]
    grad_train_new = grad_z(args, z_train, t_train, model)#[0]
    score_old, score_new, score_stable = 0, 0, 0
    '''
    g1 = torch.cat(tuple([x.reshape(1,-1) for x in grad_test]), 1)#grad_test[-3].reshape(1,-1)
    g2 = torch.cat(tuple([x.reshape(1,-1) for x in grad_train_old]), 1)#grad_train_old[-3].reshape(1,-1)#
    g3 = torch.cat(tuple([x.reshape(1,-1) for x in grad_train_new]), 1)#grad_train_new[-3].reshape(1,-1)#
    score_old = F.cosine_similarity(g1, g2)
    score_new = F.cosine_similarity(g1, g3)
    '''
    for i in range(len(grad_train_old)):
        g1, g2, g3 = grad_test[i], grad_train_old[i], grad_train_new[i]
        nm = args.dot_norm
        if args.alignment_calc == 'cosine':
            if len(g1.size()) == 1:
                score_old += F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).sum()
                score_new += F.cosine_similarity(g1.unsqueeze(0), g3.unsqueeze(0)).sum()
                score_stable += F.cosine_similarity(g2.unsqueeze(0), g3.unsqueeze(0)).sum()
            else:
                score_old += F.cosine_similarity(g1, g2).sum()
                score_new += F.cosine_similarity(g1, g3).sum()
                score_stable += F.cosine_similarity(g2, g3).sum()

        elif args.alignment_calc == 'dot':
            score_old += torch.mm(g1.reshape(1, -1), g2.reshape(-1, 1))
            score_new += torch.mm(g1.reshape(1, -1), g3.reshape(-1, 1))
    
    if args.target_class_id == args.base_class_id:
        score = score_new - score_old + score_stable
    else:
        score = score_old - score_new + score_stable
    return score
    
def calc_influence_single(args, model, z_origin, z_train, t_train, train_loader, test_loader, grad_test, gpu, recursion_depth, r, s_test_vec=None, time_logging=False):
    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    test_dataset_size = len(test_loader.dataset)
    influences = []
    
    s_test_vec = calc_s_test_single(args, model, z_origin, z_train, t_train, train_loader,
                                    gpu, recursion_depth=recursion_depth, r=r, score=True) #
    
    #for i in range(test_dataset_size):
    if time_logging:
        time_a = datetime.datetime.now()
    #grad_z_vec = grad_test[i] #
    if time_logging:
        time_b = datetime.datetime.now()
        time_delta = time_b - time_a
        logging.info(f"Time for grad_z iter:"
                     f" {time_delta.total_seconds() * 1000}")
    tmp_influence = -sum(
        [
            torch.sum(k * j).data
            for k, j in zip(grad_test, s_test_vec)
        ]) / train_dataset_size
    influences.append(tmp_influence)
    torch.cuda.empty_cache()
    #display_progress("Calc. influence function: ", i, test_dataset_size)
    
    score = torch.Tensor(influences).mean()
    #harmful = np.argsort(influences)
    #helpful = harmful[::-1]

    return score

def calc_s_test_single(args, model, z_origin, z_test, t_test, train_loader, gpu=-1, damp=0.1, scale=250.0, 
                       recursion_depth=5000, r=1, score=False):
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(args, z_origin, z_test, t_test, model, train_loader,
                                      gpu=gpu, damp=damp, scale=scale, recursion_depth=recursion_depth, score=score))

    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec

def s_test(args, z_origin, z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000, score=True):
    if score:
        v = grad_z(args, z_test, t_test, model) ###
        if args.perturbation:
            v = tuple(v[0] - grad_z(z_origin, t_test, model, gpu)[0])
    else:
        for i in range(len(z_test)):
            temp = grad_z(z_test[i], t_test[i], model, gpu) ###
            if args.perturbation:
                temp = tuple(temp[0] - grad_z(z_origin[i], t_test[i], model, gpu)[0])
            if i == 0:
                v = temp #.copy()
            else:
                v += temp

    h_estimate = v
    #out_norm = []
    #diff_norm = []
    
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop

        np.random.seed(100)
        j = np.random.randint(len(z_loader))
        x, t = z_loader.collate_fn([z_loader.dataset[j]])
        if gpu >= 0:
            x, t = x.cuda(), t.cuda()
        
        y, _ = model(x)
        loss = get_loss(args, y, t)
        params = [ p for p in model.parameters() if p.requires_grad ]
        hv = hvp(loss, params, h_estimate)
        
        # Recursively caclulate h_estimate
        with torch.no_grad():
            h_estimate = [
            _v + (1 - damp) * _h_e - _hv / scale
            for _v, _h_e, _hv in zip(v, h_estimate, hv)]
        
        if i == 0:
            last_h = h_estimate
        else:
            diff_norm_ = h_estimate[0] - last_h[0]
            diff_norm = sum([h_.norm() for h_ in diff_norm_])
            if diff_norm < 1e-2:
                #print('early break!')
                break
            else:
                last_h = h_estimate
        #out_norm.append(out_n)
        #diff_norm.append(diff_n)
        torch.cuda.empty_cache()
        #display_progress("Calc. s_test recursions: ", i, recursion_depth)
    #print(out_norm)
    #print(diff_norm)
    #print('\n')
    return h_estimate

def calc_weight_bfgs(model, z_train, t_train, H_inv):
    grad_train_ = grad_z(z_train, t_train, model)
    grad_train = grad_train_.view(-1).unsqueeze(1)
    s_test_vec = -torch.mm(H_inv, grad_train)
    return s_test_vec

def grad_z_my(args, z, t, model, gpu=0):
    model.train() #.eval()

    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y, _ = model(z)
    loss = get_loss(args, y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    result = list(grad(loss, params))[-1] ###
    model.zero_grad()
    #pdb.set_trace()
    torch.cuda.empty_cache()
    return result

def grad_z(args, z, t, model, gpu=0):
    model.train() #.eval()

    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y, _ = model(z)
    loss = get_loss(args, y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    result = list(grad(loss, params))###grad(loss, params)[-1]#list(grad(loss, params))[0]
    model.zero_grad()
    torch.cuda.empty_cache()
    return result

def hvp(y, w, v):
    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True) ###

    # Elementwise products
    elemwise_products = 0

    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True) ###

    return return_grads

def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):

    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()

def get_loss(args, y, t):
    #Calculates the loss
    if args.dataset == 'IPS':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y, t)
    else:
        criterion = nn.BCELoss()
        loss = criterion(y, t.float())
    
    return loss

def multi_hot_to_ori(args, new_data):
    data = new_data.squeeze(0)
    
    if args.dataset == 'MALWARE':
        out = [(torch.nonzero(data-1e-5)+1).squeeze(1).tolist()]
    else:
        out = []
        for item in data:
            temp = (torch.nonzero(item-1e-5)+1).squeeze(1).tolist()#-1e-5
            if len(temp) == 0:
                break
            else:
                out.append(temp)
    return out
    
def proportion_compare(args):
    '''
    0: times of code i appeals in sample with class 0
    1: times of code i appeals in sample with class 1
    2: probability of code i appeals in sample with class 0
    3: probability of code i appeals in sample with class 1
    4: normalized result of 3
    5: mutual information 0
    6: mutual information 1
    '''
    num_total = args.train_num 
    num_0 = args.class0_num 
    num_1 = args.class1_num 
    
    print('Start comparing the proportion of codes in dataset.')
    torch.set_printoptions(precision=2, sci_mode=False)
    with open('../data/' + args.dataset + '/train_data_random.json', 'r') as f:
        clean_data = json.load(f)
    with open('../data/' + args.dataset + '/poisoned_data/' + args.new_dataset_name + '.json', 'r') as f:
        poison_data = json.load(f)
    
    result_c = torch.zeros((args.code_range, 7)) + 1e-5
    result_p = torch.zeros((args.code_range, 7)) + 1e-5
    
    for item in clean_data:
        visit = item['data']
        label = item['label']
        
        t = []
        for v in visit:
            t.extend(v)
        s = list(set(t))
        for code in s:
            result_c[code-1, label] += 1
    
    for i, item in enumerate(poison_data):
        visit = item['data']
        label = item['label']
        
        t = []
        for v in visit:
            t.extend(v)
        s = list(set(t))
        for code in s:
            result_p[code-1, label] += 1
            
    result_c[:, 2] = result_c[:, 0] / num_0
    result_c[:, 3] = result_c[:, 1] / num_1
    result_c[:, 4] = result_c[:, 1] / (result_c[:, 0] + result_c[:, 1] + 0.0)
    rank_c = torch.argsort(result_c[:, 4], descending=True) + 1
    result_c[:, 5] = (result_c[:, 0] * num_0) / ((result_c[:, 0] + result_c[:, 1]) * num_total)
    result_c[:, 6] = (result_c[:, 1] * num_1) / ((result_c[:, 0] + result_c[:, 1]) * num_total)
    
    result_p[:, 2] = result_p[:, 0] / num_0
    result_p[:, 3] = result_p[:, 1] / num_1
    result_p[:, 4] = result_p[:, 1] / (result_p[:, 0] + result_p[:, 1] + 0.0)
    rank_p = torch.argsort(result_p[:, 4], descending=True) + 1
    result_p[:, 5] = (result_p[:, 0] * num_0) / ((result_p[:, 0] + result_p[:, 1]) * num_total)
    result_p[:, 6] = (result_p[:, 1] * num_1) / ((result_p[:, 0] + result_p[:, 1]) * num_total)
    
    print('Code distribution rank of clean dataset: ')
    print(rank_c[:100])
    
    print('Code distribution rank of poison dataset: ')
    print(rank_p[:100])

    with open(args.save_cleanprop_path, 'a+') as f:
        for item in result_c:
            f.write(str(item)+'\n')
        
    with open(args.save_poisonprop_path, 'a+') as f:
        for item in result_p:
            f.write(str(item)+'\n')
            
    print('Finished.')
    return None