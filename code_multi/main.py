import torch
import pdb
import time
import json
import opts
import os
import sys
import math
import pynvml
import numpy as np
import torch.nn as nn
from kmeans_pytorch import kmeans
from copy import deepcopy
import torch.nn.functional as F
from collections import Counter

#import pytorch_influence_functions as ptif
from attack_function import *
from attack_tools import *

from dataloader import get_loader, multi_hot_to_ori
#from dataloader_con import get_loader
#from visualization import visualize, final_visual

def main(args):
    
    #torch.backends.cudnn.enabled=True ###
    #torch.backends.cudnn.benchmark=True
    
    if args.dataset == 'EHR' or args.dataset == 'new_EHR':
        if args.model_type == 'lstm':
            from model import LSTM_Model as Model
        elif args.model_type == 'cnn':
            from model import TextCNN as Model
        elif args.model_type == 'fnn':
            from model import FNN_Model as Model
    elif args.dataset == 'MALWARE':
        if args.model_type == 'lstm':
            pass
        elif args.model_type == 'cnn':
            pass
        elif args.model_type == 'fnn':
            from model import FNN_malware as Model
    elif args.dataset == 'IPS':
        if args.model_type == 'fnn':
            from model import FNN_IPS as Model
    
    model = Model(args)
    params = list(model.parameters())
    if torch.cuda.is_available():
        model = model.cuda()
    
    #training
    if args.train:
        start = time.time()
        model = training_process(args, model, params) #train model
        end = time.time()
        print('EHR training time: ', str(end-start))
  
    #influence
    if args.attack:
        #load model
        model.load_state_dict(torch.load(args.reload_model_path, map_location=torch.device('cpu')))
        print('Load model from: ', args.reload_model_path)
        
        #for k,v in model.named_parameters():
        #    if k != 'mlp.weight':
        #        v.requires_grad = False
        
        print('# Model parameters:', sum(param.numel() for param in model.parameters()))
        '''
        t1_loader = get_loader(args, 'train', batch_size=1)
        t2_loader = get_loader(args, 'test', batch_size=1)
        for i, (visits, labels) in enumerate(t1_loader):
            visits = visits.cuda()
            with torch.no_grad():
                _, out = model(visits)
            out = torch.cat((out.cpu(), labels.unsqueeze(1)), dim=1)
            if i == 0:
                clean_vec = out
            else:
                clean_vec = torch.cat((clean_vec, out))
            visits, out = 0, 0
            torch.cuda.empty_cache()

        torch.save(clean_vec, args.clean_embedding_path)
        
        for i, (visits, labels) in enumerate(t2_loader):
            visits = visits.cuda()
            with torch.no_grad():
                _, out = model(visits)
            out = torch.cat((out.cpu(), labels.unsqueeze(1)), dim=1)
            if i == 0:
                test_vec = out
            else:
                test_vec = torch.cat((test_vec, out))
            visits, out = 0, 0
            torch.cuda.empty_cache()
        torch.save(test_vec, args.test_embedding_path)
        pdb.set_trace()
        '''
        #evaluate the model first
        accs, cm, pc = evaluation(args, model)

        initial_acc = np.mean(accs)
        print('Initial accuracy: {:.3f}'.format(initial_acc))
        print_confusion_matrix(cm)

        #get test data loader
        testloader = get_loader(args, 'test')
        
        #prepare training
        config = get_default_config()
        
        #start change
        total_acc = []
        total_pre = []
        total_group_acc = []
        total_time = []

        target = select_target(args, model, testloader)

        step_num = 1
        best_pre = 0.0
        num_pc = 1
        for num_c in range(args.num_changes_stepbystep):
            
            print('\nCurrent step: ' + str(num_c+1))
            start_time = time.time()
            if num_c == 0:
                visual = None
                    
            if args.BFGS:
                ids, modi_indexes, new_data, new_labels, H_inv = poison_sample_crafting(args, config, model, trainloader, testloader, len(trainloader))
            elif args.ntk:
                trainloader = get_loader(args, 'train', visual=visual)
                worst_block_id, new_data, new_labels, cluster_ids = poison_sample_crafting(args, config, model, trainloader, testloader, len(trainloader), target)
                conclusion = None
                modi_id = worst_block_id
                train_index = modi_id
                print('Worst block id: ', worst_block_id)
            else:
                #get base samples (choose samples to be changed. Not used in NTK method)
                print('Start getting base samples.')

                #if num_c == 0:
                train_index = get_base_samples(args, model, target, visual=visual) ###,ind
                #    group1 = train_index
                #else:
                #    train_index, ind = get_base_samples(args, model, target, visual=visual, group1=group1) ###
                '''
                if num_c == 0:
                    index_0 = ind
                elif num_c == 1:
                    index_1 = ind
                    for i, item in enumerate(index_1):
                        if item > args.train_num:
                            index_1[i] = index_0[item-args.train_num]
                    print('Train index overlap: ' ,len(list(set(index_0).intersection(set(index_1)))) / len(index_0))
                    break
                '''
                cluster_ids = []

                #get edited data point
                trainloader = get_loader(args, 'train', train_index=train_index, batch_size=len(train_index), visual=visual)
                
                ids, modi_indexes, new_data, new_labels, conclusion = poison_sample_crafting(args, config, model, trainloader, testloader, len(trainloader.dataset), target=target)
                modi_id = train_index
            
            end_time = time.time()
            old_data_list = []
            print('Conclusion Matrix: ')
            torch.set_printoptions(precision=3)
            print(conclusion)

            #update the model
            if args.retrain:
                if args.dataset_craft == 'aggregate':
                    args.new_dataset_name = args.new_dataset_name + '_' + str(num_c+1) #!

                new_dataloader = get_loader(args, 'train', additional_data=new_data, modi_id=modi_id, labels=new_labels, shuffle_bool=True, visual=visual) #full
                #proportion_compare(args)
                print('New number of dataset: ', len(new_dataloader.dataset))
                #visualization
                #visualize(args, model, train_index)
                target_logit = []
                for tt in target:
                    with torch.no_grad():
                        t_logit, _ = model(tt[0].cuda())
                    target_logit.append(t_logit)

                ori_model = model
                if num_c == 0:
                    original_logit = target_logit
                    print('original logit: ', original_logit)

                if args.retrain_from_scratch:
                    model = Model(args) ###retrain from scratch
                    retrain_params = list(model.parameters()) #[list(model.parameters())[-1]]
                else:
                    retrain_params = [list(model.parameters())[-1]]

                if torch.cuda.is_available():
                    model = model.cuda()
                
                visual = args.new_dataset_name
                model, avg_acc, pc, group_acc = training_process(args, model, retrain_params, target, target_logit, retrain=True, visual=visual, train_index=train_index)
                if pc > best_pre:
                    best_pre = pc
                    num_pc = num_c+1

                print('Overall best precision call: ', best_pre, 'at step: ', (num_pc), 'Improve: ', (best_pre-0.681))
                #final_visual(args, ori_model, model, train_index, cluster_ids)
                #pdb.set_trace()
            else: #update based on influence estimation
                if args.BFGS:
                    s_modi_vec = calc_weight_bfgs(model, new_data, new_label, H_inv)
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if 'mlp.weight' in name:
                                param.add_(-s_modi_vec.view(2, -1), alpha=args.change_lr)
                
                else:
                    s_modi_vec = calc_s_test_single(args, model, old_data_list, new_data, new_labels, trainloader, gpu=config['gpu'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])
                    #print(model.mlp.weight)
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if 'mlp.weight' in name:
                                vec = s_modi_vec[0].clone()
                                par = param.data.clone()
                                #param.data = par + vec * (-1/len(trainloader.dataset))
                                break      
                    par = par + vec * (1/len(trainloader.dataset))

            #evaluate the result of poisoning
            torch.cuda.empty_cache()
            
            total_time.append(end_time-start_time)
            total_acc.append(avg_acc)
            total_pre.append(pc)
            total_group_acc.append(group_acc)

            if args.dataset == 'IPS':
                correct_count = 0
                for index in range(len(avg_acc)):
                    if torch.argmax(avg_acc[index], dim=-1) == args.base_class_id:
                        correct_count += 1
                if correct_count == len(avg_acc):
                    step_num = num_c+1
                    print('Stop step: ', str(step_num), 'Sample Num: ', str(step_num*args.base_top_num/1000.0), 'Sample percent: ', str(step_num*args.base_top_num/1000.0/3.0*100), '%')
                    #print('Logit Change: ', str(original_logit-avg_acc), 'Change Rate: ', str((original_logit-avg_acc)/step_num))
                    break
            else:
                correct_count = 0
                for index in range(len(avg_acc)):
                    if int(avg_acc[index] > 0.5) == args.base_class_id:
                        correct_count += 1
                if correct_count == len(avg_acc):
                    step_num = num_c+1
                    print('Stop step: ', str(step_num), 'Sample Num: ', str(step_num*args.base_top_num/1000.0), 'Sample percent: ', str(step_num*args.base_top_num/1000.0*6.25), '%')
                    #print('Logit Drop: ', str(original_logit-avg_acc), 'Drop Rate: ', str((original_logit-avg_acc)/step_num))
                    break
            #if num_c % args.poison_eval_step == 0:
            #    accs, _ = evaluation(args, model)
            #    print('Num_changes: ', num_c+1, '\nInitial evaluation accuracy: {:.5f}'.format(initial_acc), '\nEvaluation accuracy: {:.5f}\n'.format(np.mean(accs)))

        print('\nConclude time: ', total_time)
        print('Total time: ', sum(total_time))
        print('Average time: ', sum(total_time)/(step_num*60), ' minutes')
        print('Total precison recall: ', total_pre)
        print('Total group accuracy: ', total_group_acc)
        #print('Conclude average accuracy: ', total_acc)

def clustering(args, logits, scores):
    train_index = []
    clean = torch.load(args.clean_embedding_path)
    clean_samples = clean[:, 0:-1]
    clean_labels = clean[:, -1]
    clean1, clean0 = clean_samples[clean_labels==1, :], clean_samples[clean_labels==0, :]
    if base_class_id == 0:
        clean = clean0
    else:
        clean = clean1
    top_logit_index = []

    if args.clustering_method == 'kmeans':
        cluster_ids, _ = kmeans(X=clean, num_clusters=args.cluster_num, distance='euclidean', device=torch.device('cuda:0'))
    elif args.clustering_method == 'dbscan':
        cluster_ids = DBSCAN(eps=0.1, min_samples=100).fit_predict(clean)
    elif args.clustering_method == 'knn':
        graph = kneighbors_graph(clean, 3, mode='connectivity').toarray()
        G = nx.from_numpy_matrix(graph)
        con_com = nx.connected_components(G)
        cluster_ids = torch.zeros(len(clean))
        for i, item in enumerate(con_com):
            for node in item:
                cluster_ids[node] = i
    
    base_cluster_ids = list(set(cluster_ids.tolist()))
    print('There are %d clusters in total' % len(base_cluster_ids))
    class_ids = (torch.zeros(len(logits)) - 1).long()
    real_ids = torch.where(clean_labels == base_class_id)[0]
    for i in range(len(cluster_ids)):
        class_ids[real_ids[i]] = cluster_ids[i]

    for base in base_cluster_ids:
        temp_logits = deepcopy(logits)
        for index in range(len(temp_logits)):
            if class_ids[index] != base:
                temp_logits[index] = 1.0
        temp_top = torch.topk(temp_logits, k=args.cluster_size, largest=False)[1]
        
        temp_scores = deepcopy(scores)
        for num in range(len(temp_logits)):
            if num not in temp_top:
                temp_scores[num] = 100
        train_index_ = torch.topk(temp_scores, k = int(top_num/len(base_cluster_ids)), largest=False)[1]
        train_index.extend(train_index_.tolist())

    return train_index, cluster_ids

def get_base_samples(args, model, target, visual=None, group1=None):
    def judgement(args, num, base_class_id, tlogit, t_train, group1=None):
        #with open('../data/MALWARE/gradient_top3000.json', 'r') as f:
        #    gradient_3k = json.load(f)
        #with open('../data/MALWARE/loss_top3000.json', 'r') as f:
        #    loss_3k = json.load(f)

        if args.dataset == 'IPS':
            if tlogit[0, t_train] > 0.5:
                return True
        else:
            if (base_class_id == 1 and tlogit > 0.5) or (base_class_id == 0 and tlogit < 0.5):
                if group1 is None:
                    return True
                else:
                    if num in group1 or num > args.train_num-1:
                        return True
        return False

    config = get_default_config()
    originloader = get_loader(args, 'train', visual=visual)
    testloader = get_loader(args, 'test')
            
    #choose training data to change
    top_num = args.base_top_num #100 / 200
    base_class_id = args.base_class_id
    train_index_0 = []
    train_index_1 = []
    base_value = 100
    #train_index = []

    rand_index = torch.Tensor(np.random.randint(0, len(originloader), 200)).int()
    if args.base_sample_choice != 'logit':
        target_hots = []
        for tt in target:
            target_hot = 1 - ((tt[0] - 1e-5) == 0).int()
            target_hots.append(target_hot)
        
        logits = torch.zeros(len(originloader.dataset)) + 1.0
        scores = torch.zeros(len(originloader.dataset)) + base_value

        for num in range(len(originloader.dataset)):
            score = 0
            #for num in rand_index:

            _, _, label = originloader.dataset[num]
            z, t_train = originloader.collate_fn([originloader.dataset[num]])

            if t_train == base_class_id:
                #train_index_1.append(num)
                
                if torch.cuda.is_available():
                    z_origin = z.cuda()
                with torch.no_grad():
                    tlogit, emb = model(z_origin)

                if judgement(args, num, base_class_id, tlogit, t_train, group1):
                    #logits[num] = abs(tlogit[0, t_train]-0.5)
                    #logits[num] = abs(tlogit-0.5)
                    if args.base_sample_choice == 'influence':
                        grad_test = grad_z(args, target[0], target[1], model)
                        score = calc_influence_single(args, model, z_origin, z_origin, torch.LongTensor([label]), originloader, testloader,
                                                grad_test, gpu=config['gpu'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])
                    elif args.base_sample_choice == 'align':
                        grad_train = grad_z(args, z_origin, t_train, model)
                        grad_test = grad_z(args, target[0], target[1], model)
                            
                        for i in range(len(grad_train)):
                            g1, g2 = grad_test[i], grad_train[i]
                            nm = args.dot_norm
                            if args.alignment_calc == 'cosine':
                                if len(g1.size()) == 1:
                                    score += F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).sum()
                                else:
                                    score += F.cosine_similarity(g1, g2).sum()
                            elif args.alignment_calc == 'dot':
                                if len(g1.size()) == 1:
                                    score += sum(torch.norm(g1, p=float(nm)) * torch.norm(g2, p=float(nm)) * F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)))
                                else:
                                    #score += torch.matmul(g1, g2.T).trace().sum()
                                    score += sum(torch.norm(g1, p=float(nm), dim=-1) * torch.norm(g2, p=float(nm), dim=-1) * F.cosine_similarity(g1, g2))
                    elif args.base_sample_choice == 'dist':
                        with torch.no_grad():
                            _, emb_target = model(target[0].cuda())
                        score = torch.sqrt(((emb - emb_target)**2).sum())
                    elif args.base_sample_choice == 'jac_dist':
                        z_hot = 1 - ((z - 1e-5) == 0).int()
                        for t_hot in target_hots:
                            score += - (z_hot & t_hot).sum() / (z_hot | t_hot).sum().float()
                    #if t_train == 1: #and len(train_index_1) < top_num:
                    scores[num] = score
            torch.cuda.empty_cache()
            #display_progress("Calculate scores for base samples: ", num, len(originloader.dataset))
            
        if args.base_sample_random:
            candidate_index = torch.nonzero(scores - base_value).squeeze(1)
            train_index = torch.Tensor(np.random.choice(candidate_index, top_num)).int().tolist()
        else:
            cluster_ids = []
            if args.use_clustering:
                train_index, cluster_ids = clustering()
            else:
                #top_logit_index = (torch.topk(logits, k = 5000, largest=False)[1]).tolist()

                #for num in range(len(originloader.dataset)):
                #    if num not in top_logit_index:
                #        scores[num] = 100
                train_index = torch.topk(scores, k = top_num, largest=False)[1].int().tolist() #align:False, influence:True/False ###
                #ind = torch.topk(scores, k = 500, largest=False)[1].int().tolist()
        print(train_index)
        return train_index#, ind

    elif args.base_sample_choice == 'logit':
        for num in range(len(originloader.dataset)):
            #for num in rand_index:
            z_origin, t_train = originloader.collate_fn([originloader.dataset[num]])
            
            if torch.cuda.is_available():
                z_origin = z_origin.cuda()
            tlogit, _ = model(z_origin)
            z_origin = 0

            if tlogit - 0.5 < 0.5 and tlogit - 0.5 > 0.0 and t_train == 1 and len(train_index_1) < top_num: #tlogit - 0.5 > 0.4 / tlogit - 0.5 < 0.1 and tlogit - 0.5 > 0.0
                train_index_1.append(num)
            #elif tlogit - 0.5 < 0.0 and tlogit - 0.5 > -0.1 and t_train == 0 and len(train_index_0) < top_num: #tlogit - 0.0 < 0.1 /tlogit - 0.5 < 0.0 and tlogit - 0.5 > -0.1
            #    train_index_0.append(num)
                
            if len(train_index_1) == top_num:# and len(train_index_0) == top_num:
                #if len(train_index_1) == top_num:
                break
            else:
                continue
            if torch.isnan(tlogit):
                pdb.set_trace()
    
    train_index = train_index_0 + train_index_1
    print('Num of chosen training data: ', len(train_index), ' train_0: ', len(train_index_0), ' train_1: ', len(train_index_1))

    return train_index, grad_test

def print_confusion_matrix(cm):
    if cm.size(0) == 2:
        print('{:<6} {:<2} {}'.format(' ', '0', '1'))
        print('{:<5} {} {}'.format('0', cm[0][0].item(), cm[0][1].item()))
        print('{:<5} {} {}'.format('1', cm[1][0].item(), cm[1][1].item()))
    elif cm.size(0) == 3:
        print('{:<6} {:<2} {:<2} {:<2}'.format(' ', '0', '1', '2'))
        print('{:<5} {} {} {}'.format('0', cm[0][0].item(), cm[0][1].item(), cm[0][2].item()))
        print('{:<5} {} {} {}'.format('1', cm[1][0].item(), cm[1][1].item(), cm[1][2].item()))
        print('{:<5} {} {} {}'.format('2', cm[2][0].item(), cm[2][1].item(), cm[2][2].item()))
    return None

def find_last(li):
    n_0 = 0
    for k in reversed(li):
        if k == 0:
            n_0 += 1
        else:
            break
    if n_0 == 0:
        return li.tolist()
    else:
        return li[:-n_0].tolist()

def train_eval(args, model, train_index, poison_dataloader, target):
    model.eval()
    cm_p, cm_c = torch.zeros(2, 2).int(), torch.zeros(2, 2).int()
    clean_dataloader = get_loader(args, 'train')
    print('There are {} samples to be poisoned in total'.format(str(len(train_index))))
    
    for index in train_index:
        vp, lp = poison_dataloader.collate_fn([poison_dataloader.dataset[index]])
        vc, lc = clean_dataloader.collate_fn([clean_dataloader.dataset[index]])
        
        if torch.cuda.is_available():
            vp, lp, vc, lc = vp.cuda(), lp.cuda(), vc.cuda(), lc.cuda()
        
        with torch.no_grad():
            logit_p, _ = model(vp)
            logit_c, _ = model(vc)
            
        if args.dataset == 'IPS':
            cm_p[lp.long(), torch.argmax(logit_p, dim=-1).long()] += 1
            cm_c[lc.long(), torch.argmax(logit_c, dim=-1).long()] += 1
        else:
            cm_p[lp.long(), int(logit_p>0.5)] += 1
            cm_c[lc.long(), int(logit_c>0.5)] += 1
            
        torch.cuda.empty_cache()
    
    print('The prediction results on original clean data: ')
    print_confusion_matrix(cm_c)
    print('The prediction results on poisoned data: ')
    print_confusion_matrix(cm_p)
    
    return None

def evaluation(args, model):
    torch.set_printoptions(precision=3, sci_mode=False)
    print(model)
    evalloader = get_loader(args, 'test')
    model.eval()
    accs = []
    logits = []
    cm = torch.zeros(args.class_num, args.class_num).int()
    
    for i, (visits, labels) in enumerate(evalloader):

        with torch.no_grad():
            logit, _ = model(visits.cuda())
        
        logit = logit.cpu()

        if args.dataset != 'IPS':
            acc = torch.div(((logit > 0.5).int() == labels).int().sum(), float(labels.size(0)))
            accs.append(acc)
            logits.append(logit)
            for t, p in zip(labels, (logit > 0.5).int()):
                cm[t.long(), p.long()] += 1
            
        else:
            acc = (torch.argmax(logit, dim=-1)==labels).float().mean()
            accs.append(acc)
            #logits.append(logit)
            for t, p in zip(labels, torch.argmax(logit, dim=-1)):
                cm[t.long(), p.long()] += 1
        
        if i == 0:
            base_logit = logit
        torch.cuda.empty_cache()

    print('The evaluation accuracy: ', np.mean(accs))
    #print('The logit of the base sample (true label is 0): ', base_logit)
    print_confusion_matrix(cm)###

    if args.dataset != 'IPS':
        FNR = torch.div(cm[1,0].float(), cm[1,0]+cm[1,1])
        FPR = torch.div(cm[0,1].float(), cm[0,1]+cm[0,0])
        F1 = torch.div(2*cm[1,1].float(), 2*cm[1,1]+cm[0,1]+cm[1,0])
        print('FNR: ', FNR, ' FPR: ', FPR, ' F1: ', F1)
        precision_call = torch.div(cm[1,1].float(), cm[1,0]+cm[1,1])
        print('precision_call: ', precision_call)
    else:
        precision_call = torch.div(cm[0,0].float(), cm[0,0]+cm[0,1]+cm[0,2])
        print('precision_call: ', precision_call)
        
    return accs, cm, precision_call

def training_process(args, model, params, target=[0], target_logit=None, conclusion=None, retrain=False, visual=None, train_index=None):
    if args.dataset == 'EHR' or args.dataset == 'new_EHR':
        if args.model_type == 'lstm':
            from model import LSTM_Model as Model
        elif args.model_type == 'cnn':
            from model import TextCNN as Model
        elif args.model_type == 'fnn':
            from model import FNN_Model as Model

    epoch = 1
    num_eval_avg = 5
    cm_final = torch.zeros(args.class_num,args.class_num).int()
    dataloader = get_loader(args, 'train', shuffle_bool=True, visual=visual)
    total_step = len(dataloader)
    total_train_point = len(dataloader.dataset)
    torch.set_printoptions(precision=3, sci_mode=False)
    
    print('# Model parameters:', sum(param.numel() for param in model.parameters()))
    print('Total step: ', total_step, 'Total train data: ', total_train_point)
    #print('Target: ', target[0])
    best_acc = -1.0
    best_pre = -1.0
    best_epoch = 1
    best_group_acc = -1.0
    avg_logit = torch.zeros(len(target)).tolist()
    worst_acc = 100.0
    lr = args.learning_rate
    '''
    test_criterion = nn.BCELoss(reduction='none')
    initial_loss, final_loss = torch.zeros(total_train_point), torch.zeros(total_train_point)
    gradient_norm = torch.zeros(total_train_point)
    params = [ p for p in model.parameters() if p.requires_grad ]
    '''
    while epoch < args.num_epochs+1:
    #for epoch in range(start_epoch, args.num_epochs + 1):
    
        if epoch > args.lr_decay and (epoch - 10) % 8 == 0:
            lr = lr * 0.8

        print('Learning Rate for Epoch %d: %.6f' % (epoch, lr))
        
        if args.dataset == 'EHR':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(args.alpha, args.beta))
        else:
            optimizer = torch.optim.Adam(params, lr=lr, betas=(args.alpha, args.beta), weight_decay=0.01)#, weight_decay=0.01
        #optimizer = torch.optim.SGD(params, lr=lr, momentum=0.0)
        
        print('------------------Training for Epoch %d----------------'%(epoch)) 

        model.train()

        for i, (visits, labels) in enumerate(dataloader):
            optimizer.zero_grad()
 
            #print(torch.cuda.max_memory_allocated(device=0))
            logits, _ = model(visits.cuda())

            loss = get_loss(args, logits, labels.cuda())
            if math.isnan(loss):
                pdb.set_trace()
            loss.backward()

            optimizer.step()
            
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % \
                      (epoch, args.num_epochs, i, total_step, loss))
            '''
            #logits, _ = model(visits)
            #ls = test_criterion(logits, labels.float())
            logits, _ = model(visits)
            loss = get_loss(args, logits, labels)
            #loss
            if epoch == 1:
                try:
                    initial_loss[i*args.batch_size:(i+1)*args.batch_size] = loss
                except:
                    initial_loss[i*args.batch_size:] = loss
            elif epoch == args.num_epochs:
                try:
                    final_loss[i*args.batch_size:(i+1)*args.batch_size] = loss
                except:
                    final_loss[i*args.batch_size:] = loss
            #gradient
            result = list(grad(loss, params))
            norm = torch.norm(torch.cat(tuple([x.reshape(1,-1) for x in result]), 1))
            gradient_norm[i] += norm.detach().cpu()
            '''
            logits = 0
            torch.cuda.empty_cache()
        #evaluation
        #test target
        
        logit = []
        group_logit = torch.zeros(len(target), args.class_num).squeeze(1)
        for i, tt in enumerate(target):
            with torch.no_grad():
                lt, _ = model(tt[0].cuda())
            logit.append(lt)
            group_logit[i] = lt
        if args.dataset != 'IPS':
            group_acc = ((group_logit>0.5).int() == args.target_class_id).float().mean()
        else:
            group_acc = (torch.argmax(group_logit, dim=-1) == args.target_class_id).float().mean()

        print('Intended class: ' + str(args.base_class_id) + '. Target class: ' + str(args.target_class_id))
        print('Original logit: ', target_logit, '\nLogit: ', logit)
        print('Group Acc: ', group_acc)

        if epoch > args.num_epochs - 5:
            for le in range(len(avg_logit)):
                avg_logit[le] += logit[le]
        if epoch == args.num_epochs:
            for le in range(len(avg_logit)):
                avg_logit[le] = avg_logit[le] / 5.0
            print('Average logit over final 5 epochs: ', avg_logit)
        
        #if retrain == True:
        #    train_eval(args, model, train_index, dataloader, target)
        accs, cm, precision_recall = evaluation(args, model)
        if epoch > args.num_epochs - 5:
            cm_final += cm
            
        if not retrain:
            #save model
            if np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
                print('Best accuracy: ', best_acc)
                #torch.save(model.state_dict(), args.save_model_path)
                print('The checkpoint of the model has been saved successfully: ', args.save_model_path)
        else:
            if epoch == args.num_epochs:
                torch.save(model.state_dict(), './check_point/'+args.dataset+'/retrained_model.pth')
                print('The checkpoint of the model has been saved successfully')

            #print(model.mlp.weight)
            if np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
            print('Best accuracy: ', best_acc)
            
            if group_acc > best_group_acc:
                best_group_acc = group_acc
                best_pre = precision_recall
                best_epoch = epoch
            print('Best precision call: ', best_pre, 'Improve: ', (best_pre-0.681), 'At epoch: ', best_epoch) #IPS:0.681
            
            #if group_acc == 1.0 and np.mean(accs) > 80.39:
            #    for le in range(len(avg_logit)):
            #        avg_logit[le] = logit[le]
            #    return model, avg_logit, best_pre, best_group_acc
        epoch += 1
        '''
        if (epoch == args.num_epochs+1) and cm[1,1]*cm[0,0] == 0 and args.dataset == 'EHR':
            epoch = 1
            print('Retrain does not count, do it again!\n')
            model = Model(args)
            model = model.cuda()
            best_acc = 0.0
            lr = args.learning_rate
            cm_final = torch.zeros(args.class_num,args.class_num).int()
            dataloader = get_loader(args, 'train', shuffle_bool=True, visual=visual)
        '''

    print('\n Final evaluation results (avged): ') 
    cm = torch.div(cm_final, num_eval_avg)
    print('Average accuracy: ', torch.div(torch.trace(cm), args.test_num))
    print_confusion_matrix(cm)
    if args.dataset != 'IPS':
        FNR = torch.div(cm[1,0].float(), cm[1,0]+cm[1,1])
        FPR = torch.div(cm[0,1].float(), cm[0,1]+cm[0,0])
        F1 = torch.div(2*cm[1,1].float(), 2*cm[1,1]+cm[0,1]+cm[1,0])
        print('FNR: ', FNR, ' FPR: ', FPR, ' F1: ', F1)
        
    return model, avg_logit, best_pre, best_group_acc

if __name__ == '__main__':
    args = opts.parse_opt()
    print(args)
    main(args)
    
'''
additional_data = [{} for i in range(len(new_data))]
            
for i in range(len(new_data)):
    #n_d = [find_last(li) for li in new_data.squeeze(0)]
    
    if args.ntk:
        n_d = new_data[i]
        additional_data[i]['modi_id'] = int(worst_block_id[i])
    else:
        n_d = multi_hot_to_ori(new_data[i]) ###
        train_sample_ids = torch.IntTensor(range(len(trainloader.dataset))).repeat(args.random_num*5, 1).T.reshape(-1)
        additional_data[i]['modi_id'] = train_index[train_sample_ids[i]]#train_index[train_sample_ids[i]] / train_index[i]
    
    additional_data[i]['data'] = n_d
    additional_data[i]['label'] = int(new_labels[i])
'''
'''
t1_loader = get_loader(args, 'train', batch_size=1)
t2_loader = get_loader(args, 'test', batch_size=1)
for i, (visits, labels) in enumerate(t1_loader):
    visits = visits.cuda()
    with torch.no_grad():
        _, out = model(visits)
    out = torch.cat((out.cpu(), labels.unsqueeze(1)), dim=1)
    if i == 0:
        clean_vec = out
    else:
        clean_vec = torch.cat((clean_vec, out))
    visits, out = 0, 0
    torch.cuda.empty_cache()
torch.save(clean_vec, './visual/clean_vecs.pt')

for i, (visits, labels) in enumerate(t2_loader):
    visits = visits.cuda()
    with torch.no_grad():
        _, out = model(visits)
    out = torch.cat((out.cpu(), labels.unsqueeze(1)), dim=1)
    if i == 0:
        test_vec = out
    else:
        test_vec = torch.cat((test_vec, out))
    visits, out = 0, 0
    torch.cuda.empty_cache()
torch.save(test_vec, './visual/test_vecs.pt')
pdb.set_trace()
'''