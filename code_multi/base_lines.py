import torch
import torch.nn as nn
import pdb
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import grad, Variable

def frogs(args, max_num1, ori_sample, label, model, train_loader, test_loader, target):
    maxIter = 1000 #1000
    lamda = 3 #IPS:0.2 malware:1 ehr: 
    beta = 0.001 #IPS:0.001 malware: 0.001 ehr: 
    feature_num = 0
    for i, tt in enumerate(target):
        with torch.no_grad():
            _, emb_t = model(tt[0].cuda())
        if i == 0:
            emb_target = emb_t
        else:
            emb_target += emb_t

    z_train, _ = train_loader.collate_fn([tuple([max_num1, ori_sample, label])])
    sample = deepcopy(z_train)

    for i in range(maxIter):
        
        weight_of_embed_codes = Variable(sample.data, requires_grad=True)
        with torch.backends.cudnn.flags(enabled=False):
            _, emb_z = model(weight_of_embed_codes.cuda())
        
        diff = (emb_target - emb_z).squeeze(0)
        loss = torch.mm(diff.unsqueeze(0), diff.unsqueeze(1))
        loss.backward(retain_graph=False)
        
        #forward
        gradients = weight_of_embed_codes.grad.data
        if torch.isnan(gradients).sum() > 0:
            pdb.set_trace()
        sample_f = sample - lamda * gradients

        #backward
        sample = (sample_f + lamda * beta * z_train) / (1 + beta * lamda)
    
    final_sample = back_discrete(sample)
    if args.dataset == 'MALWARE':
        code_S_t = torch.nonzero(final_sample - torch.round(z_train)).tolist()[:100]
    elif args.dataset == 'IPS':
        code_S_t = torch.nonzero(final_sample[0].permute(1,0) - torch.round(z_train[0]).permute(1,0)).tolist()[:100]
    else:
        code_S_t = torch.nonzero(final_sample[0,:len(ori_sample),:] - torch.round(z_train[0,:len(ori_sample),:])).tolist()[:100]

    return code_S_t

def witch(args, model, train_loader, test_loader, target):
    R = 8#8
    M = 250 #250
    
    min_score = 1000
    criterion = nn.BCELoss()
    for i, tt in enumerate(target):
        if i == 0:
            grad_test = grad_z(args, tt[0], tt[1], model)
        else:
            grad_test += grad_z(args, tt[0], tt[1], model)

    for (visits, labels) in train_loader:
        batch_num = int(args.base_top_num / args.batch_size)
        pass

    for num in range(R):
        loss = 0.0
        for batch_id in range(batch_num):
            try:
                sample = torch.zeros_like(visits[batch_id*args.batch_size:(batch_id+1)*args.batch_size])
                sample.data = visits[batch_id*args.batch_size:(batch_id+1)*args.batch_size].data
                label = labels[batch_id*args.batch_size:(batch_id+1)*args.batch_size]
            except:
                sample = torch.zeros_like(visits[batch_id*args.batch_size:])
                sample.data = visits[batch_id*args.batch_size:].data
                label = labels[batch_id*args.batch_size:]
            #sample.grad = torch.zeros_like(sample)
            #sample.requires_grad_()
            delta = torch.rand(sample.size())
            delta.grad = torch.zeros_like(sample)
            delta.requires_grad_()
            optimizer = torch.optim.Adam([delta], lr=0.025, betas=(args.alpha, args.beta))

            for j in range(M):
                inputs = sample + delta
                with torch.backends.cudnn.flags(enabled=False):
                    logit, _ = model(inputs.cuda())

                if args.dataset == 'IPS':
                    poison_loss = criterion(logit, F.one_hot(label, num_classes=args.class_num).float().cuda())
                else:
                    poison_loss = criterion(logit, label.float().cuda())
                poison_grad = grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)
                
                for i in range(len(poison_grad)):
                    g1, g2 = grad_test[i], poison_grad[i]
                    if len(g1.size()) == 1:
                        loss += F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).sum()
                    else:
                        loss += F.cosine_similarity(g1, g2).sum()
                
                loss.backward(retain_graph=False)
                delta.grad.sign_()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                
                poison_loss, poison_grad, loss, score = 0, 0, 0, loss.data

            #final_sample = back_discrete(inputs) 
            if score < min_score:
                print('Get better loss: ', score)
                min_score = score
                best_delta = delta
        print('Step num: ', num)

    indexes = []

    vec, ori_num = back_discrete(best_delta[0]), train_loader.dataset[i][0]
    if args.dataset == 'MALWARE':
        code_S_t = torch.nonzero(vec.unsqueeze(0)).tolist()[:100]
    elif args.dataset == 'IPS':
        code_S_t = torch.nonzero(vec.permute(1,0)).tolist()[:100]
    else:
        code_S_t = torch.nonzero(vec[:ori_num,:]).tolist()[:100]

    for i in range(len(visits)):
        indexes.append(code_S_t)

    return indexes

def back_discrete(input):
    input_ = torch.clamp(input, min=0.0, max=1.0)
    output = torch.round(input_)

    return output

def grad_z(args, z, t, model, gpu=0):
    model.train() #.eval()

    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y, _ = model(z)
    loss = get_loss(args, y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    result = list(grad(loss, params))
    model.zero_grad()
    torch.cuda.empty_cache()
    return result

def get_loss(args, y, t):
    #Calculates the loss
    if args.dataset == 'IPS':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y, t)
    else:
        criterion = nn.BCELoss()
        loss = criterion(y, t.float())
    
    return loss