import pdb
import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    
    #data and path info
    parser.add_argument('--dataset', type=str, default='EHR', help='choice of dataset: new_EHR | EHR | MALWARE | IPS')
    parser.add_argument('--model_type', type=str, default='lstm', help='type of used model: fnn | cnn | lstm')
    parser.add_argument('--embedding_name', type=str, default='PretrainedEmbedding.4', help='name of pretrained embedding file')
    parser.add_argument('--new_dataset_name', type=str, default='grad_200_witch_group7_1toH', help='name of new generated poisoned dataset.')
    parser.add_argument('--final_visual_save_name', type=str, default='mlp_gradattack_500_add_final.jpg', help='name of final visual image.')
    parser.add_argument('--save_model_path', type=str, default='./check_point/'+parser.parse_args().dataset+'/model_lstm_20_0.95.pth', help='path of model check point.')#model_lstm_20_0.95.pth (EHR) / model_fnn_decay_0.95.pth (MALWARE/IPS)
    parser.add_argument('--reload_model_path', type=str, default='./check_point/'+parser.parse_args().dataset+'/model_lstm_20_0.95.pth', help='path of model check point.') #model_cnn.pth / model_selfatt_1hot_relu.pth / model_fnn.pth
    parser.add_argument('--clean_embedding_path', type=str, default='./visual/'+parser.parse_args().dataset+'/fnn_clean_vecs.pt', help='clean training embedding vectors') #lstm_clean_vecs.pt / fnn_clean_vecs.pt
    parser.add_argument('--test_embedding_path', type=str, default='./visual/'+parser.parse_args().dataset+'/fnn_test_vecs.pt', help='clean testing embedding vectors') #lstm_test_vecs.pt / fnn_test_vecs.pt
    parser.add_argument('--save_cleanprop_path', type=str, default='./log/'+parser.parse_args().dataset+'/fnn_ntk_1000_base1_clean.txt', help='proportion of clean dataset') 
    parser.add_argument('--save_poisonprop_path', type=str, default='./log/'+parser.parse_args().dataset+'/fnn_ntk_1000_base1_poison.txt', help='proportion of poisoned dataset')
    
    parser.add_argument('--inner_method', type=str, default='dist', help='inner features choosing method (align / error / influence / dist / jac_dist)')
    parser.add_argument('--outer_method', type=str, default='dist', help='outer sample choosing method (align / error / influence / dist)')
    parser.add_argument('--attack_method', type=str, default='witch', help='the method of attacking each sample in the inner loop (atonce / fsgs/ ompgs / gradattack)')
    
    parser.add_argument('--base_class_id', type=int, default=1, help='The label of samples allowed to change')
    parser.add_argument('--target_class_id', type=int, default=1, help='The label of samples that will be misclassified as base class')
    parser.add_argument('--base_top_num', type=int, default=200, help='The number of base samples')
    #parser.add_argument('--target_num', type=int, default=5, help='The number of samples to be altered.')
    parser.add_argument('--num_changes_stepbystep', type=int, default=1, help='The number of allowed changes step by step.') #30
    parser.add_argument('--base_sample_random', type=bool, default=True, help='random choose index based on influence or alignment scores')
    parser.add_argument('--dataset_craft', type=str, default='replace', help='aggregate: add poison into clean dataset aggregately \
                                                    / replace: modify clean point in the dataset / add: add poison into clean dataset')


    #training parameters
    parser.add_argument('--num_epochs', type=int, default=30) #30
    parser.add_argument('--batch_size', type=int, default=32, help='batch size') #8(EHR) / 32(others)
    parser.add_argument('--eval_size', type=int, default=32, help='batch size') #8(EHR) / 32(others)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate for the whole model')#0.002/0.0005
    parser.add_argument('--change_lr', type=float, default=0.03, help='learning rate for posining the model')
    parser.add_argument('--log_step', type=int, default=100, help='step size for printing log info')#10
    parser.add_argument('--poison_eval_step', type=int, default=1, help='step size for evaluation after adding poisoning data point')#10
    parser.add_argument('--lr_decay', type=int, default=10, help='epoch at which to start lr decay')
    parser.add_argument('--alpha', type=float, default=0.8, help='alpha in Adam')                  
    parser.add_argument('--beta', type=float, default=0.999, help='beta in Adam')

    parser.add_argument('--train', type=bool, default=False, help='train the model or not')
    parser.add_argument('--train_shuffle', type=bool, default=False, help='shuffle the data or not')
    parser.add_argument('--attack', type=bool, default=True, help='do poisoning attack or not')
    
    #NTK
    parser.add_argument('--ntk', type=bool, default=False, help='use neural tangent kernel to get gradient')
    parser.add_argument('--block_size', type=int, default=2048, help='the size of training block')#2048
    parser.add_argument('--r', type=int, default=3, help='iteration number in perturbing the sample')
    if parser.parse_args().ntk == True:
        parser.add_argument('--indice_k', type=int, default=150, help='The number of top changable code ids when NOT change codes at once (used in get indices).') #200
    else:
        parser.add_argument('--indice_k', type=int, default=5, help='The number of top changable code ids when NOT change codes at once (used in get indices).') #200
        
    #model parameters
    if parser.parse_args().dataset == 'MALWARE':
        parser.add_argument('--code_range', type=int, default=5000, help='The range of code at MALWARE')
        parser.add_argument('--train_num', type=int, default=20700, help='The number of original training data.')
        parser.add_argument('--test_num', type=int, default=1089, help='The number of testing data.')
        parser.add_argument('--class1_num', type=int, default=10370, help='The number of class 1 training data.')
        parser.add_argument('--class0_num', type=int, default=10330, help='The number of class 0 training data.')
        parser.add_argument('--test1_num', type=int, default=522, help='The number of class 1 testing data.')
        parser.add_argument('--test0_num', type=int, default=567, help='The number of class 0 testing data.')
        parser.add_argument('--class_num', type=int, default=2, help='The number of class.')
    elif parser.parse_args().dataset == 'IPS':
        parser.add_argument('--code_range', type=int, default=1103, help='The range of code at MALWARE')
        parser.add_argument('--train_num', type=int, default=3895, help='The number of original training data.')
        parser.add_argument('--test_num', type=int, default=205, help='The number of testing data.')
        parser.add_argument('--class2_num', type=int, default=1842, help='The number of class 2 training data.')
        parser.add_argument('--class1_num', type=int, default=1057, help='The number of class 1 training data.')
        parser.add_argument('--class0_num', type=int, default=996, help='The number of class 0 training data.')
        parser.add_argument('--test2_num', type=int, default=104, help='The number of class 2 testing data.')
        parser.add_argument('--test1_num', type=int, default=54, help='The number of class 1 testing data.')
        parser.add_argument('--test0_num', type=int, default=47, help='The number of class 0 testing data.')
        parser.add_argument('--class_num', type=int, default=3, help='The number of class.')
    else:
        parser.add_argument('--code_range', type=int, default=4130, help='The range of code at EHR.')
        parser.add_argument('--train_num', type=int, default=6473, help='The number of original training data.')
        parser.add_argument('--test_num', type=int, default=340, help='The number of testing data.')
        parser.add_argument('--class1_num', type=int, default=1923, help='The number of class 1 training data.')
        parser.add_argument('--class0_num', type=int, default=4550, help='The number of class 0 training data.')
        parser.add_argument('--test1_num', type=int, default=233, help='The number of class 1 testing data.')
        parser.add_argument('--test0_num', type=int, default=107, help='The number of class 0 testing data.')
        parser.add_argument('--class_num', type=int, default=2, help='The number of class.')
        
    parser.add_argument('--code_size', type=int, default=70, help='size of medical code vector') #70
    parser.add_argument('--hidden_size', type=int, default=64, help='size of hidden vector in RNN') #512

    parser.add_argument('--multi_hot', type=bool, default=False, help='use multi_hot or not')
    parser.add_argument('--selfatt', type=bool, default=True, help='use self atention or not')

    #attack choice
    parser.add_argument('--retrain', type=bool, default=True, help='after getting the poison point, retrain the model or not')
    parser.add_argument('--retrain_from_scratch', type=bool, default=True, help='retrain from scratch on poisoned dataset or not')
    parser.add_argument('--use_clustering', type=bool, default=False, help='whether to cluster base samples or not.')
    parser.add_argument('--clustering_method', type=str, default='knn', help='use which method to cluster base samples. (kmeans / dbscan / knn)')
    parser.add_argument('--cluster_num', type=int, default=20, help='The number of clusters.')
    parser.add_argument('--cluster_size', type=int, default=500, help='size of chosen samples in each cluster from kmeans clustering.')
    
    parser.add_argument('--prototype_method', type=str, default='distance', help='the method of calculate prototype loss (distance / predict)')
    parser.add_argument('--perturbation', type=bool, default=False, help='pertubate the sample (used in the influence function)')
    parser.add_argument('--pyramid_code', type=bool, default=False, help='Use pyramid level of changable codes.')
    parser.add_argument('--base_sample_choice', type=str, default='jac_dist', help='align: use alignment score from embedding space of input data\
                        jac_dist / dist / logit: use logit from model / influence: use influence score from embedding space of input data')
    parser.add_argument('--alignment_calc', type=str, default='cosine', help='Method of calculating alignment score: cosine / dot')
    parser.add_argument('--dot_norm', type=str, default='2', help='norm value of inner product: 2,3,4,...,inf.')

    #attack parameters
    parser.add_argument('--max_code_step', type=int, default=5, help='The maximum number of changable code (used in pyramid attack).') #>1
    parser.add_argument('--code_step', type=int, default=10, help='The maximum number of changable code (used in regular ompgs).') #4 or 6
    parser.add_argument('--lamb1', type=float, default=1.0, help='The weight of loss1 (cross entropy).') #0.2
    parser.add_argument('--lamb2', type=float, default=0.0, help='The weight of loss2 (classification).') #0.2
    parser.add_argument('--num_samples_allowed', type=int, default=100, help='The number of samples allowed to change at conclude result.') #100/200, 200/400
    parser.add_argument('--num_codes_once', type=int, default=100, help='The number of samples allowed to change (at get indices).') #100
    parser.add_argument('--gradattack_group_size', type=int, default=5, help='The number of samples for each group in gradattack method.') #10
    
    #pyramid logit
    parser.add_argument('--pyramid_logit', type=bool, default=False, help='Use pyramid level of target logit for modified sample')
    parser.add_argument('--random_num', type=int, default=4, help='For each candidate with each target logit, the number of random modified plans')#4

    #influence
    parser.add_argument('--BFGS', type=bool, default=False, help='use BFGS algorithm or not')
    parser.add_argument('--iter_num', type=int, default=100, help='the number of iteration in BFGS.')
    parser.add_argument('--abs_inf', type=bool, default=False, help='Use absolute value of influence score or not.')

    
    
    args = parser.parse_args()

    return args

