import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import time, os
import scipy.io as sio
import collections
import matplotlib.pyplot as plt
from operator import truediv
from sklearn import metrics
from util import sampling, sampling_disjoint, get_device, generate_batch, print_results,draw_labelresult, draw_allresult
from data import lazyprocessing, DrawCluster, dataset_size_dict

# from HSIC_FM import FullModel
# from network.GMANet import GMA_Net
from network.MyNet import FEDFormerPyramid_v7

from PolyLoss import Poly1CrossEntropyLoss

from itertools import product
from collections import namedtuple
from collections import OrderedDict
import pandas as pd

# from torchsummary import summary
# from thop import clever_format
# from thop import profile

import datetime

device = get_device(GPU='0')
print('training on ', device)
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')


def resolve_dict(hp):
    return hp['dataset'], hp['run_times'], hp['pchannel'], hp['model'], hp['ws'], hp['epochs'], \
           hp['batch_size'], hp['learning_rate'], hp['train_proportion'], hp['train_num'], hp['outputmap'], \
           hp['only_draw_label'], hp['disjoint'], hp['drawcluster'], hp['DR_FLAG'], hp['early_stopping'], \
           hp['patience'], hp['dropcls'],hp['depth1'],hp['depth2'],hp['depth3'], hp['model_type_flag'],\
           hp['coef_branch_main'], hp['coef_branch_spe'], hp['coef_branch_spa'], hp['center_choice'], \
           hp['ce_choice'], hp['poly_epsilon']  
           
def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=0)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            n += y.shape[0]
    net.train()
    return [acc_sum / n, test_l_sum / test_num]


def train(net, train_idx, val_idx,
          ground_truth, ground_test, X_PCAMirrow,X_org,
          batch_size, ws, model_type_flag, dataset_name,
          loss, optimizer, device, epochs,
          coef_branch_main, coef_branch_spe, coef_branch_spa, 
          early_stopping=True, patience=10,
          ):
    best_acc = 0.5
    net.train()

    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    
    best_score = None
    finish_flag = False
    train_loss_min = np.Inf

    lr_adjust = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    
    for epoch in range(epochs):
        train_lorder = generate_batch(train_idx, X_PCAMirrow,X_org, ground_truth, batch_size, ws, dataset_name,
                                      shuffle=True)
        # valida_lorder = generate_batch(val_idx, X_PCAMirrow,X_org, ground_test, batch_size, ws, dataset_name,
        #                                shuffle=True)
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        
        for step, (X,V,y) in enumerate(train_lorder):
            # X is 3D patch. V is center spectral.
            batch_count, train_l_sum = 0, 0
            X = torch.Tensor(X)  # .type(torch.HalfTensor)
            V = torch.Tensor(V) 
            y = torch.Tensor(y).type(torch.LongTensor)
            V = V.to(device)
            X = X.to(device)
            y = y.to(device)
            
            if model_type_flag == 1:        
                y_hat = net(X)
                optimizer.zero_grad()
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            
            elif model_type_flag ==2:
                y_hat = net(X,V)
                y_pred = y_hat[0]
                
                l = 0.
                for j in range(3):
                    if j == 0:
                        l_main = loss(y_hat[j], y)
                        l = l_main * coef_branch_main
                        # print('l:',l)
                    elif j == 1:
                        l_Spe = loss(y_hat[j], y)
                        l += l_Spe * coef_branch_spe
                        # l += l_Spe / (l_Spe / l_main).detach()
                        # print('l:',l)
                    elif j == 2:
                        l_Spa = loss(y_hat[j], y)
                        l += l_Spa * coef_branch_spa
                        # l += l_Spa / (l_Spa / l_main).detach()
                        
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
        
            elif model_type_flag ==3:
                y_hat = net(X,V)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
        
        train_loss = train_l_sum / batch_count
        train_acc =  train_acc_sum / n*100
        train_loss_list.append(train_loss)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        
        print('epoch %d, train loss %.5f, train acc %.2f, time %.2f sec, lr: %.6f'
              % (epoch + 1, train_loss, train_acc,
                 time.time() - time_epoch, lr_adjust.get_last_lr()[0]))
        lr_adjust.step()
        
        #自己写的early_stopping
        if early_stopping:
            score = -train_loss 
  
            if best_score is None:
                best_score = score
                '''Saves model when validation loss decrease.'''
                print(f'Train loss decreased ({train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
                torch.save(net.state_dict(), './models/temp_model.pt')
                train_loss_min = train_loss
                counter = 0
                
            elif score < best_score:
                counter += 1
                print(f'EarlyStopping counter: {counter} out of {patience}')
                if counter >= patience:
                    finish_flag =  True
            else:
                best_score = score
                '''Saves model when validation loss decrease.'''
                print(f'Train loss decreased ({train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
                torch.save(net.state_dict(), './models/temp_model.pt')
                train_loss_min = train_loss
                counter = 0
                
        if finish_flag:            
            print("Early stopping")
            break
            
        # torch.save(net.state_dict(), "./models/temp_model.pt")
    
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
          % (epoch + 1, train_loss, train_acc_sum / n,
             time.time() - start))
        
def test(net, test_loder, model_type_flag):
    pred_test = []
    pred_test_index = []
    manifold = []
    
    if model_type_flag == 1:
        with torch.no_grad():
            for X, _ in test_loder:
                X = torch.Tensor(X)
                X = X.to(device)
                net.eval()
                y_hat = net(X)
                pred_test_index.extend(np.array(y_hat.cpu().detach().numpy()))
                pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
                manifold.extend(np.array(_))
        pred_test_index = np.array(pred_test_index)        
        pred_test = np.array(pred_test)
        manifold = np.array(manifold)
    
    elif model_type_flag == 2:
        with torch.no_grad():
            for X, V, _ in test_loder:
                X = torch.Tensor(X)
                X = X.to(device)
                V = torch.Tensor(V)
                V = V.to(device)
                net.eval()
                y_pred = net(X,V)
                # y_hat = y_pred
                y_hat = y_pred[0]
                pred_test_index.extend(np.array(y_hat.cpu().detach().numpy()))
                pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
                manifold.extend(np.array(_))
        pred_test_index = np.array(pred_test_index)        
        pred_test = np.array(pred_test)
        manifold = np.array(manifold)
        
    elif model_type_flag == 3:
        with torch.no_grad():
            for X, V, _ in test_loder:
                X = torch.Tensor(X)
                X = X.to(device)
                V = torch.Tensor(V)
                V = V.to(device)
                net.eval()
                y_hat = net(X,V)
                pred_test_index.extend(np.array(y_hat.cpu().detach().numpy()))
                pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
                manifold.extend(np.array(_))
        pred_test_index = np.array(pred_test_index)        
        pred_test = np.array(pred_test)
        manifold = np.array(manifold)
    return pred_test, manifold, pred_test_index
    
dataset_min_square_dict = {
    'PU': 121,
    'IN': 225,
    'HU': 144,
    'KSC': 196,
    'SV': 225,
    'BT': 169,
    'HC': 289,
}

dataset_spe_num = {
    'PU': 103,
    'IN': 200,
    'HU': 144,
    'KSC': 176,
    'SV': 204,
    'BT': 145,
    'HC': 274,
}


def loop_train_test(hyper_parameter,run_para, run_data):
    # print('hyper_parameter:',hyper_parameter)
    # run_data = [] 
    datasetname, run_times, num_PC, model_type, w, epochs, batch_size, lr, \
    train_proportion, num_list, outputmap, only_draw_label, disjoint, drawcluster, \
    DR_FLAG, early_stopping, patience,dropcls, \
    depth1, depth2, depth3, model_type_flag, coef_branch_main, coef_branch_spe,\
    coef_branch_spa, center_choice, ce_choice, poly_epsilon = resolve_dict(hyper_parameter)
    print('>' * 10, "Data set Loading", '<' * 10)
    X_PCAMirrow, ground_truth, shapelist, X_org = lazyprocessing(datasetname, num_PC=num_PC, w=w, disjoint=disjoint, dr_flag=DR_FLAG)
    classnum = np.max(ground_truth)
    print(datasetname, 'shape:', shapelist)
    
    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((run_times, classnum))
    
    # print('num_PC:',num_PC)

    print('>' * 10, "Start Training", '<' * 10)
    for run_i in range(0, run_times):
        print('round:', run_i + 1)
        
        depth_run = [depth1,depth2,depth3]
        print('depth:',depth_run)
            
        if model_type == 'MyNet7':
                net = FEDFormerPyramid_v7(img_size=w, 
                                          in_chans=num_PC, 
                                          num_classes=classnum, 
                                          embed_dim=[64,64,64],  
                                          depth=depth_run,
                                          mlp_ratio=[1,1,1], 
                                          num_heads=[8,8,8], #只影响SPA的多头
                                          drop_rate=0.1, 
                                          drop_path_rate=0.1, 
                                          dropcls=dropcls, 
                                          norm_layer=None, 
                                          uniform_drop = False,
                                          num_stages=3,
                                          init_values=0.001, 
                                          sample_ration=0.5, 
                                          channel_fft=True,
                                          org_spe_dim= dataset_spe_num[datasetname],
                                          center_choice = center_choice,
                                          ce_choice = ce_choice)
        
        
        print("load model")
        # net.half()#半精度
        net = net.to(device)
                    
        # a = torch.randn(1, 9, 9, 15).cuda()
        # summary(net, (9, 9, 15), batch_size=1)
        # # print('ok!!')
        # flops, params = profile(net, inputs=(a,))
        # flops, params = clever_format([flops, params], '%.3f')
        # print('模型参数：',params)
        # print('每一个样本浮点运算量：',flops)
        
        net.train()
        
        optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)  # 1e-4
        # optimizer = optim.RMSprop(net.parameters(),lr=lr)
        # optimizer = optim.SGD(
        #     net.parameters(),
        #     lr=lr,
        #     weight_decay=0)
        
        loss = Poly1CrossEntropyLoss(num_classes=classnum, epsilon=poly_epsilon, reduction='mean')
        
        
        if disjoint:
            train_indexes, test_indexes, val_indexes, drawlabel_indexes, drawall_indexes = sampling_disjoint(
                ground_truth)

            ground_truth_train = ground_truth[0]
            ground_truth_test = ground_truth[1]
        
            print("Training sapmles:", len(train_indexes))
            print("Testing sapmles:", len(test_indexes))
            print("Val sapmles:", len(val_indexes))
            
        
        else:
            train_indexes, test_indexes, val_indexes, drawlabel_indexes, drawall_indexes = sampling(ground_truth,
                                                                                                    train_proportion,
                                                                                                    num_list,
                                                                                                    seed=(run_i + 1) * 111)
            ground_truth_train = ground_truth
            ground_truth_test = ground_truth

            print("Training sapmles:", len(train_indexes))
            print("Testing sapmles:", len(test_indexes))

            print("Val sapmles:", len(val_indexes))
        
        test_loder = generate_batch(test_indexes, X_PCAMirrow,X_org,ground_truth_test, batch_size*3, w, datasetname,
                                    shuffle=True)
        print('>' * 10, "Start Testing", '<' * 10)
        
        net.load_state_dict(torch.load('./models/MyNet7/HC/Patchsize_17_FA_15_OA_0.988_AA_0.972_depth_111.pt'))
        # net.load_state_dict(torch.load('./models/MyNet7/IN/Patchsize_13_FA_15_OA_0.975_AA_0.985_depth_111.pt'))
        # net.load_state_dict(torch.load('./models/MyNet7/HU/Patchsize_9_FA_15_OA_0.922_AA_0.936_depth_111.pt', map_location='cuda:0'))
        
        torch.cuda.synchronize()
        tic2 = time.time()
        pred_test, manifold, pred_test_index = test(net, test_loder, model_type_flag)
        torch.cuda.synchronize()
        toc2 = time.time()
        
        collections.Counter(pred_test)
        gt_test = manifold
        
        overall_acc = metrics.accuracy_score(pred_test, gt_test[:])
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:])
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, gt_test[:])
        
        if drawcluster:
            DrawCluster(gt_test[:], pred_test_index, overall_acc, average_acc, datasetname, model_type)
        
        if not os.path.exists('./models/{}/{}'.format(model_type, datasetname)):
            os.makedirs('./models/{}/{}'.format(model_type, datasetname))
        
        torch.save(net.state_dict(), "./models/" + str(model_type) + "/" + str(datasetname) + '/' 
            + "Patchsize_" + str(w) + "_" + str(DR_FLAG) + '_' + str(num_PC) + "_OA_" + str(round(overall_acc,3)) 
            + "_AA_" + str(round(average_acc,3)) + '_depth_' + str(depth1)+ str(depth2)+ str(depth3) + '.pt')
            
        # torch.save(net.state_dict(), "./models/" + str(model_type) + "/" + str(datasetname) + '/' 
        #            + "Patchsize_" + str(w) + "_" + str(DR_FLAG) + '_' + str(num_PC) + '_' + 
        #            str(LEARN_MODE_FLAG) + "_OA_" + str(round(overall_acc,3)) 
        #            + "_AA_" + str(round(average_acc,3)) + '_depth_' + str(depth1)+ str(depth2)+ str(depth3) + '.pt')

        # net.load_state_dict(torch.load('./models/spatical/vithsi0.948.pt'))
        
        # print(confusion_matrix)
        print('OA :', overall_acc)
        print('AA :', average_acc)
        print('Kappa :', kappa)
        print("Each acc:", each_acc)
        print('test time:', toc2 - tic2)
        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[run_i, :] = each_acc
        
        if outputmap:
            print('>' * 10, "Start Drawmap", '<' * 10)
            tic3 = time.time()
            if only_draw_label:
                drawmap_loder = generate_batch(drawlabel_indexes, X_PCAMirrow,X_org, ground_truth_train, batch_size * 5, w,
                                               datasetname, shuffle=False)  # 画图的时候不要打乱,画图的时候不需要Groundtruch,
                pred_map, _, _ = test(net, drawmap_loder, model_type_flag)
                map, labelmat = draw_labelresult(labels=pred_map, index=drawlabel_indexes, dataset_name=datasetname)
            
            else:
                drawmap_loder = generate_batch(drawall_indexes, X_PCAMirrow,X_org, ground_truth_train, batch_size * 10, w,
                                               datasetname, shuffle=False)
                pred_map, _, _ = test(net, drawmap_loder, model_type_flag)
                map, labelmat = draw_allresult(labels=pred_map, dataset_name=datasetname)
            toc3 = time.time()
            print('drawmap time:', toc3 - tic3)
            if not os.path.exists('./classification_maps/{}/{}'.format(model_type, datasetname)):
                os.makedirs('./classification_maps/{}/{}'.format(model_type, datasetname))

            plt.imsave(
                './classification_maps/{}/{}/oa_{}aa_{}.png'.format(model_type, datasetname, str(round(overall_acc,3)), str(round(average_acc,3))),map)
           
if __name__ == '__main__':
    class RunBuilder():
        @staticmethod
        def get_runs(params):
            
            Run = namedtuple('Run', params.keys())
            
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
                
            return runs

    params = OrderedDict(
        ## global constant value
        output_map = [True],
        only_draw_label = [False],  # draw full map or labeled map
        disjoint = [False],
        drawcluster = [False],
        train_proportion = [0.01],
        ws = [17], #HSIC_FM是27
        pcadimension = [15],
        num_epochs=[200],
        batch_size=[64],
        lr=[0.0005],
        # model_set = ['HSIC_FM','MyNet','GMA_Net'],
        model_set = ['MyNet7'],
        # data_set = ['IN','HU','HC'],
        data_set = ['HC'],
        RUN_TOTAL = [1],
        # data_set = ['FA','PCA', 'NO'],
        DR_FLAG = ['FA'],
        early_stopping = [True],
        patience = [15],
        dropcls = [0.1],
        depth1 = [1],
        depth2 = [1],
        depth3 = [1],
        model_type_flag = [3], #1代表模型只有一个3D PATCH输入 2代表有3D patch和中心像素光谱输入,而且多损失函数 3代表2的基础上单一损失
        coef_branch_main = [1],
        coef_branch_spe = [0],
        coef_branch_spa = [0],
        center_choice = [0], #0:QKV,Q是原始中心像素向量，1：QKV,Q是特征图中心像素，2：特征图中心像素余弦距离 3：特征图中心像素欧氏距离
        #center_choice对于HU是1，对于IN, HC是0
        # coef_branch_main = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # coef_branch_spe = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # coef_branch_spa = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ce_choice = [0], #0：center-edge并联相加, 1:并联cat, 2:顺序串联center-edge 3：#顺序串联edge-center
        # loss_flag = [1], #0: CEloss, 1: Polyloss
        poly_epsilon = [0],
        # poly_epsilon = [-1.0, -0.5, 0, 0.5, 1.0, 2.0, 3.0],
    )
        
    run_data = []  
    for run in RunBuilder.get_runs(params):
        print("run_params:",run)
        num_list = []
        hp = {
            'dataset': run.data_set,
            'run_times': run.RUN_TOTAL,
            'pchannel': dataset_size_dict[run.data_set][2] if run.pcadimension == 0 else run.pcadimension,
            'model': run.model_set,
            'ws': run.ws,
            'epochs': run.num_epochs,#30,25
            'batch_size': run.batch_size,
            'learning_rate': run.lr,
            'train_proportion': run.train_proportion,
            'train_num': num_list,
            'outputmap': run.output_map,
            'only_draw_label': run.only_draw_label,
            'disjoint': run.disjoint,
            'drawcluster': run.drawcluster,
            'DR_FLAG': run.DR_FLAG,
            # 'LEARN_MODE_FLAG': run.LEARN_MODE_FLAG,
            'early_stopping': run.early_stopping,
            'patience': run.patience,
            'dropcls': run.dropcls,
            'depth1' : run.depth1,
            'depth2' : run.depth2,
            'depth3' : run.depth3,
            'model_type_flag' : run.model_type_flag,
            'coef_branch_main': run.coef_branch_main,
            'coef_branch_spe': run.coef_branch_spe,
            'coef_branch_spa': run.coef_branch_spa,
            'center_choice': run.center_choice,
            'ce_choice': run.ce_choice,
            # 'loss_flag': run.loss_flag,
            'poly_epsilon': run.poly_epsilon,
        }
        loop_train_test(hp, run, run_data)

