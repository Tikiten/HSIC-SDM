from train_test import loop_train_test
from data import dataset_size_dict
import warnings
import time

from itertools import product
from collections import namedtuple
from collections import OrderedDict

# remove abundant output
warnings.filterwarnings('ignore')

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
    output_map = [False],
    only_draw_label = [False],  # draw full map or labeled map
    disjoint = [True],
    drawcluster = [False],
    # train_proportion = [0.01],
    train_proportion = [1],
    ws = [11], #HSIC_FM是27
    pcadimension = [15],
    num_epochs=[200],
    batch_size=[64],
    lr=[0.0005],
     # model_set = ['HSIC_FM','MyNet','GMA_Net'],
    model_set = ['MyNet7'],
    # data_set = ['IN','HU','HC'],
    data_set = ['IN'],
    RUN_TOTAL = [5],
    # data_set = ['FA','PCA', 'NO'],
    DR_FLAG = ['FA'],
    early_stopping = [True],
    patience = [15],
    dropcls = [0.1],
    depth1 = [1],
    depth2 = [1],
    depth3 = [1],
    model_type_flag = [3], #1代表模型只有一个3D PATCH输入 2代表有3D patch和中心像素光谱输入,而且多损失函数，3代表2的基础上单一损失
    coef_branch_main = [1],
    coef_branch_spe = [0],
    coef_branch_spa = [0],
    center_choice = [0], #0:QKV,Q是原始中心像素向量，1：QKV,Q是特征图中心像素，2：特征图中心像素余弦距离 3：特征图中心像素欧氏距离
    # coef_branch_main = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # coef_branch_spe = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # coef_branch_spa = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ce_choice = [0], #0：center-edge并联相加, 1:并联cat, 2:顺序串联center-edge 3：#顺序串联edge-center
    # loss_flag = [1], #0: CEloss, 1: Polyloss
    poly_epsilon = [0],
    # poly_epsilon = [-1.0, -0.5, 0, 0.5, 1.0, 2.0, 3.0],
)

if __name__ == '__main__':
    run_data = []  
    for run in RunBuilder.get_runs(params):
        print("run_params:",run)
        num_list = []
        # num_list = [200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200]
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
        
