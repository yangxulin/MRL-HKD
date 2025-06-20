import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from MRLHKD.utils.dataloader import get_clinical_gene_WSIs_and_labels
from MRLHKD.model.MRLHKDModel import MRLHKDModel
import time
import pickle

start = time.time()
cancer_type = 'BLCA'
clinical_x_path = 'D:\\TCGA\\' + cancer_type + '\\clinical\\clinical_x.xlsx'
clinical_label_path = 'D:\\TCGA\\' + cancer_type + '\\clinical\\clinical_label.xlsx'
gene_path = 'D:\\TCGA\\'+ cancer_type + '\\gene_select\\gene.xlsx'
WSIs_path = 'D:\\TCGA\\'+ cancer_type + '\\WsiVaeSample'
hyper_path = 'D:\\python\\multimodal\\Test16\\hyperparameter\\' + cancer_type + '.pkl'
hyper_tune = 'D:\\python\\multimodal\\Test16\\hyperparameter\\' + cancer_type + '.xlsx'
clinical_x, gene, WSIs, clinical_label = get_clinical_gene_WSIs_and_labels(clinical_x_path, gene_path, WSIs_path,
                                                                           clinical_label_path)
clinical_x = clinical_x.astype(np.float32) # shape: [N,number of clinical feature]
gene = gene.values.astype(np.float32) # shape: [N,80]
WSIs = WSIs.astype(np.float32) # shape: [N,40,256]
clinical_label = clinical_label.values.astype(np.float32)
c_index_test_5_fold = []
mae_test_5_fold = []
hyper_list = []
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
# torch.backends.cudnn.enabled = False
random_state = 2025


def train():
    skf_train = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf_train.split(clinical_x, clinical_label[:, 1])):
        if i >= 1:
            break
        train_clinical_x = clinical_x[train_index]
        train_gene_x = gene[train_index]
        train_WSIs_x = WSIs[train_index]
        train_label = clinical_label[train_index]
        learning_rate_list = [0.01, 0.001, 0.0001]
        dropout_list = [0.1, 0.3, 0.5]
        output_size_list = [32, 64, 128, 256]
        hidden_dim_list = [64, 128, 256, 512]
        alpha_list = [0.01, 0.1, 1.0]
        beta_list = [0.1, 1.0, 10.0]
        tuning_para_data = pd.DataFrame(columns=['learning_rate', 'dropout', 'output_size', 'hidden_dim','alpha', 'beta','c-index'])
        for lr in learning_rate_list:
            for dropout in dropout_list:
                for output_size in output_size_list:
                    for hidden_dim in hidden_dim_list:
                        for alpha in alpha_list:
                            for beta in beta_list:
                                # 5-fold cross validation is used to adjust hyperparameters
                                skf_tune = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
                                inner_cindex = []
                                for i, (inner_train_index, val_index) in enumerate(skf_tune.split(train_clinical_x, train_label[:, 1])):
                                    print(f"Inner Fold {i}:")
                                    inner_train_clinical_x = train_clinical_x[inner_train_index]
                                    inner_train_gene_x = train_gene_x[inner_train_index]
                                    inner_train_WSIs_x = train_WSIs_x[inner_train_index]
                                    inner_train_label = train_label[inner_train_index]
                                    val_clinical_x = train_clinical_x[val_index]
                                    val_gene_x = train_gene_x[val_index]
                                    val_WSIs_x = train_WSIs_x[val_index]
                                    val_label = train_label[val_index]
                                    print('Training model.......')
                                    Model = MRLHKDModel(learning_rate=lr, output_size=output_size, hidden_dim=hidden_dim,
                                                     alpha=alpha, beta=beta, dropout=dropout, epochs=300,
                                                    batch_size=1024,optimizer="Adam", val_size=0.2, device=device)
                                    Model.fit(inner_train_clinical_x, inner_train_gene_x, inner_train_WSIs_x, inner_train_label)
                                    val_cindex = Model.get_c_index(val_clinical_x, val_gene_x, val_WSIs_x, val_label)
                                    inner_cindex.append(val_cindex)
                                val_cindex = np.mean(inner_cindex)
                                print('LR:{}, output_size:{}, dropout:{}, hidden_dim:{},alpha:{}, beta:{}, C-index:{}'.format(lr, dropout, output_size, hidden_dim, alpha, beta, val_cindex))
                                alist = [lr, dropout, output_size, hidden_dim, alpha, beta, val_cindex]
                                tuning_para_data.loc[len(tuning_para_data)] = alist
        # Optimal hyperparameters
        max_c_index = tuning_para_data['c-index'].max()
        line = tuning_para_data[tuning_para_data['c-index'] == max_c_index]
        best_LR = line['learning_rate'].values[0]
        best_dropout = line['dropout'].values[0]
        best_output_size = line['output_size'].values[0]
        best_hidden_dim = line['hidden_dim'].values[0]
        best_alpha = line['alpha'].values[0]
        best_beta = line['beta'].values[0]
        print('Optimal hyperparameters LR:{}, dropout:{}, output_size:{}, hidden_dim:{}, alpha:{}, beta:{}'.format(best_LR, best_dropout, best_output_size, best_hidden_dim, best_alpha, best_beta))
        tuning_para_data.to_excel(hyper_tune, index=False, sheet_name='Sheet1')
        # 保存最优超参数
        hyper_dict = {}
        hyper_dict['learning_rate'] = best_LR
        hyper_dict['dropout'] = best_dropout
        hyper_dict['output_size'] = best_output_size
        hyper_dict['hidden_dim'] = best_hidden_dim
        hyper_dict['alpha'] = best_alpha
        hyper_dict['beta'] = best_beta
        hyper_list.append(hyper_dict)
        f_save = open(hyper_path, 'wb')
        pickle.dump(hyper_list, f_save)
        f_save.close()
        end = time.time()
        print(end - start)


def test():
    f_read = open(hyper_path, "rb")
    hyper = pickle.load(f_read)[0]
    hyper['output_size'] = int(hyper['output_size'])
    hyper['hidden_dim'] = int(hyper['hidden_dim'])
    # Repeat the 5-fold cross validation five times
    skf_outer = RepeatedStratifiedKFold(n_splits=5, random_state=random_state, n_repeats=5)
    for i, (train_index, test_index) in enumerate(skf_outer.split(clinical_x, clinical_label[:, 1])):
        print(f"Test Fold {i}:")
        train_clinical_x = clinical_x[train_index]
        train_gene_x = gene[train_index]
        train_WSIs_x = WSIs[train_index]
        train_label = clinical_label[train_index]
        test_clinical_x = clinical_x[test_index]
        test_gene_x = gene[test_index]
        test_WSIs_x = WSIs[test_index]
        test_label = clinical_label[test_index]
        Model = MRLHKDModel(**hyper, epochs=300, batch_size=1024, optimizer="Adam", val_size=0.2, device=device)
        Model.fit(train_clinical_x, train_gene_x, train_WSIs_x, train_label)
        test_cindex = Model.get_c_index(test_clinical_x, test_gene_x, test_WSIs_x, test_label)
        test_mae = Model.get_mae(test_clinical_x, test_gene_x, test_WSIs_x, test_label)
        c_index_test_5_fold.append(test_cindex)
        mae_test_5_fold.append(test_mae)

    end = time.time()
    print(c_index_test_5_fold)
    print(mae_test_5_fold)
    print(np.mean(c_index_test_5_fold), np.std(c_index_test_5_fold))
    print(np.mean(mae_test_5_fold), np.std(mae_test_5_fold))
    print(end - start)


train()