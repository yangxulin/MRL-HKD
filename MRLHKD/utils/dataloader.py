import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper


def get_clinical_gene_WSIs_and_labels(clinical_x_path, gene_path, WSIs_path,clinical_label_path):
    clinical_x = pd.read_excel(clinical_x_path, sheet_name='Sheet1')
    clinical_x_feature_num = clinical_x.shape[1] - 1
    clinical_label = pd.read_excel(clinical_label_path, sheet_name='Sheet1')
    clinical_label_feature_num = clinical_label.shape[1] - 1
    clinical = pd.merge(clinical_x, clinical_label, left_on='Unnamed: 0', right_on='Unnamed: 0', how='inner')
    gene = pd.read_excel(gene_path, sheet_name='Sheet1')
    gene_feature_num = gene.shape[1] - 1
    # print(gene_feature_num)
    clinical_gene = pd.merge(clinical, gene, left_on='Unnamed: 0', right_on='Unnamed: 0', how='inner')
    patient_id = []
    path = []
    for root, dirs, files in os.walk(WSIs_path):
        for file in files:
            patient_id.append(file.split('.')[0])
            path.append(os.path.join(root, file))

    WSI_data = pd.DataFrame(path)
    WSI_data['Unnamed: 0'] = patient_id
    WSI_data_num = WSI_data.shape[1] - 1
    all_data = pd.merge(clinical_gene, WSI_data, left_on='Unnamed: 0', right_on='Unnamed: 0', how='inner')
    clinical_x_end = clinical_x_feature_num + 1
    clinical_labe_end = clinical_x_end + clinical_label_feature_num
    gene_end = clinical_labe_end + gene_feature_num
    wsi_end = gene_end+WSI_data_num
    clinical_x = all_data.iloc[:, 1:clinical_x_end]
    clinical_label = all_data.iloc[:, clinical_x_end:clinical_labe_end]
    gene = all_data.iloc[:, clinical_labe_end:gene_end]
    WSIs = all_data.iloc[:, gene_end:wsi_end]
    ans = []
    for item in WSIs.iloc[0:, :].values:
        ans.append(np.load(item[0]))
    WSIs = np.stack(ans, axis=0)
    # print(clinical_x['AGE'].mean())
    cols_standardize = ['AGE']
    # BLCA,BRCA
    cols_leave = ['SEX', 'RACE', 'TNM', 'PATH_M_STAGE', 'PATH_N_STAGE', 'PATH_T_STAGE',
                'RADIATION_THERAPY']
    # LUAD
    # cols_leave = ['SEX', 'TNM', 'PATH_M_STAGE', 'PATH_N_STAGE', 'PATH_T_STAGE',
    #               'RADIATION_THERAPY']
    # UCEC
    # cols_leave = ['SEX', 'RACE',
    #             'RADIATION_THERAPY']
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [([col], OneHotEncoder(handle_unknown='ignore', drop='first')) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)
    clinical_x = x_mapper.fit_transform(clinical_x)
    return clinical_x, gene, WSIs, clinical_label



class MyDataset(Dataset):
    def __init__(self, clinical_x, gene, WSIs, clinical_label):
        self.clinical_x = torch.from_numpy(clinical_x).float()
        self.clinical_label = torch.from_numpy(clinical_label).float()
        self.gene = torch.from_numpy(gene).float()
        self.wsis = torch.from_numpy(WSIs).float()

    def __getitem__(self, index):
        return self.clinical_x[index], self.gene[index], self.wsis[index], self.clinical_label[index]

    def __len__(self):
        return len(self.clinical_x)


def make_dataloader(clinical_x, gene, WSIs, clinical_label, batch_size):
    dataset = MyDataset(clinical_x, gene, WSIs, clinical_label)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=True, num_workers=0, drop_last=False)


if __name__ == '__main__':
    clinical_x_path = r'D:\TCGA\BLCA\clinical\clinical_x.xlsx'
    gene_path = r'D:\TCGA\BLCA\gene_select\gene.xlsx'
    WSIs_path = r'D:\TCGA\BLCA\WsiVaeSample'
    clinical_label_path = r'D:\TCGA\BLCA\clinical\clinical_label.xlsx'
    clinical_x, gene, WSIs, clinical_label = get_clinical_gene_WSIs_and_labels(clinical_x_path, gene_path, WSIs_path, clinical_label_path)
    print(clinical_label.shape[1])
    print(clinical_x.shape[1])
    print(gene.shape[1])
    print(WSIs.shape[1])
    for epoch in range(100):
        train_loader = make_dataloader(clinical_x, gene.values, WSIs, clinical_label.values, 32)
        for _, (train_clinical_x, train_gene, train_WSIs, train_clinical_label) in enumerate(train_loader):
            print(train_clinical_x.shape)
            print(train_gene.shape)
            print(train_WSIs.shape)
            print(train_clinical_label.shape)
            break
        break