import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy import io as scio
from Augmentation import *
from torch.utils.data import DataLoader

def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)     #A+ A的转置
        d = torch.sum(A, 1)   #对A的第1维度求和
        d = 1 / torch.sqrt(d + 1e-10)    #d的-1/2次方
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K,device):
    support = []
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
            temp = torch.eye(A.shape[1])
            temp = temp.to(device)
            support.append(temp)
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class eegDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.augment:
            feature = scale_frequency_bands(feature)
            feature = add_periodic_perturbation(feature)

        return feature.clone().detach().float(), label.clone().detach().long()


def load_DE_SEED(load_path):
    datasets = scio.loadmat(load_path)
    DE = datasets['DE']  # 形状为（通道数，样本数，频带数）
    dataAll = np.transpose(DE, [1, 0, 2])  # 转换为（样本数，通道数，频带数）
    labelAll = datasets['labelAll'].flatten()
    labelAll = labelAll
    return dataAll, labelAll


def load_dataloader(data_train, data_test, label_train, label_test, augment=False):
    batch_size = 32
    train_dataset = eegDataset(data_train, label_train, augment=augment)
    test_dataset = eegDataset(data_test, label_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader