import scipy.io
import os
import numpy as np

"""
prepared for dependent
"""


# 配置路径
extract_path = r'/mnt/DATA-2/DEAP/Valance/'
label_path = r'/mnt/DATA-2/DEAP/Valance/label.mat'
save_path = r'/mnt/DATA-2/DEAP/1/'

# 加载标签
label = scipy.io.loadmat(label_path)
labels = label['label'][0]  # 标签数组，假设 shape 为 (24,)

# 配置参数
train_per_label = 3  # 每个标签在训练集中 trial 的数量
test_per_label = 2   # 每个标签在测试集中 trial 的数量
random_seed = 42     # 随机种子

# 获取文件列表
file_list = os.listdir(extract_path)

for file_name in file_list:
    if '_' not in file_name:  # 跳过非特征文件
        continue

    # 加载当前文件
    input_file_path = os.path.join(extract_path, file_name)
    S = scipy.io.loadmat(input_file_path)

    # 获取所有键并按 `de_LDS1` 到 `de_LDS24` 排序
    eeg_keys = [key for key in S.keys() if 'de_LDS' in key]
    sorted_keys = sorted(eeg_keys, key=lambda x: int(x.split('de_LDS')[-1]))  # 按数字排序

    # 将数据和标签按 trial 索引分组
    DE_data = [S[key] for key in sorted_keys]  # 每个键对应的数据 (62, 样本数, 5)
    trial_labels = labels  # 直接使用已经加载的标签

    # 数据划分和全局标准化
    def stratified_split_and_zscore(data, labels, train_per_label, test_per_label, random_seed):
        np.random.seed(random_seed)
        unique_labels = np.unique(labels)
        train_data, test_data = [], []
        train_labels, test_labels = [], []

        # 按标签平衡划分训练集和测试集
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]  # 获取当前标签的所有索引
            np.random.shuffle(label_indices)  # 打乱索引
            train_indices = label_indices[:train_per_label]  # 前 `train_per_label` 个作为训练数据
            test_indices = label_indices[train_per_label:train_per_label + test_per_label]  # 后 `test_per_label` 个作为测试数据

            # 收集数据
            for idx in train_indices:
                train_data.append(data[idx])  # 直接添加每个 trial 的数据
                train_labels.extend([labels[idx]] * data[idx].shape[1])  # 根据样本数扩展标签

            for idx in test_indices:
                test_data.append(data[idx])  # 直接添加每个 trial 的数据
                test_labels.extend([labels[idx]] * data[idx].shape[1])  # 根据样本数扩展标签

        # 将 trial 数据拼接成训练集和测试集 (62, 总样本数, 5)
        train_data = np.concatenate(train_data, axis=1)
        test_data = np.concatenate(test_data, axis=1)


        # 全局 z-score 标准化
        combined_data = np.concatenate([train_data, test_data], axis=1)
        mean = combined_data.mean(axis=(1, 2), keepdims=True)
        std = combined_data.std(axis=(1, 2), keepdims=True)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        # 转换为 (样本数, 通道数, 特征数) 格式
        train_data = np.transpose(train_data, (1, 0, 2))  # (样本数, 62, 5)
        test_data = np.transpose(test_data, (1, 0, 2))    # (样本数, 62, 5)

        # 转换标签为 NumPy 数组
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        return train_data, test_data, train_labels, test_labels

    # 调用划分函数
    DE_train, DE_test, label_train, label_test = stratified_split_and_zscore(
        DE_data, trial_labels, train_per_label, test_per_label, random_seed
    )

    # 保存为 .mat 文件
    result_dict = {
        "DE_train": DE_train,
        "label_train": label_train,
        "DE_test": DE_test,
        "label_test": label_test,
        "label": "experiment"
    }

    output_file_path = os.path.join(save_path, file_name)
    scipy.io.savemat(output_file_path, result_dict)

    print(f"Processed: {input_file_path} -> {output_file_path}")
