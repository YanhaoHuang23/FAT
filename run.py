import copy
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2  # Convert to MB
    return size_all_mb

def train_independent(train_loader, test_loader, model, criterion, optimizer, num_epochs, sub_name):
    print('开始在', device, '上训练...')
    alpha = 0.2
    acc_test_best = 0.0
    n = 0
    model_best = None
    for ep in range(num_epochs):
        model.train()
        n += 1
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels.long().to(device)

            # 前向传播
            # ----------- 1) 生成 lam &amp; 随机乱序索引 -----------
            lam = np.random.beta(alpha, alpha)  # from numpy or torch.distributions
            rand_index = torch.randperm(images.size(0)).to(device)

            # ----------- 2) 生成混合的输入 -----------
            images_shuffled = images[rand_index, :]
            images_mix = lam * images + (1 - lam) * images_shuffled

            # 对应的标签
            labels_shuffled = labels[rand_index]

            # ----------- 3) 前向传播 (用混合输入) -----------
            output = model(images_mix)

            loss = lam * criterion(output, labels) + (1 - lam) * criterion(output, labels_shuffled)

            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Epoch {}, batch {}, loss: {:.4f}, accuracy: {:.4f}'.format(ep + 1,
                                                                              batch_id,
                                                                              total_loss / batch_id,
                                                                              accuracy))
            batch_id += 1

        print('Epoch {} 总损失: {:.4f}'.format(ep + 1, total_loss))

        acc_test = evaluate(test_loader, model)

        if acc_test >= acc_test_best:
            n = 0
            acc_test_best = acc_test
            model_best = copy.deepcopy(model)

        # 提前停止条件
        if n >= num_epochs // 10 and acc_test < acc_test_best - 0.1:
            print('######################### 重新加载最佳模型 #########################')
            n = 0
            model = copy.deepcopy(model_best)
        # 输出目前最佳测试准确率
        print('>>> 目前最佳测试准确率: {:.4f}'.format(acc_test_best))

    return acc_test_best


def train_dependent(train_loader, test_loader, model, criterion, optimizer, num_epochs, sub_name):
    print('开始在', device, '上训练...')

    acc_test_best = 0.0
    n = 0
    model_best = None
    for ep in range(num_epochs):
        model.train()
        n += 1
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels.long().to(device)

            # 前向传播
            output = model(images)
            loss = criterion(output, labels)

            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Epoch {}, batch {}, loss: {:.4f}, accuracy: {:.4f}'.format(ep + 1,
                                                                              batch_id,
                                                                              total_loss / batch_id,
                                                                              accuracy))
            batch_id += 1

        print('Epoch {} 总损失: {:.4f}'.format(ep + 1, total_loss))

        acc_test = evaluate(test_loader, model)

        if acc_test >= acc_test_best:
            n = 0
            acc_test_best = acc_test
            model_best = copy.deepcopy(model)

        # 提前停止条件
        if n >= num_epochs // 10 and acc_test < acc_test_best - 0.1:
            print('######################### 重新加载最佳模型 #########################')
            n = 0
            model = copy.deepcopy(model_best)
        # 输出目前最佳测试准确率
        print('>>> 目前最佳测试准确率: {:.4f}'.format(acc_test_best))

    return acc_test_best


def evaluate(test_loader, model):
    print('开始在', device, '上测试...')
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float().to(device)
            labels = labels.long().to(device)

            output = model(images)
            pred = output.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += len(labels)
    accuracy = correct / total
    print('测试准确率: {:.4f}'.format(accuracy))
    return accuracy