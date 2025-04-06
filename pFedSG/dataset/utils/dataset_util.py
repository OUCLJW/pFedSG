import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_size = 0.75  # merge original training set and test set, then split it manually.
least_samples = batch_size / (1 - train_size)  # least samples for each client
alpha = 0.1  # for Dirichlet distribution


def check(config_path, train_path, test_path, num_clients, num_classes, niid=False,
          balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data, num_clients, num_classes,niid=False, balance=False, partition=None, class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    total_samples_per_client = 4000  # 总数据量 4000，分为 3000 训练 + 1000 测试

    dataset_content, dataset_label = data

    dataidx_map = {}

    # 为每个客户端分配两个独特的类别
    classes_per_client_idx = [np.random.choice(range(num_classes), class_per_client, replace=False) for _ in
                              range(num_clients)]

    idxs_per_class = {i: np.where(np.array(dataset_label) == i)[0] for i in range(num_classes)}

    for client in range(num_clients):
        client_idxs = []
        for c in classes_per_client_idx[client]:
            client_idxs += list(
                np.random.choice(idxs_per_class[c], total_samples_per_client // class_per_client, replace=True))
        dataidx_map[client] = client_idxs

    # 分配数据给每个客户端
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        class_counts = np.bincount(y[client], minlength=num_classes)
        statistic[client] = list(enumerate(class_counts))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Total size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples per label: ", statistic[client])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y, train_samples_per_client=3000, test_samples_per_client=1000):
    # 分割数据集
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        idxs = np.arange(len(X[i]))
        np.random.shuffle(idxs)  # Shuffle indexes before splitting
        # Ensure there are enough samples for train and test
        idxs = np.tile(idxs, (train_samples_per_client + test_samples_per_client) // len(idxs) + 1)

        # Split indexes for training and testing data
        train_idxs = idxs[:train_samples_per_client]
        test_idxs = idxs[train_samples_per_client:train_samples_per_client + test_samples_per_client]

        X_train, y_train = X[i][train_idxs], y[i][train_idxs]
        X_test, y_test = X[i][test_idxs], y[i][test_idxs]

        train_data.append({'x': X_train, 'y': y_train})
        test_data.append({'x': X_test, 'y': y_test})

        num_samples['train'].append(len(y_train))
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")