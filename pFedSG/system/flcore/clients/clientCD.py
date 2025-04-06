import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data,read_deal_data
from sklearn.decomposition import TruncatedSVD
from utils.privacy import initialize_dp, get_dp_params


class clientCD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.truncated_data = None
        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        #        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size,
        #                       self.rand_percent, self.layer_idx, self.eta, self.device)

        self.n_components = args.n_components
        # 添加此行以执行SVD截断

    #       print(processed_data)
    import numpy as np

    def pad_sequences(self,sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0):
        """
        该函数用于填充序列至相同的长度。
        """
        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

        if maxlen is None:
            maxlen = np.max(lengths)

        # 根据最大长度进行填充或裁剪
        is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
        if isinstance(value, str) and dtype != object and not is_dtype_str:
            raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                             "You should set `dtype=object` for variable length strings."
                             .format(dtype, type(value)))

        x = np.full((len(sequences), maxlen) + sequences[0].shape[1:], value, dtype=dtype)
        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # 空序列直接跳过
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" is not understood' % truncating)

            # 填充
            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" is not understood' % padding)
        return x

    def preprocess_data(self,trainloader):
        # 假设 trainloader 是一个列表，包含了所有客户端的数据，每个元素都是一个(tensor数据, 类别)
        # 首先，提取所有的tensor数据
        sequences = [x[0].numpy() for x in trainloader]  # 将tensor转换为numpy数组
        # 应用pad_sequences函数
        padded_sequences = self.pad_sequences(sequences, padding='post', value=0)
        # 将填充后的序列转换回tensor
#        padded_sequences_tensor = torch.Tensor(padded_sequences)
#        print("Padded Sequences Shape:", np.array(padded_sequences).shape)
        return padded_sequences

    def preprocess_images(self, images):
        # 将每个图像平展
        N, C, H, W = images.shape
        images_reshaped = images.reshape(N, -1)  # 形状变为 (N, C*H*W)
        return images_reshaped
    def truncate_svd(self, n_components):
        """
        对数据集进行SVD截断，并保留前n_components个奇异值。

        参数：
        - n_components: 保留的奇异值的数量，也是截断后数据的特征维度。
        """
        trainloader = read_client_data(self.dataset, self.id, is_train=True)
        preprocessed_data = self.preprocess_data(trainloader)
        trainloader_reshaped = self.preprocess_images(preprocessed_data)
#        print("Preprocessed Trainloader Shape:", trainloader_reshaped.shape)
        # 计算SVD
        U, Sigma, Vt = np.linalg.svd(trainloader_reshaped, full_matrices=False)
        # 保留前n_components个奇异值
#        Sigma = np.array([...])
#        print("Sigma Shape:", Sigma.shape)

        Sigma_truncated = np.diag(Sigma[:n_components])
        U_truncated = U[:, :n_components]

        Vt_truncated = Vt[:n_components, :]
#        U_truncated = np.array([...])
#        Vt_truncated = np.array([...])
        # 计算截断SVD后的数据
        self.truncated_data = np.dot(U_truncated, np.dot(Sigma_truncated, Vt_truncated))

    def get_truncated_data(self):
        """
        获取截断SVD后的数据
        """
#        print(self.n_components)
        self.truncate_svd(self.n_components)
        return self.truncated_data

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        # 假设model_origin是从服务端下载的模型（基准模型）

        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        #        分布计算后正则化过程
        #        with torch.no_grad():
        #            loss2 = 0
        #            loss1 = 0
        #            a = 0
        #            b = 0
        #            for i, (x, y) in enumerate(trainloader):
        #                if type(x) == type([]):
        #                    x[0] = x[0].to(self.device)
        #                else:
        #                    x = x.to(self.device)
        #                y = y.to(self.device)
        #                if self.train_slow:
        #                    time.sleep(0.1 * np.abs(np.random.rand()))
        #                output = self.model(x)
        #                loss2 += self.loss(output, y)
        #                a = a + 1

        #            for i, (x, y) in enumerate(trainloader):
        #                if type(x) == type([]):
        #                    x[0] = x[0].to(self.device)
        #                else:
        #                    x = x.to(self.device)
        #                y = y.to(self.device)
        #                if self.train_slow:
        #                    time.sleep(0.1 * np.abs(np.random.rand()))
        #                output = self.pmodel(x)
        #                loss1 += self.loss(output, y)
        #                b = b + 1

        #        lambda_personalization = 0
        #        if lambda_personalization <= (loss2/a - loss1/b):
        #            lambda_personalization = (loss2-loss1)
        #        else:
        #            lambda_personalization = 0
        # Calculate personalization regularization term

        for step in range(max_local_epochs):
            #            print("lambda_personalization = ",lambda_personalization)
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                #Calculate personalization regularization term
                #personalization_reg = 0
                #for param, base_param in zip(self.model.parameters(), self.pmodel.parameters()):
                #personalization_reg += torch.norm(param - base_param, 2) ** 2

                #loss += lambda_personalization * personalization_reg  # 加入个性化正则化项
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)