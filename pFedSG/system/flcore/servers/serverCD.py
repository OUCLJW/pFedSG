import time
from flcore.clients.clientCD import clientCD
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
from utils.data_utils import read_client_data
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

#        param1 = [p.data for p in params1.parameters()]
#        flat_params1 = torch.cat([p.view(-1) for p in param1])
#        param2 = [p.data for p in params2.parameters()]
#        flat_params2 = torch.cat([p.view(-1) for p in param2])
#        manhattan_distance = np.sum(np.abs(flat_params1 - flat_params2))
#        similarity = 1 / (1 + manhattan_distance)
#        return similarity

#余弦
def compute_subspace_similarity(U_i, U_j):
    flat_U_i = U_i.flatten()
    flat_U_j = U_j.flatten()
    max_len = max(len(flat_U_i), len(flat_U_j))
    flat_U_i = np.pad(flat_U_i, (0, max_len - len(flat_U_i)), 'constant')
    flat_U_j = np.pad(flat_U_j, (0, max_len - len(flat_U_j)), 'constant')
    dot_product = np.dot(flat_U_i, flat_U_j)
    norm_U_i = np.linalg.norm(flat_U_i)
    norm_U_j = np.linalg.norm(flat_U_j)
    if norm_U_i == 0 or norm_U_j == 0:
        return 0
    similarity = dot_product / (norm_U_i * norm_U_j)
    return similarity

#L2
#def compute_subspace_similarity(U_i, U_j):
#    flat_U_i = U_i.flatten()
#    flat_U_j = U_j.flatten()
#    max_len = max(len(flat_U_i), len(flat_U_j))
#    flat_U_i = np.pad(flat_U_i, (0, max_len - len(flat_U_i)), 'constant')
#    flat_U_j = np.pad(flat_U_j, (0, max_len - len(flat_U_j)), 'constant')
#    return 1/np.sqrt(np.sum((flat_U_i - flat_U_j) ** 2))

#曼哈顿
#def compute_subspace_similarity(U_i, U_j):
#    flat_U_i = U_i.flatten()
#    flat_U_j = U_j.flatten()
#    max_len = max(len(flat_U_i), len(flat_U_j))
#    flat_U_i = np.pad(flat_U_i, (0, max_len - len(flat_U_i)), 'constant')
#    flat_U_j = np.pad(flat_U_j, (0, max_len - len(flat_U_j)), 'constant')
#    manhattan_distance = np.sum(np.abs(flat_U_i - flat_U_j))
#    similarity = 1 / (1 + manhattan_distance)
#    print(similarity)
#    return similarity

#def compute_subspace_similarity(U_i, U_j):
#    flat_U_i = U_i.flatten()
#    flat_U_j = U_j.flatten()
#    max_len = max(len(flat_U_i), len(flat_U_j))
#    flat_U_i = np.pad(flat_U_i, (0, max_len - len(flat_U_i)), 'constant')
#    flat_U_j = np.pad(flat_U_j, (0, max_len - len(flat_U_j)), 'constant')
#    hamming_distance = np.count_nonzero(flat_U_i != flat_U_j)
#    return hamming_distance
class ServerCD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCD)
        self.kr = args.kr


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # 存储来自各客户端的子空间表示
        self.client_subspaces = {}  # 使用字典来存储客户端ID和其子空间
        self.client_similarity_weights = {}  # 初始化一个字典来存储权重
        self.similar_clients = {}
        self.its_weights = {}
#    def receive_subspace(self, client_id, U_k):
#        """
#        用于接收来自客户端的子空间U_k，并存储下来。
#        """
#      self.client_subspaces[client_id] = U_k

    def compute_all_similarities(self):
        """
        计算并返回所有客户端子空间之间的两两相似度。
        """
        client_ids = list(self.client_subspaces.keys())
        num_clients = len(client_ids)
        similarity_matrix = np.zeros((num_clients, num_clients))

        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                sim = compute_subspace_similarity(self.client_subspaces[client_ids[i]],
                                                       self.client_subspaces[client_ids[j]])
#                print(i,j,sim)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix, client_ids

    def select_top_similar_clients(self, k):
        """
        根据每个客户端之间的相似度选择最相似的k个客户端，并存储在一个字典中。
        """
        similarity_matrix, client_ids = self.compute_all_similarities()
        similar_clients = {}
        for i in range(20):
            for j in range(20):
                print(i,j,similarity_matrix[i][j])


        for i, client_id in enumerate(client_ids):
            sim_vector = similarity_matrix[i]
            top_k_indices = np.argpartition(-sim_vector, kth=len(sim_vector) - k)[:k]
            # 因为我们对sim_vector取了负数，所以现在需要将角标转换回原数组中最大值的角标
            # 这里我们只需要取前k个角标即可
            top_k_indices = np.take_along_axis(np.argsort(-sim_vector), top_k_indices, axis=0)
            # 输出结果
            # 从索引转换为客户端ID
            similar_clients[client_id] = [client_ids[idx] for idx in top_k_indices]

            # 打印结果以验证
            print(f"Client ID: {client_id}, Top {k} Similar Clients: {similar_clients[client_id]}")

        return similar_clients
    #重写包含自身的权重
#余弦相似度计算和对比方法
#    def compute_and_store_weights(self):
#        self.client_similarity_weights = {}  # 初始化一个字典来存储权重
#        for client_id, top_k_similar_clients in self.similar_clients.items():
#            weights = []
#            current_client_params = self.uploaded_models[client_id]
#            for similar_client_id in top_k_similar_clients:
#                similar_client_params = self.uploaded_models[similar_client_id]
#                similarity = self.compute_cosine_similarity(current_client_params, similar_client_params)
#                weights.append(similarity)
#            its_weights = self.compute_cosine_similarity(current_client_params, current_client_params)
#            self.its_weights[client_id] = its_weights
#            # 将计算好的权重列表存储在字典中
#            self.client_similarity_weights[client_id] = weights
#    def compute_cosine_similarity(self, params1, params2):
#        param1 = [p.data for p in params1.parameters()]
#        flat_params1 = torch.cat([p.view(-1) for p in param1])
#        param2 = [p.data for p in params2.parameters()]
#        flat_params2 = torch.cat([p.view(-1) for p in param2])
#        dot_product = np.dot(flat_params1, flat_params2)
#        norm_params1 = np.linalg.norm(flat_params1)
#        norm_params2 = np.linalg.norm(flat_params2)
#        if norm_params1 == 0 or norm_params2 == 0:
#            return 0
#        similarity = dot_product / (norm_params1 * norm_params2)
#        return similarity
#   曼哈顿距离对比
#    def compute_and_store_weights(self):
#        self.client_similarity_weights = {}
#        for client_id, top_k_similar_clients in self.similar_clients.items():
#            weights = []
#            current_client_params = self.uploaded_models[client_id]
#            for similar_client_id in top_k_similar_clients:
#                similar_client_params = self.uploaded_models[similar_client_id]
#                similarity = self.compute_manhattan_similarity(current_client_params, similar_client_params)
#                weights.append(similarity)
#            its_weights = self.compute_manhattan_similarity(current_client_params, current_client_params)
#            self.its_weights[client_id] = its_weights
#            self.client_similarity_weights[client_id] = weights
#    def compute_manhattan_similarity(self, params1, params2):
#        param1 = [p.data for p in params1.parameters()]
#        flat_params1 = torch.cat([p.view(-1) for p in param1])
#        param2 = [p.data for p in params2.parameters()]
#        flat_params2 = torch.cat([p.view(-1) for p in param2])
#        manhattan_distance = np.sum(np.abs(flat_params1 - flat_params2))
#        similarity = 1 / (1 + manhattan_distance)
#        return similarity
#   正弦相似度对比方法
#    def compute_and_store_weights(self):
#        self.client_similarity_weights = {}
#        for client_id, top_k_similar_clients in self.similar_clients.items():
#            weights = []
#            current_client_params = self.uploaded_models[client_id]
#            for similar_client_id in top_k_similar_clients:
#                similar_client_params = self.uploaded_models[similar_client_id]
#                similarity = self.compute_sine_similarity(current_client_params, similar_client_params)
#                weights.append(similarity)
#            its_weights = self.compute_sine_similarity(current_client_params, current_client_params)
#            self.its_weights[client_id] = its_weights
#            self.client_similarity_weights[client_id] = weights

#    def compute_sine_similarity(self, params1, params2):
#        cosine_similarity = self.compute_cosine_similarity(params1, params2)
#        sine_similarity = np.sqrt(1 - cosine_similarity ** 2)
#        return sine_similarity

#    def compute_cosine_similarity(self, params1, params2):
#        param1 = [p.data for p in params1.parameters()]
#        flat_params1 = torch.cat([p.view(-1) for p in param1])

#        param2 = [p.data for p in params2.parameters()]
#        flat_params2 = torch.cat([p.view(-1) for p in param2])

        # 计算点积和范数
#        dot_product = torch.dot(flat_params1, flat_params2)
#        norm_params1 = torch.norm(flat_params1)
#        norm_params2 = torch.norm(flat_params2)

        # 避免除以零的情况
#        if norm_params1 == 0 or norm_params2 == 0:
#            return 0

#        similarity = dot_product / (norm_params1 * norm_params2)
#        return similarity
#    def compute_and_store_weights(self):
#        self.client_similarity_weights = {}
#        for client_id, top_k_similar_clients in self.similar_clients.items():
#            weights = []
#            current_client_params = self.uploaded_models[client_id]
#            for similar_client_id in top_k_similar_clients:
#                similar_client_params = self.uploaded_models[similar_client_id]
#                similarity = self.compute_hamming_distance(current_client_params, similar_client_params)
#                weights.append(similarity)
#            its_weights = self.compute_hamming_distance(current_client_params, current_client_params)
#            self.its_weights[client_id] = its_weights
#        self.client_similarity_weights[client_id] = weights

 #   def compute_hamming_distance(self, params1, params2):
 #       param1 = [p.data for p in params1.parameters()]
 #       flat_params1 = torch.cat([p.view(-1) for p in param1])
 #       param2 = [p.data for p in params2.parameters()]
 #       flat_params2 = torch.cat([p.view(-1) for p in param2])
 #       hamming_distance = np.count_nonzero(flat_params1 != flat_params2)
 #       return hamming_distance
    def compute_hamming_distance(self, params1, params2):
        param1 = [p.data for p in params1.parameters()]
        flat_params1 = torch.cat([p.view(-1) for p in param1])

        param2 = [p.data for p in params2.parameters()]
        flat_params2 = torch.cat([p.view(-1) for p in param2])

        # 汉明距离计算：不相等元素的数量
        hamming_distance = torch.sum((flat_params1 != flat_params2).float())

        # 将汉明距离转换为权重
        # 注意：这里使用1/(1+distance)作为权重，确保权重在0到1之间
        weight = 1 / (1 + hamming_distance.item())

        return weight

    def compute_and_store_weights(self):
        self.client_similarity_weights = {}  # 初始化一个字典来存储权重
        for client_id, top_k_similar_clients in self.similar_clients.items():
            weights = []
            current_client_params = self.uploaded_models[client_id]
            for similar_client_id in top_k_similar_clients:
                similar_client_params = self.uploaded_models[similar_client_id]
                similarity = self.compute_cosine_similarity(current_client_params, similar_client_params)
                weights.append(similarity)
            its_weights = self.compute_cosine_similarity(current_client_params, current_client_params)
            self.its_weights[client_id] = its_weights
            # 将计算好的权重列表存储在字典中
            self.client_similarity_weights[client_id] = weights
    def compute_cosine_similarity(self, params1, params2):
        param1 = [p.data for p in params1.parameters()]
        flat_params1 = torch.cat([p.view(-1) for p in param1])
        param2 = [p.data for p in params2.parameters()]
        flat_params2 = torch.cat([p.view(-1) for p in param2])
        dot_product = np.dot(flat_params1, flat_params2)
        norm_params1 = np.linalg.norm(flat_params1)
        norm_params2 = np.linalg.norm(flat_params2)
        if norm_params1 == 0 or norm_params2 == 0:
            return 0
        similarity = dot_product / (norm_params1 * norm_params2)
        return similarity
    def train(self):
        self.selected_clients = self.select_clients()
        print("Initializing client subspaces and computing initial similarities...")
        # ...[接收客户端子空间代码]...
        for client in self.selected_clients:
            self.client_subspaces[client.id] = client.get_truncated_data()
    #        c = client.get_truncated_data
#            self.receive_subspace(self, client.id, c)
#        self.receive_subspace(self, client_id, U_k)
        print("Selecting the top similar clients for each client...")
        print("kr=",self.kr)
#        print(self.client_subspaces)
        self.similar_clients = self.select_top_similar_clients(self.kr)

#        for client in self.selected_clients:
#            print("k的最相似客户端")
#            print(similar_clients[client.id])
        # 初始化权重为0
#        self.initialize_weights_to_zero()
#         服务器得做截断操作还得做截断操作

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
#            if i ==0:
#                print("Initializing client subspaces and computing initial similarities...")
                # ...[接收客户端子空间代码]...
#                for client in self.selected_clients:
#                    self.client_subspaces[client.id] = client.get_truncated_data()
                #            self.receive_subspace(self, client.id, client.get_truncated_data())
                #        self.receive_subspace(self, client_id, U_k)
#                print("Selecting the top similar clients for each client...")
#                print("kr=", self.kr)
#                print(self.client_subspaces)
#                similar_clients = self.select_top_similar_clients(self.kr)

#                for id in self.uploaded_ids:
#                    print("k的最相似客户端")
#                    print(similar_clients[id])
#                # 初始化权重为0
            print(f"\n-------------Round number: {i}-------------")
            if i > 0:
                self.send_modelsCD()

            for client in self.selected_clients:
                client.train()

            if i % self.eval_gap == 0:
                print("\nEvaluate global model")
                self.evaluate()
            self.receive_models()
#            self.generation_test_evaluate()
            # 跳过第一次计算权重，因为我们已经将它们初始化为0。

            print("Computing new weights and aggregating personalized models...")
            self.compute_and_store_weights()
            # 个性化聚合部分
            self.personalized_aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])



            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()


    def personalized_aggregate_parameters(self):
        """
        使用存储的相似度权重为每个客户端聚合个性化模型。
        """
#        self.client_similarity_weights[client_id] = weights 相似度权重键值对
        # 遍历每个客户端进行个性化聚合
        for client_id in self.uploaded_ids:
            # 获取当前客户端模型的参数
            client_model_params = self.model_to_params(self.uploaded_models[client_id])
            aggregated_params = [0]*( len(client_model_params))
#            aggregated_params = []
            # 从相似客户端权重中获取权重
            weights = self.client_similarity_weights.get(client_id, [])
            total_weight = sum(weights)+self.its_weights[client_id]
            print(total_weight)
            for i, weight in enumerate(weights):
                if weight > 0:
                    # 获取与当前客户端相似的客户端模型参数
                    similar_client_params = self.model_to_params(self.uploaded_models[self.similar_clients[client_id][i]])
                    # 聚合相似客户端参数
#                    aggregated_params = self.aggregate_weighted_parameters(client_model_params, similar_client_params, weight, total_weight)
                    # 更新当前客户端的模型参数
#                    aggregated_params = []

                    for i in range(len(client_model_params)):
                        # 聚合第i部分的参数
#                        weighted_param = client_model_params[i].clone()
                        if len(similar_client_params) > 0:
                            #                total_weight = sum(weights)
                            #                for sim_params, weight in zip(similar_params, weights):
                            weighted_param = (similar_client_params[i] * weight) / (total_weight)
                            aggregated_params[i] += weighted_param
            for i in range(len(client_model_params)):
                # 聚合第i部分的参数
                weighted_param = client_model_params[i].clone()
                aggregated_params[i] += (client_model_params[i] * self.its_weights[client_id]) / total_weight

#                aggregated_params.append(weighted_param)
            self.set_params_to_model(self.uploaded_models[client_id], aggregated_params)
            # 获取与当前客户端最相似的客户端模型参数
#            similar_client_params = [self.model_to_params(self.uploaded_models[similar_client_id])
#                                     for similar_client_id in self.similar_clients[client_id]]

            # 对于当前客户端，聚合个性化模型参数
#            new_params = self.aggregate_weighted_parameters(client_model_params, similar_client_params, weights)

            # 设置新的个性化模型参数到客户端
#            self.set_params_to_model(self.uploaded_models[client_id], new_params)

    def aggregate_weighted_parameters(self, base_params, similar_params, weights, total_weight):
        """
        使用权重聚合类似客户端的模型参数。
        """
        # 聚合相似客户端参数
        aggregated_params = []
        for i in range(len(base_params)):
            # 聚合第i部分的参数
            weighted_param = base_params[i].clone()
            if len(similar_params) > 0:
#                total_weight = sum(weights)
#                for sim_params, weight in zip(similar_params, weights):
                weighted_param += (similar_params[i] * weights) / (1 + weights)
            aggregated_params.append(weighted_param)
        return aggregated_params

    def model_to_params(self, model):
        """
        从模型中提取参数。
        """
        return [param.data for param in model.parameters()]

    def set_params_to_model(self, model, params):
        """
        将新的参数设置回模型中。
        """
        for old_param, new_param in zip(model.parameters(), params):
            old_param.data = new_param.clone()



#下发操作要重写
    # 服务器处理模型请求的简化代码
    def initialize_weights_to_zero(self):
        # 假定self.weights是一个字典，
        # 其中键是客户端ID，值是对应的权重
        self.weights = {client_id: 0 for client_id in self.similar_clients.keys()}
        # 如果你有一个不同的数据结构存储权重，
        # 需要按照实际方式创建所有0权重
    #泛化性测试
