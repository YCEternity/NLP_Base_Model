# -*-coding:utf-8-*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 构建输入输出数据
def random_batch():
    random_inputs = []
    random_labels = []
    # np.random.choice（ndarray, size, replace）从一维
    # 的数组里随机抽取数字，组成size大小的数组，replace=True
    # 表示可以抽取相同的数字
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        # np.eye生成一个对角矩阵
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])
        random_labels.append(skip_grams[i][1])  # 上下文词

    return random_inputs, random_labels


# Word2Vec
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Linear(voc_size, embedding_size, bias=False)  # voc_size > embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)  # embedding_size > voc_size Weight

    def forward(self, X):
            # X : [batch_size, voc_size]
            hidden_layer = self.W(X)
            output_layer = self.WT(hidden_layer)
            return output_layer


if __name__ == '__main__':
    batch_size = 2
    embedding_size = 2  # 嵌入词向量的维度
    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    # 将上面的单词逐个分开
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    # 将分词后的结果去重
    word_list = list(set(word_list))
    # 对单词建立索引，for循环里面先取索引，再取单词
    word_dict = {w: i for i, w in enumerate(word_list)}
    # 计算词典长度
    voc_size = len(word_list)

    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


