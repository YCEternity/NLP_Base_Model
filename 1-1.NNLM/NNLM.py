# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.optim as optim


# 构建输入输出数据
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # 将句子中的每个词分词

        # :1表示去每个句子里面的前两个单词作为输入
        # 再通过word_dict取出这两个单词的下标，作为整个网络的输入
        input = [word_dict[n] for n in word[:-1]]
        # target取预测单词的下标
        target = word_dict[word[-1]]

        # 输入数据集
        input_batch.append(input)
        # 输出数据集
        target_batch.append(target)
    return input_batch, target_batch


# 定义网络结构，继承nn.Module
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # 计算词向量表，大小为len(word_dict) * m
        self.C = nn.Embedding(n_class, m)
        # 初始化网络参数
        """公式如下：
            hiddenout = tanh(a + X*H)
            y = b + X*H + hiddenout*U
        """
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        """
        X: [batch_size, n_step]
        """
        # 根据词向量表，将输入数据转换为三维数据
        # 将每个单词替换为相应的词向量
        X = self.C(X)  # [batch_size, n_step] => [batch_size, n_step, m]
        # 把替换后的词向量表的相同行进行拼接
        # view的第一个参数为-1，表示自动判断需要合并成几行
        X = X.view(-1, n_step * m)
        tanh = torch.tanh(self.d + self.H(X))  # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh)  # [batch_size, n_class]
        return output


if __name__ == '__main__':
    n_step = 2  # 计算步长
    n_hidden = 2  # 隐藏层参数量
    m = 2  # embedding size，即嵌入词向量的维度

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    # 将上面的单词逐个分开
    word_list = " ".join(sentences).split()
    # 将分词后的结果去重
    word_list = list(set(word_list))
    # 对单词建立索引，for循环里面先取索引，再取单词
    word_dict = {w:i for i, w in enumerate(word_list)}
    # 反向简历索引
    number_dict = {i:w for i, w in enumerate(word_list)}
    # 计算词典长度
    n_class = len(word_dict)

    model = NNLM()
    # 分类问题用交叉熵做损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器使用Adam,所谓优化器，即用某种方法取更新网络中的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    input_batch, target_batch = make_batch(sentences)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # 开始训练
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if(epoch + 1)%1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        # 反向传播更新梯度
        loss.backward()
        optimizer.step()

    # 预测
    # max（）表示取的是最内层维度中最大的那个数的值和索引，[1]表示取索引
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # test
    # squeeze()表示把数组中维度为1的维度去掉
    print([sen.split()[:n_step] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

