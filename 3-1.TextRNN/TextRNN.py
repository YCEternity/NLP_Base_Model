import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input，最后一个词不要
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        # np.eye()构建一个n_class*n_class的对角矩阵，[input]表示提取出第input-1行，因为是从0开始的
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        # input_size指每个词用多少维的向量去编码, hidden_size指的是输出词的维度
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        # fully connected layer
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    # number of cells(= number of Step)，一句话里有多少个词作为输入
    n_step = 2
    # number of hidden units in one cell
    # 输入的数据，输出之后，要转换为多少维
    n_hidden = 5

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    # 把词转换为一个索引
    word_dict = {w: i for i, w in enumerate(word_list)}
    # 把索引转换为一个词
    number_dict = {i: w for i, w in enumerate(word_list)}
    # 有word_list这么多的类
    n_class = len(word_dict)
    batch_size = len(sentences)

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()

        # hidden : [num_layers * num_directions, batch, hidden_size]
        hidden = torch.zeros(1, batch_size, n_hidden)
        # input_batch : [batch_size, n_step, n_class]
        output = model(hidden, input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])