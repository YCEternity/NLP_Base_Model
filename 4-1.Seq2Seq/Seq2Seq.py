import argparse
import numpy as np
import torch
import torch.nn as nn


def make_batch():
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))  # 都取第一个，不够的用P （pad）补齐 => manPP，women

        input = [num_dic[n] for n in seq[0]]  # encoder input => 'm','a','n','?','?','E'
        output = [num_dic[n] for n in ('S' + seq[1])]  # decoder input，加上S表示开始 => 'S','w','o','m','e','n'
        target = [num_dic[n] for n in (seq[1] + 'E')]  # decoder output，加上E表示结束 => 'w','o','m','e','n','E'

        # encoder input 和decoder input使用one-hot形式
        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)  # decoder output不用 one-hot，只用索引就行

    # 转换为tensor，decoder output的Tensor必须是LongTensor类型，因为计算交叉熵的时候，第二个参数需要是LongTensor类型的
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


# Model很简单，先训练encoder、 decoder，再把维度匹配上，最后得到output
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        # 用传统的RNN训练encoder decoder，用LSTM会比较慢
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1) # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = dec_input.transpose(0, 1) # dec_input: [max_len(=n_step, time step), batch_size, n_class]

        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.dec_cell(dec_input, enc_states)

        model = self.fc(outputs) # model : [max_len+1(=6), batch_size, n_class]
        return model


if __name__ == '__main__':
    n_step = 5
    n_hidden = 128

    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    # 字母转换为索引
    num_dic = {n: i for i, n in enumerate(char_arr)}
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    n_class = len(num_dic)  # 分类问题。有多少个字母，九有多少个类
    batch_size = len(seq_data)

    model = Seq2Seq()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, output_batch, target_batch = make_batch()

    for epoch in range(5000):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, batch_size, n_hidden)

        optimizer.zero_grad()
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1, batch_size, n_class]
        output = output.transpose(0, 1) # [batch_size, max_len+1(=6), n_class]
        loss = 0
        for i in range(0, len(target_batch)):
            # output[i] : [max_len+1, n_class, target_batch[i] : max_len+1]
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    def translate(word):
        input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]])

        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, 1, n_hidden)
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1(=6), batch_size(=1), n_class]

        predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))