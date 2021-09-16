'''
函数说明:  升级版也放在这里了
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 17:05:53
'''
import math
import numpy as np
from matplotlib.pyplot import imshow
from numpy.core.fromnumeric import mean, std
%matplotlib inline
np.random.seed(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from emo_utils import *

X_test, Y_test = read_csv('data/tesss.csv')

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


def custompad(X,max_len=10):
    X = torch.Tensor(X)
    m=X.shape[0]
    pad = torch.zeros(1,50)
    for i in range(max_len-m):
        X = torch.cat([X,pad],dim=0)
    return X
def sentence_to_avg(sentences, word_to_vec_map):
    res=[]
    totallen=[]
    for sentence in sentences:
        # 取出句子中的每一个单词，并且转换成小写形式
        words = [i.lower() for i in sentence.split()]
        avg = []
        # 遍历每一个单词，并且转换成Glove向量，然后将每个向量累加起来
        for w in words:
            tw = word_to_vec_map[w]
            avg.append(tw)
        avg = custompad(avg)
        res.append(avg)
        totallen.append(len(words))
    final = torch.stack(res,dim=0)
    return final,totallen

X_test,test_lenth = sentence_to_avg(X_test,word_to_vec_map)
X_test = pack(X_test,test_lenth,batch_first=True,enforce_sorted=False)
def calacc(X_test,Y_test,net):
    out = torch.max(net(torch.FloatTensor(X_test)),dim=1)[1]
    acc = np.sum(out.numpy()==Y_test)/Y_test.shape[0]
    print('accuracy = {}%'.format(acc*100))
    
class traindata(Dataset):
    def __init__(self) -> None:
        super().__init__()
        X, self.Y = read_csv('data/train_emoji.csv')
        self.X,self.lens=sentence_to_avg(X,word_to_vec_map)
        
    def __getitem__(self, index):
        return self.X[index],self.lens[index],self.Y[index]

    def __len__(self):
        return self.X.shape[0]

def my_collate(batch):
    data = torch.stack([item[0] for item in batch])
    lens = [item[1] for item in batch]
    res = pack(data,lens,batch_first=True,enforce_sorted=False)
    target = [item[2] for item in batch]
    target = torch.LongTensor(target)
    return [res, target]

Xdata = traindata()
datas = dataloader.DataLoader(Xdata,32,True,collate_fn=my_collate)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(50,128,1)
        self.lstm2 = nn.LSTM(128,256,1)
        self.dp = nn.Dropout(0.5)
        self.dense = nn.Linear(256,512)
        # self.dense2 = nn.Linear(256,512)
        self.fc = nn.Linear(512,5)

    def forward(self,x,h=None,c=None):
        hp=(h,c) if h!=None and c!=None else None
        x,hid = self.lstm(x,hp)
        x,hid = self.lstm2(x,hp)
        x = unpack(x,batch_first=True)
        seq_len_indices = [length - 1 for length in x[1]]
        batch_indices = [i for i in range(x[0].shape[0])]
        x = x[0][batch_indices, seq_len_indices, :]
        x = self.dp(x)
        x = F.relu(self.dense(x))
        x = self.dp(x)
        # x = F.relu(self.dense2(x))
        x = self.fc(x)
        return x

net = Net()
optimizer = torch.optim.Adam(net.parameters())
lossfunc = nn.CrossEntropyLoss()

for i in range(1001):
    net.train()
    for _,(x,y) in enumerate(datas):
        out = net(x)
        loss = lossfunc(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(i%50==0):
        print('episode{} finished,loss = {}'.format(i,loss.item()))
    if(i%100==0):
        net.eval()
        out = torch.max(net(X_test),dim=1)[1]
        acc = np.sum(out.numpy()==Y_test)/Y_test.shape[0]
        print('accuracy = {}%'.format(acc*100))

X_my_sentence = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"]) 

X_my_sentences,lens=sentence_to_avg(X_my_sentence,word_to_vec_map)
o = net(pack(X_my_sentences,lens,batch_first=True,enforce_sorted=False))
pred = torch.max(o,dim=1)[1].numpy()
print_predictions(X_my_sentence, pred)


