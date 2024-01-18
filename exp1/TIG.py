import GetData
import RNN
import GRU
import LSTM
import MTMv2
import torch
import torch.nn as nn
import numpy as np

INPUT_SIZE = 1 # lstm 的输入维度
DEVICE =  "cuda"
H_SIZE = 32 # of lstm 隐藏单元个数
EPOCHS = 100000 # 总共训练次数
h_state = torch.zeros(1,H_SIZE) # 隐藏层状态
cell = torch.zeros(1,H_SIZE)
torch.set_printoptions(precision=8)
print(DEVICE)

def cal_pred_of_proto(feature, proto, exists_flag, tau=1.0):
    # calculate distance
    feature = feature.unsqueeze(1) # [B，1，CH，D，H，W]
    print(feature,feature.shape)
    proto = proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [1, c, CH, 1, 1,1]dist = torch.norm(proto - feature, dim=2) # [B，C，D，H，W]
    print(proto, proto.shape)
    dist = torch.norm(proto - feature,dim = 2) # [B,C,D,H,W]
    print(dist, dist.shape)
    # calculate weight
    weight = torch.exp(-dist/tau) # [B，C，D，H，W]
    print(weight, weight.shape)
    weight = weight * exists_flag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [B, C, D, H, W]
    print(weight, weight.shape)
    weight = weight / torch.sum(weight, dim=1, keepdim=True) # [B, ,D,H, W] TODO: 这步计算导致 weight nan
    print(weight, weight.shape)
    return weight


# B = 1 CH = 2 D = 1 H =2 W =2 C = 2
cal_pred_of_proto(
    feature=torch.tensor([
        [
            [
                [
                 [3,4]
                ],[
                [5,6]
            ]
            ]
        ],[
            [
                [
                 [7,8]
                ],[
                [9,10]
                ]
            ]
        ]
    ]),
    proto=torch.tensor([[1,2],[3,4]]),
    exists_flag=torch.tensor([1,2])
)