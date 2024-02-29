
import torch
import torch.nn as nn

def cal_pred_of_proto(feature, proto, exists_flag, tau=1.0):
    # calculate distance
    print("feature:")
    print(feature.shape)
    feature = feature.unsqueeze(1) # [B，1，CH，jiduoD，H，W]
    print("feature:")
    print(feature.shape)
    proto = proto.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [1, c, CH, 1, 1,1]
    print("proto:")
    print(proto.shape)
    print(proto - feature,(proto - feature).shape)
    dist = torch.norm(proto - feature,dim = 2) # [1, c, CH, 1, 1,1] - [B，1，CH，jiduoD，H，W] = [B,C,CH,jiduoD,H,W] norm -> [B,C,jiduoD,H,W]
    print("dist:")
    print(dist, dist.shape)
    # calculate weight
    print("weight:")
    weight = torch.exp(-dist/tau) # [B，C，jiduoD，H，W]
    print("weight:")
    print(weight, weight.shape)
    weight = weight * exists_flag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [B, C, jiduoD, H, W]
    #exists_flag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) [1,C,1,1,1]
    print("weight:")
    print(weight, weight.shape)
    weight = weight / torch.sum(weight, dim=1, keepdim=True) # [B, C,jiduoD,H, W] TODO: 这步计算导致 weight nan
    print("weight:")
    print(weight, weight.shape)
    return weight


# B = 1 CH = 2 jiduoD = 1 H =2 W =2 C = 2
for i in range(10):
    try:
        x = 1/0
    except Exception as e:
        print("发生了一个异常:", e)
        print(i)
        i = i - 2
        continue  # 退出程