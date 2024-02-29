from database import MyDataset
import torch

areaList = ['jiduo', 'huashengdun', 'niuyue', 'shengbidebao', 'xichang', 'xinjiapo']
if __name__ == '__main__':
    DEVICE_ID = "cuda:2"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    for area in areaList:
        for v in [0.5, 1, 1.5, 2]:
            MyDataset.MyDataset(area, DEVICE, v)
