from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 用于进行可视化
from transweather_model import Transweather


modelviz = Transweather()
sampledata = torch.rand(1,3, 224, 224)
summary(modelviz,sampledata),
# # 创建输入
# sampledata = torch.rand(1, 3, 224, 224)
# # 看看输出结果对不对
# out = modelviz(sampledata)
# print(out)  # 测试有输出，网络没有问题

# # 1. 来用tensorflow进行可视化
# with SummaryWriter("./log", comment="sample_model_visualization") as sw:
#     sw.add_graph(modelviz, sampledata)