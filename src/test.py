import random
import numpy as np
from torchvision import datasets, transforms, models
import torch


def text_save(filename, epoch, loss_type, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    # file_path = "/home/zhw/code/advexmp/Projection/result" + filename + '-' + str(epoch) + ".txt"
    file_path = "C:\\Users\\ER2\\OneDrive\\桌面\\code\\advexmp\\Projection\\result\\" + filename + '-' + str(epoch) + '-' + loss_type + ".txt"
    r_file = open(file_path, 'w+')          #会对原有内容清空并有读写权限
    # file = open(filename, 'a')

    for i in range(len(data)):
        # s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        # s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        s = str(data[i])
        s = s.replace("'", '').replace(',', '') +'\n'

        # s = s.replace("'", '') + '\n'
        r_file.write(s)
    r_file.close()
    print("保存文件成功")

def load_txt(filename, loss_type):
    r_list = []
    for epoch in range(3):
        file_path = "C:\\Users\\ER2\\OneDrive\\桌面\\code\\advexmp\\Projection\\result\\" + filename + '-' + '0' + '-' + loss_type + ".txt"
        r = np.loadtxt(file_path)
        r_list.append(r)
    r_list = np.stack(r_list)
    r_list = r_list.reshape(-1)
    print(r_list)
    print("读取成功")

    # r_file.close()

# a = []
# for i in range(64):
#     a.append(i)
# a = []
# print(a)
# a = torch.tensor([2, 3])
# b = torch.tensor([2, 1])
# print(torch.dot(a, b))
# print(a)
# print(b)
# # a = a.reshape(2, 3)
# b = np.exp(a)
# print(5e-4)
# # text_save('aaa', 1, 'loss',  a)
# # if __name__ == '__main__':
# #     load_txt('ce_loss', 'project_lossnew')
# # a = torch.arange(1)
# # # a = a.view(2, -1)
# # a = a.item()
# # print(200%100)
# b = a[:, 0]
# print(b)
# print(torch.mul(3,2))
a = torch.tensor([2])
print(a.shape)
# data_train = datasets.FashionMNIST
