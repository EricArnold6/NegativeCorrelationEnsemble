import numpy as np
import matplotlib.pyplot as plt

n_epoch = 27
filename = 'proj_loss'
loss_type = 'project_norm_sum'

def load_txt(filename, loss_type, n_epoch):              #合并多个txt文件，返回一个list
    # r_list = []
    file_path = "C:\\Users\\ER2\\OneDrive\\桌面\\code\\advexmp\\Projection\\result\\" + filename + '-' + str(n_epoch) + '-' + loss_type + ".txt"
    r = np.loadtxt(file_path)
    # r_list.extend(r)
    print(r)
    print("读取成功")
    return r

if __name__ == '__main__':
    y1 = load_txt(filename, loss_type, n_epoch)
    y2 = load_txt(filename='ce_loss', loss_type='project_norm_sum', n_epoch=27)
    plt.plot(y1)
    plt.plot(y2)
    plt.axis([0, 40])
    plt.legend(['project_loss', 'ce_loss'])
    plt.show()

