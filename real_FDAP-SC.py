import time
from turtle import color
import matplotlib as mpl
import numpy as np
import os
import cv2
from skimage import data
from skimage import color
from skimage import img_as_float
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import *
from sklearn import datasets, preprocessing
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import normalized_mutual_info_score, rand_score, adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons, make_friedman1, make_friedman2
from scipy import stats
from sklearn.utils import as_float_array
from scipy.optimize import linear_sum_assignment


def get_results(labels,cluster):
    TP, TN, FP, FN = 0, 0, 0, 0
    n = len(labels)
    # a lookup table
    for i in range(n):
        for j in range(i + 1, n):
            same_label = (labels[i] == labels[j])
            same_cluster = (cluster[i] == cluster[j])
            if same_cluster:
                if same_label:
                    TP += 1
                else:
                    FP += 1
            elif same_label:
                FN += 1
            else:
                TN += 1
    return TP,TN,FP,FN
def affiar(v1,v2):
    return -np.sqrt(np.sum(np.power(v1-v2,2),axis=1))
def distance(v1,v2,axis=0):
    return np.sqrt(np.sum(np.power(v1-v2,2),axis))
def AP(S,kind,preference=None,convergence_iter=50,max_iter=1000,damping=0.55):#0.55 median
    size = S.shape[0]
    if preference is None:
        preference = np.median(S)


    S.flat[::size+1] = preference
    random_state = np.random.RandomState(0)
    S += ((np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100) *
          random_state.randn(size,size))

    A = np.zeros((size,size))
    R = np.zeros((size,size))
    E = np.zeros((kind,convergence_iter))#存储标签矩阵
    tmp = np.zeros((size,size))#中间矩阵
    ind = np.arange(size)

    #迭代R
    for iter in range(max_iter):
        np.add(A,S,tmp)
        max_index = np.argmax(tmp,axis = 1)
        max1 = tmp[ind,max_index]
        tmp[ind,max_index] = -np.Inf
        max2 = np.max(tmp,axis = 1)
        np.subtract(S,max1[:,None],tmp)
        tmp[ind,max_index] = S[ind,max_index] - max2


        R = (1 - damping) * tmp + damping * R


        #迭代A
        np.maximum(R,0,tmp)
        tmp.flat[::size+1] = R.flat[::size+1]

        tmp = np.sum(tmp,axis=0) - tmp
        temp = np.diag(tmp).copy()
        tmp = np.minimum(tmp,0)
        tmp.flat[::size+1] = temp
        A = (1 - damping) * tmp + damping * A

        I = np.argsort(np.diag(A + R))
        I = I[::-1]
        I = I[:kind]
        '''
        改
        '''
        E[:,iter%convergence_iter] = I
        if iter >= convergence_iter:
            panduan = 1
            for i in range(kind):
                if np.where(E[i,:]==E[i,0])[0].shape[0] != convergence_iter:
                    panduan = 0
            if (panduan and kind > 0) or (iter == max_iter):
                print("在第%d次达到收敛"%iter)
                break
    #获取前k个最大值

    # I = np.where(I)[0]#获取初始聚类下标
    K = I.size
    if K > 0:
        label_one = np.argmax(S[:,I],axis=1)
        label_one[I] = np.arange(K)
        for k in range(K):
            label_sample = np.where(label_one == k)[0]
            j = np.argmax(np.sum(S[label_sample[:,np.newaxis],label_sample],axis=0))#更新后的聚类下标
            I[k] = label_sample[j]#在每个初试聚类中找到与类中的点的相似度和最高的点
        label_final = np.argmax(S[:,I],axis=1)
        label_final[I] = np.arange(K)
        labels = I[label_final]
        clusters = np.unique(labels)
        labels = np.searchsorted(clusters,labels)
    else:
        labels = np.empty((size,1))
        labels.fill(np.nan)
    return labels
def scc(data):
    size,dim = data.shape
    dis = np.zeros((size,size))
    snn = np.zeros((size,size))
    front = time.time()
    for i in range(size):
        dis[i,:] = distance(data,data[i,:],axis=1)
    dis1 = np.argsort(dis,axis=1)
    neigh_bor_sum = 0
    neighbor = []
    reverse_neighbor = []
    for i in range(size):
        reverse_neighbor.append([])
    for i in range(size):
        neighbor_i = []
        neighbor_i.append(dis1[i,0])
        neighbor.append(neighbor_i)
        reverse_neighbor[dis1[i,0]].append(i)
    last_sum = 0
    for i in range(1, size):
        nb = np.zeros(size)
        max = 0
        neigh_bor_temp = []#当前的邻居
        for k in range(size):
            neighbor[k].append(dis1[k,i])
            reverse_neighbor[dis1[k,i]].append(k)
        for k in range(size):
            final_neighbor_temp = list(set(neighbor[k])&set(reverse_neighbor[k]))
            nb[k] = len(final_neighbor_temp)
            if nb[k]>max:
                max = nb[k]
                neigh_bor_temp = final_neighbor_temp.copy()
        last = 0
        while True:
            len_temp = len(neigh_bor_temp)
            if last == len_temp - 1:
                break
            else:
                for h in range(last,len_temp):
                    for k in neighbor[neigh_bor_temp[h]]:
                        if k not in neigh_bor_temp and neigh_bor_temp[h] in neighbor[k]:
                            neigh_bor_temp.append(k)
                last = len_temp - 1
        if len(neigh_bor_temp) == size:
            neigh_bor_sum = i
            break
        else:
            if len(neigh_bor_temp) == last_sum:
                zero_point = list(set(np.arange(size)) - set(neigh_bor_temp))
                zero_sum = size - len(neigh_bor_temp)
                weights = np.zeros(zero_sum)
                weights.fill(np.inf)
                for k in range(zero_sum):
                    for h in neighbor[zero_point[k]]:
                        if h == zero_point[k]:
                            pass
                        else:
                            if int(np.where(dis1[zero_point[k]] == h)[0]) > int(np.where(dis1[h] == zero_point[k])[0]):
                                max_ = int(np.where(dis1[zero_point[k]] == h)[0])
                            else:
                                max_ = int(np.where(dis1[h] == zero_point[k])[0])
                            if max_ < weights[k]:
                                weights[k] = max_
                neigh_bor_sum = int(np.max(weights))
                if neigh_bor_sum < i:
                    neigh_bor_sum = i
                break
            else:
                last_sum = len(neigh_bor_temp)
    last = time.time()
    print(last - front)
    print(neigh_bor_sum)
    neighbor = []
    reverse_neighbor = []
    for i in range(size):
        neigh_bor_temp = []
        for j in range(0,neigh_bor_sum+1):
            neigh_bor_temp.append(dis1[i,j])
        neighbor.append(neigh_bor_temp)
    for i in range(size):
        reverse_neighbor_temp = []
        for j in range(size):
            for h in range(0,neigh_bor_sum+1):
                if dis1[j,h]  == i:
                    reverse_neighbor_temp.append(j)
        reverse_neighbor.append(reverse_neighbor_temp)
    final_neighbor = []
    for i in range(size):
        final_neighbor.append(list(set(reverse_neighbor[i]) & set(neighbor[i])))
    std_size = np.zeros(size)
    for i in range(size):
        std_size[i] = np.std(neighbor[i],ddof=1)
    for i in range(size):
        for j in range(i, size):
            temp_1 = list(set(reverse_neighbor[i]) & set(reverse_neighbor[j]))
            temp_2 = list(set(neighbor[i]) & set(neighbor[j]))
            length1 = len(temp_1)
            length2 = len(temp_2)
            dis_temp1 = 0
            dis_temp2 = 0
            for h in range(length1):
                dis_temp1 += dis[i, temp_1[h]] + dis[j, temp_1[h]]
            for h in range(length2):
                dis_temp2 += dis[i, temp_2[h]] + dis[j, temp_2[h]]
            if length1 > 0 and length2 > 0:
                snn[i, j] = length1 ** 2 * length2 ** 2 / (dis_temp1 * dis_temp2)
                snn[j, i] = snn[i, j]
    return snn
def spectral_clustering(data,k):
    size = data.shape[0]
    S = scc(data)
    D = np.zeros((size, size))
    for i in range(size):
        D[i, i] = np.sum(S[i, :])
    D = np.linalg.inv((np.power(D, 0.5)))
    L = D.dot(np.dot(S, D))
    eigvals, eigvecs = np.linalg.eigh(L)
    indices = np.argsort(eigvals)
    indices = indices[::-1]
    indices = indices[:k]
    data_new = eigvecs[:,indices]
    for i in range(size):
        if np.sum(data_new[i,:] == 0):
            pass
        else:
            data_new[i,:] = data_new[i,:] / np.sqrt(np.sum(data_new[i,:]**2))
    s = np.zeros((size,size))
    for i in range(size):
        s[i,:] = affiar(data_new,data_new[i,:])
    return AP(s,k)

def read_directory(directory_name):#返回那个路径的所有文件,每一个文件夹下的pm文件
    faces_addr = []
    for file_name in os.listdir(directory_name):
        faces_addr.append(directory_name + "/" + file_name)
    return faces_addr

def load_faces_data(path):
    faces = []
    for i in range(1,41):#获取每个文件夹下的图片的路径，总共有400个路径放在faces里
        file_addr = read_directory(path + "/s" + str(i))
        for addr in file_addr:
            faces.append(addr)
    images = []
    label = []
    for index,face in enumerate(faces):
        image = cv2.imread(face,0)
        images.append(image)
        label.append(int(index/10+1))
    return images,label
def plot_image(images):
    # fig,axes = plt.subplots(1,1)
    aspect = 1.
    n = 10  # number of rows
    m = 10  # numberof columns
    bottom = 0.1;
    left = 0.05
    top = 1. - bottom;
    right = 1. - 0.18
    fisasp = (1 - bottom - (1 - top)) / float(1 - left - (1 - right))
    # widthspace, relative to subplot size
    wspace = 0  # set to zero for no spacing
    hspace = wspace / float(aspect)
    # fix the figure height
    figheight = 50  # inch
    figwidth = (m + (m - 1) * wspace) / float((n + (n - 1) * hspace) * aspect) * figheight * fisasp

    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        wspace=wspace, hspace=hspace)
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[int(i / 10) * 10 + i % 10], cmap="gray")
        ax.axis('off')
    plt.show()

def colorize(image, hue, saturation=1):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)

def plot_image2(images,label):
    # fig,axes = plt.subplots(1,1)

    aspect = 1.
    n = 20  # number of rows
    m = 10  # numberof columns
    bottom = 0.1;
    left = 0.05
    top = 1. - bottom;
    right = 1. - 0.18
    fisasp = (1 - bottom - (1 - top)) / float(1 - left - (1 - right))
    # widthspace, relative to subplot size
    wspace = 0  # set to zero for no spacing
    hspace = wspace / float(aspect)
    # fix the figure height
    figheight = 50  # inch
    figwidth = (m + (m - 1) * wspace) / float((n + (n - 1) * hspace) * aspect) * figheight * fisasp

    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        wspace=wspace, hspace=hspace)
    hue_rotations = np.linspace(0, 2, 20)
    for i,ax in enumerate(axes.flat):
        image1 = color.gray2rgb(images[int(i / 10) * 10 + i % 10])
        image1 = colorize(image1, hue_rotations[label[i]], saturation=0.5)
        ax.imshow(image1)
        ax.axis('off')
    plt.show()
def PCA_images(images,label):
    image_data = []
    label_ = []
    #95%
    #100,60是0.954,0.868,0.886,LDP是0.871,0.741,0.773,RNN是0.876,0.716,0.758(5)，DPC是0.630,0.051,0.111,KDPC是0.940,0.836,0.855,SNNDPC是0.871,0.737,0.766(5),
    # Dcore是0.888,0.759,0.784( r1=0.9,r2=0.81,r=0.92,t1=1,t2=0)
    #200，111，HORC是0.909,0.746,0.762，LDP是0.845,0.568,0.612(只有18类),RNN是0.720,0.217,0.383(5)，DPC是0.685,0.073,0.108，KDPC是0.895,0.636,0.676，SNNDPC是0.822,0.504,0.557(5)
    # Dcore是0.858,0.636,0.657( r1=0.9,r2=0.81,r=0.88,t1=1,t2=0)
    for i in range(200):
        data = images[int(i/10)*10+i%10].flatten()#把100*112*92的三维数组变成100*10304的二维数组
        label_.append(label[int(i/10)*10+i%10])
        image_data.append(data)
    X = np.array(image_data)#每个图像数据降到一维后的列表
    data = pd.DataFrame(X)#打印的话，X可以显示列和行号的
    pca = PCA(.95)
    pca.fit(X)
    PCA_data = pca.transform(X)
    expained_variance_ratio = pca.explained_variance_ratio_.sum()#计算保留原始数据的多少

    # 看降到i维的保留原始数据的曲线图
    # expained_variance_ratio = []
    # for i in range(1,200):
    #     pca = PCA(n_components=i).fit(X)#构建pca降维器
    #     expained_variance_ratio.append(pca.explained_variance_ratio_.sum())#计算每次降维，所带的数据是原始数据的多少
    #     print(i,pca.explained_variance_ratio_.sum())
    # plt.plot(range(1,160),expained_variance_ratio)
    # plt.show()
    # V = pca.components_#pca中间的转换矩阵，让10304*100转换成100*98的矩阵

    return PCA_data,label_

if __name__ == "__main__":
    path = "../data/olivetti"
    images, label = load_faces_data(path)
    data, label_ = PCA_images(images, label)
    # plot_image(images)
    label_result = spectral_clustering(data,20)
    plot_image2(images,label_result)
    print(label_result)
    C = lambda x, y: normalized_mutual_info_score(x, y)
    RI = lambda x, y: rand_score(x, y)
    print("RI指标为:%f" % RI(label_, label_result))
    print("NMI指标为:%f" % C(label_, label_result))
# if __name__ == "__main__":#同心圆
#     x, y = make_circles(n_samples=500, shuffle=True,
#                         noise=0.03, random_state=2, factor=0.6)
#     plt.scatter(x[:,0],x[:,1],c='gray')
#     plt.show()
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(x)
#     label = spectral_clustering(data,2)
#     RI = lambda x, y: rand_score(x, y)
#     print(RI(y,label))
#     plt.plot(x[label == 0, 0], x[label == 0, 1], 'ro')
#     plt.plot(x[label == 1, 0], x[label == 1, 1], 'bo')
#     plt.show()
#
#
# if __name__ == "__main__":#双月形
#     x, y = make_moons(n_samples=500, noise=0.05, random_state=3)
#     # plt.scatter(x[:,0],x[:,1],c='gray')
#     # plt.show()
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(x)
#     label = spectral_clustering(x, 2)
#     # plt.scatter(x[:, 0], x[:, 1], c=label)
#     # plt.show()
# #
# if __name__ == "__main__":#spiral
#     filename = "../Artificial/spiral.txt"
#     data = []
#     label = []
#     with open(filename, "r") as f:
#         while True:
#             line = f.readline()
#             m = 0
#             data_tmp = []
#             line = line.strip('\n')
#             if not line:
#                 break
#             for i in line.split(' '):
#                 if m == 0:
#                     label.append(float(i))
#                 else:
#                     data_tmp.append(float(i))
#                 m = m + 1
#             data.append(data_tmp)
#     data = np.array(data)
#     size, dim = data.shape
#     # plt.scatter(data[:,0],data[:,1],c='gray')
#     # plt.show()
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data1 = min_max_scaler.fit_transform(data)
#     label_result = spectral_clustering(data1, 2)
#     # plt.scatter(data[:, 0], data[:, 1], c=label_result)
#     # plt.show()

# if __name__ == "__main__":#ThreeCircles
#     filename = "../Artificial/ThreeCircles.txt"
#     data = []
#     label = []
#     with open(filename, "r") as f:
#         while True:
#             line = f.readline()
#             m = 0
#             data_tmp = []
#             line = line.strip('\n')
#             if not line:
#                 break
#             for i in line.split(' '):
#                 if m == 0:
#                     label.append(float(i))
#                 else:
#                     data_tmp.append(float(i))
#                 m = m + 1
#             data.append(data_tmp)
#     data = np.array(data)
#     # plt.scatter(data[:, 0], data[:, 1], c=label)
#     # plt.show()
#     size, dim = data.shape
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data1 = min_max_scaler.fit_transform(data)
#     label_result = spectral_clustering(data1, 3)
#     # plt.scatter(data[:, 0], data[:, 1], c=label_result)
#     # plt.show()
# if __name__ == "__main__":#four lines
#     x1, y1 = make_blobs(n_samples=500, n_features=2, centers=3, random_state=20)
#     transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
#     x1 = np.dot(x1, transformation)
#     # plt.scatter(x1[:, 0], x1[:, 1], c='gray')
#     # plt.show()
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(x1)
#     label = spectral_clustering(data, 3)
#     # plt.scatter(x1[:, 0], x1[:, 1], c=label)
#     # plt.show()
# if __name__ == "__main__":#ThreeCircles
#     filename = "../Artificial/VDD2.txt"
#     data = []
#     label = []
#     with open(filename, "r") as f:
#         while True:
#             line = f.readline()
#             m = 0
#             data_tmp = []
#             line = line.strip('\n')
#             if not line:
#                 break
#             for i in line.split(' '):
#                 if m == 2:
#                     label.append(float(i))
#                 else:
#                     data_tmp.append(float(i))
#                 m = m + 1
#             data.append(data_tmp)
#     data = np.array(data)
#     plt.scatter(data[:, 0], data[:, 1], c=label)
#     plt.show()
#     size, dim = data.shape
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data1 = min_max_scaler.fit_transform(data)
#     label_result = spectral_clustering(data1, 3)
#     # print(label_result)
#     # plt.scatter(data[:, 0], data[:, 1], c=label_result)
#     # plt.show()
# if __name__ == "__main__":
#     print("iris")
#     np.set_printoptions(threshold=np.inf)
#     data = datasets.load_iris()['data']
#     data = np.array(data)
#     label = datasets.load_iris()['target']
#     size,dim = data.shape
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#     C = lambda x, y: normalized_mutual_info_score(x, y)
#     RI = lambda x, y: rand_score(x, y)
#     label_result = spectral_clustering(data,3)
#     print("RI指标为:%f" % RI(label, label_result))
#     print("NMI指标为:%f" % C(label, label_result))
#     TP, TN, FP, FN = get_results(label,label_result)
#     print("TP,TN,FP,FN",(TP,TN,FP,FN))
# if __name__ == "__main__":
#     print("wine")
#     start = time.time()
#     np.set_printoptions(threshold=np.inf)
#     data = datasets.load_wine()['data']
#     data = np.array(data)
#     label = datasets.load_wine()['target']
#
#     size = data.shape[0]
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#     label = np.array(label)
#
#     label_result = spectral_clustering(data,3)
#     end = time.time()
#     print(end - start)
# C = lambda x, y: normalized_mutual_info_score(x, y)
#     RI = lambda x, y: rand_score(x, y)
#     print("RI指标为:%f" % RI(label, label_result))
#     print("NMI指标为:%f" % C(label, label_result))
#     TP, TN, FP, FN = get_results(label, label_result)
#     print("TP,TN,FP,FN", (TP, TN, FP, FN))
# if __name__ == "__main__":#14
#     print("seg")
#     np.set_printoptions(threshold=np.inf)
#     filename = '../data/segmentation.data'
#     data = []
#     label = []
#     name = ['BRICKFACE','SKY','FOLIAGE','CEMENT','WINDOW','PATH','GRASS']
#     with open(filename,'r') as f:
#         while True:
#             m = 0
#             data_temp = []
#             lines = f.readline()
#             if not lines:
#                 f.close()
#                 break
#             else:
#                 for i in lines.split(','):
#                     m = m + 1
#                     if m == 1:
#                         label.append(name.index(i))
#                     else:
#                         data_temp.append(float(i))
#             data.append(data_temp)
#     data = np.array(data)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#     label = np.array(label)
#     C = lambda x, y: normalized_mutual_info_score(x, y)
#     RI = lambda x, y: rand_score(x, y)
#     meanRI = 0
#     label_result = spectral_clustering(data,7)
#     print("RI指标为:%f" % RI(label, label_result))
#     print("NMI指标为:%f"%C(label, label_result))
#     TP, TN, FP, FN = get_results(label, label_result)
#     print("TP,TN,FP,FN", (TP, TN, FP, FN))
# if __name__ == "__main__":
#     print("glass")
#     filename = "../data/glass.data"
#     data = []
#     label = []
#     k = 6
#     with open(filename, "r") as f:
#         while True:
#             line = f.readline()
#             m = 0
#             data_tmp = []
#             line = line.strip('\n')
#             if not line:
#                 break
#             for i in line.split(','):
#                 if m == 0:
#                     pass
#                 elif m == 10:
#                     if i > '4':
#                         label.append(int(i) - 2)
#                     else:
#                         label.append(int(i) - 1)
#                 else:
#                     data_tmp.append(float(i))
#                 m = m + 1
#             data.append(data_tmp)
#     data = np.array(data)
#     size, dim = data.shape
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#     label = np.array(label)
#     C = lambda x, y: normalized_mutual_info_score(x, y)
#     RI = lambda x, y: rand_score(x, y)
#     meanRI = 0
#     label_result = spectral_clustering(data,6)
#     print("RI指标为:%f" % RI(label, label_result))
#     print("NMI指标为:%f" % C(label, label_result))
#     TP, TN, FP, FN = get_results(label, label_result)
#     print("TP,TN,FP,FN", (TP, TN, FP, FN))
# if __name__ == "__main__":
#     print("wdbc")
#     filename = "../data/wdbc.data"
#     data = []
#     data_label = []
#     with open(filename,"r") as f:
#         while True:
#             data_tmp = []
#             k = 0
#             lines = f.readline()
#             if not lines:
#                 break
#             lines = lines.strip('\n')
#             for i in lines.split(","):
#                 k = k + 1
#                 if k == 1:
#                     pass
#                 elif k == 2:
#                     if(i=="M"):
#                         data_label.append(0)
#                     else:
#                         data_label.append(1)
#                 else:
#                     data_tmp.append(float(i))
#             data.append(data_tmp)
#     data = np.array(data)
#     data_label = np.array(data_label)
#     k = 2
#     C = lambda x, y: normalized_mutual_info_score(x, y)
#     RI = lambda x, y: rand_score(x, y)
#     size, dim = data.shape
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#     label_result = spectral_clustering(data, k)
#     print("RI指标为%f,NMI指标为%f"%(RI(label_result,data_label),C(label_result,data_label)))
#     TP, TN, FP, FN = get_results(label_result, data_label)
#     print("TP,TN,FP,FN", (TP, TN, FP, FN))
# if __name__ == "__main__":
#     print("seeds")
#     filename = "../data/seeds_dataset.txt"
#     data = []
#     data_label = []
#     one_index = 0
#     two_index = 0
#     k = 3
#     with open(filename, "r") as f:
#         while True:
#             line = f.readline()
#             m = 0
#             data_tmp = []
#             if not line:
#                 break
#             for i in line.split('	'):
#                 if (m == 7):
#                     data_label.append(int(i)-1)
#                 else:
#                     data_tmp.append(float(i))
#                 m = m + 1
#             data.append(data_tmp)
#     data = np.array(data)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#     data_label = np.array(data_label)
#     C = lambda x, y: normalized_mutual_info_score(x, y)
#     RI = lambda x, y: rand_score(x, y)
#     meanRI = 0
#     label_result = spectral_clustering(data, k)
#     print("RI指标为:%f" % RI(data_label, label_result))
#     print("NMI指标为:%f" % C(data_label, label_result))
#     TP, TN, FP, FN = get_results(data_label, label_result)
#     print("TP,TN,FP,FN", (TP, TN, FP, FN))
# #
# if __name__ == "__main__":#聚类成功率为: 0.9357541899441341,NMI指标为: 0.8572918716587341
#     np.set_printoptions(threshold=1e6)
#     filename = "../data/dermatology.data"
#     data = []
#     label = []
#     k = 6
#     miss_index = np.zeros(366)
#     with open(filename,"r") as f:
#         h = 0
#         while True:
#             lines = f.readline()
#             data_temp = []
#             if not lines:
#                 break
#             else:
#                 lines = lines.strip('\n')
#                 m = 0
#                 if '?' in lines:
#                     data_temp.append(-1)
#                     miss_index[h] = 1
#                 else:
#                     for i in lines.split(','):
#                         if m == 34:
#                             label.append(float(i))
#                         else:
#                             data_temp.append(float(i))
#                         m += 1
#                     data.append(data_temp)
#             h += 1
#     data = np.array(data)
#     label = np.array(label)
#     mean = 0
#     size,dim = data.shape
#     for i in range(size):
#         if miss_index[i] == 1:
#             pass
#         else:
#             mean += data[i,33]
#     for i in range(size):
#         if miss_index[i] == 1:
#             data[i,33] = mean/358
#     C = lambda x,y:normalized_mutual_info_score(x,y)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#     label_result = spectral_clustering(data,k)
#     RI = lambda x, y: rand_score(x, y)
#     print("dermatology")
#     print("NMI指标为:",C(label_result,label))
#     print("RI指标为:%f" % RI(label, label_result))
#     TP, TN, FP, FN = get_results(label, label_result)
#     print("TP,TN,FP,FN", (TP, TN, FP, FN))
