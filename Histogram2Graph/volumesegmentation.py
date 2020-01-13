# divide data into several parts based on class

import numpy as np

labels = np.load('cluster_label.npy')
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

for i in range(len(labels)):
    print(i, labels[i])

dimension = [480, 720, 120] # x y z
file_path = 'I:\\science data\\4 Combustion\\jet_0051\\'
histogram_size = 256
#volume_array = np.fromfile(file_path + 'jet_mixfrac_0051.raw', dtype=np.float32)
volume_array = np.fromfile(file_path + 'jet_mixfrac_0051.raw', dtype=np.float32)
label_array = np.fromfile(file_path + 'jet_mixfrac_0051_label.raw', dtype=np.int)
for i in range(-1, n_clusters_):

    buf_volume_array = np.zeros(dtype=np.float32, shape=volume_array.shape)
    buf_index_array = np.array(np.where(labels == i))
    # print(buf_index_array)
    for j in buf_index_array[0]:
        buf = np.where(label_array == j)
        # print(buf)
        buf_volume_array[buf] = volume_array[buf]
    # buf_index = np.where(label_array == np.where(labels == i))
    # print(np.where(labels == i))
    # buf_volume_array[buf_index] = volume_array[buf_index]
    buf_volume_array.tofile('jet_mixfrac_0051_super_voxles'+str(len(buf_index_array[0]))+'_part_'+str(i)+'.raw')
    print("Cluster ", i , " in ", n_clusters_, " has been saved.")