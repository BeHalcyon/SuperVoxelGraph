import csv
import numpy as np
def loadLabelId(csv_file_name):
    labeled_data = []
    with open(csv_file_name) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        labeled_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            labeled_data.append(row[0])

    labeled_data = [int(x) for x in labeled_data]  # 将数据从string形式转换为float形式
    labeled_data = np.array(labeled_data)  # 将list数组转化成array数组便于查看数据结构
    return labeled_data

# labeled_data = loadLabelId("tooth_labeled.csv")
labeled_data = loadLabelId("manix_labeled.csv")


dimension = [256, 256, 230] # x y z
file_path = 'E:/CThead/manix/MANIX.raw'
histogram_size = 256
volume_array = np.fromfile(file_path, dtype=np.uint8)
label_array = np.fromfile('I:\\supergraph buffer\\manix_label_int.raw', dtype=np.int)

buf_volume_array = np.zeros(dtype=np.uint8, shape=volume_array.shape)

for j in labeled_data:
    buf = np.where(label_array == j)
    # print(buf)
    buf_volume_array[buf] = volume_array[buf]
# buf_index = np.where(label_array == np.where(labels == i))
# print(np.where(labels == i))
# buf_volume_array[buf_index] = volume_array[buf_index]
buf_volume_array.tofile('spheres_test.raw')

print("volume has been transferred.")