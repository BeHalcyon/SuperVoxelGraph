import networkx as nw
import numpy as np

dimension = [480, 720, 120] # x y z
file_path = 'I:\\science data\\4 Combustion\\jet_0051\\'
histogram_size = 256

label_array = np.fromfile(file_path + 'jet_mixfrac_0051_label.raw', dtype=np.int)
volume_array = np.fromfile(file_path + 'jet_mixfrac_0051.raw', dtype=np.float32)
volume_min_value = np.min(volume_array)
volume_max_value = np.max(volume_array)
print(volume_max_value, volume_min_value)


label_number = np.max(label_array) + 1
label_histogram_array = np.zeros((label_number, histogram_size), dtype=np.float)
print(volume_array.shape)
print(volume_array)

edge_weight_array = np.zeros((label_number, label_number), dtype=np.int)
print("SuperVoxel block number : ", label_number)
print(label_array.shape)

#针对每个体素向周围扩展，遇到相邻的数值说明有相邻边
x_offset = [1, -1, 0, 0, 0, 0]
y_offset = [0, 0, 1, -1, 0, 0]
z_offset = [0, 0, 0, 0, 1, -1]
for i in range(0, label_array.shape[0]):
    #针对每个体素，上下左右进行探索label是否一致，不一致则赋值一致
    x = i % dimension[0]
    z = int(i / (dimension[0]*dimension[1]))
    y = int((i % (dimension[0]*dimension[1])) / dimension[0])

    for j in range(0, 6):
        new_x = x + x_offset[j]
        new_y = y + y_offset[j]
        new_z = z + z_offset[j]

        if new_x < 0 or new_x >= dimension[0] or new_y < 0 or new_y >= dimension[1] or new_z < 0 or new_z >= dimension[2]:
            continue
        #两者label不一致，则设定为联通
        neighbor_index = new_z*dimension[0]*dimension[1]+new_y*dimension[0]+new_x

        if label_array[i] != label_array[neighbor_index]:
            # if neighbor_index > 8235:
            #     print("test:", label_array[neighbor_index], label_array[i], new_y, new_z)
            edge_weight_array[label_array[i]][label_array[neighbor_index]] = 1
            edge_weight_array[label_array[neighbor_index]][label_array[i]] = 1

    #存储每个label对应的数值。该数值从0~255变化的统计直方图
    regular_value = int((volume_array[i]-volume_min_value)*255/(volume_max_value-volume_min_value))
    label_histogram_array[label_array[i]][regular_value] += 1

    if i%100000 == 0:
        # print(label_array[i], label_histogram_array[label_array[i]])
        print("Process ", i*100/label_array.shape[0], "%.")

# label_histogram_array = label_histogram_array/label_histogram_array.max(axis=1)

for i in range(label_number):
    max_value = np.max(label_histogram_array[i])
    print(max_value)
    if max_value == 0:
        continue
    for j in range(histogram_size):
        label_histogram_array[i][j] /= max_value

print(label_histogram_array)

#保存当前label_histogram_array

np.savetxt("label_histogram_array.txt", label_histogram_array) #缺省按照'%.18e'格式保存数据，以空格分隔
np.savetxt("edge_weight_array.txt", edge_weight_array, fmt='%d')
#封装为networkx