# encoding: utf-8
import time
from tools import *

import numpy as np

if __name__ == "__main__":
    time_start = time.time()


    hp = parse_args("SupervoxelGraph --- Supervoxel to voxel")
    f = open(hp.configure_file)
    json_content = json.load(f)



    workspace = json_content["workspace"]
    supervoxel_id_file = json_content["volume2supervoxel"]["supervoxel_id_file"]
    volume_file = json_content["volumes"]["file_names"][0]
    dtype = json_content["volumes"]["dtype"]


    volume_raw_data = readVolumeRaw(json_content['volumes']['file_path'] + volume_file, dtype)
    # read label int file
    label_int_data = readVolumeRaw(workspace + supervoxel_id_file, 'int')


    # transfer predict volumes to voxel form.
    is_predicted_supervoxel_transferred = True
    # is_groundtruth_supervoxel_transferred = True
    predict_voxel_based_segmentation_array = None

    if is_predicted_supervoxel_transferred:
        # predict labels
        predict_label_supervoxel_file = workspace + json_content['model']['predict_label_supervoxel_file']
        predict_label_supervoxel_array = np.load(predict_label_supervoxel_file)

        n_clusters_ = len(set(predict_label_supervoxel_array)) - (
            1 if -1 in predict_label_supervoxel_array else 0)  # 获取分簇的数目
        print("Number of labeled type : {}".format(n_clusters_))
        print("Max labeled type id : {}".format(max(set(predict_label_supervoxel_array))))
        predict_voxel_based_segmentation_array = volumeSegmentation(predict_label_supervoxel_array, label_int_data)
        predict_voxel_based_segmentation_array = predict_voxel_based_segmentation_array.astype(np.int32)
        predict_voxel_based_segmentation_array.tofile(workspace + json_content['model']['predict_label_voxel_file'])
        print("predicted supervoxel array has been saved in voxel form ")

        labelVoxel2Nii(hp, predict_voxel_based_segmentation_array+1, hp.predict_label_nii_file)


        # analysis for groundtruth. generation of groundtruth labeled voxel fraw file.
    if hp.labeled_type == 2:
        # groundtruth labels
        groundtruth_label_supervoxel_file = workspace + json_content['model']['groundtruth_label_supervoxel_file']
        groundtruth_label_supervoxel_array = np.load(groundtruth_label_supervoxel_file).astype(np.int32)

        n_clusters_ = len(set(groundtruth_label_supervoxel_array)) - (
            1 if -1 in groundtruth_label_supervoxel_array else 0)  # 获取分簇的数目
        print("Number of labeled type : {}".format(n_clusters_))
        print("Max labeled type id : {}".format(max(set(groundtruth_label_supervoxel_array))))
        groundtruth_voxel_based_segmentation_array = volumeSegmentation(groundtruth_label_supervoxel_array, label_int_data)
        groundtruth_voxel_based_segmentation_array = groundtruth_voxel_based_segmentation_array.astype(np.int32)
        groundtruth_voxel_based_segmentation_array.tofile(workspace + json_content['model']['groundtruth_label_voxel_file'])
        print("groundtruth supervoxel array has been saved in voxel form ")

        from sklearn.metrics import f1_score, precision_score, recall_score

        def metricMeasure(y_true, y_pred):


            f1 = f1_score(y_true, y_pred, average='macro')
            p = precision_score(y_true, y_pred, average='macro')
            r = recall_score(y_true, y_pred, average='macro')
            return p, r, f1

        precision_sorce, recall_score, f1_score = metricMeasure(predict_voxel_based_segmentation_array,
                                                                groundtruth_voxel_based_segmentation_array)
        print("Voxel precision score : {}".format(precision_sorce))
        print("Voxel recall score : {}".format(recall_score))
        print("Voxel F1 score : {}".format(f1_score))
        from sklearn import metrics
        acc = metrics.accuracy_score(predict_voxel_based_segmentation_array,
                                                                         groundtruth_voxel_based_segmentation_array)
        print("voxel accuracy socre : {}".format( acc))

        logger = logger_config(hp.workspace+"metric_log.txt")
        logger.info("supervoxelgraph: precision: {}, recall: {}, f1: {}, accuracy : {}".format(precision_sorce, recall_score, f1_score, acc))


    # do feature classification for different classes.
    else:
        for i in range(n_clusters_):
            buf_volume_array = np.zeros(dtype=volume_raw_data.dtype, shape=volume_raw_data.shape)
            buf_index_array = np.array(np.where(predict_voxel_based_segmentation_array == i))
            label_number = len(buf_index_array[0])

            for j in buf_index_array[0]:
                buf_volume_array[j] = volume_raw_data[j]

            # for j in buf_index_array[0]:
            #     buf = np.where(label_int_data == j)
            #     # print(buf)
            #     buf_volume_array[buf] = volume_raw_data[buf]
            #
            #     if j % 100 == 0:
            #         print("Process of indexing the index of predicted labels' voxels : {:.2f}%".format(
            #             j * 100 / (len(buf_index_array[0]))))

            if label_number > 0:
                buf_volume_array.tofile(
                    workspace + 'feature_part_' + str(i) +'voxel_number_' + str(label_number) + '.raw')
                print("Cluster {} in {} has {} labels, and it has been saved.".format(i, n_clusters_, label_number))


    time_end = time.time()
    all_time = int(time_end - time_start)

    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute, 's')



