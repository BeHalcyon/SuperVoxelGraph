Question:
1. Maybe the information entropy is not necessaray for training? The generated supervoxels have similar information for the reason that the voxels are highly consistent in the supervoxel.
2. It's a huge difference between the origin gradient values,  so the gradient can be normalized ?

Devlog 20200629
1. Use ITK-SNAP to label data.
2. Find the law: the more nodes, the greater results can be achieved; the histogram size=64 may be a good choice.
3. Implement the binary GCN model, and the results is not good than normal GCN.
4. Several algorithm are implemented: svm, simple nural network, knn, random forest.
5. The lower the histogram size and node number, the higher efficiency the model is. But it may cause a low model quality.
6. Add metrics of precision, recall and f1 score (metricMeasure method in module.py) to test set.(It should add cross validation set rather than test set?)
7. Add ground truth for spheres data set.
8. Add niilabel2csv.py to transfer nii.gz file to labeled csv file.


Devlog 20200608
1. Test the histogram size of less then 256.
2. The log histogram is used in further exploration.
3. Get the conclusion: the smaller the histogram size, the greater the training quantity the model is.
4. Get the conclusion: the inforamtion entropy and gradient are not necessary for training GCN, and they can be aborted.
5. Set 1/10 of the ground truth labels as train set and get a better result than training the train set labeled by user.

Devlog 20200605
1. Add ground truth for spheres volume.
2. TODO: Test the histogram size of less then 256.

Devlog 20200604
1. Perform tensorflow GPU test.
2. log normalization for histogram seems to be a good choice.

Devlog 20200603
1. Support multiple histogram calculation methods: origin normalization, log normalization, one hot.

Devlog 20200602
1. Provide LabelSuperVoxelQt project for user to label the train set.
2. Abort the using of gcn.label_type_number in json file.
3. Extend the SLIC to generate more super-voxels.

