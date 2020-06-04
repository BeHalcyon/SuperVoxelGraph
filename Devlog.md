Question:
1. Maybe the information entropy is not necessaray for training? The generated supervoxels have similar information for the reason that the voxels are highly consistent in the supervoxel.
2. It's a huge difference between the origin gradient values,  so the gradient can be normalized ?


Devlog 20200604
1. Perform tensorflow GPU test.
2. log normalization for histogram seem to be a good choice.
3. TODO: Add ground truth for spheres volume.
4. TODO: Test the histogram size of less then 256.

Devlog 20200603
1. Support multiple histogram calculation methods: origin normalization, log normalization, one hot.

Devlog 20200602
1. Provide LabelSuperVoxelQt project for user to label the train set.
2. Abort the using of gcn.label_type_number in json file.
3. Extend the SLIC to generate more super-voxels.

