
# SLIC3DSuperVoxel (SLIC algorithm for volume data in 3D space)

Three works are achieved:

**STEP 1. Calculating super-voxels (SLIC3DSuperVoxel)**

This module needs vifo file for calculating super-voxels. The output file is the raw file with the same format with volume data.
Input informaiton:
```c
*.vifo: the input volume data.
```

Output information:
```c
*_label.raw: the output file with int value from 0 to N-1, where N is the number of super-voxels
*_boundary.raw: the output file with int value from origin volume value and 511, where 511 is boundary for super-voxels.
```

**STEP 2. Calculating graph for super-voxels (SuperVoxel2Graph)**

The SuperVoxel2Graph transfers calucated super-voxels be STEP 1 to graph where nodes represent the super-voxels and edges are measured by neigboring nodes. The edge weight is set to 1 by default. Note that each node carries addtional information of the normalized scalar histogram with size 256.

Input informaiton:
```c
*.vifo: the input volume data and dimension information
```

Middle information:
```c
edge_weight_array.txt: 2D array with the side length of super-voxel number where arr[i][j] = 1 represents super-voxel i and j are neighboring.
label_histogram_array.txt: the normalized scalar histogram for each super-voxel.
```

Output information:
```c
*_supervoxels.gexf: graph file for the volume data.
```

**STEP 3. Merging similar super-voxels using graph-based volume segmentation** (Not completed)

Compute the edge weight using the **chi-squared distance** between 1D intensity histograms of the two super-voxels. Each histogram uses a total of 64 bins across the entire scalar range of the input volume.

The segementation algorithm is under testing and is not convincing. There is an error in the measurement of the weight for edges and internal variance of regions. Current internal variance is measured by the maximum gradient difference of the edges in region, while the origin method is using the maximum edge weight of MST of the region.


Using:
```c
SLIC3DSuperVoxel.exe --TAG=parameter_value
```

Example:
```c
SLIC3DSuperVoxel.exe --vifo_path="C:/mixfrac.vifo" --merge=0
```
![cmdline](https://github.com/XiangyangHe/SuperVoxelGraph/tree/master/image/cmdline.png)


## vifo file format:
-----------------------------------------------------
```cpp
1                   //volume number
uchar               //volume data type in "uchar" "float" "ushort"
500 500 100         //volume dimensions in x y z direction
1 1 1               //volume spaces in x y z direction
SPEEDf21.raw        //volume relative path
```

Results for STEP 1:
-----------------------------------------------------

![xy plane](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/design%20sketch_xyplane.png)
![yz plane](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/design%20sketch_yzplane.png)
![3D results](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/design%20sketch_volumerendering.png)
![3D results](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/design%20sketch_asteroid_tev.png)
![3D results](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/design%20sketch_MANIX.png)
![3D results](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/design%20sketch_tooth.png)


Results for STEP 2 (jet_mixfrix_0051):
-----------------------------------------------------
![super-voxel graph](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/super-voxel-graph.png)
![super-voxel graph](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/super-voxel-graph2.png)


Results for STEP 3:
-----------------------------------------------------
(A modification for distance measurement to achieve irregular super-voxels)
![image](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/merged_combustion.png)
![image](https://github.com/XiangyangHe/SuperVoxelGraph/blob/master/image/merged_H.png)

Reference:
-----------------------------------------------------
- [1] SLIC Superpixels Compared to State-of-the-Art Superpixel Methods
- [2] FeatureLego: Volume Exploration Using Exhausting Clustering of Super-Voxels
- [3] Efficient Graph-Based Image Segmentation
