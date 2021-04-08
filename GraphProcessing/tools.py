import argparse
import json
def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)

    # Debug 20200522 hxy : add configure_file

    parser.add_argument('--configure_file',
                        default=r'D:\project\science_project\SLIC3DSuperVoxel\x64\Release\workspace\lung_groundtruth_supervoxel.json',
                        help='configure json file')
    parser.add_argument('--csv_file', nargs='*', help='csv file export by gephi')

    args = parser.parse_args()
    configure_json_file = args.configure_file
    f = open(configure_json_file)
    json_content = json.load(f)


    args.workspace = json_content["workspace"]

    # volumes
    args.file_path = json_content["volumes"]["file_path"]
    args.dtype = json_content["volumes"]["dtype"]
    # args.space = json_content["volumes"]["space"]
    args.dimension = json_content["volumes"]["dimension"]
    args.data_byte_order = json_content["volumes"]["data_byte_order"]
    args.file_names = json_content["volumes"]["file_names"]
    args.downsample = json_content["volumes"]["downsample"]

    #volume2supervoxel
    args.supervoxel_id_file = json_content["volume2supervoxel"]["supervoxel_id_file"]

    # supervoxelnodefeature
    args.histogram_size = json_content["supervoxelnodefeature"]["histogram_size"]
    args.is_histogram_stored = json_content["supervoxelnodefeature"]["is_histogram_stored"]
    args.is_gradient_stored = json_content["supervoxelnodefeature"]["is_gradient_stored"]
    args.is_entropy_stored = json_content["supervoxelnodefeature"]["is_entropy_stored"]
    args.is_barycenter_stored = json_content["supervoxelnodefeature"]["is_barycenter_stored"]
    args.label_histogram_file = json_content["supervoxelnodefeature"]["label_histogram_file"]
    args.edge_weight_file = json_content["supervoxelnodefeature"]["edge_weight_file"]

    # supervoxelgraph
    args.histogram_size = json_content["supervoxelgraph"]["histogram_size"]
    args.graph_file = json_content["supervoxelgraph"]["graph_file"]
    args.graph_node_feature_file = json_content["supervoxelgraph"]["graph_node_feature_file"]
    args.labeled_graph_file = json_content["supervoxelgraph"]["labeled_graph_file"]
    args.labeled_type = json_content["supervoxelgraph"]["label_type"]
    args.labeled_file = json_content["supervoxelgraph"]["labeled_file"]

    # model
    args.vector_dimension = json_content["model"]["vector_dimension"]
    args.dimension = json_content["model"]["dimension"]
    args.epochs = json_content["model"]["epochs"]
    args.warmup_steps = json_content["model"]["warmup_steps"]
    args.label_type_number = json_content["model"]["label_type_number"]
    args.ratio = json_content["model"]["ratio"]
    args.node_embedding_file = json_content["model"]["node_embedding_file"]

    args.type = args.labeled_type
    return args
