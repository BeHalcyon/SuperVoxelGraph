#pragma once
#include "VMUtils/include/VMUtils/json_binding.hpp"
struct DataPathJSONStruct : public vm::json::Serializable<DataPathJSONStruct>
{
	VM_JSON_FIELD(std::string, vifo_file);
	VM_JSON_FIELD(std::string, file_prefix);
	VM_JSON_FIELD(int, volume_index);
};

struct FileNameJSONStruct : public vm::json::Serializable<FileNameJSONStruct>
{
	VM_JSON_FIELD(std::string, merged_label_file);
	VM_JSON_FIELD(std::string, merged_boundary_file);
	VM_JSON_FIELD(std::string, label_file);
	VM_JSON_FIELD(std::string, boundary_file);

	VM_JSON_FIELD(std::string, gradient_file);
	
	VM_JSON_FIELD(std::string, label_histogram_file);
	VM_JSON_FIELD(std::string, edge_weight_file);
	
	VM_JSON_FIELD(std::string, graph_file);
};

struct Volume2SuperVoxelJSONStruct : public vm::json::Serializable<Volume2SuperVoxelJSONStruct>
{
	VM_JSON_FIELD(int, cluster_number);
	VM_JSON_FIELD(int, output_super_voxel_label);
	VM_JSON_FIELD(int, output_super_voxel_boundary);
	VM_JSON_FIELD(int, output_gradient);
	
	VM_JSON_FIELD(int, k_threshold);
	
	VM_JSON_FIELD(int, is_merge);
	VM_JSON_FIELD(int, output_merged_label);
	VM_JSON_FIELD(int, output_merged_boundary);
	VM_JSON_FIELD(double, compactness_factor);
	
};

struct SuperVoxel2HistogramJSONStruct : public vm::json::Serializable<SuperVoxel2HistogramJSONStruct>
{
	VM_JSON_FIELD(int, histogram_size);
};

struct Histogram2GraphJSONStruct : public vm::json::Serializable<Histogram2GraphJSONStruct>
{
	VM_JSON_FIELD(int, histogram_size);
};

struct ConfigureJSONStruct : public vm::json::Serializable<ConfigureJSONStruct>
{
	VM_JSON_FIELD(DataPathJSONStruct, data_path);
	VM_JSON_FIELD(FileNameJSONStruct, file_name);
	VM_JSON_FIELD(Volume2SuperVoxelJSONStruct, volume2supervoxel);
	VM_JSON_FIELD(SuperVoxel2HistogramJSONStruct, supervoxel2histogram);
	VM_JSON_FIELD(Histogram2GraphJSONStruct, histogram2graph);
};
