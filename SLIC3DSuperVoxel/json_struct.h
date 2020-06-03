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
	VM_JSON_FIELD(std::string, node_embedding_file);
	VM_JSON_FIELD(std::string, labeled_graph_file);
	VM_JSON_FIELD(std::string, labeled_voxel_file);
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
	VM_JSON_FIELD(int, histogram_size);			// the histogram size of each supervoxel
	VM_JSON_FIELD(int, histogram_stored_type);	// Whether store the historam, default to set to 1 to store origin histogram, 2 for log, 3 for one-hot
	VM_JSON_FIELD(int, is_gradient_stored);		// Whether store the non-normalized gradient
	VM_JSON_FIELD(int, is_entropy_stored);		// Whether store the entropy
	VM_JSON_FIELD(int, is_barycenter_stored);	// Whether store the baycenter
};

struct Histogram2GraphJSONStruct : public vm::json::Serializable<Histogram2GraphJSONStruct>
{
	VM_JSON_FIELD(int, histogram_size);
};

struct GCNJSONStruct : public vm::json::Serializable<GCNJSONStruct>
{
	VM_JSON_FIELD(int, vector_dimension);
	VM_JSON_FIELD(int, epochs);
	VM_JSON_FIELD(double, learning_rate);
	VM_JSON_FIELD(int, warmup_steps);
	VM_JSON_FIELD(int, label_type_number);
	VM_JSON_FIELD(double, ratio);
};

struct ConfigureJSONStruct : public vm::json::Serializable<ConfigureJSONStruct>
{
	VM_JSON_FIELD(DataPathJSONStruct, data_path);
	VM_JSON_FIELD(FileNameJSONStruct, file_name);
	VM_JSON_FIELD(Volume2SuperVoxelJSONStruct, volume2supervoxel);
	VM_JSON_FIELD(SuperVoxel2HistogramJSONStruct, supervoxel2histogram);
	VM_JSON_FIELD(Histogram2GraphJSONStruct, histogram2graph);
	VM_JSON_FIELD(GCNJSONStruct, gcn);
};
