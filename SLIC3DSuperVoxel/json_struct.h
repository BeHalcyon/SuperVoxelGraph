#pragma once
#include "VMUtils/include/VMUtils/json_binding.hpp"
#include <vector>

struct VolumesJSONStruct : public vm::json::Serializable<VolumesJSONStruct>
{
	VM_JSON_FIELD(std::string, dtype);
	VM_JSON_FIELD(std::vector<int>, space);
	VM_JSON_FIELD(std::vector<int>, dimension);
	VM_JSON_FIELD(std::string, file_path);
	VM_JSON_FIELD(std::string, data_byte_order);
	VM_JSON_FIELD(std::vector<std::string>, file_names);
	VM_JSON_FIELD(int, downsample);
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


	VM_JSON_FIELD(std::string, merged_label_file);
	VM_JSON_FIELD(std::string, merged_boundary_file);
	// origin label_file
	VM_JSON_FIELD(std::string, supervoxel_id_file);
	VM_JSON_FIELD(std::string, boundary_file);
	VM_JSON_FIELD(std::string, gradient_file);
};


struct SuperVoxelNodeFeatureJSONStruct : public vm::json::Serializable<SuperVoxelNodeFeatureJSONStruct>
{
	VM_JSON_FIELD(int, histogram_size);			// the histogram size of each supervoxel
	VM_JSON_FIELD(int, histogram_stored_type);	// Whether store the histogram, default to set to 1 to store origin histogram, 2 for log, 3 for one-hot
	VM_JSON_FIELD(int, is_gradient_stored);		// Whether store the non-normalized gradient
	VM_JSON_FIELD(int, is_entropy_stored);		// Whether store the entropy
	VM_JSON_FIELD(int, is_barycenter_stored);	// Whether store the baycenter

	VM_JSON_FIELD(std::string, label_histogram_file);
	VM_JSON_FIELD(std::string, edge_weight_file);

};


struct SuperVoxelGraphJSONStruct : public vm::json::Serializable<SuperVoxelGraphJSONStruct>
{
	VM_JSON_FIELD(int, histogram_size);
	VM_JSON_FIELD(std::string, graph_file);
	VM_JSON_FIELD(std::string, labeled_graph_file);
};

struct ModelJSONStruct : public vm::json::Serializable<ModelJSONStruct>
{
	VM_JSON_FIELD(int, vector_dimension);
	//VM_JSON_FIELD(int, dimension);
	VM_JSON_FIELD(int, epochs);
	VM_JSON_FIELD(double, learning_rate);
	VM_JSON_FIELD(int, warmup_steps);
	VM_JSON_FIELD(int, label_type_number);
	VM_JSON_FIELD(double, ratio);
	VM_JSON_FIELD(std::string, node_embedding_file);
};



struct ConfigureJSONStruct : public vm::json::Serializable<ConfigureJSONStruct>
{
	VM_JSON_FIELD(std::string, workspace);
	VM_JSON_FIELD(VolumesJSONStruct, volumes);
	VM_JSON_FIELD(Volume2SuperVoxelJSONStruct, volume2supervoxel);
	VM_JSON_FIELD(SuperVoxelNodeFeatureJSONStruct, supervoxelnodefeature);
	VM_JSON_FIELD(SuperVoxelGraphJSONStruct, supervoxelgraph);
	VM_JSON_FIELD(ModelJSONStruct, model);
};
