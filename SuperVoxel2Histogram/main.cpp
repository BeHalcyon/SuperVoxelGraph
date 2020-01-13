#include "../SLIC3DSuperVoxel/json_struct.h"
#include "../SLIC3DSuperVoxel/VMUtils/include/VMUtils/cmdline.hpp"
#include <string>
#include <fstream>
#include <vector>
#include "../SLIC3DSuperVoxel/resource.h"
#include "../SLIC3DSuperVoxel/SourceVolume.h"
#include "../SLIC3DSuperVoxel/SourceVolume.cpp"


void readInfoFile(const std::string& infoFileName, int& data_number, std::string& datatype, hxy::my_int3& dimension, hxy::my_double3& space,
                  std::vector<std::string>& file_list)
{
	file_list.clear();

	std::ifstream inforFile(infoFileName);

	inforFile >> data_number;
	inforFile >> datatype;
	inforFile >> dimension.x >> dimension.y >> dimension.z;
	//Debug 20190520 ����sapce�ӿ�
	inforFile >> space.x >> space.y >> space.z;
	const std::string filePath = infoFileName.substr(0, infoFileName.find_last_of('/') == -1 ?
		infoFileName.find_last_of('\\') + 1 : infoFileName.find_last_of('/') + 1);
	std::cout << (filePath.c_str()) << std::endl;
	for (auto i = 0; i < data_number; i++)
	{
		std::string rawFileName;
		inforFile >> rawFileName;
		std::string volumePath = filePath + rawFileName;
		file_list.push_back(volumePath);
	}
	std::cout << "Info file name : \t\t" << infoFileName.c_str() << std::endl;
	std::cout << "Volume number : \t\t" << data_number << std::endl;
	std::cout << "data type : \t\t\t" << datatype.c_str() << std::endl;
	std::cout << "Volume dimension : \t\t" << "[" << dimension.x << "," << dimension.y << "," << dimension.z << "]" << std::endl;
	std::cout << "Space dimension : \t\t" << "[" << space.x << "," << space.y << "," << space.z << "]" << std::endl;
	for (auto i = 0; i < data_number; i++)
	{
		std::cout << "Volume " << i << " name : \t\t" << file_list[i].c_str() << std::endl;
	}


	std::cout << "Info file has been loaded successfully." << std::endl;
}


void saveVector(const vector<vector<float>>& arr, const string& file_name)
{
	ofstream writer(file_name);
	for(auto& i: arr)
	{
		for(auto j=0;j<i.size()-1;j++)
		{
			writer << i[j] << " ";
		}
		writer << i[i.size() - 1] << std::endl;
	}
	vm::println("Label histogram has been saved.");
}
void saveVector(const vector<vector<int>>& arr, const string& file_name)
{
	ofstream writer(file_name);
	for (auto& i : arr)
	{
		for (auto j = 0; j < i.size() - 1; j++)
		{
			writer << i[j] << " ";
		}
		writer << i[i.size() - 1] << std::endl;
	}
	vm::println("edge histogram has been saved.");
}


int main(int argc, char* argv[])
{
	clock_t time_begin = clock();
	
	if (argc <= 1)
	{
		vm::println("Please use command line parameter.");
		return 0;
	}

	cmdline::parser a;
	a.add<std::string>("configure_file", 'c', "json configure file", false, 
		"D:\\project\\science_project\\SLIC3DSuperVoxel\\x64\\Release\\workspace\\jet_mixfrac_supervoxel.json");
	a.parse_check(argc, argv);
	ConfigureJSONStruct configure_json;
	const auto configure_json_file = a.get<std::string>("configure_file");
	try
	{
		std::ifstream input_file(configure_json_file);
		input_file >> configure_json;

		const auto file_prefix = configure_json.data_path.file_prefix;

		std::string					infoFileName = configure_json.data_path.vifo_file;
		int							data_number;
		std::string					datatype;
		hxy::my_int3				dimension;
		hxy::my_double3				space;
		std::vector<std::string>	file_list;

		auto histogram_size = configure_json.supervoxel2histogram.histogram_size;

		readInfoFile(infoFileName, data_number, datatype, dimension, space, file_list);
		SourceVolume source_volume(file_list, dimension.x, dimension.y, dimension.z, datatype, 256, histogram_size);
		source_volume.loadRegularVolume(); //[0, 255] data
		auto volume_index = configure_json.data_path.volume_index;
		// auto volume_data = source_volume.getRegularVolume(volume_index);

		auto volume_data = source_volume.getDownsamplingVolume(volume_index);

		auto labeled_volume_file = file_prefix + configure_json.file_name.label_file;
		SourceVolume labeled_volume({ labeled_volume_file }, dimension.x, dimension.y, dimension.z, "int", 256, 256);
		auto labeled_data = labeled_volume.getOriginVolume(0);

		auto max_index = max_element((*labeled_data).begin(), (*labeled_data).end());
		auto label_number = *max_index + 1;



		std::vector<std::vector<float>> label_histogram_array(label_number);
		for (auto& i : label_histogram_array) { i.resize(histogram_size, 0.0); }

		vm::println("SuperVoxel block number : {}", label_number);

		std::vector<std::vector<int>> edge_weight_array(label_number);
		for (auto& i : edge_weight_array) { i.resize(label_number, 0); }


		std::vector<int> x_offset = { 1, -1, 0, 0, 0, 0 };
		std::vector<int> y_offset = { 0, 0, 1, -1, 0, 0 };
		std::vector<int> z_offset = { 0, 0, 0, 0, 1, -1 };

		auto volume_size = (*labeled_data).size();

		for (auto i = 0; i < volume_size; i++)
		{
			auto x = i % dimension.x;
			auto z = i / (dimension.x * dimension.y);
			auto y = (i % (dimension.x * dimension.y)) / dimension.x;

			for (auto j = 0; j < 6; j++)
			{
				auto new_x = x + x_offset[j];
				auto new_y = y + y_offset[j];
				auto new_z = z + z_offset[j];

				if (new_x < 0 || new_x >= dimension.x || new_y < 0 || new_y >= dimension.y || new_z < 0 || new_z >= dimension.z)
					continue;
				auto neighbor_index = new_z * dimension.x * dimension.y + new_y * dimension.x + new_x;

				if ((*labeled_data)[i] != (*labeled_data)[neighbor_index])
				{
					edge_weight_array[(*labeled_data)[i]][(*labeled_data)[neighbor_index]] = 1;
					edge_weight_array[(*labeled_data)[neighbor_index]][(*labeled_data)[i]] = 1;
				}
			}
			label_histogram_array[(*labeled_data)[i]][(*volume_data)[i]] ++;

			if (i % 100000 == 0)
				vm::println("Process {} %.", (i*1.0 / volume_size)*100);
		}

		for (auto i = 0; i < label_number; i++)
		{
			// auto buf_max_index = max_element(label_histogram_array[i].begin(), label_histogram_array[i].end());
			float max_value = 0;
			for (auto j : label_histogram_array[i])
			{
				max_value = std::max(j, max_value);
			}

			//if (*buf_max_index == 0) continue;
			if (max_value == 0) continue;
			for (auto& j : label_histogram_array[i])
			{
				j /= max_value;
			}
		}

		// 保存
		saveVector(label_histogram_array, file_prefix + configure_json.file_name.label_histogram_file);
		saveVector(edge_weight_array, file_prefix + configure_json.file_name.edge_weight_file);

	}
	catch(std::exception& e)
	{
		vm::println("{}", e.what());
	}
	
	clock_t time_end = clock();

	vm::println("Time for supervoxel2histogram : {}s.", (time_end - time_begin) / 1000.0);
}
