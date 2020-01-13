#include <iostream>
#include <fstream>
#include <string>

#include "SourceVolume.h"
#include "SLIC3D.h"
#include "graph_segmentation.h"
#include "cmdline.h"

#include "json_struct.h"
#include <time.h>



void readInfoFile(const string& infoFileName, int& data_number, string& datatype, hxy::my_int3& dimension, hxy::my_double3& space,
	vector<string>& file_list)
{
	file_list.clear();

	ifstream inforFile(infoFileName);

	inforFile >> data_number;
	inforFile >> datatype;
	inforFile >> dimension.x >> dimension.y >> dimension.z;
	//Debug 20190520 ����sapce�ӿ�
	inforFile >> space.x >> space.y >> space.z;
	const string filePath = infoFileName.substr(0, infoFileName.find_last_of('/') == -1?
		infoFileName.find_last_of('\\') + 1: infoFileName.find_last_of('/')+1);
	std::cout << (filePath.c_str()) << std::endl;
	for (auto i = 0; i < data_number; i++)
	{
		string rawFileName;
		inforFile >> rawFileName;
		string volumePath = filePath + rawFileName;
		file_list.push_back(volumePath);
	}
	std::cout << "Info file name : \t\t" << infoFileName.c_str() << std::endl;
	std::cout << "Volume number : \t\t" << data_number << std::endl;
	std::cout << "data type : \t\t\t" << datatype.c_str() << std::endl;
	std::cout << "Volume dimension : \t\t" << "[" << dimension.x << "," << dimension.y << "," << dimension.z << "]" << std::endl;
	std::cout << "Space dimension : \t\t" << "[" << space.x << "," << space.y << "," << space.z << "]" << std::endl;
	for (auto i = 0; i < data_number; i++)
	{
		std::cout << "Volume "<<i<<" name : \t\t" << file_list[i].c_str() << std::endl;
	}


	std::cout << "Info file has been loaded successfully." << std::endl;
	
}

int readLabelFile(const string& label_file_name, int * label_array, const int& sz)
{
	ifstream in(label_file_name, std::ios::in | std::ios::binary);
	unsigned char *contents = nullptr;
	int k_number = 0;

	int max_value = -0xffffff;
	int min_value = 0xffffff;
	if (in)
	{
		in.seekg(0, std::ios::end);
		const long int fileSize = in.tellg();
		contents = static_cast<unsigned char*>(malloc(static_cast<size_t>(fileSize + 1)));
		in.seekg(0, std::ios::beg);
		in.read(reinterpret_cast<char*>(contents), fileSize);
		in.close();
		contents[fileSize] = '\0';
		std::cout << "Load gradient data successfully.\nThe file path is : " << label_file_name.c_str() << std::endl;

		std::cout << "The gradient file size is : " << fileSize << std::endl;


		const long long volume_length = sz;


		for (auto x = 0; x < volume_length; ++x)
		{
			const auto src_idx = sizeof(int) * (x);
			memcpy(&label_array[x], &contents[src_idx], sizeof(int));
			max_value = label_array[x] > max_value ? label_array[x] : max_value;
			min_value = label_array[x] < min_value ? label_array[x] : min_value;
		}
		std::cout << "Max value : " << max_value << " min_value : " << min_value << std::endl;
	}
	else
	{
		std::cout << "The label file " << label_file_name.c_str() << " fails loaded." << std::endl;
	}
	free(contents);

	return max_value - min_value + 1;
}

void doSuperVoxelMerge(
					int *					merged_label,
					const unsigned char*	volume_data, 
					const int &				dimension,
					const int*				label,
					const double*			gradient_array,
					const int &				width,
					const int &				height, 
					const int &				depth,
					const int &				label_number, 
					const int &				k_threshold,
					const string&			file_path = "")
{
	std::cout << "Begin super-voxel merged algorithm..." << std::endl;
	int minimum_segment_size = 10;

	std::cout << "Threshold, minimum segment size and k number: " << k_threshold << "\t" << minimum_segment_size << std::endl;

	GraphSegmentationMagicThreshold magic(k_threshold);

	GraphSegmentation segmenter;
	segmenter.setMagic(&magic);

	segmenter.buildGraph(volume_data, dimension, label, label_number, gradient_array, width, height, depth);
	segmenter.oversegmentGraph();
	//segmenter.enforceMinimumSegmentSize(minimum_segment_size);

	const auto sz = width * height * depth;
	for (auto i = 0; i < sz; i++) merged_label[i] = -1;
	segmenter.deriveLabels(merged_label);

	if(!file_path.empty())
		segmenter.saveMergeLabels(merged_label, width, height, depth, file_path);
}

void doAlgorithmWithoutCmdLine()
{

	//string			infoFileName = "F:/CThead/manix/manix.vifo";
	//string			infoFileName = "F:/atmosphere/timestep21_float/multiple_variables.vifo";
	string			infoFileName = "I:/science data/4 Combustion/jet_0051/mixfrac.vifo";

	int				data_number;
	string			datatype;
	hxy::my_int3	dimension;
	hxy::my_double3	space;
	vector<string>	file_list;

	readInfoFile(infoFileName, data_number, datatype, dimension, space, file_list);

	auto file_path = file_list[0].substr(0, file_list[0].find_last_of('.'));

	SourceVolume source_volume(file_list, dimension.x, dimension.y, dimension.z, datatype);

	//source_volume.loadVolume();	//origin data
	source_volume.loadRegularVolume(); //[0, 255] data
	//source_volume.loadDownSamplingVolume(); //[0, histogram_dimension] data

	auto volume_data = source_volume.getRegularVolume(0);


	//----------------------------------
	// Initialize parameters
	//----------------------------------
	int k = 8196;//Desired number of superpixels.
	double m = 20;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	int* klabels = new int[dimension.x*dimension.y*dimension.z];
	int num_labels(0);
	SLIC3D slic_3d;
	slic_3d.PerformSLICO_ForGivenK((*volume_data).data(), dimension.x, dimension.y, dimension.z, klabels, num_labels, k, m);

	//string label_file = "F:/CThead/manix/manix_label.raw";
	//string label_file = "F:/atmosphere/timestep21_float/_SPEEDf21_label.raw";

	slic_3d.SaveSuperpixelLabels(klabels, dimension.x, dimension.y, dimension.z, file_path+"_label.raw");

	slic_3d.SaveGradient(file_path + "_gradient.raw");


	auto segment_boundary_array = new int[dimension.x*dimension.y*dimension.z];
	slic_3d.DrawContoursAroundSegments(segment_boundary_array, klabels, dimension.x, dimension.y, dimension.z);
	slic_3d.SaveSegmentBouyndaries(segment_boundary_array, dimension.x, dimension.y, dimension.z, file_path + "_boundary.raw");


	int k_threshold = 100000;
	auto merged_label = new int[dimension.x*dimension.y*dimension.z];
	doSuperVoxelMerge(merged_label, (*volume_data).data(), 
		source_volume.getRegularDimenion(),
		klabels, 
		slic_3d.getGradient().data(),
		dimension.x, dimension.y, dimension.z, 
		num_labels, k_threshold, file_path + "_merged_label.raw");

	slic_3d.DrawContoursAroundSegments(segment_boundary_array, merged_label, dimension.x, dimension.y, dimension.z);
	slic_3d.SaveSegmentBouyndaries(segment_boundary_array, dimension.x, dimension.y, dimension.z, file_path + "_merged_boundary.raw");

	delete[] segment_boundary_array;
	delete[] klabels;
}

void doAlgorithmWithCmdLine(int argc, char* argv[])
{
	// create a parser
	cmdline::parser a;

	a.add<string>("vifo_path", 'p', "vifo file path", true, "");


	a.add<int>("cluster_number", 'c', "initial cluster number", false, 8196, cmdline::range(100, 6553500));

	// cmdline::oneof() can restrict options.
	a.add<bool>("super_voxel_label", 'l', "whether output super-voxel label as a raw file (int) or not", false, false,
		cmdline::oneof<bool>(true, false));

	a.add<bool>("super_voxel_boundary", 'b', "whether output super-voxel boundary as a raw file (int) or not", false, true,
		cmdline::oneof<bool>(true, false));

	a.add<bool>("gradient", 'g', "whether output gradient as a raw file (float) or not", false, false,
		cmdline::oneof<bool>(true, false));

	a.add<bool>("merge", 'm', "whether do super-voxel merged algorithm or not", true, false,
		cmdline::oneof<bool>(true, false));


	a.add<int>("k_threshold", 'k', "the k threshold when doing super-voxel merged algorithm", false, 100000, 
		cmdline::range(0, 655350000));

	a.add<bool>("merged_label", 'L', "whether output merged label as a raw file (int) or not", false, false,
		cmdline::oneof<bool>(true, false));

	a.add<bool>("merged_boundary", 'B', "whether output merged boundary as a raw file (int) or not", false, true,
		cmdline::oneof<bool>(true, false));

	// Call add method without a type parameter.
	//a.add("non_parameter", '\0', "gzip when transfer");

	a.parse_check(argc, argv);

	string			infoFileName = a.get<string>("vifo_path");
	int				data_number;
	string			datatype;
	hxy::my_int3	dimension;
	hxy::my_double3	space;
	vector<string>	file_list;

	readInfoFile(infoFileName, data_number, datatype, dimension, space, file_list);

	auto file_path = file_list[0].substr(0, file_list[0].find_last_of('.'));

	SourceVolume source_volume(file_list, dimension.x, dimension.y, dimension.z, datatype);
	                                                                                                                    
	//source_volume.loadVolume();	//origin data
	source_volume.loadRegularVolume(); //[0, 255] data
	//source_volume.loadDownSamplingVolume(); //[0, histogram_dimension] data

	auto volume_data = source_volume.getRegularVolume(0);

	//----------------------------------
	// Initialize parameters
	//----------------------------------
	int k = a.get<int>("cluster_number");
	double m = 20;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	int* klabels = new int[dimension.x*dimension.y*dimension.z];
	int num_labels(0);
	SLIC3D slic_3d;
	slic_3d.PerformSLICO_ForGivenK((*volume_data).data(), dimension.x, dimension.y, dimension.z, klabels, num_labels, k, m);
	if (a.get<bool>("super_voxel_label"))
	{
		slic_3d.SaveSuperpixelLabels(klabels, dimension.x, dimension.y, dimension.z, file_path + "_label.raw");
	}
	if (a.get<bool>("gradient"))
	{
		slic_3d.SaveGradient(file_path + "_gradient.raw");
	}
	auto segment_boundary_array = new int[dimension.x*dimension.y*dimension.z];
	slic_3d.DrawContoursAroundSegments(segment_boundary_array, klabels, dimension.x, dimension.y, dimension.z);

	if (a.get<bool>("super_voxel_boundary"))
	{

		slic_3d.SaveSegmentBouyndaries(segment_boundary_array, 
			dimension.x, dimension.y, dimension.z, file_path + "_boundary.raw");
	}

	if (a.get<bool>("merge")) {
		int k_threshold = a.get<int>("k_threshold");
		auto merged_label = new int[dimension.x*dimension.y*dimension.z];

		string buf_file_path = "";
		if (a.get<bool>("merged_label")) buf_file_path = file_path + "_merged_label.raw";

		doSuperVoxelMerge(merged_label, (*volume_data).data(),
			source_volume.getRegularDimenion(),
			klabels,
			slic_3d.getGradient().data(),
			dimension.x, dimension.y, dimension.z,
			num_labels, k_threshold, buf_file_path);

		if (a.get<bool>("merged_boundary")) {
			slic_3d.DrawContoursAroundSegments(segment_boundary_array, merged_label, dimension.x, dimension.y, dimension.z);
			slic_3d.SaveSegmentBouyndaries(segment_boundary_array, dimension.x, dimension.y, dimension.z, file_path + "_merged_boundary.raw");
		}
		delete[] merged_label;
	}
	delete[] segment_boundary_array;
	delete[] klabels;
}

void doAlgorithmWithJsonConfigure(int argc, char* argv[])
{
	// create a parser
	cmdline::parser a;

	a.add<string>("configure_file", 'c', "json configure file", true, "");

	// Call add method without a type parameter.
	//a.add("non_parameter", '\0', "gzip when transfer");

	a.parse_check(argc, argv);

	ConfigureJSONStruct configure_json;

	std::string		configure_json_file = a.get<std::string>("configure_file");

	try
	{
		std::ifstream input_file(configure_json_file);
		input_file >> configure_json;

		

		string			infoFileName = configure_json.data_path.vifo_file;
		int				data_number;
		string			datatype;
		hxy::my_int3	dimension;
		hxy::my_double3	space;
		vector<string>	file_list;

		readInfoFile(infoFileName, data_number, datatype, dimension, space, file_list);

		//auto file_path = file_list[0].substr(0, file_list[0].find_last_of('.'));

		SourceVolume source_volume(file_list, dimension.x, dimension.y, dimension.z, datatype);

		//source_volume.loadVolume();	//origin data
		source_volume.loadRegularVolume(); //[0, 255] data
		//source_volume.loadDownSamplingVolume(); //[0, histogram_dimension] data

		auto volume_index = configure_json.data_path.volume_index;
		
		auto volume_data = source_volume.getRegularVolume(volume_index);

		//----------------------------------
		// Initialize parameters
		//----------------------------------
		//int k = a.get<int>("cluster_number");
		auto k = configure_json.volume2supervoxel.cluster_number;
		//double m = 20;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
		double m = configure_json.volume2supervoxel.compactness_factor;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
		int* klabels = new int[dimension.x * dimension.y * dimension.z];
		int num_labels(0);
		SLIC3D slic_3d;
		slic_3d.PerformSLICO_ForGivenK((*volume_data).data(), dimension.x, dimension.y, dimension.z, klabels, num_labels, k, m);

		auto file_prefix = configure_json.data_path.file_prefix;

		//if (a.get<bool>("super_voxel_label"))
		if (configure_json.volume2supervoxel.output_super_voxel_label)
		{
			auto output_label_file = file_prefix + configure_json.file_name.label_file;
			slic_3d.SaveSuperpixelLabels(klabels, dimension.x, dimension.y, dimension.z, output_label_file);
		}
		//if (a.get<bool>("gradient"))
		if (configure_json.volume2supervoxel.output_gradient)
		{
			auto output_gradient_file = file_prefix + configure_json.file_name.gradient_file;
			slic_3d.SaveGradient(output_gradient_file);
		}
		auto segment_boundary_array = new int[dimension.x * dimension.y * dimension.z];
		slic_3d.DrawContoursAroundSegments(segment_boundary_array, klabels, dimension.x, dimension.y, dimension.z);

		if (configure_json.volume2supervoxel.output_super_voxel_boundary)
		{
			auto output_boundary_file = file_prefix + configure_json.file_name.boundary_file;
			slic_3d.SaveSegmentBouyndaries(segment_boundary_array,
				dimension.x, dimension.y, dimension.z, output_boundary_file);
		}

		if (configure_json.volume2supervoxel.is_merge) {
			//int k_threshold = a.get<int>("k_threshold");
			int k_threshold = configure_json.volume2supervoxel.k_threshold;
			auto merged_label = new int[dimension.x * dimension.y * dimension.z];

			string output_merged_label_file = "";
			//if (a.get<bool>("merged_label")) buf_file_path = file_path + "_merged_label.raw";
			if (configure_json.volume2supervoxel.output_merged_label) output_merged_label_file = file_prefix + configure_json.file_name.merged_label_file;

			doSuperVoxelMerge(merged_label, (*volume_data).data(),
				source_volume.getRegularDimenion(),
				klabels,
				slic_3d.getGradient().data(),
				dimension.x, dimension.y, dimension.z,
				num_labels, k_threshold, output_merged_label_file);

			// if (a.get<bool>("merged_boundary")) {
			if (configure_json.volume2supervoxel.output_merged_boundary) {
				auto output_merged_boundary_file = file_prefix + configure_json.file_name.merged_boundary_file;
				slic_3d.DrawContoursAroundSegments(segment_boundary_array, merged_label, dimension.x, dimension.y, dimension.z);
				slic_3d.SaveSegmentBouyndaries(segment_boundary_array, dimension.x, dimension.y, dimension.z, output_merged_boundary_file);
			}
			delete[] merged_label;
		}
		delete[] segment_boundary_array;
		delete[] klabels;

	}
	catch (std::exception & e)
	{
		vm::println("{}", e.what());
	}
}

int main(int argc, char* argv[])
{
	clock_t time_begin = clock();
	if(argc <= 1)
	{
		doAlgorithmWithoutCmdLine();
	}
	else
	{
		//doAlgorithmWithCmdLine(argc, argv);
		doAlgorithmWithJsonConfigure(argc, argv);
	}
	clock_t time_end = clock();

	vm::println("Time for volume2supervoxel : {}s.", (time_end - time_begin) / 1000.0);
}
