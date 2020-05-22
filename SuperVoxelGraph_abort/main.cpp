#include "graph_segmentation.h"
#include <iostream>
#include <string>
#include <fstream>
using namespace std;


void readGradientFile(const string& gradient_file_name, double * gradient_array, const int& sz)
{
	ifstream in(gradient_file_name, std::ios::in | std::ios::binary);
	unsigned char *contents = nullptr;
	if (in)
	{
		in.seekg(0, std::ios::end);
		const long int fileSize = in.tellg();
		contents = static_cast<unsigned char*>(malloc(static_cast<size_t>(fileSize + 1)));
		in.seekg(0, std::ios::beg);
		in.read(reinterpret_cast<char*>(contents), fileSize);
		in.close();
		contents[fileSize] = '\0';
		std::cout << "Load gradient data successfully.\nThe file path is : " << gradient_file_name.c_str() << std::endl;

		std::cout << "The gradient file size is : " << fileSize << std::endl;


		const long long volume_length = sz;
		double max_value = -0xffffff;
		double min_value = 0xffffff;

		for (auto x = 0; x < volume_length; ++x)
		{
			const auto src_idx = sizeof(float) * (x);
			float buf = 0.0;
			memcpy(&buf, &contents[src_idx], sizeof(float));
			gradient_array[x] = buf;
			//std::cout << gradient_array[x]<< std::endl;			
			max_value = gradient_array[x] > max_value ? gradient_array[x] : max_value;
			min_value = gradient_array[x] < min_value ? gradient_array[x] : min_value;
		}
		std::cout << "Max value : " << max_value<< " min_value : " << min_value << std::endl;
	}
	else
	{
		std::cout << "The gradient file " << gradient_file_name.c_str() << " fails loaded." << std::endl;
	}
	free(contents);
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


void saveMergeLabels(
	const int*					merged_labels,
	const int&					width,
	const int&					height,
	const int&					depth,
	const string&				filename)
{
	int sz = width * height * depth;

	ofstream outfile(filename.c_str(), ios::binary);
	for (int i = 0; i < sz; i++)
	{
		outfile.write((char*)(&merged_labels[i]), sizeof(int));
	}
	outfile.close();
	std::cout << "Merged label file for the super-voxels has been saved." << std::endl;
}

void readVolumeFile(const string& label_file_name, unsigned char * volume_array, const int& sz)
{
	ifstream in(label_file_name, std::ios::in | std::ios::binary);
	unsigned char *contents = nullptr;
	int k_number = 0;

	float max_value = -0xffffff;
	float min_value = 0xffffff;
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

		float * label_array = new float[sz];

		for (auto x = 0; x < volume_length; ++x)
		{
			const auto src_idx = sizeof(float) * (x);
			memcpy(&label_array[x], &contents[src_idx], sizeof(float));
			max_value = label_array[x] > max_value ? label_array[x] : max_value;
			min_value = label_array[x] < min_value ? label_array[x] : min_value;
		}
		std::cout << "Max value : " << max_value << " min_value : " << min_value << std::endl;

		auto factor = 1.0 / (max_value - min_value);
		for(auto x = 0;x<volume_length;x++)
		{
			volume_array[x] = (label_array[x] - min_value) *255.0*factor;
		}

	}
	else
	{
		std::cout << "The label file " << label_file_name.c_str() << " fails loaded." << std::endl;
	}
	free(contents);
}


int main(int argc, char* argv[])
{
	int width = 480;
	int height = 720;
	int depth = 120;
	auto sz = width * height * depth;
	double * gradient_array = new double[sz];
	int * label_array = new int[sz];

	const string gradient_file_name = "J:/science data/4 Combustion/jet_0051/jet_mixfrac_0051_gradient.raw";
	const string label_file_name = "J:/science data/4 Combustion/jet_0051/jet_mixfrac_0051_label.raw";
	const string merged_label_file_name = "J:/science data/4 Combustion/jet_0051/jet_mixfrac_0051_merged_label.raw";
	const string volume_file_name = "J:/science data/4 Combustion/jet_0051/jet_mixfrac_0051.raw";

	readGradientFile(gradient_file_name, gradient_array, sz);
	int k_number = readLabelFile(label_file_name, label_array, sz);

	double threshold = 200;
	if(argc == 2)
	{
		threshold = atof(argv[1]);
	}
	int minimum_segment_size = 10;

	//getchar();

	std::cout << "Threshold, minimum segment size and k number: " << threshold << "\t" << minimum_segment_size<<"\t" <<k_number << std::endl;

	GraphSegmentationMagicThreshold magic(threshold);

	GraphSegmentation segmenter;
	segmenter.setMagic(&magic);

	unsigned char* volume_array = new unsigned char[sz];
	readVolumeFile(volume_file_name, volume_array, sz);


	segmenter.buildGraph(volume_array, 256, label_array, k_number, gradient_array, width, height, depth);
	segmenter.oversegmentGraph();
	//segmenter.enforceMinimumSegmentSize(minimum_segment_size);

	int* merged_label = new int[sz];
	for (auto i = 0; i < sz; i++) merged_label[i] = -1;
	segmenter.deriveLabels(merged_label);

	saveMergeLabels(merged_label,width, height, depth, merged_label_file_name);
}