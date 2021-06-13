#pragma once
#include <vector>
#include "../SLIC3DSuperVoxel/VMUtils/include/VMUtils/fmt.hpp"
#include <fstream>
#include <cmath>

struct Sphere
{
	std::vector<size_t> central_point = { 0, 0, 0 };
	double radius = 0;
	double value;
	double range;
};


template<typename T>
class VolumeGenerator
{
public:

	VolumeGenerator(const std::vector<size_t> & dimension, const std::vector<double>& space, const std::string type="uchar");

	void addSphere(const std::vector<size_t>& central_point, double radius, T value);
	void addSphere(const Sphere & sphere);

	void saveVolume(std::string file_path = "");
	void saveGroundTruthLabelVolume(std::string file_path);

	inline void clearVolume() { volume_data.resize(volume_size, 0); shape_number = 0; }

private:
	void calcSphereBoundingBox(const Sphere& sphere, std::vector<int>& min_bounding_box, std::vector<int>& max_bounding_box);
	bool isInSphere(const std::vector<size_t>& point, const Sphere& sphere);
private:
	std::vector<T> volume_data;
	//Extension 20200605 The vector contains shape_number+1 cluster, 0 for background, and [1,shape_number] for spheres.
	// TODO : This is not fit with the current project.
	std::vector<int> volume_ground_true_data;

	std::vector<size_t> dimension;
	std::vector<double> space;
	size_t volume_size;
	int shape_number = 0;
	std::string type;
};

template <typename T>
VolumeGenerator<T>::VolumeGenerator(const std::vector<size_t>& dimension, const std::vector<double>& space, const std::string type)
{
	if(dimension.size()!=3)
	{
		vm::println("Please indicate the dimension.");
		exit(-1);
	}
	this->dimension = dimension;
	this->space = space;

	volume_size = dimension[0] * dimension[1] * dimension[2];

	volume_data.resize(volume_size, 0);
	volume_ground_true_data.resize(volume_size, 0);

	this->type = type;
}

template <typename T>
void VolumeGenerator<T>::addSphere(const std::vector<size_t>& central_point, double radius, T value)
{
	addSphere({ central_point, radius , static_cast<double>(value) });
}

template <typename T>
void VolumeGenerator<T>::addSphere(const Sphere& sphere)
{
	std::vector<int> min_bounding_box(3, 0);
	std::vector<int> max_bounding_box(3, 0);

	calcSphereBoundingBox(sphere, min_bounding_box, max_bounding_box);

	vm::println("Min box for sphere {}: {}", shape_number, min_bounding_box);
	vm::println("Max box for sphere {}: {}", shape_number, max_bounding_box);


	auto& point_coor = sphere.central_point;

	for(size_t k=min_bounding_box[2];k<=max_bounding_box[2];k++)
	{
		for (size_t j = min_bounding_box[1]; j <= max_bounding_box[1]; j++)
		{
			for (size_t i = min_bounding_box[0]; i <= max_bounding_box[0]; i++)
			{
				auto index = k * dimension[0] * dimension[1] + j * dimension[0] + i;

				if(isInSphere({i,j,k},sphere))
				{
					auto distance = sqrt((point_coor[0] - i) * (point_coor[0] - i) +
						(point_coor[1] - j) * (point_coor[1] - j) +
						(point_coor[2] - k) * (point_coor[2] - k));
					auto value = sphere.value + (distance  / sphere.radius) * sphere.range;

					if (value < 0) value = 0;

					volume_data[index] = static_cast<T>(value);
					volume_ground_true_data[index] = shape_number+1;
				}
			}
		}
	}
	shape_number++;
}

template <typename T>
void VolumeGenerator<T>::saveVolume(std::string file_path)
{
	// Origin volume
	const auto file_name = "spheres_" + type + "_" + std::to_string(dimension[0])
		+ "_" + std::to_string(dimension[1]) +
		"_" + std::to_string(dimension[2]) + "_" + std::to_string(shape_number) ;

	std::ofstream writer(file_path + file_name + ".raw", std::ios::binary);

	writer.write(reinterpret_cast<const char*>(volume_data.data()), sizeof(T) * volume_size);
	writer.close();

	if (file_path == "") file_path = "./";
	vm::println("{} has been generated in path {}", file_name, file_path);


	// vifo
	std::ofstream vifo_writer(file_path + file_name + ".vifo");

	vifo_writer << 1 << std::endl;
	vifo_writer << type << std::endl;
	vifo_writer << dimension[0] << " " << dimension[1] << " " << dimension[2] << std::endl;
	vifo_writer << space[0]<<" "<< space[1]<< " " << space[2] << std::endl;
	vifo_writer << file_name + ".raw" << std::endl;
	vifo_writer.close();
}

template <typename T>
void VolumeGenerator<T>::saveGroundTruthLabelVolume(std::string file_path)
{
	// Origin volume
	const auto file_name = "spheres_" + type + "_" + std::to_string(dimension[0])
		+ "_" + std::to_string(dimension[1]) +
		"_" + std::to_string(dimension[2]) + "_" + std::to_string(shape_number);

	std::ofstream writer(file_path + file_name + "_ground_truth_voxel_label_int.raw", std::ios::binary);

	writer.write(reinterpret_cast<const char*>(volume_ground_true_data.data()), sizeof(int) * volume_size);
	writer.close();
	if (file_path == "") file_path = "./";
	vm::println("Ground truth {} volume has been generated in path {}", file_name, file_path);


	//std::ofstream writer_csv(file_path + file_name + "ground_truth_label_int.csv");
	//writer_csv << "VoxelPos,LabelID\n";
	//for(auto i=0;i<volume_ground_true_data.size();i++)
	//{
	//	writer_csv << i << "," << volume_ground_true_data[i] << std::endl;
	//}
	//writer_csv.close();
	//vm::println("Ground truth {} csv has been generated in path {}", file_name, file_path);
}

template <typename T>
void VolumeGenerator<T>::calcSphereBoundingBox(const Sphere& sphere, std::vector<int>& min_bounding_box,
	std::vector<int>& max_bounding_box)
{
	for(auto i=0;i<min_bounding_box.size();i++)
	{
		min_bounding_box[i] = sphere.central_point[i] - sphere.radius;
		if (min_bounding_box[i] < 0) min_bounding_box[i] = 0;
		else if (min_bounding_box[i] >= dimension[i]) min_bounding_box[i] = dimension[i] - 1;
	}
	for (auto i = 0; i < max_bounding_box.size(); i++)
	{
		max_bounding_box[i] = sphere.central_point[i] + sphere.radius;
		if (max_bounding_box[i] < 0) max_bounding_box[i] = 0;
		else if (max_bounding_box[i] >= dimension[i]) max_bounding_box[i] = dimension[i] - 1;
	}
}

template <typename T>
bool VolumeGenerator<T>::isInSphere(const std::vector<size_t>& point, const Sphere& sphere)
{
	size_t sum = 0;
	for(auto i=0;i < point.size();i++)
	{
		const auto buf = point[i] - sphere.central_point[i];
		sum += buf * buf;
	}
	return (sum <= (sphere.radius * sphere.radius));
}

