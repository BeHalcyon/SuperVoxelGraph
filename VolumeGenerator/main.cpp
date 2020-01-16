#include "VolumeGenerator.h"
#include <string>
#include <iostream>
#include "../SLIC3DSuperVoxel/VMUtils/include/VMUtils/cmdline.hpp"
#include "../SLIC3DSuperVoxel/VMUtils/include/VMUtils/json_binding.hpp"

struct SphereJsonStruct : public vm::json::Serializable<SphereJsonStruct>
{
	VM_JSON_FIELD(std::vector<size_t>, central_point);
	VM_JSON_FIELD(double, radius);
	VM_JSON_FIELD(double, value);
};

struct VolumeGeneratorJSONStruct : public vm::json::Serializable<VolumeGeneratorJSONStruct>
{
	VM_JSON_FIELD(std::string, generator_path);
	VM_JSON_FIELD(std::vector<size_t>, dimension);
	VM_JSON_FIELD(std::vector<double>, space);
	VM_JSON_FIELD(std::string, volume_type);
	VM_JSON_FIELD(std::vector<SphereJsonStruct>, sphere);
};




int main(int argc, char* argv[])
{
	if(argc<=1)
	{
		vm::println("Please use command line parameter.");
		//return 0;
	}

	try
	{

		cmdline::parser a;
		a.add<std::string>("configure_json", 'c', "configure json file for volume generator",
			false, "D:/project/science_project/SLIC3DSuperVoxel/x64/Release/workspace/volume_generator.json");
		
		a.parse_check(argc, argv);

		auto configure_json_file = a.get<std::string>("configure_json");
		
		VolumeGeneratorJSONStruct configure_json;
		
		std::ifstream reader(configure_json_file);
		reader >> configure_json;
		
		std::string generator_path = configure_json.generator_path;
		std::vector<size_t> dimension=configure_json.dimension;
		std::vector<double> space =  configure_json.space;
		std::string volume_type = configure_json.volume_type;


		//VolumeGenerator<unsigned char>
		auto sphere_number = configure_json.sphere.size();
		std::vector<Sphere> spheres;
		for(auto i=0;i<sphere_number;i++)
		{
			auto & buf_json_sphere = configure_json.sphere[i];
			spheres.push_back({ buf_json_sphere.central_point, buf_json_sphere.radius ,buf_json_sphere.value});
		}

		if(volume_type == "uchar")
		{
			VolumeGenerator<unsigned char> volume_generator(dimension, space, volume_type);
			for (auto& sphere : spheres) volume_generator.addSphere(sphere);
			volume_generator.saveVolume(generator_path);
		}
		else if(volume_type == "ushort")
		{
			VolumeGenerator<unsigned short> volume_generator(dimension, space, volume_type);
			for (auto& sphere : spheres) volume_generator.addSphere(sphere);
			volume_generator.saveVolume(generator_path);
		}
		else if(volume_type == "float")
		{
			VolumeGenerator<float> volume_generator(dimension, space, volume_type);
			for (auto& sphere : spheres) volume_generator.addSphere(sphere);
			volume_generator.saveVolume(generator_path);
		}
		else
		{
			vm::println("volume type error.");
			return 0;
		}

		
		
	}
	catch (std::exception & e)
	{
		vm::println("{}", e.what());
	}


	
	
	
	


	
}