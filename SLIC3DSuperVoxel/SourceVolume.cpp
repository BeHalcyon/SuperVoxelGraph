#include "SourceVolume.h"
#include <iostream>
#include <fstream>
#include <algorithm>


/**
 * \brief A template function to get the origin volume data(target_data).
 */


template <class VolumeType>
void loadAndTransVolume(vector<string> volume_file_name, hxy::my_int3& volume_res, std::vector<hxy::myreal>& min_value, std::vector<hxy::myreal>& max_value,
	std::vector<std::vector<VolumeType>>& origin_data, std::vector<std::vector<hxy::myreal>>& target_data)
{
	origin_data.clear();
	origin_data.resize(volume_file_name.size());
	target_data.clear();
	target_data.resize(volume_file_name.size());
	for (auto i = 0; i < volume_file_name.size(); i++)
	{
		std::ifstream in(volume_file_name[i], std::ios::in | std::ios::binary);
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
			std::cout << "Load data successfully.\nThe file path is : " << volume_file_name[i].c_str() << std::endl;

			std::cout << "The file size is : " << fileSize << std::endl;

			
			const long long volume_length = volume_res.z*volume_res.y*volume_res.x;
			origin_data[i].resize(volume_length);
			target_data[i].resize(volume_length);
			for (auto x = 0; x < volume_length; ++x)
			{
				int src_idx = sizeof(VolumeType) * (x);
				memcpy(&origin_data[i][x], &contents[src_idx], sizeof(VolumeType));
				max_value[i] = origin_data[i][x]> max_value[i] ? origin_data[i][x] : max_value[i];
				min_value[i] = origin_data[i][x]< min_value[i] ? origin_data[i][x] : min_value[i];

				target_data[i][x] = origin_data[i][x];
			}
			std::cout << "Max value : " << max_value[i] << " min_value : " << min_value[i] <<std::endl;
		}
		else
		{
			std::cout << "The file " << volume_file_name[i].c_str() << " fails loaded." << std::endl;
		}
		free(contents);
	}
}


SourceVolume::SourceVolume(std::string type_name, int regular_histogram_dimension)
	: type_name(type_name), regular_histogram_dimension(regular_histogram_dimension), volume_number(0),
	  volume_length(0),
	  volume_res(0, 0, 0), fixed_index(0), test_index(0)
{
	is_data_in_memory = false;
	is_regular_data_generated = false;
	is_down_sampling_data_generated = false;
	is_histogram_distribution_calculated = false;
}

SourceVolume::SourceVolume(vector<string> file_name, int x, int y, int z, std::string type_name, int regular_histogram_dimension, int downing_sampling_histogram_dimension)
	:type_name(type_name), regular_histogram_dimension(regular_histogram_dimension), volume_length(0), 
	fixed_index(0), test_index(0), downing_sampling_histogram_dimension(downing_sampling_histogram_dimension)
{
	is_data_in_memory = false;
	is_regular_data_generated = false;
	is_down_sampling_data_generated = false;
	is_histogram_distribution_calculated = false;

	setFileAndRes(file_name, x, y, z);
}


SourceVolume::SourceVolume()
{
}

void SourceVolume::loadVolume()
{
	if (is_data_in_memory)
	{
		std::cout << "The origin data has already been loaded." << std::endl;
		return;
	}
	min_value = std::vector<hxy::myreal>(volume_number, 1e8);
	max_value = std::vector<hxy::myreal>(volume_number, -1e8);

	if(type_name=="uchar")
	{
		std::vector<std::vector<unsigned char>> origin_data;
		loadAndTransVolume(volume_file_name, volume_res, min_value, max_value, origin_data, volume_data);
	}
	else if(type_name=="ushort")
	{
		std::vector<std::vector<unsigned short>> origin_data;
		loadAndTransVolume(volume_file_name, volume_res, min_value, max_value, origin_data, volume_data);
	}
	else if(type_name=="float")
	{
		std::vector<std::vector<float>> origin_data;
		loadAndTransVolume(volume_file_name, volume_res, min_value, max_value, origin_data, volume_data);
	}
	is_data_in_memory = true;
	std::cout << "The origin volume data has been calculated successfully." << std::endl;
}

void SourceVolume::loadRegularVolume()
{
	if(!is_data_in_memory)
	{
		loadVolume();
	}
	if(is_regular_data_generated)
	{
		std::cout << "The regular data has already been loaded." << std::endl;
		return;
	}
	//The next step under the condition that the origin data has been loaded to memory.
	regular_data.clear();
	regular_data.resize(volume_number);
	//const long long volume_length = volume_res.z*volume_res.y*volume_res.x;
	for(auto i=0;i<volume_number;i++)
	{
		regular_data[i].resize(volume_length);
		for(auto index = 0;index<volume_length;index++)
		{
			regular_data[i][index] = (volume_data[i][index] - min_value[i]) / (max_value[i] - min_value[i])*(regular_histogram_dimension - 1);
		}
	}
	is_regular_data_generated = true;
	std::cout << "The regular volume data has been calculated successfully." << std::endl;
}

void SourceVolume::loadDownSamplingVolume()
{
	if (!is_data_in_memory)
	{
		loadVolume();
	}
	if (is_down_sampling_data_generated)
	{
		std::cout << "The down sampling data has already been loaded." << std::endl;
		return;
	}

	//The next step under the condition that the origin data has been loaded to memory.
	down_sampling_data.clear();
	down_sampling_data.resize(volume_number);
	down_sampling_uchar_data.clear();
	down_sampling_uchar_data.resize(volume_number);
	for (auto i = 0;i<volume_number;i++)
	{
		down_sampling_data[i].resize(volume_length);
		down_sampling_uchar_data[i].resize(volume_length);
		for (auto index = 0;index<volume_length;index++)
		{
			down_sampling_data[i][index] = (volume_data[i][index] - min_value[i]) / (max_value[i] - min_value[i])*(downing_sampling_histogram_dimension - 1);
			down_sampling_uchar_data[i][index] = down_sampling_data[i][index];
		}
	}
	is_down_sampling_data_generated = true;
	std::cout << "The down sampling data has been calculated successfully." << std::endl;
}

std::vector<unsigned char>* SourceVolume::getRegularVolume(int index)
{
	if(index<0||index>=volume_number)
	{
		std::cout << "The index out of bounds in getRegularVolume" << std::endl;
		exit(-1);
	}
	if (!is_regular_data_generated)
	{
		loadRegularVolume();
	}
	return &(regular_data[index]);
}

std::vector<int>* SourceVolume::getDownsamplingVolume(int index)
{
	if (index<0 || index >= volume_number)
	{
		std::cout << "The index out of bounds." << std::endl;
		exit(-1);
	}
	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	return &(down_sampling_data[index]);
}
std::vector<unsigned char>* SourceVolume::getDownsamplingUcharVolume(int index)
{
	if (index<0 || index >= volume_number)
	{
		std::cout << "The index out of bounds." << std::endl;
		exit(-1);
	}
	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	return &(down_sampling_uchar_data[index]);
}
/**
 * \brief Get the down sampling volume \note Only two variables are supposed.
 * \param fixed_index 
 * \param test_index 
 * \return 
 */
std::vector<int>* SourceVolume::getDownsamplingVolume(int fixed_index, int test_index)
{
	if (fixed_index<0 || fixed_index >= volume_number || test_index<0 || test_index>volume_number)
	{
		std::cout << "The index out of bounds." << std::endl;
		exit(-1);
	}
	this->fixed_index = fixed_index;
	this->test_index = test_index;
	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	combination_down_sampling_data.clear();
	combination_down_sampling_data.resize(volume_length);
	for(auto i=0;i<volume_length;i++)
	{
		//Calculate the result data;
		const int fixed_value = down_sampling_data[fixed_index][i];
		const int test_value = down_sampling_data[test_index][i];
		combination_down_sampling_data[i] = fixed_value*downing_sampling_histogram_dimension + test_value;
	}
	std::cout << "The down sampling data for index combination [" << fixed_index << ", " << test_index << "] has been calculated." << std::endl;
	return &combination_down_sampling_data;
}

/**
 * \brief Get the down sampling volume \note Multivariate variables are supposed.
 * \param index_array 
 * \return 
 */

std::vector<int>* SourceVolume::getDownsamplingVolume(const vector<int>& index_array)
{

	for (int i : index_array)
	{
		if (i < 0 || i >= volume_number)
		{
			std::cout << "The index out of bounds." << std::endl;
			exit(-1);
		}
	}

	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	combination_down_sampling_data = std::vector<int>(volume_length, 0);
	for (auto i = 0; i < volume_length; i++)
	{
		for (auto index : index_array)
		{
			combination_down_sampling_data[i] += pow(downing_sampling_histogram_dimension, index_array.size() - index - 1)
				*down_sampling_data[index][i];
		}
	}
	std::cout << "The down sampling data for multivariate data has been calculated." << std::endl;
	return &combination_down_sampling_data;
}

long long SourceVolume::getVolumeSize() const
{
	return volume_length;
}

int SourceVolume::getVolumeNumber() const
{
	return volume_file_name.size();
}


void SourceVolume::calcHistogramDistribution()
{
	if(volume_number<2)
	{
		std::cout << "Histogram distribution for univariate data is not supported." << std::endl;
		return;
	}
	//if(is_histogram_distribution_calculated)
	//{
	//	std::cout << "The histogram distribution has already been calculated." << std::endl;
	//	return;
	//}
	if(!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}

	double max_count = 0;
	double second_max_count = 0;

	regularization_histogram_distribution.clear();
	regularization_histogram_distribution.resize(downing_sampling_histogram_dimension);
	origin_histogram_distribution.clear();
	origin_histogram_distribution.resize(downing_sampling_histogram_dimension);

	for(auto i=0;i<downing_sampling_histogram_dimension;i++)
	{
		regularization_histogram_distribution[i].resize(downing_sampling_histogram_dimension);
		origin_histogram_distribution[i].resize(downing_sampling_histogram_dimension);
	}
	for(auto i=0;i<volume_length;i++)
	{
		origin_histogram_distribution[down_sampling_data[fixed_index][i]][down_sampling_data[test_index][i]]++;
	}
	for (auto i = 0;i<downing_sampling_histogram_dimension;i++)
	{
		for (auto j = 0;j<downing_sampling_histogram_dimension;j++)
		{
			max_count = std::max(max_count, static_cast<double>(origin_histogram_distribution[i][j]));
		}
	}
	for (auto i = 0;i<downing_sampling_histogram_dimension;i++)
	{
		for (auto j = 0;j<downing_sampling_histogram_dimension;j++)
		{
			if (origin_histogram_distribution[i][j]<max_count)
				second_max_count = std::max(second_max_count, static_cast<double>(origin_histogram_distribution[i][j]));
		}
	}
	//Save the histogram array into csv file.
	std::ofstream histogram_distribution_file("./result/histogram_distribution.csv");
	histogram_distribution_file << "name" << ",";
	for (int i = 0;i<downing_sampling_histogram_dimension - 1;i++)
		histogram_distribution_file << i << ",";
	histogram_distribution_file << downing_sampling_histogram_dimension - 1 << std::endl;
	for (auto i = 0;i < downing_sampling_histogram_dimension;i++)
	{
		histogram_distribution_file << i << ",";
		double max_value1 = 0.0f;
		for (auto j = 0;j < downing_sampling_histogram_dimension - 1;j++)
		{
			histogram_distribution_file << origin_histogram_distribution[i][j] << ",";
			regularization_histogram_distribution[i][j] = (origin_histogram_distribution[i][j] == 0) ? 0 : log(origin_histogram_distribution[i][j]) / log(second_max_count);
			regularization_histogram_distribution[i][j] = std::min(regularization_histogram_distribution[i][j], 1.0);
		}
		histogram_distribution_file << origin_histogram_distribution[i][downing_sampling_histogram_dimension - 1] << std::endl;
		regularization_histogram_distribution[i][downing_sampling_histogram_dimension - 1] = 
			(origin_histogram_distribution[i][downing_sampling_histogram_dimension - 1] == 0) ? 
			0 : log(origin_histogram_distribution[i][downing_sampling_histogram_dimension - 1]) / log(second_max_count);
		regularization_histogram_distribution[i][downing_sampling_histogram_dimension - 1] = 
			std::min(regularization_histogram_distribution[i][downing_sampling_histogram_dimension - 1], 1.0);
	}
	histogram_distribution_file.close();
	std::cout << "The histogram file has been saved." << std::endl;
	//is_histogram_distribution_calculated = true;
}

vector<vector<hxy::myreal>>  SourceVolume::getRegularizationHistogramDistribution()
{
	//Debug 20190527
	//if(!is_histogram_distribution_calculated)
	//{
	calcHistogramDistribution();
	//}
	return regularization_histogram_distribution;
}

vector<vector<int>>  SourceVolume::getOriginHistogramDistribution()
{
	//Debug 20190527
	//if (!is_histogram_distribution_calculated)
	//{
		calcHistogramDistribution();
	//}
	return origin_histogram_distribution;
}

vector<int> SourceVolume::getComposeHistogramDistribution(const vector<int>& multivariate_id_array)
{
	if (volume_number < 2)
	{
		std::cout << "Histogram distribution for univariate data is not supported." << std::endl;
		exit(-1);
	}
	histogram_for_multivariate.clear();
	const auto multivariate_id_array_size = multivariate_id_array.size();
	const auto length = pow(downing_sampling_histogram_dimension, multivariate_id_array_size);
	histogram_for_multivariate.resize(length);

	for (auto i = 0; i < histogram_for_multivariate.size(); i++) histogram_for_multivariate[i] = 0;

	for(auto i=0;i<volume_length;i++)
	{
		//histogram_for_multivariate[i] = 0;
		auto buf_value = 0;
		for(auto j=0;j< multivariate_id_array_size;j++)
		{
			buf_value += down_sampling_data[multivariate_id_array[j]][i] *
				pow(downing_sampling_histogram_dimension, multivariate_id_array_size-1-j);
		}
		histogram_for_multivariate[buf_value]++;
	}
	std::cout << "The histogram for multivariate data has been calculated." << std::endl;
	return histogram_for_multivariate;
}


void SourceVolume::setFileAndRes(vector<string>& file_name, int x, int y, int z)
{
	volume_file_name = file_name;
	volume_number = file_name.size();

	volume_res.x = x;
	volume_res.y = y;
	volume_res.z = z;

	volume_length = volume_res.z*volume_res.y*volume_res.x;
}

void SourceVolume::setFileAndRes(vector<string>& file_name, hxy::my_int3& resolution)
{
	volume_file_name = file_name;
	volume_number = file_name.size();

	volume_res = resolution;
	volume_length = volume_res.z*volume_res.y*volume_res.x;
}

/**
* \brief Get the index array based on fixed variable.
* \param index_array If the fix_value is in the range of min_value and max_value, index_array[fix_value] = test_value
* \param max_fixed_value
* \param min_fixed_value
*/
//TODO the other function has not been implemented.
void SourceVolume::getIndexArrayBasedFixedVariable(vector<int>& index_array, const int max_fixed_value,
	const int min_fixed_value)
{
	//index_array 表示在第一个变量范围内，第二个变量的数据分布结果。如果第一个变量在指定范围内，则index_array[当前体素的第二个变量]=1
	//Debug 20190221 fix the initial value of index_array to -1

	if(volume_number<2)
	{
		std::cout << "The volume number is not supported." << std::endl;
		//exit(-1);
		return;
	}

	index_array = vector<int>(downing_sampling_histogram_dimension, -1);
	std::cout << "The histogram dimension of multiple volume is: " << downing_sampling_histogram_dimension << std::endl;

	for (auto i = 0; i < volume_length; ++i)
	{
		auto& fixed_value = down_sampling_data[fixed_index][i];
		if (fixed_value >= min_fixed_value&&fixed_value <= max_fixed_value)
		{
			index_array[fixed_value] = down_sampling_data[test_index][i];
		}
	}
	std::cout << "Get index array end."<<std::endl;

}


void SourceVolume::deleteData()
{
	if (is_data_in_memory)
	{
		volume_data.clear();
		is_data_in_memory = false;
	}
	if (is_regular_data_generated)
	{
		regular_data.clear();
		is_regular_data_generated = false;
	}
	if (is_down_sampling_data_generated)
	{
		down_sampling_data.clear();
		is_down_sampling_data_generated = false;
	}
	if (is_histogram_distribution_calculated)
	{
		regularization_histogram_distribution.clear();
		origin_histogram_distribution.clear();
		is_histogram_distribution_calculated = false;
	}
	max_value.clear();
	min_value.clear();
	combination_down_sampling_data.clear();
}

SourceVolume::~SourceVolume()
{
}
