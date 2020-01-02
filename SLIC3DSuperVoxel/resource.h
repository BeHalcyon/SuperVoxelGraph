#pragma once

namespace hxy
{
	typedef double myreal;
	const int REGULAR_DIMENSION = 256;

	class my_int3
	{
	public:
		int x = 0;
		int y = 0;
		int z = 0;
	public:
		my_int3(int x = 0, int y = 0, int z = 0) :
			x(x), y(y), z(z)
		{
		}
	};

	class my_double3
	{
	public:
		double x = 0;
		double y = 0;
		double z = 0;
	public:
		my_double3(double x = 0, double y = 0, double z = 0) :
			x(x), y(y), z(z)
		{
		}
	};

}