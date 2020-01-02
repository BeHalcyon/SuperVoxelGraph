#include "SLIC3D.h"
#include <iostream>


SLIC3D::SLIC3D()
{
	m_volumevec = nullptr;
}


SLIC3D::~SLIC3D()
{
	delete[] m_volumevec;
}


//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void SLIC3D::PerformSLICO_ForGivenK(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	const int					depth,
	int*						klabels,	//size is width*height*depth
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m)//weight given to spatial distance
{
	vector<double> kseedintensity(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> kseedsz(0);

	//--------------------------------------------------
	m_width = width;
	m_height = height;
	m_depth = depth;

	int sz = m_width * m_height*m_depth;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------
	// if (1)//LAB
	// {
	// 	DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	// }
	// else//RGB
	// {
	// 	m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
	// 	for (int i = 0; i < sz; i++)
	// 	{
	// 		m_lvec[i] = ubuff[i] >> 16 & 0xff;
	// 		m_avec[i] = ubuff[i] >> 8 & 0xff;
	// 		m_bvec[i] = ubuff[i] & 0xff;
	// 	}
	// }
	//��ʼ������
	m_volumevec = new double[sz];
	for (auto i = 0; i < sz; i++) m_volumevec[i] = ubuff[i];
	//--------------------------------------------------

	bool perturbseeds(true);
	//edgemag��ʾͼ�е�ÿ���ڵ��Ӧ���ݶȲ�
	vector<double> edgemag(0);
	//�������ص���ݶ�ֵ�������ص�ı仯�̶�
	if (perturbseeds) DetectLabEdges(m_volumevec, m_width, m_height, m_depth, edgemag);
	//��ȡ��Ӧ�����ӵ㣬�ҽ����ӵ�����
	GetLABXYSeeds_ForGivenK(kseedintensity, kseedsx, kseedsy, kseedsz, K, perturbseeds, edgemag);

	//����̫С������£����һ��С��ֵ
	int STEP = pow(double(sz) / double(K), 1.0/3) + 2.0;//adding a small value in the even the STEP size is too small.
	//PerformSuperpixelSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, m);
	PerformSuperpixelSegmentation_VariableSandM(kseedintensity, kseedsx, kseedsy, kseedsz, klabels, STEP, 10);
	numlabels = kseedintensity.size();

	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, m_depth, nlabels, numlabels, K);
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;
}


void SLIC3D::PerformSLICO_ForGivenK(
	const unsigned char*			ubuff,
	const int					width,
	const int					height,
	const int					depth,
	int*						klabels,	//size is width*height*depth
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m)//weight given to spatial distance
{
	vector<double> kseedintensity(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> kseedsz(0);

	//--------------------------------------------------
	m_width = width;
	m_height = height;
	m_depth = depth;

	int sz = m_width * m_height*m_depth;
	//klabels = new int[sz];
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------
	// if (1)//LAB
	// {
	// 	DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	// }
	// else//RGB
	// {
	// 	m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
	// 	for (int i = 0; i < sz; i++)
	// 	{
	// 		m_lvec[i] = ubuff[i] >> 16 & 0xff;
	// 		m_avec[i] = ubuff[i] >> 8 & 0xff;
	// 		m_bvec[i] = ubuff[i] & 0xff;
	// 	}
	// }
	//��ʼ������
	m_volumevec = new double[sz];
	for (auto i = 0; i < sz; i++) m_volumevec[i] = ubuff[i];
	//--------------------------------------------------

	bool perturbseeds(true);
	edgemag.clear();
	//�������ص���ݶ�ֵ�������ص�ı仯�̶�
	if (perturbseeds) DetectLabEdges(m_volumevec, m_width, m_height, m_depth, edgemag);
	std::cout << "Seed points has been initialized." << std::endl;
	//��ȡ��Ӧ�����ӵ㣬�ҽ����ӵ�����
	GetLABXYSeeds_ForGivenK(kseedintensity, kseedsx, kseedsy, kseedsz, K, perturbseeds, edgemag);
	std::cout << "Seed points has been perturbed." << std::endl;

	//����̫С������£����һ��С��ֵ
	int STEP = pow(double(sz) / double(K), 1.0 / 3) + 2.0;//adding a small value in the even the STEP size is too small.
	//PerformSuperpixelSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, m);
	PerformSuperpixelSegmentation_VariableSandM(kseedintensity, kseedsx, kseedsy, kseedsz, klabels, STEP, 10);
	std::cout << "K-means clustering has been calculated." << std::endl;


	numlabels = kseedintensity.size();
	std::cout << "Cluster number : \t\t\t" << numlabels << std::endl;
	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, m_depth, nlabels, numlabels, K);
	std::cout << "Cluster number after merged : \t\t" << numlabels << std::endl;
	std::cout << "Small clusters have been merged." << std::endl;
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;
	std::cout << "SLIC algorithm for volume data in 3D space has been executed." << std::endl;
}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
///		����EnforceLabelConnectivityʵ������ͨ��������¸�ֵ��֮ǰ���е�labelֵ��
///		�������ĵ�ֵ��ͨ�������������ͳһ�����صı�ţ���0��ʼһֱ�����һ������
///		�أ�Ȼ���ͳ��ÿ����������ռ�����أ����̫С�Ļ�����кϲ�
//===========================================================================
void SLIC3D::EnforceLabelConnectivity(
	const int*					labels,//input labels that need to be corrected to remove stray labels //��Ҫ������������
	const int&					width,
	const int&					height,
	const int&					depth,
	int*						nlabels,//new labels
	int&						numlabels,//the number of labels changes in the end if segments are removed
	const int&					K) //the number of superpixels desired by the user
{
	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	//const int dx4[4] = { -1,  0,  1,  0 };
	//const int dy4[4] = { 0, -1,  0,  1 };


	const int dx6[6] = { -1,  1,  0,  0,  0,  0 };
	const int dy6[6] = {  0,  0, -1,  1,  0,  0 };
	const int dz6[6] = {  0,  0,  0,  0, -1,  1 };

	const int sz = width * height * depth;
	//ÿ��С�������صĴ�С
	const int SUPSZ = sz / K;
	//nlabels.resize(sz, -1);
	//nlabelsΪ������������������������
	for (int i = 0; i < sz; i++) nlabels[i] = -1;
	int label(0);
	//���������鶼������ͼƬ��ô��
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int* zvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label

	for(auto i=0;i<depth;i++)
	{
		for(auto j=0;j<height;j++)
		{
			for(auto k=0;k<width;k++)
			{
				if(nlabels[oindex] < 0)
				{
					nlabels[oindex] = label;
					//--------------------
					// Start a new segment
					//--------------------
					xvec[0] = k;
					yvec[0] = j;
					zvec[0] = i;

					//-------------------------------------------------------
					// Quickly find an adjacent label for use later if needed
					//-------------------------------------------------------
					//Ѱ��һ�����ڵ������label
					{
						for (int n = 0; n < 6; n++)
						{
							int x = xvec[0] + dx6[n];
							int y = yvec[0] + dy6[n];
							int z = zvec[0] + dz6[n];
							//�鿴���������Ƿ��д��ڵ���0��nlabels������еĻ�����adjlabel
							if ((x >= 0 && x < width) && (y >= 0 && y < height) && (z >= 0 && z < depth))
							{
								int nindex = z*height*width + y * width + x;
								if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
							}
						}
					}

					int count(1);
					for (int c = 0; c < count; c++)
					{
						for (int n = 0; n < 6; n++)
						{
							int x = xvec[c] + dx6[n];
							int y = yvec[c] + dy6[n];
							int z = zvec[c] + dz6[n];

							if ((x >= 0 && x < width) && (y >= 0 && y < height) && (z >= 0 && z < depth))
							{
								int nindex = z * height*width + y * width + x;
								//�����������û�д���� && ����������ص�label���������ص�labelһ��
								if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
								{
									//��������ȥ�����Ǽ�¼������Щ��
									xvec[count] = x;
									yvec[count] = y;
									zvec[count] = z;
									//������������չ
									nlabels[nindex] = label;
									//��Ϊ���count��һֱ���ӣ��������Ҳ��һֱ��չ��ֱ����Χû���������ؿ�������չ��Ϊֹ
									count++;
								}
							}
						}
					}

					//-------------------------------------------------------
					// If segment size is less then a limit, assign an
					// adjacent label found before, and decrement label count.
					//-------------------------------------------------------
					//�����С�Ļ�����Ҫֱ�����ڽ�����ϲ�
					//��ά����£������ؿ�����Ҫ�޸ģ�ԭʼ��count <= SUPSZ >> 2
					if (count <= SUPSZ >> 2)
					{
						for (int c = 0; c < count; c++)
						{
							int ind = zvec[c]*height*width + yvec[c] * width + xvec[c];
							nlabels[ind] = adjlabel;
						}
						label--;
					}
					label++;
				}
				oindex++;
			}
		}
	}
	numlabels = label;

	delete[] xvec;
	delete[] yvec;
	delete[] zvec;
}


//===========================================================================
///	PerformSuperpixelSegmentation_VariableSandM
///
///	Magic SLIC - no parameters
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
/// This function picks the maximum value of color distance as compact factor
/// M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
/// So no need to input a constant value of M and S. There are two clear
/// advantages:
///
/// [1] The algorithm now better handles both textured and non-textured regions
/// [2] There is not need to set any parameters!!!
///
/// SLICO (or SLIC Zero) dynamically varies only the compactness factor S,
/// not the step size S.
//===========================================================================
void SLIC3D::PerformSuperpixelSegmentation_VariableSandM(
	vector<double>&				kseedintensity,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	vector<double>&				kseedsz,
	int*						klabels,
	const int&					STEP,
	const int&					NUMITR)
{
	int sz = m_width * m_height * m_depth;
	const int numk = kseedintensity.size();
	//double cumerr(99999.9);
	int numitr(0);

	//----------------
	int offset = STEP;
	if (STEP < 10) offset = STEP * 1.5;
	//----------------

	//vector<double> sigmal(numk, 0);
	//vector<double> sigmaa(numk, 0);
	//vector<double> sigmab(numk, 0);
	vector<double> sigmaintensity(numk, 0);

	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<double> sigmaz(numk, 0);

	vector<int> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	//vector<double> distxy(sz, DBL_MAX);
	//vector<double> distlab(sz, DBL_MAX);

	vector<double> distxyz(sz, DBL_MAX);
	vector<double> distintensity(sz, DBL_MAX);

	vector<double> distvec(sz, DBL_MAX);
	
	//vector<double> maxlab(numk, 10 * 10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	//�������ֵ���൱�ڿ������ȵ�Ȩ�أ� mԽ����Ȩ��ԽС����������Ϊ��ֵ��
	vector<double> maxintensity(numk, 10*2);

	//vector<double> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	vector<double> maxxyz(numk, STEP*STEP*STEP);
	

	double invxywt = 1.0 / (STEP*STEP*STEP);//NOTE: this is different from how usual SLIC/LKM works

	while (numitr < NUMITR)
	{
		//------
		//cumerr = 0;
		numitr++;
		//------

		distvec.assign(sz, DBL_MAX);
		for (int n = 0; n < numk; n++)
		{
			int z1 = max(0.0, kseedsz[n] - offset);
			int z2 = min(static_cast<double>(m_depth), kseedsz[n] + offset);
			int y1 = max(0.0, kseedsy[n] - offset);
			int y2 = min(static_cast<double>(m_height), kseedsy[n] + offset);
			int x1 = max(0.0, kseedsx[n] - offset);
			int x2 = min(static_cast<double>(m_width), kseedsx[n] + offset);


			for (auto z = z1; z < z2; z++)
			{
				for (auto y = y1; y < y2; y++)
				{
					for (auto x = x1; x < x2; x++)
					{
						int i = z * m_height*m_width + y * m_width + x;
						_ASSERT(z < m_depth&&z >= 0 && y < m_height&&y >= 0 && x < m_width&&x >= 0);
						distintensity[i] = (m_volumevec[i] - kseedintensity[n])*(m_volumevec[i] - kseedintensity[n]);
						distxyz[i] = (z - kseedsz[n])*(z - kseedsz[n]) + (y - kseedsy[n])*(y - kseedsy[n]) + (x - kseedsx[n])*(x - kseedsx[n]);

						//Debug 20191108 ���޸ľ���ӳ�䣬����invxywt����Ч���;����Ȩ�أ�����maxintensity[n]����Ч��߻Ҷ�ֵ��Ȩ��
						double dist = distintensity[i] / maxintensity[n] + distxyz[i] * invxywt;//only varying m, prettier superpixels

						if (dist < distvec[i])
						{
							distvec[i] = dist;
							klabels[i] = n;
						}
					}
				}
			}
		}

		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		if (0 == numitr)
		{
			maxintensity.assign(numk, 1);
			maxxyz.assign(numk, 1);
		}
		{
			for (int i = 0; i < sz; i++)
			{
				if (maxintensity[klabels[i]] < distintensity[i]) 
					maxintensity[klabels[i]] = distintensity[i];
				if (maxxyz[klabels[i]] < distxyz[i]) 
					maxxyz[klabels[i]] = distxyz[i];
			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmaintensity.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		sigmaz.assign(numk, 0);
		clustersize.assign(numk, 0);

		for (int j = 0; j < sz; j++)
		{
			int temp = klabels[j];
			_ASSERT(klabels[j] >= 0);
			sigmaintensity[klabels[j]] += m_volumevec[j];
			
			sigmaz[klabels[j]] += (j / (m_height*m_width));
			auto buf = j % (m_height*m_width);
			sigmax[klabels[j]] += (buf%m_width);
			sigmay[klabels[j]] += (buf/m_width);

			clustersize[klabels[j]]++;
		}

		{
			for (int k = 0; k < numk; k++)
			{
				//_ASSERT(clustersize[k] > 0);
				if (clustersize[k] <= 0) clustersize[k] = 1;
				inv[k] = 1.0 / double(clustersize[k]);//computing inverse now to multiply, than divide later
			}
		}

		{
			for (int k = 0; k < numk; k++)
			{
				kseedintensity[k] = sigmaintensity[k] * inv[k];
				kseedsx[k] = sigmax[k] * inv[k];
				kseedsy[k] = sigmay[k] * inv[k];
				kseedsz[k] = sigmaz[k] * inv[k];
			}
		}

		std::cout << "Iteration " << numitr << "/" << NUMITR << " has been executed." << std::endl;
	}
}


// Debug 20191102
void SLIC3D::DetectLabEdges(
	const double*				volumevec,
	const int&					width,
	const int&					height,
	const int&					depth,
	vector<double>&				edges)
{
	int sz = width * height * depth;

	edges.resize(sz, 0);

	for (int q = 1; q < depth - 1; q++)
	{
		for (int j = 1; j < height - 1; j++)
		{
			for (int k = 1; k < width - 1; k++)
			{
				auto i = q * height*width + j * width + k;
				//dx���x�����ϵ��ݶ�ƽ��
				double dx = (volumevec[i - 1] - volumevec[i + 1])*(volumevec[i - 1] - volumevec[i + 1]);
				//dy���y�����ϵ��ݶ�ƽ��
				double dy = (volumevec[i - width] - volumevec[i + width])*(volumevec[i - width] - volumevec[i + width]);
				//dz���z�����ϵ��ݶ�ƽ��
				double dz = (volumevec[i - width*height] - volumevec[i + width * height])*(volumevec[i - width * height] - volumevec[i + width * height]);

				//edges��ʾ���ص����ֵ�ݶȴ�С
				//edges[i] = (sqrt(dx) + sqrt(dy));
				edges[i] = (dx + dy + dz);
			}

		}
	}
}

//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC3D::GetLABXYSeeds_ForGivenK(
	//l a b x y��ʾ���ӵ����Ϣ
	vector<double>&				kseedintensity,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	vector<double>&				kseedsz,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width * m_height * m_depth;
	//step��ʾ���ӵ�֮��ľ���
	//double step = sqrt(double(sz) / double(K));
	double step = pow(double(sz) / double(K), 1.0/3);
	int T = step;
	//xoff��yoff��ʾ��ʼ�����ӵ�λ��
	int xoff = step / 2;
	int yoff = step / 2;
	int zoff = step / 2;

	int n(0); int r(0);


	for(auto z = 0;z<m_depth;z++)
	{
		int Z = z * step + zoff;
		if (Z > m_depth - 1) break;
		//r��һ��ѭ�����������Ϊż������������Ϊ�������Ա�֤��������ĳ�ʼ���ӵ㲻һ��
		if (!(r & 0x1)) r = 1;
		//else r = 0;
		for(auto y =0;y<m_height;y++)
		{
			int Y = y * step + yoff;
			if (Y > m_height - 1) break;
			
			for(auto x=0;x<m_width;x++)
			{
				//int X = x*step + xoff;//square grid
				int X = x * step + (xoff << (r & 0x1));//hex grid
				if (X > m_width - 1) break;

				int i = Z*m_height*m_width + Y * m_width + X;

				kseedintensity.push_back(m_volumevec[i]);
				kseedsx.push_back(X);
				kseedsy.push_back(Y);
				kseedsz.push_back(Z);
				n++;
			}
			r++;
		}
	}

	//�������ӵ㣬���ӵ��΢С�ƶ�
	if (perturbseeds)
	{
		PerturbSeeds(kseedintensity, kseedsx, kseedsy, kseedsz, edgemag);
	}
}

//===========================================================================
///	PerturbSeeds ���ӵ��΢С�ƶ� ȡ��Χ26������
//===========================================================================
void SLIC3D::PerturbSeeds(
	vector<double>&				kseedintensity,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	vector<double>&				kseedsz,
	const vector<double>&		edges)
{
	//const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };
	
	const int dx26[26] = { -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1 };
	const int dy26[26] = { -1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1 };
	const int dz26[26] = { -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };

	int numseeds = kseedintensity.size();

	for (int n = 0; n < numseeds; n++)
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oz = kseedsz[n];//original z

		int oind = oz*m_height*m_width + oy * m_width + ox;

		//�����ǰ�ڵ�Ĳ����Ը�С���򱣴�
		int storeind = oind;
		for (int i = 0; i < 26; i++)
		{
			int nx = ox + dx26[i];//new x
			int ny = oy + dy26[i];//new y
			int nz = oz + dz26[i];//new z

			if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height && nz>=0 && nz <m_depth)
			{
				int nind = nz*m_height*m_width + ny * m_width + nx;
				if (edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		//��̬�����µ����ӵ�
		if (storeind != oind)
		{
			kseedsz[n] = storeind / (m_width*m_height);
			auto buf = storeind % (m_width*m_height);
			kseedsx[n] = buf % m_width;
			kseedsy[n] = buf / m_width;

			kseedintensity[n] = m_volumevec[storeind];
		}
	}
}


//===========================================================================
///	SaveSuperpixelLabels
///
///	Save labels in raster scan order.
//===========================================================================
void SLIC3D::SaveSuperpixelLabels(
	const int*					labels,
	const int&					width,
	const int&					height,
	const int&					depth,
	const string&				filename)
{
	int sz = width * height * depth;

	ofstream outfile(filename.c_str(), ios::binary);
	for (int i = 0; i < sz; i++)
	{
		outfile.write((char*)(&labels[i]), sizeof(int));
	}
	outfile.close();
	std::cout << "Label file for the super-voxels has been saved." << std::endl;
}


//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void SLIC3D::DrawContoursAroundSegments(
	int*					ubuff,
	const int*				labels,
	const int&				width,
	const int&				height,
	const int&				depth,
	const int&				boundary_value)
{
	const int dx26[26] = { -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1 };
	const int dy26[26] = { -1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1 };
	const int dz26[26] = { -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };

	auto sz = width * height * depth;

	vector<bool> is_taken(sz, false);

	auto main_index(0);

	for(auto i=0;i<depth;i++)
	{
		for (auto j = 0; j < height; j++)
		{
			for (auto k = 0; k < width; k++)
			{
				auto np(0);
				for (auto idx = 0; idx < 26; idx++)
				{
					const auto x = k + dx26[idx];
					const auto y = j + dy26[idx];
					const auto z = i + dz26[idx];
					if ((x >= 0 && x < width) && (y >= 0 && y < height) && (z >= 0 && z < depth))
					{
						const auto index = z*height*width + y * width + x;

						if (!is_taken[index])//comment this to obtain internal contours
						{
							if (labels[main_index] != labels[index]) np++;
						}
					}
				}
				if (np > 1)//change to 3 or 5 for thinner lines
				{
					ubuff[main_index] = boundary_value;
					is_taken[main_index] = true;
				}
				else
				{
					ubuff[main_index] = m_volumevec[main_index];
				}
				main_index++;
			}
		}
	}
	std::cout << "Segment boundary array for the super-voxels has been executed." << std::endl;
}



void SLIC3D::SaveSegmentBouyndaries(
	const int*					ubuff,
	const int&					width,
	const int&					height,
	const int&					depth,
	const string&				filename)
{
	int sz = width * height * depth;

	ofstream outfile(filename.c_str(), ios::binary);
	for (int i = 0; i < sz; i++)
	{
		outfile.write((char*)(&ubuff[i]), sizeof(int));
	}
	outfile.close();
	std::cout << "Segment boundary file for the super-voxels has been saved." << std::endl;
}


void SLIC3D::SaveGradient(
	const string&				filename)
{
	
	ofstream outfile(filename.c_str(), ios::binary);
	for (int i = 0; i < edgemag.size(); i++)
	{
		float buf = edgemag[i];
		outfile.write((char*)(&buf), sizeof(float));
	}
	outfile.close();
	std::cout << "Gradient file for the super-voxels has been saved." << std::endl;
}

