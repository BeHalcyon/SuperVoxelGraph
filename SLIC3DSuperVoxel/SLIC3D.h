#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
using namespace std;




class SLIC3D
{
public:
	SLIC3D();
	virtual ~SLIC3D();


	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
	void PerformSLICO_ForGivenK(
		const unsigned int*			ubuff,
		const int					width,
		const int					height,
		const int					depth,
		int*						klabels,
		int&						numlabels,
		const int&					K,
		const double&				m);


	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
	void PerformSLICO_ForGivenK(
		const unsigned char*			ubuff,
		const int					width,
		const int					height,
		const int					depth,
		int*						klabels,
		int&						numlabels,
		const int&					K,
		const double&				m);


	//============================================================================
	// Save superpixel labels in a text file in raster scan order
	//============================================================================
	void SaveSuperpixelLabels(
		const int*					labels,
		const int&					width,
		const int&					height,
		const int&					depth,
		const string&				filename);


	//=================================================================================
	/// DrawContoursAroundSegments
	///
	/// Internal contour drawing option exists. One only needs to comment the if
	/// statement inside the loop that looks at neighbourhood.
	//=================================================================================
	void DrawContoursAroundSegments(
		int*					ubuff,
		const int*				labels,
		const int&				width,
		const int&				height,
		const int&				depth,
		const int&				boundary_value = 0x1ff);


	//============================================================================
	// Save superpixel labels in a text file in raster scan order
	//============================================================================
	void SaveSegmentBouyndaries(
		const int*				ubuff, 
		const int&				width,
		const int&				height,
		const int&				depth,
		const string&			filename);

	void SaveGradient(
		const string&			filename);

	vector<double>& getGradient()
	{
		return edgemag;
	}
private:
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(
		const double*				volumevec,
		const int&					width,
		const int&					height,
		const int&					depth,
		vector<double>&				edges);


	//============================================================================
	// Pick seeds for superpixels when number of superpixels is input.
	//============================================================================
	void GetLABXYSeeds_ForGivenK(
		vector<double>&				kseedintensity,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		vector<double>&				kseedsz,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edges);

	//===========================================================================
	///	PerturbSeeds 种子点的微小移动
	//===========================================================================
	void PerturbSeeds(
		vector<double>&				kseedintensity,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		vector<double>&				kseedsz,
		const vector<double>&		edges);


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
	void PerformSuperpixelSegmentation_VariableSandM(
		vector<double>&				kseedintensity,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		vector<double>&				kseedsz,
		int*						klabels,
		const int&					STEP,
		const int&					NUMITR);

	//===========================================================================
	///	EnforceLabelConnectivity
	///
	///		1. finding an adjacent label for each new component at the start
	///		2. if a certain component is too small, assigning the previously found
	///		    adjacent label to this component, and not incrementing the label.
	///		函数EnforceLabelConnectivity实现了连通区域的重新赋值，之前所有的label值是
	///		聚类中心的值，通过这个函数将其统一超像素的编号，从0开始一直到最后一个超像
	///		素，然后会统计每个超像素所占的像素，如果太小的话会进行合并
	//===========================================================================
	void EnforceLabelConnectivity(
		const int*					labels,//input labels that need to be corrected to remove stray labels //需要被纠正的数组
		const int&					width,
		const int&					height,
		const int&					depth,
		int*						nlabels,//new labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user

private:
	int										m_width;	//x
	int										m_height;	//y
	int										m_depth;	//z

	//double*									m_lvec;
	//double*									m_avec;
	//double*									m_bvec;

	double*									m_volumevec;

	//double**								m_lvecvec;
	//double**								m_avecvec;
	//double**								m_bvecvec;

	//edgemag表示图中的每个节点对应的梯度差
	vector<double>							edgemag;
};

