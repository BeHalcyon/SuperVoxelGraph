#ifndef GRAPH_SEGMENTATION_H
#define	GRAPH_SEGMENTATION_H

//#include <opencv2/opencv.hpp>
#include "image_graph.h"
#include <iostream>

#define RAND() ((float) std::rand() / (RAND_MAX))

 /** \brief Interface to be implemented by a concerete distance. The distance defines
  * how the weights between nodes in the image graph are computed. See the paper
  * by Felzenswalb and Huttenlocher for details. Essentially, derived classes
  * only need to overwrite the () operator.
  * \author David Stutz
  */
class GraphSegmentationDistance {
public:
	/** \brief Constructor.
	 */
	GraphSegmentationDistance() {};

	/** \brief Destructor.
	 */
	virtual ~GraphSegmentationDistance() {};

	/** \brief Compute the distance given 2 nodes.
	 * \param[in] n first node
	 * \param[in] m second node
	 */
	virtual float operator()(const ImageNode & n, const ImageNode & m) = 0;

};

/** \brief Manhatten (i.e. L1) distance.
 * \author David Stutz
 */
class GraphSegmentationManhattenRGB : public GraphSegmentationDistance {
public:
	/** \brief Constructor; sets normalization constant.
	 */
	GraphSegmentationManhattenRGB() {
		// Normalization.
		D = 255 + 255 + 255;
	}

	/** \brief Compute the distance given 2 nodes.
	 * \param[in] n first node
	 * \param[in] m second node
	 */
	virtual float operator()(const ImageNode & n, const ImageNode & m) {
		float dr = std::abs(n.max_w - m.max_w);
		//float dg = std::abs(n.g - m.g);
		//float db = std::abs(n.b - m.b);

		return (dr/* + dg + db*/);
	}

private:

	/** \brief Normalization term. */
	float D;

};

/** \brief Euclidean RGB distance.
 * \author David Stutz
 */
class GraphSegmentationEuclideanRGB : public GraphSegmentationDistance {
public:
	/** \brief Constructor; sets normalization constant.
	 */
	GraphSegmentationEuclideanRGB() {
		// Normalization.
		D = std::sqrt(255 * 255 + 255 * 255 + 255 * 255);
	}

	/** \brief Compute the distance given 2 nodes.
	 * \param[in] n first node
	 * \param[in] m second node
	 */
	virtual float operator()(const ImageNode & n, const ImageNode & m) {
		float dr = n.max_w - m.max_w;
		//float dg = n.g - m.g;
		//float db = n.b - m.b;

		return std::sqrt(dr*dr /*+ dg * dg + db * db*/);
	}

private:

	/** \brief Normalization term. */
	float D;

};

/** \brief The magic part of the graph segmentation, i.e. s given two nodes decide
 * whether to add an edge between them (i.e. merge the corresponding segments).
 * See the paper by Felzenswalb and Huttenlocher for details.
 * \author David Stutz
 */
class GraphSegmentationMagic {
public:
	/** \brief Constructor.
	 */
	GraphSegmentationMagic() {};

	/** \brief Decide whether to merge the two segments corresponding to the
	 * given nodes or not.
	 * \param[in] S_n node representing the first segment
	 * \param[in] S_m node representing the second segment
	 * \param[in] e the edge between the two segments
	 * \rturn true if merge
	 */
	virtual bool operator()(const ImageNode & S_n, const ImageNode & S_m,
		const ImageEdge & e) = 0;

};

/**
 * The original criterion employed by [2].
 */
class GraphSegmentationMagicThreshold : public GraphSegmentationMagic {
public:
	/** \brief Constructor; sets the threshold.
	 * \param[in] c the threshold to use
	 */
	GraphSegmentationMagicThreshold(float c) : c(c) {};

	/** \brief Decide whether to merge the two segments corresponding to the
	 * given nodes or not.
	 * \param[in] S_n node representing the first segment
	 * \param[in] S_m node representing the second segment
	 * \param[in] e the edge between the two segments
	 * \rturn true if merge
	 */
	virtual bool operator()(const ImageNode & S_n, const ImageNode & S_m,
		const ImageEdge & e) {


		float threshold = std::min(S_n.max_w + c / S_n.n, S_m.max_w + c / S_m.n);

		
		//std::cout << S_n.id << "\t" << S_m.id << "\t" << "\t" << e.w << "\t" <<
		//	S_n.max_w << "\t" << S_m.max_w << "\t" << 
		//	S_n.max_w + c / S_n.n << "\t" << S_m.max_w + c / S_m.n << "\t"  << std::endl;


		
		//std::cout << "threshold and max weight for s_n and s_m : \t" << threshold <<"\t" << S_n.max_w << "\t" << S_m.max_w<<"\t"<<e.w<< std::endl;
		return e.w < threshold;
	}

private:

	/** \brief T hreshold. */
	float c;

};

/** \brief Implementation of graph based image segmentation as described in the
 * paper by Felzenswalb and Huttenlocher.
 * \author David Stutz
 */
class GraphSegmentation {
public:
	/** \brief Default constructor; uses the Manhatten distance.
	 */
	GraphSegmentation() : //distance(new GraphSegmentationManhattenRGB()),
		magic(new GraphSegmentationMagicThreshold(1)) {

	};

	/** \brief Destructor.
	 */
	virtual ~GraphSegmentation() {};

	/** \brief Set the distance to use.
	 * \param[in] _distance pointer to a GraphSegmentationDistance to use
	 */
	void setDistance(GraphSegmentationDistance* _distance) {
		//distance = _distance;
	}

	/** \brief Set the magic part of graph segmentation.
	 * \param[in] _magix pointer to a GraphSegmentationMagic to use
	 */
	void setMagic(GraphSegmentationMagic* _magic) {
		magic = _magic;
	}

	/** \brief Build the graph nased on the image, i.e. compute the weights
	 * between pixels using the underlying distance.
	 * \param[in] image image to oversegment
	 */
	void buildGraph(const unsigned char * volume_data,
		const int&		dimension, 
		const int*		label,
		const int&		k_number, 
		const double*	gradient,
		const int &		width,
		const int &		height, 
		const int &		depth);

	/** \brief Oversegment the given graph.
	 */
	void oversegmentGraph();

	/** \brief Enforces the given minimum segment size.
	 * \pram[in] M minimum segment size in pixels
	 */
	void enforceMinimumSegmentSize(int M);

	/** \brief Derive labels from the produced oversegmentation.
	 * \pram[in] merge_label the 1D array need to allocate memory outside. 
	 */
	void deriveLabels(int * merged_label);

	/** \brief Save the merged labels from the produced oversegmentation.
	 */
	void saveMergeLabels(
		const int*					merged_labels,
		const int&					width,
		const int&					height,
		const int&					depth,
		const std::string&			filename);


protected:
	/** \brief Image depth */
	int D;

	/** \brief Image height. */
	int H;

	/** \brief Image width */
	int W;



	/** \brief The constructed and segmented image graph. */
	ImageGraph graph;

	/** \brief The underlying distance to use. */
	//GraphSegmentationDistance* distance;

	/** \brief The magic part of graph segmentation. */
	GraphSegmentationMagic* magic;

};

#endif	/* GRAPH_SEGMENTATION_H */
