#ifndef IMAGE_GRAPH_H
#define	IMAGE_GRAPH_H

#include <assert.h>
#include <vector>
#include <map>
#include <algorithm>

 /** \brief Represents an edge between two pixels in an image.
  * Each edge is characterized by a weight and the adjacent nodes.
  * \author David Stutz
  */
class ImageEdge {
public:
	/** \brief Default constructor.
	 */
	ImageEdge() : n(0), m(0), w(0) {};

	/** \brief Index of first node. */
	unsigned long int n;

	/** \brief Index of second node. */
	unsigned long int m;

	/** \brief Edge weight. */
	double w;

};

/** \brief Class for sorting edges according to weight.
 * \author David Stutz
 */
class ImageEdgeSorter {
public:
	/** \brief Compare to edges according to their weights.
	 * \param[in] g first edge
	 * \param[in] h second edge
	 * \return true if h.w greater than h.w
	 */
	inline bool operator()(const ImageEdge & g, const ImageEdge h) {
		return (h.w > g.w);
	}
};

/** \brief Represents a pixel in a video. Each pixel is represented by its
 * color which is needed to compute the weights between pixels.
 * \author David Stutz
 */

#define MAX_HISTOGRAM_SIZE 64

class ImageNode {
public:
	/** \brief Default constructor.
	 */
	ImageNode(std::vector<int>& voxels) :/* voxels(voxels),*/ l(0), n(1), id(0), max_w(0) {

	};

	ImageNode() : l(0), n(1), id(0), max_w(0) {
		voxels.clear();
		voxel_histogram.clear();
		voxel_histogram.resize(MAX_HISTOGRAM_SIZE, 0);
	};

	/** \brief voxel histogram array */
	std::vector<int> voxel_histogram;

	/** \brief voxel position [z,y,x] array in 1D */
	std::vector<int> voxels;


	/** \brief The label of the pixel. */
	unsigned long int l; // label, i.e. the index of the node this node belongs to

	/** \brief Size of node after merging with other nodes. */
	unsigned long int n;

	/** \brief Id of the node. */
	unsigned long int id;

	/** \brief Maximum weight in the region. */
	double max_w;

};

/** \brief Represents an image graph, consisting of one node per pixel which are
 * 4-connected.
 * \author David Stutz
 */
class ImageGraph {
public:
	/** \brief Default constructor.
	 */
	ImageGraph() {
		K = 0;
	};

	/** \brief Constructs an image graph with the given exact number of nodes.
	 * \param[in] N number of nodes to allocate
	 */
	ImageGraph(int N) {
		nodes = std::vector<ImageNode>(N);
		K = N;
	}

	/** \brief Assignment operator.
	 * \param[in] graph graph to copy
	 */
	void operator=(const ImageGraph & graph) {
		nodes = graph.nodes;
		edges = graph.edges;
		K = graph.K;
	}

	/** \brief Set the node of the given index.
	 * \param[in] n index of node
	 * \param[in] node
	 */
	void setNode(int n, ImageNode & node) {
		nodes[n] = node;
	}

	/** \brief Add a new node.
	 * \param[in] node
	 */
	void addNode(ImageNode & node) {
		nodes.push_back(node);
		K++;
	}

	/** \brief Add a new edge.
	 * \param[in] edge
	 */
	void addEdge(ImageEdge & edge) {
		edges.push_back(edge);
	}

	/** \param[in] Get the n-th node.
	 * \param[in] n node index
	 * \return node at index n
	 */
	ImageNode & getNode(int n) {
		assert(n >= 0 && n < static_cast<int>(nodes.size()));
		return nodes[n];
	}

	/** \brief Get the e-th edge in the current sorting.
	 * \param[in] e edge index
	 */
	ImageEdge & getEdge(int e) {
		assert(e >= 0 && e < static_cast<int>(edges.size()));
		return edges[e];
	}

	/** \brief Get the number of nodes.
	 * \return number of nodes
	 */
	int getNumNodes() {
		return nodes.size();
	}

	/** \brief Get the number of edges.
	 * \return number of edges
	 */
	int getNumEdges() {
		return edges.size();
	}
	/** \brief Get number of connected components.
	 * \return
	 */
	int getNumComponents() {
		return K;
	}

	/** \brief Sort the edges by weight.
	 */
	void sortEdges() {
		std::sort(edges.begin(), edges.end(), ImageEdgeSorter());
	}

	/** \brief When two nodes get merged, the first node is assigned the id of the second
	 * node as label. By traversing this labeling, the current component of each
	 * node (that is, pixel) can easily be identified and the label can be updated
	 * for efficiency.
	 * \param[in] node node to find component for
	 * \return node representing found component
	 */
	ImageNode & findNodeComponent(ImageNode & n) {

		// Get component of node n.
		int l = n.l;
		int id = n.id;

		while (l != id) {
			id = nodes[l].id;
			l = nodes[l].l;
		}

		ImageNode & S = nodes[l];
		assert(S.l == S.id);

		// Save latest component.
		n.l = S.id;

		return S;
	}

	/** \brief Merge two pixels (that is merge two nodes).
	 *
	 * Depending on the used "Distance", some lines may be commented out
	 * to speed up the algorithm.
	 *
	 * \param[in] S_n first node
	 * \param[in] S_m second node
	 * \param[in] e corresponding edge
	 */
	void merge(ImageNode & S_n, ImageNode & S_m, ImageEdge & e) {
		S_m.l = S_n.id;

		// Update cound.
		S_n.n += S_m.n;

		// Update maximum weight.
		S_n.max_w = std::max(std::max(S_n.max_w, S_m.max_w), e.w);

		// Update component count.
		K--;

		// Update voxel histogram

		for (auto i=0;i<S_m.voxel_histogram.size();i++)
		{
			S_n.voxel_histogram[i] += S_m.voxel_histogram[i];
			S_n.voxels.push_back(S_m.voxels[i]);
		}

		// for (auto voxel : S_m.voxels)
		// {
		// 	S_n.voxels.push_back(voxel);
		// }

		// Release S_m
		S_m.voxel_histogram.clear();
		S_m.voxels.clear();
	}

private:

	/** \brief Number of components. */
	int K;

	/** \brief All edges in this graph. */
	std::vector<ImageEdge> edges;

	/** \brief All nodes in this graph. */
	std::vector<ImageNode> nodes;

};

#endif	/* IMAGE_GRAPH_H */
