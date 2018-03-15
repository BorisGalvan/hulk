/* ---------------------------------------------------------------------
 * HULK - Hexahedra from Unique Location in Konvex Polyhedra
 *
 * Copyright (C) 2015,2016 - the authors
 *
 * Authors: Gunnar Jansen, University of Neuchatel, 2015
 *         Reza Sohrabi, University of Neuchatel, 2015
 *
 */

#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/base/std_cxx11/bind.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/filtered_iterator.h>
#include <boost/any.hpp>
#include <boost/foreach.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>
#include <iostream>

namespace hulk
{
using namespace dealii;

const double PLANE_THICKNESS_EPSILON = 1e-3;
const int POINT_BEHIND_PLANE = -1;
const int POINT_IN_FRONT_OF_PLANE = 1;
const int POINT_ON_PLANE = 0;
const int POLYGON_BEHIND_PLANE = -1;
const int POLYGON_IN_FRONT_OF_PLANE = 1;
const int POLYGON_STRADDLING_PLANE = 0;
const int POLYGON_COPLANAR_WITH_PLANE = 2;

const int COPLANAR_WITH_PLANE = 2;
const int IN_FRONT_OF_PLANE = 1;
const int BEHIND_PLANE = -1;
const int STRADDLING_PLANE = 0;

const int POINT_INSIDE = 1;
const int POINT_OUTSIDE = 0;

const double EPSILON = 0;

namespace Parameters
{
struct MiscParam
{
	/** \brief Name of the output file, either relative to the present */
	std::string outfile;
	/** \brief Number of refinement cycles */
	unsigned int max_cycle;
	/** \brief Max Number of cells */
	unsigned int max_cells;
	/** \brief Number of input STL */
	unsigned int num_stl;
	/** \brief Number of input txt */
	unsigned int num_txt;
	/** \brief Number of input faults */
	unsigned int num_faults;
	/** \brief Coarsening Iterations */
	unsigned int coarsening_iterations;
	/** \brief Normal direction of input STL */
	bool normal_outside;
	/** \brief Use only global refinement */
	bool refine_global;
	/** \brief Number of initial refinement cycles */
	unsigned int initial_refinement;
	/** \brief Analyse the meshing in each cycle */
	bool analysis_mode;


	/** \brief Max x coodirnate of bounding box */
	double max_x;
	/** \brief Max x coodirnate of bounding box */
	double max_y;
	/** \brief Max x coodirnate of bounding box */
	double max_z;
	/** \brief Max x coodirnate of bounding box */
	double min_x;
	/** \brief Max x coodirnate of bounding box */
	double min_y;
	/** \brief Max x coodirnate of bounding box */
	double min_z;

	/** \brief Vector for input file names */
	std::vector<std::string> infiles;
	/** \brief Vector for input txt file names */
	std::vector<std::string> infilestxt;
	/** \brief Vector for corresponding material ids */
	std::vector<unsigned int> materialids;

	/** \brief Constructor for the AllParameters struct
	 *
	 * The constructor takes a pointer to the parameter file as input arguments.
	 */
	MiscParam(const std::string &input_file);

	/** \brief Declare miscellaneous parameters for the ParameterHandler
	 *
	 */
	static void
	declare_parameters(ParameterHandler &prm);

	/** \brief Parse miscellaneous parameters for the ParameterHandler
	 *
	 */
	void
	parse_parameters(ParameterHandler &prm);
};

void MiscParam::declare_parameters(ParameterHandler &prm)
{
	prm.declare_entry ("Outfile",
			"solution",
			Patterns::Anything(),
			"Name of the output file, either relative to the present"
			"path or absolute");
	prm.declare_entry ("Cycles",
			"6",
			Patterns::Integer(),
			"Number of refinement cycles");
	prm.declare_entry ("Max Cells",
			"1000000",
			Patterns::Integer(),
			"Maximum number of mesh cells");
	prm.declare_entry ("NStl",
			"1",
			Patterns::Integer (0, 50),
			"Number of input STL"); //Make the restriction here.
	prm.declare_entry ("Ntxt",
			"1",
			Patterns::Integer (0, 50),
			"Number of input txt"); //Make the restriction here.
	prm.declare_entry ("Nfaults",
			"1",
			Patterns::Integer (0, 5000),
			"Number of input faults"); //Make the restriction here.
	prm.declare_entry ("Coarsening",
			"2",
			Patterns::Integer(),
			"Number of coarsening cycles");
	prm.declare_entry ("NormalDirectionOutside",
			"false",
			Patterns::Bool(),
			"Is the direction of the normals outside?");
	prm.declare_entry ("RefineGlobalOnly",
			"false",
			Patterns::Bool(),
			"Should only global refinement be used?");
	prm.declare_entry ("InitialRefinement",
			"2",
			Patterns::Integer(),
			"Amount of initial refinement (2 => 512 cells)");
	prm.declare_entry ("AnalysisMode",
			"false",
			Patterns::Bool(),
			"Perform a mesh analysis in each cycle");

	prm.enter_subsection("BoundingBox");
	prm.declare_entry("Xmax","13000.0",
			Patterns::Double(),
			"maximum X coordinate of the bounding box");
	prm.declare_entry("Ymax","9000.0",
			Patterns::Double(),
			"maximum Y coordinate of the bounding box");
	prm.declare_entry("Zmax","-6000.0",
			Patterns::Double(),
			"maximum Z coordinate of the bounding box");
	prm.declare_entry("Xmin","0.0",
			Patterns::Double(),
			"minimum X coordinate of the bounding box");
	prm.declare_entry("Ymin","0.0",
			Patterns::Double(),
			"minimum Y coordinate of the bounding box");
	prm.declare_entry("Zmin","0.0",
			Patterns::Double(),
			"minimum Z coordinate of the bounding box");
	prm.leave_subsection();

	for (int i=0; i<50; ++i)
	{
		std::string sectionname("STL_IN-" +
				boost::lexical_cast<std::string>(i));
		prm.enter_subsection(sectionname);
		prm.declare_entry("Input File", "Sediments-1.stl",
				Patterns::Anything(),
				"Name of the file to read in.");
		prm.declare_entry("Id", "1",
				Patterns::Integer (1, 255),
				"Material ID");
		prm.leave_subsection();
	}

	for (int i=0; i<50; ++i)
	{
		std::string sectionname("TXT_IN-" +
				boost::lexical_cast<std::string>(i));
		prm.enter_subsection(sectionname);
		prm.declare_entry("Input File", "Sediments-1.stl",
				Patterns::Anything(),
				"Name of the file to read in.");
		prm.declare_entry("Id", "1",
				Patterns::Integer (1, 255),
				"Material ID");
		prm.declare_entry("Nfaults", "1",
				Patterns::Integer (1, 255),
				"Number of Faults");
		prm.leave_subsection();
	}
}

void MiscParam::parse_parameters(ParameterHandler &prm)
{
	outfile = prm.get ("Outfile");
	max_cycle = prm.get_integer("Cycles");
	max_cells = prm.get_integer("Max Cells");
	num_stl = prm.get_integer("NStl");
	num_txt = prm.get_integer("Ntxt");
	num_faults = prm.get_integer("Nfaults");
	coarsening_iterations = prm.get_integer("Coarsening");
	normal_outside = prm.get_bool("NormalDirectionOutside");
	refine_global = prm.get_bool("RefineGlobalOnly");
	initial_refinement = prm.get_integer("InitialRefinement");
	analysis_mode = prm.get_bool("AnalysisMode");

	prm.enter_subsection("BoundingBox");
	max_x = prm.get_double("Xmax");
	max_y = prm.get_double("Ymax");
	max_z = prm.get_double("Zmax");
	min_x = prm.get_double("Xmin");
	min_y = prm.get_double("Ymin");
	min_z = prm.get_double("Zmin");
	prm.leave_subsection();

	for (unsigned int j=0; j<num_stl; ++j)
	{
		std::string sectionname("STL_IN-" +
				boost::lexical_cast<std::string>(j));
		prm.enter_subsection(sectionname);
		std::string ifile = prm.get("Input File");
		infiles.push_back(ifile);
		int id = prm.get_integer("Id");
		materialids.push_back(id);
		prm.leave_subsection();
	}

	for (unsigned int j=0; j<num_txt; ++j)
	{
		std::string sectionname("TXT_IN-" +
				boost::lexical_cast<std::string>(j));
		prm.enter_subsection(sectionname);
		std::string ifile = prm.get("Input File");
		infilestxt.push_back(ifile);
		int id = prm.get_integer("Id");
		for (unsigned int jj=0; jj<num_faults; jj++)
		{
			materialids.push_back(id+jj);
		}
		prm.leave_subsection();
	}
}

MiscParam::MiscParam(const std::string &input_file)
{
	ParameterHandler prm;
	declare_parameters(prm);
	prm.read_input(input_file);
	parse_parameters(prm);
}
}


class Polygon
{
public:
	Polygon();
	Polygon(int numVerts, Point<3> *points);

	int getNumVertices();
	Point<3> getVertex(int iVert);

	bool querySelected();
	void setSelected();
	void clearSelected();

private:
	const int numVerts;
	std::vector<Point<3>> vertices;
	bool selected = false;
};



Polygon::Polygon(int numVerts, Point<3> *points)
:
																		  numVerts(numVerts)
{
	for (int i=0; i<numVerts; i++){
		vertices.push_back(points[i]);
	}
}



int Polygon::getNumVertices(){
	return numVerts;
}



Point<3> Polygon::getVertex(int iVert){
	return vertices[iVert];
}



bool Polygon::querySelected(){
	return selected;
}



void Polygon::setSelected(){
	selected = true;
}


void Polygon::clearSelected(){
	selected = false;
}


struct Plane {
	Tensor<1,3> n; // Plane normal. Points x on the plane satisfy Dot(n,x) = d
	double d; // d = dot(n,p) for a given point p on the plane
};



// Given three noncollinear points (ordered ccw), compute plane equation
Plane ComputePlane(Tensor<1,3> a, Tensor<1,3> b, Tensor<1,3> c)
{
	Plane p;
	Tensor<1,3> ba = b-a;
	Tensor<1,3> ca = c-a;

	cross_product(p.n, ba, ca);
	p.n = p.n / p.n.norm();
	p.d = p.n*a;
	return p;
}



Plane GetPlaneFromPolygon(Polygon *poly)
{
	Plane p;
	Tensor<1,3> a = poly->getVertex(0);
	Tensor<1,3> b = poly->getVertex(1);
	Tensor<1,3> c = poly->getVertex(2);
	p = ComputePlane(a,b,c);
	return p;
}



Point<3> IntersectEdgeAgainstPlane(Point<3> a, Point<3> b, Plane p)
																		{
	Tensor<1,3> ab = b - a;
	double t = (p.d - p.n*a) / (p.n*ab);
	// If t in [0..1] compute and return intersection point
	Point<3> q;
	if (t >= 0.0f && t <= 1.0f) {
		q = a + t * ab;
	}
	return q;
																		}



// Classify point p to a plane thickened by a given thickness epsilon
int ClassifyPointToPlane(Point<3> p, Plane plane)
{
	// Compute signed distance of point from plane
	double dist = plane.n*p - plane.d;
	// Classify p based on the signed distance
	if (dist > PLANE_THICKNESS_EPSILON)
		return POINT_IN_FRONT_OF_PLANE;

	if (dist < -PLANE_THICKNESS_EPSILON)
		return POINT_BEHIND_PLANE;

	return POINT_ON_PLANE;
}


// Return value specifying whether the polygon ‘poly’ lies in front of,
// behind of, on, or straddles the plane ‘plane’
int ClassifyPolygonToPlane(Polygon *poly, Plane plane)
{
	// Loop over all polygon vertices and count how many vertices
	// lie in front of and how many lie behind of the thickened plane
	int numInFront = 0, numBehind = 0;
	int numVerts = poly->getNumVertices();
	for (int i = 0; i < numVerts; i++) {
		Point<3> p = poly->getVertex(i);
		switch (ClassifyPointToPlane(p, plane)) {
		case POINT_IN_FRONT_OF_PLANE:
			numInFront++;
			break;
		case POINT_BEHIND_PLANE:
			numBehind++;
			break;
		}
	}
	// If vertices on both sides of the plane, the polygon is straddling
	if (numBehind != 0 && numInFront != 0)
		return POLYGON_STRADDLING_PLANE;
	// If one or more vertices in front of the plane and no vertices behind
	// the plane, the polygon lies in front of the plane
	if (numInFront != 0)
		return POLYGON_IN_FRONT_OF_PLANE;
	// Ditto, the polygon lies behind the plane if no vertices in front of
	// the plane, and one or more vertices behind the plane
	if (numBehind != 0)
		return POLYGON_BEHIND_PLANE;
	// All vertices lie on the plane so the polygon is coplanar with the plane
	return POLYGON_COPLANAR_WITH_PLANE;
}



void SplitPolygon(Polygon &poly, Plane plane, Polygon **frontPoly, Polygon **backPoly)
{
	int numFront = 0, numBack = 0;
	Point<3> frontVerts[100], backVerts[100];

	// Test all edges (a, b) starting with edge from last to first vertex
	int numVerts = poly.getNumVertices();
	Point<3> a = poly.getVertex(numVerts - 1);

	int aSide = ClassifyPointToPlane(a, plane);
	// Loop over all edges given by vertex pair (n - 1, n)
	for (int n = 0; n < numVerts; n++) {
		Point<3> b = poly.getVertex(n);

		int bSide = ClassifyPointToPlane(b, plane);
		if (bSide == POINT_IN_FRONT_OF_PLANE) {
			if (aSide == POINT_BEHIND_PLANE) {
				// Edge (a, b) straddles, output intersection point to both sides
				Point<3> i = IntersectEdgeAgainstPlane(b, a, plane);
				assert(ClassifyPointToPlane(i, plane) == POINT_ON_PLANE);
				frontVerts[numFront++] = backVerts[numBack++] = i;
			}
			// In all three cases, output b to the front side
			frontVerts[numFront++] = b;
		} else if (bSide == POINT_BEHIND_PLANE) {
			if (aSide == POINT_IN_FRONT_OF_PLANE) {
				// Edge (a, b) straddles plane, output intersection point
				Point<3> i = IntersectEdgeAgainstPlane(a, b, plane);
				assert(ClassifyPointToPlane(i, plane) == POINT_ON_PLANE);
				frontVerts[numFront++] = backVerts[numBack++] = i;
			} else if (aSide == POINT_ON_PLANE) {
				// Output a when edge (a, b) goes from ‘on’ to ‘behind’ plane
				backVerts[numBack++] = a;
			}
			// In all three cases, output b to the back side
			backVerts[numBack++] = b;
		} else {
			// b is on the plane. In all three cases output b to the front side
			frontVerts[numFront++] = b;
			// In one case, also output b to back side
			if (aSide == POINT_BEHIND_PLANE)
				backVerts[numBack++] = b;
		}
		// Keep b as the starting point of the next edge
		a = b;
		aSide = bSide;
	}
	// Create (and return) two new polygons from the two vertex lists
	*frontPoly = new Polygon(numFront, frontVerts);
	*backPoly = new Polygon(numBack, backVerts);
}



class BSPNode
{
public:
	BSPNode();
	BSPNode(bool front);
	BSPNode(BSPNode *frontTree, BSPNode *backTree, Plane plane, Tensor<1,3> center);

	BSPNode* child[2];
	Plane plane;
	Tensor<1,3> center;

	bool IsLeaf();
	bool IsSolid();
private:
	const bool leaf = true;
	bool solid = false;
};



bool BSPNode::IsLeaf(){
	return leaf;
}



bool BSPNode::IsSolid(){
	return solid;
}


BSPNode::BSPNode(bool front)
:
																		  leaf(true),
																		  solid(true)
{
	if (front)
		solid = false;
	else
		solid = true;
}



BSPNode::BSPNode(BSPNode* frontTree, BSPNode *backTree, Plane plane, Tensor<1,3> center)
:
																		  child{frontTree,backTree},
																		  plane(plane),
																		  center(center),
																		  leaf(false),
																		  solid(false)
																		  {
																		  }


																		  // Constructs BSP tree from an input vector of polygons. Pass ‘depth’ as 0 on entry
																		  BSPNode* BuildBSPTree(std::vector<Polygon *> &polygons, int depth, bool front)
																		  {
																			  // Return leaf if tree is empty or maximum depth is reached
																			  if (polygons.empty())
																				  return new BSPNode(front);

																			  // Get number of polygons in the input vector
																			  int numPolygons = polygons.size();


																			  // Select best possible partitioning plane based on the input geometry
																			  Plane splitPlane = GetPlaneFromPolygon(polygons[0]);

																			  std::vector<Polygon *> frontList, backList;
																			  std::vector<Polygon *> splitList;

																			  // Test each polygon against the dividing plane, adding them
																			  // to the front list, back list, or both, as appropriate
																			  for (int i = 1; i < numPolygons; i++) {

																				  Polygon *poly = polygons[i], *frontPart, *backPart;
																				  switch (ClassifyPolygonToPlane(poly, splitPlane)) {
																				  case COPLANAR_WITH_PLANE:
																					  //Do not do anything in this case.
																					  break;
																				  case BEHIND_PLANE:
																					  backList.push_back(poly);
																					  break;
																				  case IN_FRONT_OF_PLANE:
																					  frontList.push_back(poly);
																					  break;
																				  case STRADDLING_PLANE:
																					  // Split polygon to plane and send a part to each side of the plane
																					  SplitPolygon(*poly, splitPlane, &frontPart, &backPart);
																					  frontList.push_back(frontPart);
																					  backList.push_back(backPart);
																					  splitList.push_back(frontPart);
																					  splitList.push_back(backPart);
																					  break;
																				  }
																			  }

																			  // Recursively build child subtrees and return new tree root combining them
																			  BSPNode *backTree = BuildBSPTree(backList, depth + 1, false);
																			  BSPNode *frontTree = BuildBSPTree(frontList, depth + 1, true);

																			  //TODO: A possible memory leak is found by valgrind because we delete the memory that also frontList and backList
																			  //      are pointing to. I think this should not be a "real" problem. However recoding would make it safer!
																			  std::for_each(splitList.begin(), splitList.end(), std::default_delete<Polygon>());

																			  Point<3> Tp1 = polygons[0]->getVertex(0);
																			  Point<3> Tp2 = polygons[0]->getVertex(1);
																			  Point<3> Tp3 = polygons[0]->getVertex(2);

																			  Tensor<1,3> C = (Tp1+Tp2+Tp3)/3 ;

																			  return new BSPNode(frontTree, backTree, splitPlane, C);
																		  }

																		  // Find minimum and maximum points
																		  void FindMinMaxPoints(std::vector<std::vector<double>> &minmaxvec,std::vector<Polygon *> &polygons)
																		  {

																		  }

																		  int PointInSolidSpace(BSPNode *node, Point<3> p)
																		  {
																			  while (!node->IsLeaf()) {
																				  // Compute distance of point to dividing plane
																				  double dist = node->plane.n*p - node->plane.d;
																				  // Traverse front of tree when point in front of plane, else back of tree
																				  node = node->child[dist <= EPSILON];
																			  }
																			  // Now at a leaf, inside/outside status determined by solid flag
																			  return node->IsSolid() ? POINT_INSIDE : POINT_OUTSIDE;
																		  }



																		  Tensor<1,3> DistancePointToTree(BSPNode *node, Point<3> p)
																		{
																			  double minDist = 9999;
																			  Tensor<1,3> minDistVec;
																			  double k;
																			  while (!node->IsLeaf()) {
																				  // Compute distance of point to dividing plane
																				  double dist = node->plane.n*p - node->plane.d;

																				  Tensor<1,3> center_dist = (node->center-p);
																				  double weight = std::fabs(center_dist.norm());

																				  if( fabs(dist)*weight < fabs(minDist)){
																					  minDist = dist*weight;
																					  k = (node->plane.n[0] * p[0] + node->plane.n[1] * p[1] + node->plane.n[2] * p[2] - node->plane.d) / (node->plane.n[0]*node->plane.n[0] + node->plane.n[1]*node->plane.n[1] + node->plane.n[2]*node->plane.n[2]);
																					  minDistVec = k*node->plane.n;
																				  }

																				  // Traverse front of tree when point in front of plane, else back of tree
																				  node = node->child[dist <= EPSILON];
																			  }
																			  // Now at a leaf, inside/outside status determined by solid flag
																			  return minDistVec;
																		}



																		  Tensor<1,3> DistancePointToGeometry(std::vector<Polygon*>polygons, Point<3> p)
																		{
																			  double minDist = 9999;
																			  Tensor<1,3> minDistVec;
																			  minDistVec = 0;

																			  // Get number of polygons in the input vector
																			  int numPolygons = polygons.size();

																			  // Test each polygon against the dividing plane, adding them
																			  // to the front list, back list, or both, as appropriate
																			  for (int i = 0; i < numPolygons; i++) {

																				  Polygon *poly = polygons[i];

																				  Point<3> Tp1 = poly->getVertex(0);
																				  Point<3> Tp2 = poly->getVertex(1);
																				  Point<3> Tp3 = poly->getVertex(2);

																				  Tensor<1,3> C = (Tp1+Tp2+Tp3)/3 ;
																				  //Plane plane = GetPlaneFromPolygon(poly);

																				  // Compute distance of point to dividing plane
																				  Tensor<1,3> pC = C - p;
																				  double dist = pC.norm();

																				  if( fabs(dist) < fabs(minDist)){
																					  minDist = dist;
																					  minDistVec = pC;
																				  }

																			  }
																			  // Now at a leaf, inside/outside status determined by solid flag
																			  return minDistVec;
																		}



																		  Point<3> populate_point(char* facet)
																		{
																			  Point<3> p;

																			  char f1[4] = {facet[0],
																					  facet[1],facet[2],facet[3]};

																			  char f2[4] = {facet[4],
																					  facet[5],facet[6],facet[7]};

																			  char f3[4] = {facet[8],
																					  facet[9],facet[10],facet[11]};

																			  float xx = *((float*) f1 );
																			  float yy = *((float*) f2 );
																			  float zz = *((float*) f3 );

																			  p(0) = double(xx);
																			  p(1) = double(yy);
																			  p(2) = double(zz);

																			  return p;
																		}



																		  template <int dim>
																		  void set_boundary_ids_box (Triangulation<dim> &triangulation)
																		  {
																			  for (typename Triangulation<dim>::active_cell_iterator
																					  cell=triangulation.begin_active();
																					  cell!=triangulation.end(); ++cell)
																				  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
																				  {
																					  if (cell->face(f)->at_boundary())
																					  {
																						  const Point<dim> face_center = cell->face(f)->center();
																						  if (face_center[dim-1] == 0)
																							  cell->face(f)->set_boundary_indicator (1);
																						  else if (face_center[dim-1] == 4)
																							  cell->face(f)->set_boundary_indicator (2);

																						  if (face_center[0] == 0)
																							  cell->face(f)->set_boundary_indicator (3);
																						  else if (face_center[0] == 4)
																							  cell->face(f)->set_boundary_indicator (4);

																					  }
																				  }
																		  }

																		  void deleteBSP(BSPNode *root){
																			  if(root->child[0]==NULL && root->child[1]==NULL)  //leaf node, delete it!!
																				  free(root);
																			  delete(root->child[0]);
																			  delete(root->child[1]);
																		  }

																		  double compute_area(Tensor<1,3> a,Tensor<1,3> b,Tensor<1,3> c,Tensor<1,3> d)
																		  {
																			  Tensor<1,3> result;
																			  cross_product(result,(c-a),(d-b));

																			  return 0.5*result.norm();
																		  }



																		  template <typename T> int sgn(T val) {
																			  return (T(0) < val) - (val < T(0));
																		  }



																		  template <int dim>
																		  class STLRefine
																		  {
																		  public:
																			  STLRefine (const std::string &parameter_file);
																			  ~STLRefine();
																			  void run_refinement();

																		  private:
																			  void read_input_stl (int i);
																			  void read_stl(std::string fname, std::vector<Polygon *> &v);
																			  void read_input_txt (int i);
																			  void read_txt(std::string fname, std::vector<std::vector<Polygon *>> &v);
																			  void build_trees(int i);
																			  void min_max_points(int i);
																			  void move_mesh();
																			  void analyse_mesh();
																			  void collect_distorted_coarse_cells();
																			  void create_box_grid (Triangulation<dim> &triangulation) const;
																			  void set_material_ids (Triangulation<dim> &triangulation) const;
																			  bool is_inside(Point<3> p,int id) const;
																			  bool is_inside(Point<3> p, BSPNode* node) const;
																			  void output_results (const unsigned int cycle) const;

																			  MPI_Comm                                  mpi_communicator;

																			  Parameters::MiscParam parameters;

																			  Triangulation<dim>   triangulation;


																			  ConditionalOStream                        pcout;
																			  TimerOutput                               computing_timer;

																			  std::vector<std::vector<Polygon *>> polyvec;
																			  std::vector<BSPNode*> bspvec;
																			  std::vector<std::vector<double>> minmaxvec;

																			  std::map<int,int> bspmap;
																			  std::map<int,int> polymap;
																		  };

																		  template <int dim>
																		  STLRefine<dim>::~STLRefine()
																		  {
																			  for(auto it = bspvec.begin(); it!=bspvec.end(); ++it){
																				  deleteBSP(*it);
																			  }
																		  }

																		  template <int dim>
																		  STLRefine<dim>::STLRefine (const std::string &parameter_file)
																		  :
																		  mpi_communicator (MPI_COMM_WORLD),
																		  parameters(parameter_file),
																		  pcout (std::cout,
																				  (Utilities::MPI::this_mpi_process(mpi_communicator)
																		  == 0)),
																		  computing_timer (mpi_communicator,
																				  pcout,
																				  TimerOutput::summary,
																				  TimerOutput::wall_times),
																				  polyvec(parameters.num_stl),
																				  bspvec(parameters.num_stl+parameters.num_faults)
																				  {}



																		  template <int dim>
																		  void STLRefine<dim>::read_input_stl (int i)
																		  {
																			  std::string fname = parameters.infiles[i];
																			  read_stl(fname,polyvec[i]);
																		  }


																		  template <int dim>
																		  void STLRefine<dim>::read_stl(std::string fname, std::vector<Polygon*> &v)
																		  {
																			  std::cout << "Reading file " << fname << std::endl;
																			  std::ifstream myFile (
																					  fname.c_str(), std::ios::in | std::ios::binary);

																			  char header_info[80] = "";
																			  unsigned long nTriLong = 0;

																			  //read 80 byte header
																			  if (myFile) {
																				  myFile.read (header_info, 80);
																				  std::cout <<"header: " << header_info << std::endl;
																			  }
																			  else{
																				  std::cout << "Error reading file." << std::endl;
																			  }

																			  //read 4-byte ulong
																			  if (myFile) {
																				  uint32_t toRestore=0;
																				  myFile.read((char*)&toRestore,sizeof(toRestore));
																			  }
																			  else{
																				  std::cout << "Error reading file." << std::endl;
																				  exit(0);
																			  }

																			  //now read in all the triangles
																			  while(myFile){
																				  nTriLong += 1;
																				  char facet[50];

																				  if (myFile)
																				  {
																					  //read one 50-byte triangle
																					  myFile.read (facet, 50);

																					  //populate each point of the triangle
																					  //Point<3> nn = populate_point(facet);
																					  //facet + 12 skips the triangle's unit normal
																					  Point<3> p[3];
																					  //This is for outward pointing normals in input file
																					  if (parameters.normal_outside){
																						  p[0] = populate_point(facet+12);
																						  p[1] = populate_point(facet+24);
																						  p[2] = populate_point(facet+36);
																					  }
																					  else{
																						  //This is for inward pointing normals in input file (e.g. Geomodeller)
																						  p[0] = populate_point(facet+12);
																						  p[2] = populate_point(facet+24);
																						  p[1] = populate_point(facet+36);
																					  }
																					  if (p[0] == p[1] || p[0] == p[2] || p[1] == p[2] )
																					  {
																						  std::cout << "They are the same." << std::endl;
																						  exit(0);
																					  }
																					  //add a new triangle to the array
																					  v.push_back( new Polygon(3,p) );

																				  }
																			  }
																			  myFile.close();
																			  std::cout <<"n Tri: " << nTriLong << std::endl;
																			  return;
																		  }

																		  template <int dim>
																		  void STLRefine<dim>::read_input_txt (int i)
																		  {
																			  std::string fname = parameters.infilestxt[i];

																			  read_txt(fname,polyvec);
																			  //std::cout<<"from the other side"<<std::endl;
																		  }


																		  template <int dim>
																		  void STLRefine<dim>::read_txt(std::string fname, std::vector<std::vector<Polygon *>> &v)
																		  {

																			  unsigned int ii=0,jj=0,kk=0;
																			  std::vector<Polygon *> vv;
																			  typedef boost::tokenizer< boost::char_separator<char> > Tokenizer;
																			  std::vector< std::string > vec;
																			  boost::char_separator<char> sep{" "};
																			  std::string line;
																			  Point<3> PP[4];

																			  std::ifstream in(fname.c_str());
																			  if (in.is_open()){
																				  std::cout << "Reading file " << fname << std::endl;
																			  }
																			  if (!in.is_open()){
																				  std::cerr<<"unable to open file"<<std::endl;
																				  exit(1);
																			  }


																			  while (getline(in,line))
																			  {

																				  Tokenizer tok{line,sep};
																				  vec.assign(tok.begin(),tok.end());

																				  if(vec[0].find("P")){


																					  kk=ii%24;
																					  jj=ii%4;

																					  Point<3> p1(boost::lexical_cast<double>(vec[0]),boost::lexical_cast<double>(vec[1]),boost::lexical_cast<double>(vec[2]));

																					  PP[jj]=p1;

																					  ii++;

																					  if(jj==3){
																						  vv.push_back(new Polygon(4,PP));
																					  }
																					  if(kk==23){

																						  v.push_back(vv);
																						  vv.clear();

																					  }


																				  }
																			  }


																			  return;
																		  }

																		  /* template <int dim>
						  void STLRefine<dim>::read_txt(std::string fname, std::vector<std::vector<Polygon *>> &v)
						  {

							  unsigned int ii=0,jj=0,kk=0,iii=0;
							  std::vector<Polygon *> vv;
							  typedef boost::tokenizer< boost::char_separator<char> > Tokenizer;
							  std::vector< std::string > vec;
							  boost::char_separator<char> sep{" "};
							  std::string line;
							  Point<3> PP[3];

							  std::ifstream in(fname.c_str());
							  if (in.is_open()){
								  std::cout << "Reading file " << fname << std::endl;
							  }
							  if (!in.is_open()){
								  std::cerr<<"unable to open file"<<std::endl;
								  exit(1);
							  }


							  while (getline(in,line))
							  {

								  Tokenizer tok{line,sep};
								  vec.assign(tok.begin(),tok.end());

								  if(vec[0].find("P")){


									  kk=ii%36;
									  jj=ii%3;

									  Point<3> p1(boost::lexical_cast<double>(vec[0]),boost::lexical_cast<double>(vec[1]),boost::lexical_cast<double>(vec[2]));


									  std::cout<<vec[0]<<"  "<<vec[1]<<"  "<<vec[2]<<"  "<<std::endl;
									  PP[jj]=p1;

									  ii++;
									  if(jj==2){

										  vv.push_back(new Polygon(3,PP));
										  iii++;

									  }
									  if(kk==35){

										  v.push_back(vv);
										  vv.clear();

									  }


								  }
							  }


							  return;
						  }*/

																		  template <int dim>
																		  void STLRefine<dim>::build_trees (int i)
																		  {
																			  //std::cout<<" num Polygones "<<polyvec[i].size()<<std::endl;
																			  bspvec[i]= BuildBSPTree(polyvec[i], 0,0);

																		  }

																		  template <int dim>
																		  void STLRefine<dim>::min_max_points(int i)
																		  {
																			  //std::cout<<" num Polygones "<<polyvec[i].size()<<std::endl;
																			  FindMinMaxPoints(minmaxvec,polyvec[i]);

																		  }

																		  template <int dim>
																		  void STLRefine<dim>::move_mesh ()
																		  {

																			  double almost_infinite_length = 0;
																			  for (typename Triangulation<dim>::cell_iterator
																					  cell=triangulation.begin(0); cell!=triangulation.end(0); ++cell)
																				  almost_infinite_length += cell->diameter();

																			  std::vector<double> minimal_length (triangulation.n_vertices(),
																					  almost_infinite_length);

																			  // also note if a vertex is at the boundary
																			  std::vector<bool>   at_boundary (triangulation.n_vertices(), false);
																			  for (typename Triangulation<dim>::active_cell_iterator
																					  cell=triangulation.begin_active(); cell!=triangulation.end(); ++cell)
																				  if (cell->is_locally_owned())
																				  {
																					  if (dim>1)
																					  {
																						  for (unsigned int i=0; i<GeometryInfo<dim>::lines_per_cell; ++i)
																						  {
																							  const typename Triangulation<dim>::line_iterator line
																							  = cell->line(i);

																							  minimal_length[line->vertex_index(0)]
																							                 = std::min(line->diameter(),
																							                		 minimal_length[line->vertex_index(0)]);
																							  minimal_length[line->vertex_index(1)]
																							                 = std::min(line->diameter(),
																							                		 minimal_length[line->vertex_index(1)]);
																						  }
																					  }
																				  }

																			  std::vector<bool> vertex_touched (triangulation.n_vertices(),
																					  false);

																			  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active() , endc = triangulation.end();
																			  for (; cell != endc; ++cell)
																				  if (cell->is_locally_owned())
																				  {

																					  int id = cell->material_id();
																					  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id;
																					  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id;
																					  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id;
																					  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id;
																					  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id;
																					  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id;

																					  if(((id != neighbor_id_x) ||
																							  (id != neighbor_id_xx)||
																							  (id != neighbor_id_y)||
																							  (id != neighbor_id_yy)||
																							  (id != neighbor_id_z)||
																							  (id != neighbor_id_zz)))
																					  {
																						  if ( id == 0)
																							  continue;

																						  for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
																						  {
																							  if (vertex_touched[cell->vertex_index(v)] == false)
																							  {
																								  vertex_touched[cell->vertex_index(v)] = true;

																								  Tensor<1,3> min_disp;
																								  min_disp = 0;


																								  // Tensor<1,3> disp= DistancePointToGeometry(polyvec[polymap.at(id)], cell->vertex(v));
																								  Tensor<1,3> disp;
																								  if (bspmap.find(id) != bspmap.end() )
																									  disp= DistancePointToTree(bspvec[bspmap.at(id)], cell->vertex(v));

																								  Tensor<1,3> disp2 = disp;
																								  disp2 /= disp2.norm();

																								  const unsigned global_vertex_no = cell->vertex_index(v);
																								  {
																									  if (disp.norm() < 0.3*minimal_length[global_vertex_no])
																										  min_disp = disp;
																									  else
																										  min_disp = 0.3*minimal_length[global_vertex_no]*disp2;
																								  }

																								  cell->vertex(v) -= min_disp;  //Maybe . here for the BSPTree and + for the Polygons??


																							  }
																						  }
																					  }
																				  }

																			  // Correct hanging nodes if necessary
																			  if (dim>=2)
																			  {
																				  // We do the same as in GridTools::transform
																				  //
																				  // exclude hanging nodes at the boundaries of artificial cells:
																				  // these may belong to ghost cells for which we know the exact
																				  // location of vertices, whereas the artificial cell may or may
																				  // not be further refined, and so we cannot know whether
																				  // the location of the hanging node is correct or not
																				  typename Triangulation<dim>::active_cell_iterator
																				  cell = triangulation.begin_active(),
																				  endc = triangulation.end();
																				  for (; cell!=endc; ++cell)
																					  if (!cell->is_artificial())
																						  for (unsigned int face=0;
																								  face<GeometryInfo<dim>::faces_per_cell; ++face)
																							  if (cell->face(face)->has_children() &&
																									  !cell->face(face)->at_boundary())
																							  {
																								  // this face has hanging nodes
																								  if (dim==2)
																									  cell->face(face)->child(0)->vertex(1)
																									  = (cell->face(face)->vertex(0) +
																											  cell->face(face)->vertex(1)) / 2;
																								  else if (dim==3)
																								  {
																									  cell->face(face)->child(0)->vertex(1)
                    																		   = .5*(cell->face(face)->vertex(0)
                    																				   +cell->face(face)->vertex(1));
																									  cell->face(face)->child(0)->vertex(2)
                    																		   = .5*(cell->face(face)->vertex(0)
                    																				   +cell->face(face)->vertex(2));
																									  cell->face(face)->child(1)->vertex(3)
                    																		   = .5*(cell->face(face)->vertex(1)
                    																				   +cell->face(face)->vertex(3));
																									  cell->face(face)->child(2)->vertex(3)
                    																		   = .5*(cell->face(face)->vertex(2)
                    																				   +cell->face(face)->vertex(3));

																									  // center of the face
																									  cell->face(face)->child(0)->vertex(3)
																									  = .25*(cell->face(face)->vertex(0)
																											  +cell->face(face)->vertex(1)
																											  +cell->face(face)->vertex(2)
																											  +cell->face(face)->vertex(3));
																								  }
																							  }
																			  }


																			  cell = triangulation.begin_active();
																			  endc = triangulation.end();
																			  for (; cell != endc; ++cell)
																				  if (cell->is_locally_owned())
																				  {
																					  Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
																					  for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
																						  vertices[i] = cell->vertex(i);

																					  Tensor<0,dim> determinants[GeometryInfo<dim>::vertices_per_cell];
																					  GeometryInfo<dim>::alternating_form_at_vertices (vertices,
																							  determinants);

																					  for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
																						  if (determinants[i] <= 1e-9 * std::pow (cell->diameter(),
																								  1.*dim))
																						  {
																							  std::cout << "Distorted" << std::endl;
																							  break;
																						  }
																				  }
																		  }



																		  template <int dim>
																		  void STLRefine<dim>::create_box_grid(Triangulation<dim> &triangulation) const
																		  {
																			  Point<3> p1(parameters.min_x,parameters.min_y,parameters.min_z);
																			  Point<3> p2(parameters.max_x,parameters.max_y,parameters.max_z);
																			  GridGenerator::hyper_rectangle 	(triangulation, p1,p2);

																			  triangulation.refine_global (1);
																			  triangulation.signals.post_refinement.connect
																			  (std_cxx11::bind (&STLRefine<dim>::set_material_ids,
																					  std_cxx11::cref(*this),
																					  std_cxx11::ref(triangulation)));
																		  }



																		  template <int dim>
																		  void STLRefine<dim>::set_material_ids (Triangulation<dim> &triangulation) const
																		  {
																			  for (typename Triangulation<dim>::active_cell_iterator
																					  cell=triangulation.begin_active();
																					  cell!=triangulation.end(); ++cell)
																			  {
																				  if (cell->is_locally_owned())
																				  {
																					  const Point<3> center = cell->center();

																					  int id = cell->material_id();
																					  /* Commented out for the LUSI with faults example
        if (is_inside(center,id))
        {
          id = id;
        }
        else if(is_inside(center,id-10))
        {
            id = id-10;
        }
        else if(is_inside(center,id+10))
              id = id+10;
        else
																					   */
																					  {

																						  for (int i=0; i<parameters.num_stl; i++)
																						  {

																							  id = parameters.materialids[i];
																							  if (is_inside(center,id))
																							  {
																								  break;
																							  }
																							  id = 0;
																						  }
																						  for (int i=0; i<parameters.num_txt; i++)
																						  {
																							  for (int ii=0; ii<parameters.num_faults; ii++)
																							  {
																								  id = parameters.materialids[ii];
																								  if (is_inside(center,id))
																								  {
																									  break;
																								  }
																								  id = 0;
																							  }
																						  }
																					  }
																					  cell->set_material_id(id);
																				  }
																			  }
																		  }



																		  template <int dim>
																		  bool STLRefine<dim>::is_inside(Point<3> p, int id) const
																		  {
																			  bool result = false;
																			  if (bspmap.find(id) != bspmap.end() )
																				  result = PointInSolidSpace(bspvec[bspmap.at(id)],p);

																			  return result;
																		  }



																		  Tensor<1,3> calcNormal( Point<3> a, Point<3> b, Point<3> c )
																		{

																			  Tensor<1,3> ba = b-a;
																			  Tensor<1,3> ca = c-a;
																			  Tensor<1,3> n;
																			  cross_product(n, ba, ca);
																			  //n = n / n.norm();
																			  return n;
																		}



																		  template <int dim>
																		  void STLRefine<dim>::output_results (const unsigned int cycle) const
																		  {
																			  std::ofstream output_file("output/mesh.msh");
																			  GridOut().write_msh (triangulation, output_file);


																			  DataOut<dim> data_out;
																			  data_out.attach_triangulation(triangulation);

																			  Vector<double> material(triangulation.n_active_cells());
																			  Vector<double> boundary(triangulation.n_active_cells());
																			  Assert (material.size() == triangulation.n_active_cells(),
																					  ExcDimensionMismatch (material.size(),
																							  triangulation.n_active_cells()));
																			  Assert (material.size() == triangulation.n_active_cells(),
																					  ExcDimensionMismatch (material.size(),
																							  triangulation.n_active_cells()));

																			  unsigned int index = 0;
																			  for (typename Triangulation<dim, dim>::active_cell_iterator
																					  cell = triangulation.begin_active();
																					  cell!=triangulation.end(); ++cell, ++index)
																				  if (cell->is_locally_owned())
																				  {
																					  int id = cell->material_id();
																					  int id_int = id/10;
																					  material[index] = id_int;

																					  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id+1;
																					  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id+1;
																					  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id+1;
																					  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id+1;
																					  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id+1;
																					  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id+1;
																					  if(id != neighbor_id_x || id != neighbor_id_xx ||
																							  id != neighbor_id_y ||id != neighbor_id_yy ||
																							  id != neighbor_id_z ||id != neighbor_id_zz)
																						  boundary[index] = 1;
																				  }
																			  Assert (index == material.size(), ExcInternalError());
																			  Assert (index == boundary.size(), ExcInternalError());
																			  data_out.add_data_vector (material, "material");
																			  data_out.add_data_vector (boundary, "boundary");

																			  Vector<double> level(triangulation.n_active_cells());
																			  {
																				  typename Triangulation<dim>::active_cell_iterator
																				  cell = triangulation.begin_active(),
																				  endc = triangulation.end();
																				  for (unsigned int index=0; cell!=endc; ++cell, ++index)
																					  if (cell->is_locally_owned())
																					  {
																						  level(index) = cell->level();
																					  }
																			  }
																			  data_out.add_data_vector (level, "level");


																			  Vector<double> aspect_ratio (triangulation.n_active_cells());
																			  {
																				  typename Triangulation<dim>::active_cell_iterator
																				  cell = triangulation.begin_active(),
																				  endc = triangulation.end();
																				  for (unsigned int index=0; cell!=endc; ++cell, ++index)
																					  if (cell->is_locally_owned())
																					  {
																						  // Compute the cell's apect ratio
																						  double L[dim];
																						  double max_found_ratio = 0;
																						  for(int i=0; i<dim;i++)
																							  L[i] = cell->extent_in_direction(i);

																						  for(int i=0; i<dim;i++)
																							  for(int j=0; j<dim;j++)
																								  if(L[i]/L[j] > max_found_ratio)
																									  max_found_ratio = L[i]/L[j];

																						  aspect_ratio(index) = max_found_ratio;

																					  }
																			  }
																			  data_out.add_data_vector (aspect_ratio, "aspect_ratio");

																			  Vector<double> skewness (triangulation.n_active_cells());
																			  {
																				  typename Triangulation<dim>::active_cell_iterator
																				  cell = triangulation.begin_active(),
																				  endc = triangulation.end();
																				  for (unsigned int index=0; cell!=endc; ++cell, ++index)
																					  if (cell->is_locally_owned())
																					  {
																						  double max_theta = 0;
																						  for (unsigned int face=0;
																								  face<GeometryInfo<dim>::faces_per_cell; ++face)
																						  {

																							  Tensor<1,3> u = calcNormal(cell->face(face)->vertex(0),cell->face(face)->vertex(1),cell->face(face)->vertex(3));
																							  Tensor<1,3> v = cell->center()-cell->face(face)->center();

																							  double value =  std::fabs(u*v / (u.norm()*v.norm()));

																							  if (value > max_theta)
																								  max_theta = value;
																						  }

																						  skewness(index) = max_theta;


																					  }
																			  }
																			  data_out.add_data_vector (skewness, "skewness");


																			  Vector<double> distorsion (triangulation.n_active_cells());
																			  {
																				  typename Triangulation<dim>::active_cell_iterator
																				  cell = triangulation.begin_active(),
																				  endc = triangulation.end();
																				  for (unsigned int index=0; cell!=endc; ++cell, ++index)
																					  if (cell->is_locally_owned())
																					  {
																						  Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
																						  for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
																							  vertices[i] = cell->vertex(i);

																						  Tensor<0,dim> determinants[GeometryInfo<dim>::vertices_per_cell];
																						  GeometryInfo<dim>::alternating_form_at_vertices (vertices,
																								  determinants);

																						  double min_determinant = 1;
																						  for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
																							  if (determinants[i] <= min_determinant)
																							  {
																								  min_determinant = determinants[i];
																							  }

																						  if (min_determinant <= 1e-8 * std::pow (cell->diameter(), 1.*dim))
																							  distorsion(index) = (1e-9 * std::pow (cell->diameter(),
																									  1.*dim)) / min_determinant;
																						  else
																							  distorsion(index) = 0;
																					  }
																			  }
																			  data_out.add_data_vector (distorsion, "distorsion");

																			  data_out.build_patches ();

																			  const std::string filename = parameters.outfile;
																			  std::ofstream output (("output/" + filename + Utilities::int_to_string (cycle, 2) + ".vtu").c_str());
																			  data_out.write_vtu (output);

																			  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
																			  {
																				  std::vector<std::string> filenames;
																				  for (unsigned int i=0;
																						  i<Utilities::MPI::n_mpi_processes(mpi_communicator);
																						  ++i)
																					  filenames.push_back (parameters.outfile + Utilities::int_to_string (cycle, 2) + ".vtu");

																				  const std::string mfilename = parameters.outfile;
																				  std::ofstream master_output (("output/" + mfilename+  Utilities::int_to_string(cycle, 2) + ".pvtu").c_str());
																				  data_out.write_pvtu_record (master_output, filenames);
																			  }
																		  }



																		  template <int dim>
																		  void STLRefine<dim>::analyse_mesh ()
																		  {

																			  double disp_sum = 0;
																			  for (typename Triangulation<dim>::active_cell_iterator
																					  cell = triangulation.begin_active();
																					  cell!=triangulation.end(); ++cell)
																				  if (cell->is_locally_owned())
																				  {
																					  int id = cell->material_id();
																					  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id;
																					  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id;
																					  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id;
																					  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id;
																					  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id;
																					  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id;
																					  if(id != neighbor_id_x || id != neighbor_id_xx ||
																							  id != neighbor_id_y ||id != neighbor_id_yy ||
																							  id != neighbor_id_z ||id != neighbor_id_zz)
																					  {
																						  if (bspmap.find(id) != bspmap.end() )
																						  {
																							  Tensor<1,3> disp= DistancePointToTree(bspvec[bspmap.at(id)], cell->center());
																							  disp_sum += disp.norm();
																						  }
																					  }
																				  }
																			  double mean_disp = disp_sum / triangulation.n_active_cells();

																			  std::map<int,double> volumes;
																			  volumes.insert(std::pair<int,double>(0,0));
																			  std::map<int,double> surfaces;
																			  surfaces.insert(std::pair<int,double>(0,0));

																			  for (auto it = 0; it < parameters.materialids.size(); ++it){
																				  volumes.insert(std::pair<int,double>(parameters.materialids[it],0));
																				  surfaces.insert(std::pair<int,double>(parameters.materialids[it],0));
																			  }

																			  for (typename Triangulation<dim>::active_cell_iterator
																					  cell = triangulation.begin_active();
																					  cell!=triangulation.end(); ++cell)
																				  if (cell->is_locally_owned())
																				  {
																					  int id =(int) cell->material_id();
																					  volumes.at(id) += cell->measure();
																				  }

																			  for (typename Triangulation<dim>::active_cell_iterator
																					  cell = triangulation.begin_active();
																					  cell!=triangulation.end(); ++cell)
																				  if (cell->is_locally_owned())
																				  {
																					  int id = cell->material_id();
																					  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id+1;
																					  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id+1;
																					  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id+1;
																					  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id+1;
																					  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id+1;
																					  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id+1;

																					  // Compute cell surfaces along the boundaries
																					  if(id != neighbor_id_x){
																						  surfaces.at(id) += compute_area(cell->face(0)->vertex(0),cell->face(0)->vertex(1),cell->face(0)->vertex(3),cell->face(0)->vertex(2));
																					  }
																					  if(id != neighbor_id_xx)
																						  surfaces.at(id) += compute_area(cell->face(1)->vertex(0),cell->face(1)->vertex(1),cell->face(1)->vertex(3),cell->face(1)->vertex(2));
																					  if(id != neighbor_id_y)
																						  surfaces.at(id) += compute_area(cell->face(2)->vertex(0),cell->face(2)->vertex(1),cell->face(2)->vertex(3),cell->face(2)->vertex(2));
																					  if(id != neighbor_id_yy)
																						  surfaces.at(id) += compute_area(cell->face(3)->vertex(0),cell->face(3)->vertex(1),cell->face(3)->vertex(3),cell->face(3)->vertex(2));
																					  if(id != neighbor_id_z)
																						  surfaces.at(id) += compute_area(cell->face(4)->vertex(0),cell->face(4)->vertex(1),cell->face(4)->vertex(3),cell->face(4)->vertex(2));
																					  if(id != neighbor_id_zz)
																						  surfaces.at(id) += compute_area(cell->face(5)->vertex(0),cell->face(5)->vertex(1),cell->face(5)->vertex(3),cell->face(5)->vertex(2));
																				  }


																			  pcout << "   Analysis of the material parts:"
																					  << std::endl;
																			  pcout << "      Volume:" << std::endl;
																			  for (auto it=volumes.begin(); it!=volumes.end(); ++it)
																				  pcout <<"      "<< it->first << " => " << it->second << '\n';
																			  pcout << "      Surface:" << std::endl;
																			  for (auto it=surfaces.begin(); it!=surfaces.end(); ++it)
																				  pcout <<"      "<< it->first << " => " << it->second << '\n';
																			  pcout << "      Min/Max cell diameter: " << GridTools::minimal_cell_diameter(triangulation)
																			  << " / " << GridTools::maximal_cell_diameter(triangulation) << std::endl;
																			  pcout << "      Mean distance to BSP tree: " << mean_disp << std::endl;

																		  }




																		  template <int dim>
																		  void STLRefine<dim>::run_refinement ()
																		  {

																			  // i number of STL files
																			  for (int i=0; i<parameters.num_stl; i++)
																			  {

																				  {
																					  TimerOutput::Scope t0(computing_timer, "reading stl");
																					  read_input_stl(i);
																				  }
																				  {
																					  TimerOutput::Scope t1(computing_timer, "building tree");
																					  min_max_points(i);
																					  build_trees(i);

																				  }
																				  polymap.insert(std::pair<int,int>(parameters.materialids[i],i));
																				  bspmap.insert(std::pair<int,int>(parameters.materialids[i],i));

																			  }

																			  for (int i=0; i<parameters.num_txt; i++){
																				  {

																					  TimerOutput::Scope t0(computing_timer, "reading txt");
																					  read_input_txt(i);

																				  }

																				  {
																					  TimerOutput::Scope t1(computing_timer, "building tree");
																					  for(unsigned int ii=0; ii<polyvec.size();ii++){
																						  min_max_points(ii);
																						  build_trees(ii);

																					  }
																				  }

																				  for(unsigned int ii=0; ii<polyvec.size();ii++) {
																					  //TODO: here is an issue I cannot use the same parameter id for different shells
																					  //when std map is used the first element of the pair is the index of the element.
																					  // which means that if the parameters.materialids	is the same for two shells it will save in the already existent position and will not
																					  // save in a new position. The lenght of the maps stays the same
																					  //polymap.insert(std::pair<int,int>(parameters.materialids[i],ii));
																					  //bspmap.insert(std::pair<int,int>(parameters.materialids[i],ii)
																					  // I will introduce a dummy material flag
																					  polymap.insert(std::pair<int,int>(parameters.materialids[ii],ii));
																					  bspmap.insert(std::pair<int,int>(parameters.materialids[ii],ii));
																				  }

																			  }

																			  {
																				  TimerOutput::Scope t2(computing_timer, "refinement");
																				  const unsigned int n_cycles = parameters.max_cycle;
																				  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
																				  {
																					  pcout << "Cycle " << cycle << ':' << std::endl;

																					  if (cycle == 0)
																					  {
																						  create_box_grid(triangulation);
																						  triangulation.refine_global (parameters.initial_refinement);

																						  /*int n_flagged = 0;
												  for (typename Triangulation<dim, dim>::active_cell_iterator
														  cell = triangulation.begin_active();
														  cell!=triangulation.end(); ++cell)
													  if (cell->is_locally_owned())
													  {
														  int id = cell->material_id();
														  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id;
														  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id;
														  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id;
														  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id;
														  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id;
														  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id;
														  if(id == neighbor_id_x && id == neighbor_id_xx &&
																  id == neighbor_id_y && id == neighbor_id_yy &&
																  id == neighbor_id_z && id == neighbor_id_zz)
														  {
															  cell->set_coarsen_flag();

															  //std::cout<<"hello coarsening"<<std::endl;
															  // n_flagged -=1;
														  }
													  }
												  triangulation.execute_coarsening ();*/

																					  }
																					  else if (parameters.refine_global == 1)
																					  {
																						  triangulation.refine_global(1);

																					  }
																					  else{
																						  int n_flagged = 0;
																						  for (typename Triangulation<dim, dim>::active_cell_iterator
																								  cell = triangulation.begin_active();
																								  cell!=triangulation.end(); ++cell)
																							  if (cell->is_locally_owned())
																							  {
																								  int id = cell->material_id();
																								  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id;
																								  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id;
																								  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id;
																								  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id;
																								  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id;
																								  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id;
																								  if(id != neighbor_id_x || id != neighbor_id_xx ||
																										  id != neighbor_id_y ||id != neighbor_id_yy ||
																										  id != neighbor_id_z ||id != neighbor_id_zz)
																								  {
																									  cell->set_refine_flag();
																									  n_flagged +=1;
																									  // if (DistancePointToTree(bspvec[0], cell->center()).norm() < 0.0025*max_diameter){
																									  double px = std::fabs(parameters.max_x - parameters.min_x);
																									  double py = std::fabs(parameters.max_y - parameters.min_y);
																									  double pz = std::fabs(parameters.max_z - parameters.min_z);

																									  int id = cell->material_id();
																									  if (bspmap.find(id) != bspmap.end() )
																										  if (DistancePointToTree(bspvec[bspmap.at(id)], cell->center()).norm() < 0.004*std::fmin(px,std::fmin(py,pz))){
																											  cell->clear_refine_flag();
																											  n_flagged -=1;
																										  }
																								  }
																								  if(id == neighbor_id_x && id == neighbor_id_xx &&
																										  id == neighbor_id_y && id == neighbor_id_yy &&
																										  id == neighbor_id_z && id == neighbor_id_zz)
																								  {
																									  cell->set_coarsen_flag();

																									  //std::cout<<"hello coarsening"<<std::endl;
																									  // n_flagged -=1;
																								  }
																							  }

																						  if ((triangulation.n_active_cells() + 8.0*n_flagged) < parameters.max_cells){
																							  triangulation.execute_coarsening_and_refinement ();
																							  if(cycle==n_cycles-1 || (triangulation.n_active_cells() + 8.0*n_flagged) > parameters.max_cells){
																								  pcout << "  Number of estimated active cells."
																										  << std::endl
																										  << "   (" << triangulation.n_global_active_cells() + 8.0*n_flagged << ")"
																										  << std::endl;

																								  for(int ii=0;ii<parameters.coarsening_iterations;ii++){
																									  for (typename Triangulation<dim, dim>::active_cell_iterator
																											  cell = triangulation.begin_active();
																											  cell!=triangulation.end(); ++cell)
																										  if (cell->is_locally_owned())
																										  {
																											  int id = cell->material_id();
																											  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id;
																											  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id;
																											  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id;
																											  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id;
																											  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id;
																											  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id;
																											  if(id == neighbor_id_x && id == neighbor_id_xx &&
																													  id == neighbor_id_y && id == neighbor_id_yy &&
																													  id == neighbor_id_z && id == neighbor_id_zz)
																											  {
																												  cell->set_coarsen_flag();

																												  //std::cout<<"hello coarsening"<<std::endl;
																												  // n_flagged -=1;
																											  }
																										  }
																									  triangulation.execute_coarsening_and_refinement ();
																									  pcout << "New number of estimated active cells."
																											  << std::endl
																											  << "   (" << triangulation.n_global_active_cells() + 8.0*n_flagged << ")"
																											  << std::endl;
																								  }
																							  }
																						  }
																						  else{
																							  pcout << "  Number of estimated active cells is bigger than max cells."
																									  << std::endl
																									  << "   (" << triangulation.n_global_active_cells() + 8.0*n_flagged << ")"
																									  << std::endl;

																							  if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
																							  {
																								  output_results(cycle);
																							  }
																							  break;
																						  }

																						  /*if ((triangulation.n_active_cells() + 8.0*n_flagged) > parameters.max_cells){
																	  pcout << "  Number of estimated active cells is bigger than max cells."
																			  << std::endl
																			  << "   (" << triangulation.n_global_active_cells() + 8.0*n_flagged << ")"
																			  << std::endl;

																	  if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
																	  {
																		  output_results(cycle);
																	  }
																	  break;
																  }
																  else{
																	  triangulation.execute_coarsening_and_refinement ();
																	  if(cycle==n_cycles-1 || (triangulation.n_active_cells() + 8.0*n_flagged) > parameters.max_cells){
																		  pcout << "  Now Number of estimated active cells is bigger than max cells."
																		  																			  << std::endl
																		  																			  << "   (" << triangulation.n_global_active_cells() + 8.0*n_flagged << ")"
																		  																			  << std::endl;
																		  for (typename Triangulation<dim, dim>::active_cell_iterator
																				  cell = triangulation.begin_active();
																				  cell!=triangulation.end(); ++cell)
																			  if (cell->is_locally_owned())
																			  {
																				  int id = cell->material_id();
																				  int neighbor_id_x  = (!cell->at_boundary(0)) ? cell->neighbor(0)->material_id() : id;
																				  int neighbor_id_xx = (!cell->at_boundary(1)) ? cell->neighbor(1)->material_id() : id;
																				  int neighbor_id_y  = (!cell->at_boundary(2)) ? cell->neighbor(2)->material_id() : id;
																				  int neighbor_id_yy = (!cell->at_boundary(3)) ? cell->neighbor(3)->material_id() : id;
																				  int neighbor_id_z  = (!cell->at_boundary(4)) ? cell->neighbor(4)->material_id() : id;
																				  int neighbor_id_zz = (!cell->at_boundary(5)) ? cell->neighbor(5)->material_id() : id;
																				  if(id == neighbor_id_x && id == neighbor_id_xx &&
																						  id == neighbor_id_y && id == neighbor_id_yy &&
																						  id == neighbor_id_z && id == neighbor_id_zz)
																				  {
																					  cell->set_coarsen_flag();

																					  //std::cout<<"hello coarsening"<<std::endl;
																					  // n_flagged -=1;
																				  }
																			  }
																		  triangulation.execute_coarsening_and_refinement ();
																		  pcout << "New number of estimated active cells."
																		  																		  																			  << std::endl
																		  																		  																			  << "   (" << triangulation.n_global_active_cells() + 8.0*n_flagged << ")"
																		  																		  																			  << std::endl;
																	  }
																  }
																						   */

																					  }


																					  pcout << "   Number of active cells:       "
																							  << triangulation.n_global_active_cells()
																							  << std::endl;


																					  computing_timer.print_summary ();

																					  pcout << std::endl;

																					  if (parameters.analysis_mode && Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
																					  {
																						  pcout << "Analysis:" << std::endl;
																						  TimerOutput::Scope t4(computing_timer, "output");
																						  analyse_mesh();
																						  output_results (cycle);
																					  }

																				  }
																			  }


																			  /*{
        TimerOutput::Scope t3(computing_timer, "move_mesh");
        pcout << "Move mesh:" << std::endl;
        move_mesh();
        set_material_ids(triangulation);
      }*/

																			  if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
																			  {
																				  pcout << "Output & Analysis:" << std::endl;
																				  TimerOutput::Scope t4(computing_timer, "output");
																				  analyse_mesh();
																				  output_results (parameters.max_cycle);
																			  }


																			  computing_timer.print_summary ();
																		  }



																		  void
																		  print_usage_message (ConditionalOStream pcout)
																		  {
																			  static const char *message
																			  =
																					  "\n"
																					  "========================================================================\n"
																					  "hulk - Generating hexahedra meshes for geologic structures.\n"
																					  "\n"
																					  "Copyright (c) 2015\n"
																					  "Author: Gunnar Jansen, Reza Sohrabi - University of Neuchatel\n"
																					  "\n"
																					  "Usage:\n"
																					  "    ./hulk [-p] parameter_file \n"
																					  "========================================================================\n"
																					  "\n";
																			  pcout << message;
																		  }

																		  const std::string
																		  parse_command_line (const int     argc,
																				  char *const *argv,
																				  ConditionalOStream pcout,
																				  ConditionalOStream pcerr)
																		  {
																			  print_usage_message (pcout);

																			  std::string parameter_file;
																			  if (argc < 2)
																			  {
																				  pcerr << "Warning: No input file specified. Trying to load default configuration."
																						  "\n"
																						  "Once the simulation is starting to asseble the system please abort the simulation and make"
																						  "\n"
																						  "the neccessary changes to the created files."
																						  "\n"
																						  "Press [Enter] to continue." << std::endl;
																				  std::cin.get();
																				  parameter_file = "default_input.prm";
																				  return parameter_file;
																			  }

																			  std::list<std::string> args;
																			  for (int i=1; i<argc; ++i)
																				  args.push_back (argv[i]);

																			  while (args.size())
																			  {
																				  if (args.front() == std::string("-p"))
																				  {
																					  if (args.size() == 1)
																					  {
																						  pcerr << "Warning: Flag '-p' must be followed by the "
																								  << "name of a parameter file."
																								  << std::endl;
																						  exit (1);
																					  }
																					  args.pop_front ();
																					  parameter_file = args.front ();
																					  args.pop_front ();

																				  }
																				  else if (args.front() == std::string("-su"))
																				  {
																					  if (args.size() == 1)
																					  {
																						  pcerr << "Warning: Flag '-su' must be followed by the "
																								  << "some super user commands to the program."
																								  << std::endl;
																						  exit (1);
																					  }
																					  pcerr << "------------------------------------------------------------------------\n"
																							  << "Warning: Running in super user (expert) mode.\n"
																							  << "This should only be used for development and debugging.\n"
																							  << "Results optained with this option need to be treated with special care.\n"
																							  << "\n"
																							  << "Note: The '-su' flag needs to be last before the expert mode flags.\n"
																							  << "      Otherwise the program will most likely crash.\n"
																							  <<"------------------------------------------------------------------------\n"
																							  << std::endl;
																					  break;
																				  }
																				  else
																				  {
																					  pcerr << "Error: Specified options not known!"
																							  << std::endl;
																					  args.pop_front();
																					  exit (1);
																				  }
																			  }
																			  if (parameter_file.empty())
																			  {
																				  pcerr << "Error: Neccessary parameter file or dimension options not set!"
																						  << std::endl;
																				  exit(1);
																			  }

																			  return parameter_file;
																		  }
}

int main(int argc, char *argv[])
{
	try
	{
		using namespace dealii;
		using namespace hulk;

		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		MPI_Comm mpi_communicator (MPI_COMM_WORLD);
		const unsigned int this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator));

		ConditionalOStream pcout(std::cout, this_mpi_process == 0);
		ConditionalOStream pcerr(std::cerr, this_mpi_process == 0);

		deallog.depth_console (0);

		std::string parameter_file;

		parameter_file = parse_command_line (argc, argv,pcout,pcerr);

		{
			STLRefine<3> refine_3d(parameter_file);

			refine_3d.run_refinement ();
		}
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "Aborting now!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
