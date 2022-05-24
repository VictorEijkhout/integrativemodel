/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** template_lulesh.cxx : 
 ****     mode-independent template LULESH
 ****
 ****************************************************************/

#include <memory>
using std::shared_ptr;

#include "template_common_header.h"
#include "lulesh_functions.h"

/*! \page lulesh LULESH Solver

*/
class mpi_lulesh_element_to_local_kernel : public mpi_kernel,virtual public kernel {
public:
  mpi_lulesh_element_to_local_kernel( shared_ptr<object>elements,shared_ptr<object>local_nodes)
    : mpi_kernel(elements,local_nodes),kernel(elements,local_nodes)
  {
    int dim = elements->get_same_dimensionality(local_nodes->get_dimensionality());
    set_name("element_to_local_nodes");
    set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	( dim, 
	  function< shared_ptr<multi_indexstruct>(const multi_indexstruct&) >{
	    [] (const multi_indexstruct&i) -> shared_ptr<multi_indexstruct> {
	      return signature_struct_element_to_local(i); } } ) );
    set_localexecutefn
      ( [dim] ( kernel_function_args ) -> void {
          return element_to_local_function( kernel_function_call,dim ); } );
  };
};

class mpi_lulesh_local_to_global_kernel : public mpi_kernel,virtual public kernel {
 public:
  mpi_lulesh_local_to_global_kernel
  ( shared_ptr<object> local_nodes,shared_ptr<object> global_nodes)
    : mpi_kernel(local_nodes,global_nodes),kernel(local_nodes,global_nodes)
  {
    int dim = global_nodes->get_same_dimensionality(local_nodes->get_dimensionality());
    set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	( dim,
	  function< shared_ptr<multi_indexstruct>(const multi_indexstruct &g) >{
	    [local_nodes] (const multi_indexstruct &g) -> shared_ptr<multi_indexstruct> {
	      auto enc = local_nodes->get_enclosing_structure();
	      return signature_local_from_global( g,enc ); } } ) );
    set_localexecutefn
      ( [local_nodes] ( kernel_function_args ) -> void {
	  local_to_global_function
	    ( kernel_function_call,local_nodes->get_enclosing_structure() ); } );
  };
};

class mpi_lulesh_global_to_local_kernel : public mpi_kernel,virtual public kernel {
public:
  mpi_lulesh_global_to_local_kernel
  ( shared_ptr<object> global_nodes,shared_ptr<object> local_nodes )
    : mpi_kernel(global_nodes,local_nodes),kernel(global_nodes,local_nodes)
  {
    int dim = global_nodes->get_same_dimensionality(local_nodes->get_dimensionality());
    set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	(dim,
	 function< shared_ptr<multi_indexstruct>( const multi_indexstruct & ) >{
	   [] ( const multi_indexstruct &i ) -> shared_ptr<multi_indexstruct>{
	     return signature_global_node_to_local(i); } } ) );
    auto 
      local_nodes_global_domain = local_nodes->get_enclosing_structure(),
      global_nodes_global_domain = global_nodes->get_enclosing_structure();
    set_localexecutefn
      ( [local_nodes_global_domain] ( kernel_function_args ) -> void {
	  function_global_node_to_local( kernel_function_call,local_nodes_global_domain ); } );
  };
};

class mpi_lulesh_local_nodes_to_element_kernel : public mpi_kernel, virtual public kernel {
public:
  mpi_lulesh_local_nodes_to_element_kernel
  ( shared_ptr<object> local_nodes,shared_ptr<object> elements )
    : mpi_kernel(local_nodes,elements),kernel(local_nodes,elements)
  {
    int dim = elements->get_same_dimensionality(local_nodes->get_dimensionality());
    set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	( dim,
	  function< shared_ptr<multi_indexstruct>(const multi_indexstruct &g) >{
	    [local_nodes,dim] (const multi_indexstruct &g) -> shared_ptr<multi_indexstruct> {
	      auto enc = local_nodes->get_enclosing_structure();
	      return signature_local_to_element( dim,g ); } } ) );
    set_localexecutefn
      ( [local_nodes] ( kernel_function_args ) -> void {
	  local_node_to_element_function( kernel_function_call ); } );
  };
};

class lulesh_environment : public IMP_environment {
protected:
public:
  lulesh_environment(int argc,char **argv) : IMP_environment(argc,argv) {
    if (has_argument("help")) print_options(); // this is broken
  };
};

/****
 **** Main program
 ****/

//! \test We have a test for a lulesh equation without collectives. See \subpage lulesh.
//! \todo make the data setting mode-independent
int main(int argc,char **argv) {

  try {
    environment::print_application_options =
      [] () {
	printf("Lulesh solver options:\n");
	printf("  -dim n : space dimension\n");
	printf("  -elocal nnn: local number of elements per dimension\n");
	printf("  -trace : print stuff\n");
	printf("\n");
      };

    /* The environment does initializations, argument parsing, and customized printf
     */
    IMP_environment *env = new lulesh_environment(argc,argv);
    env->set_name("lulesh");
    int elocal = env->iargument("elocal",100); // element per processor per dimension
    int dim = env->iargument("dim",2);   
    int trace = env->has_argument("trace");

    IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
    auto layout = arch->get_proc_layout(dim);
    decomposition decomp = mpi_decomposition(arch,layout);

#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    int mytid = arch->mytid();
    auto mycoord = decomp.coordinate_from_linear(mytid);
    double timestamp;
#endif
    int ntids = arch->nprocs();

    /****
     **** Make objects:
     ****/
    shared_ptr<object> elements,elements_back, local_nodes,local_nodes_back, global_nodes;

    /*
     * Elements
     */
    domain_coordinate elements_domain(dim);
    for (int id=0; id<dim; id++)
      elements_domain.set(id,elocal*(layout.coord(id)));
    auto elements_dist =
      shared_ptr<distribution>( new IMP_block_distribution(decomp,elements_domain) );
    elements = shared_ptr<object>( new IMP_object(elements_dist) );
    elements->set_name(fmt::format("elements{}D",dim));
    elements_back = shared_ptr<object>( new IMP_object(elements_dist) );
    elements_back->set_name(fmt::format("elements_back{}D",dim));

    /*
     * Local nodes
     */
    domain_coordinate local_nodes_domain(dim);
    for (int id=0; id<dim; id++ )
      local_nodes_domain.set(id,2*elements_domain[id] );
    auto local_nodes_dist = 
      shared_ptr<distribution>( new IMP_block_distribution(decomp,local_nodes_domain) );
    local_nodes = shared_ptr<object>( new IMP_object(local_nodes_dist) );
    local_nodes->set_name("local_nodes");
    local_nodes_back = shared_ptr<object>( new IMP_object(local_nodes_dist) );
    local_nodes_back->set_name("local_nodes_back");

    domain_coordinate global_nodes_domain(dim);
    for (int id=0; id<dim; id++)
      global_nodes_domain.set(id,elocal*layout.coord(id)+1 );
    auto global_nodes_dist =
      shared_ptr<distribution>( new IMP_block_distribution(decomp,global_nodes_domain) );
    global_nodes = shared_ptr<object>( new IMP_object(global_nodes_dist) );
    global_nodes->set_name("global_nodes");

    /****
     **** Kernels
     ****/
    shared_ptr<kernel> init_elements, element_to_local_nodes, local_to_global_nodes,
      global_to_local_nodes,local_nodes_to_element;
    
#if 0
    init_elements = shared_ptr<kernel>( new IMP_origin_kernel(elements) );
    init_elements->set_localexecutefn( &vecsetlinear );
#endif

    /*
     * Elements to local nodes
     */
#if 0
    element_to_local_nodes = shared_ptr<kernel>( new IMP_kernel(elements,local_nodes) );
    element_to_local_nodes->set_name("element_to_local_nodes");
    element_to_local_nodes->last_dependency().set_signature_function_function
      ( multi_sigma_operator
	( dim, 
	  function< shared_ptr<multi_indexstruct>(const multi_indexstruct&) >{
	    [] (const multi_indexstruct&i) -> shared_ptr<multi_indexstruct> {
	      return signature_struct_element_to_local(i); } } ) );
    //&signature_struct_element_to_local ) ); // coordinate?
    element_to_local_nodes->set_localexecutefn
      ( [dim] ( kernel_function_args ) -> void {
          return element_to_local_function( kernel_function_call,dim ); } );
#endif
    
    /*
     * Local nodes to global nodes
     */
    local_to_global_nodes = shared_ptr<kernel>
      ( new IMP_lulesh_local_to_global_kernel(local_nodes,global_nodes) );
#if 0
    local_to_global_nodes->set_last_dependency().set_signature_function_function
      ( multi_sigma_operator
	( dim,
	  function< shared_ptr<multi_indexstruct>(const multi_indexstruct &g) >{
	    [local_nodes] (const multi_indexstruct &g) -> shared_ptr<multi_indexstruct> {
	      auto enc = local_nodes->get_enclosing_structure();
	      return signature_local_from_global( g,enc ); } } ) );
    local_to_global_nodes->set_localexecutefn
      ( [local_nodes] ( kernel_function_args ) -> void {
	  local_to_global_function( kernel_function_call,local_nodes->get_enclosing_structure() ); } );
#endif

    /*
     * Global nodes back to local nodes
     */
    global_to_local_nodes = shared_ptr<kernel>
      ( new IMP_lulesh_global_to_local_kernel(global_nodes,local_nodes_back) );
#if 0
    {
      global_to_local_nodes->set_last_dependency().set_signature_function_function
	( multi_sigma_operator
	  (dim,
	   function< shared_ptr<multi_indexstruct>( const multi_indexstruct & ) >{
	     [] ( const multi_indexstruct &i ) -> shared_ptr<multi_indexstruct>{
	       return signature_global_node_to_local(i); } } ) );
      auto 
	local_nodes_global_domain = local_nodes->get_enclosing_structure(),
	global_nodes_global_domain = global_nodes->get_enclosing_structure();
      global_to_local_nodes->set_localexecutefn
	( [local_nodes_global_domain] ( kernel_function_args ) -> void {
	    function_global_node_to_local( kernel_function_call,local_nodes_global_domain ); } );
    }
#endif
  
    /****
     **** Lulesh algorithm
     ****/
  
    //snippet lulesh algorithm
    algorithm lulesh(decomp);

    elements = shared_ptr<object>( new IMP_object(elements_dist) );
    { // init
      init_elements = shared_ptr<kernel>( new IMP_origin_kernel(elements) );
      init_elements->set_localexecutefn( &vecsetlinear );
      lulesh.add_kernel(init_elements);
    }
    for (int it=0; it<10; it++) {
      element_to_local_nodes = shared_ptr<kernel>
	( new IMP_lulesh_element_to_local_kernel(elements,local_nodes) );
      lulesh.add_kernel(element_to_local_nodes);

      local_to_global_nodes = shared_ptr<kernel>
	( new IMP_lulesh_local_to_global_kernel(local_nodes,global_nodes) );
      lulesh.add_kernel(local_to_global_nodes);

      global_to_local_nodes = shared_ptr<kernel>
	( new IMP_lulesh_global_to_local_kernel(global_nodes,local_nodes) ); // back?
      lulesh.add_kernel(global_to_local_nodes);

      elements = shared_ptr<object>( new IMP_object(elements_dist) );
      local_nodes_to_element = shared_ptr<kernel>
	( new IMP_lulesh_local_nodes_to_element_kernel(local_nodes,elements) );
      lulesh.add_kernel(local_nodes_to_element);
    }

    // if (trace) {
    //   lulesh.add_kernel
    // 	( shared_ptr<kernel>( new IMP_trace_kernel(elements,"elements") ) );
    //   lulesh.add_kernel
    // 	( shared_ptr<kernel>( new IMP_trace_kernel(local_nodes,"local_nodes") ) );
    //   lulesh.add_kernel
    // 	( shared_ptr<kernel>( new IMP_trace_kernel(global_nodes,"global_nodes") ) );
    //   lulesh.add_kernel
    // 	( shared_ptr<kernel>( new IMP_trace_kernel(local_nodes_back,"local_nodes_back") ) );
    // }


#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    timestamp = MPI_Wtime();
#endif
    lulesh.analyze_dependencies();
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    timestamp = MPI_Wtime()-timestamp;
    if (mytid==0)
      cout << "Analysis time: " << timestamp << endl;
    timestamp = MPI_Wtime();
#endif
    lulesh.execute();
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    timestamp = MPI_Wtime()-timestamp;
    if (mytid==0)
      cout << "Execution time: " << timestamp << endl;
#endif
  } catch (string c) {
    fmt::print("Error in lulesh main: <<{}>>\n",c);
  } catch (...) {
    fmt::print("Error in lulesh main\n");
  }
  //snippet end

  //  delete env;

  return 0;
}
