// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** product_base.h: Header file for the product derived class
 ****
 ****************************************************************/
#ifndef PRODUCT_BASE_H
#define PRODUCT_BASE_H 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
using namespace std;

#include <omp.h>
#include <mpi.h>

#include <imp_base.h>
#include <omp_base.h>
#include <mpi_base.h>

/*! \page product MPI/OMP product backend

  A product of MPI and OpenMP is the easiest way for hybrid programming.
  At first glance we construct MPI distributions and objects, but when
  we execute a task, we invoke an \ref omp_algorithm. Of course this
  task queue will be flat since it contains the tasks of just one kernel.

  The product backend is thus somewhat uninteresting; the main point
  is the aggregation of MPI messages, and the load balancing through
  possible oversubscription of the OMP distribution over the hardware threads.

  Regarding the implementation, most product classes are the same as MPI classes, only
  - the \ref product_distribution contains an \ref omp_distribution
  - the \ref product_task contains an \ref omp_algorithm for the local execution.

  Currently we have unit tests working for
  - unittest_distribution.cxx : block and replicated distributions,
    operations on distributions;
    test[2] investigates the structure of a product distributions
  - unittest_struct: testing the mpi messaging, structure of embedded algorithm,
    shift left

  The unit tests are broken for 
  - modulo operators. This is because MPI is broken there.

  \todo write a product shift kernel analogous to struct[36]
 */

/****
 **** Basics
 ****/

class product_environment : public mpi_environment {
private:
public:
  product_environment(int argc,char **argv);
};

/****
 **** Architecture
 ****/

/*!
  Only one object of this class is created, namely in the environment.
  Each object then has its private \ref product_architecture object,
  which has just a pointer to this unique architecture data object.
 */
class product_architecture : public mpi_architecture {
protected:
  //int mpi_ntids,mpi_mytid,nthreads;
public :
  product_architecture( int mytid,int ntids,int nthreads )
    : mpi_architecture(mytid,ntids) {
    type = architecture_type::ISLANDS;
    set_name("product-architecture");
    embedded_architecture = new omp_architecture(nthreads);
    embedded_architecture->set_protocol_is_embedded();
    embedded_architecture->set_name("omp-in-product-architecture");
  };
};

/*!
  A product decomposition is an mpi decomposition with embedded an omp decomposition.
 */
class product_decomposition : public mpi_decomposition {
public:
  product_decomposition() : mpi_decomposition() {};
  //! Multi-d decomposition from explicit processor grid layout
  product_decomposition( architecture *arch,processor_coordinate *grid)
    : mpi_decomposition(arch,grid) {
    embedded_decomposition =
      std::shared_ptr<decomposition>( new omp_decomposition( arch->get_embedded_architecture() ) );
  };
  product_decomposition( architecture *arch )
    : mpi_decomposition(arch) {
    embedded_decomposition =
      std::shared_ptr<decomposition>( new omp_decomposition( arch->get_embedded_architecture() ) );
  };
  virtual std::string as_string() override;
};

/****
 **** Distribution
 ****/

/*!
  A product distribution is an \ref mpi_distribution with an embedded
  \ref omp_distribution. The latter is not created in the base class
  but in the derived classes, so that its type is set correctly.
*/
class product_distribution : virtual public mpi_distribution,virtual public distribution {
// protected:
//   std::shared_ptr<distribution> embedded_distribution{nullptr};
public:
  //! basic constructor: create mpi and openmp member distributions
  product_distribution( const decomposition &d)
    : mpi_distribution(d),distribution(d) {
    if (d.get_architecture_type()!=architecture_type::ISLANDS)
      throw(std::string("Is not a product architecture"));
  };
  //! Not technically a copy constructor.
  product_distribution( std::shared_ptr<distribution> other)
    : mpi_distribution( other->get_structure() ),
      distribution( other->get_decomposition() ) {
    if (other->get_architecture_type()!=architecture_type::ISLANDS)
      throw(std::string("Is not a product architecture"));
    auto pother = dynamic_cast<product_distribution*>(other.get());
    if (pother==nullptr)
      throw(fmt::format("Could not upcast <<{}>> to product",other->as_string()));
    embedded_distribution =
      std::shared_ptr<distribution>( new omp_distribution( pother->get_omp_distribution() ) );
    set_dist_factory();
  };
  //! constructor from parallel_indexstruct
  product_distribution( const parallel_structure &struc)
    : mpi_distribution(struc),distribution(struc) {
    //fmt::print("Product dist has type {}\n",type_as_string());
    if (struc.get_architecture_type()!=architecture_type::ISLANDS)
      throw(std::string("Is not a product architecture"));
    set_dist_factory();
  };

  void set_dist_factory();
  std::shared_ptr<distribution> get_omp_distribution();
  //! \todo this can only word if pouter==me; test for that.
  index_int embedded_volume( processor_coordinate *pouter,processor_coordinate *pinner ) {
    auto embedded = get_omp_distribution();
    return embedded->volume(pinner);
  };

  // // Factory for objects
  // virtual std::shared_ptr<object> new_object();
  // virtual std::shared_ptr<object> new_object( double* );
};

class product_block_distribution
  : public product_distribution,virtual public mpi_block_distribution,
    virtual public distribution {
public:
  product_block_distribution(const decomposition &d,int o,index_int l,index_int g);
  product_block_distribution(const decomposition &d,index_int l,index_int g)
    : product_block_distribution(d,1,l,g) {};
  product_block_distribution(const decomposition &d,index_int g)
    : product_block_distribution(d,-1,g) {};
  //! Multi-d constructor takes an endpoint vector: array of global sizes
  product_block_distribution(const decomposition &d,std::vector<index_int> endpoint)
    : mpi_block_distribution(d,endpoint),product_distribution(d),
      mpi_distribution(d),distribution(d) {
    d.get_same_dimensionality(endpoint.size()); };
};

/*!
  A replicated scalar is the output of a all-reduction operation.
  The cleverness here is taken care of by 
  \ref parallel_indexstruct::create_from_replicated_local_size
  (called in the constructor)
  which puts the same indices on each processor.
*/
class product_replicated_distribution
  : public product_distribution,virtual public mpi_replicated_distribution,
    virtual public distribution {
public:
  product_replicated_distribution(const decomposition &d,index_int s)
    : product_distribution(d),
      mpi_replicated_distribution(d,s),mpi_distribution(d),
      distribution(d) {
    const auto &omp_decomp = d.get_embedded_decomposition();
    parallel_structure parallel(omp_decomp);
    parallel.create_from_replicated_local_size(s); // this sets type to REPLICATED
    embedded_distribution = std::shared_ptr<distribution>( new omp_distribution(parallel) );
  };
  //! Create without integer arguments corresponds to a replicated single element
  product_replicated_distribution(const decomposition &d)
    : product_replicated_distribution(d,1) {};
};

/*!
  Product gathered distributions describe the gathered result of each omp thread
  on each mpi process having s elements. This translates to a \ref mpi_gathered_distribution
  of s*omp_nprocs elements for each mpi process. Each omp process then replicates
  this full storage of size s*product_nprocs.
 */
class product_gathered_distribution
  : public product_distribution,virtual public mpi_gathered_distribution,
    virtual public distribution {
public:
  //! Create a gather of s elements, with k ortho
  product_gathered_distribution(const decomposition &d,int k,index_int s)
    : product_distribution(d),
      mpi_gathered_distribution(d,k,s*d.embedded_nprocs()),mpi_distribution(d),
      distribution(d) {
    // s elements for each omp proc
    int procno = d.mytid();
    auto pcoord = d.coordinate_from_linear(procno);
    auto domain = get_processor_structure(pcoord);
    const auto &omp_decomp = d.get_embedded_decomposition();
    parallel_structure parallel(omp_decomp);
    parallel.create_from_replicated_local_size(s*d.product_nprocs());
    embedded_distribution = std::shared_ptr<distribution>( new omp_distribution(parallel) );
    embedded_distribution->set_enclosing_structure
      ( *(this->get_enclosing_structure().get()) );
  };
  //! Create with a single integer arguments corresponds to ortho=1
  product_gathered_distribution(const decomposition &d,index_int s)
    : product_gathered_distribution(d,1,s) {};
  //! Create without integer arguments corresponds to one element per processor
  product_gathered_distribution(const decomposition &d)
    : product_gathered_distribution(d,1,1) {};
};

class product_cyclic_distribution
  : public product_distribution,virtual public mpi_cyclic_distribution,
    virtual public distribution {
public:
  product_cyclic_distribution(const decomposition &d,index_int lsize,index_int gsize)
    : product_distribution(d),
      mpi_cyclic_distribution(d,lsize,gsize),mpi_distribution(d),
      distribution(d) {
    const auto &omp_decomp = d.get_embedded_decomposition();
    parallel_structure parallel(omp_decomp);
    parallel.create_cyclic(lsize,gsize); // this sets type to CYCLIC
    embedded_distribution = std::shared_ptr<distribution>( new omp_distribution(parallel) );
  };
};

/****
 **** Sigma object
 ****/

/****
 **** Object
 ****/

/*!
  We let the data of a product object be allocated by the MPI distribution
  component of the product distribution.
*/
class product_object : public mpi_object {
private:
protected:
public:
  void set_factory() {
  };
  product_object( std::shared_ptr<distribution> d )
    : mpi_object(d) { set_factory(); };
  product_object( std::shared_ptr<distribution> d,
		  std::shared_ptr<std::vector<double>> x,
		  index_int offset=0 )
    : mpi_object(d,x) { set_factory(); };
};

/****
 **** Message
 ****/

/****
 **** Task
 ****/

/*!
  A product task is an MPI task, except that it executes a complete \ref omp_algorithm
  as its local function. Other functions, in particular \ref allocate_halo_vector 
  are completely inherited from the MPI version.

  \todo try constructor delegating for the node queue & local stuff
 */
class product_task : public mpi_task {
private:
  std::shared_ptr<object> omp_inobject,omp_outobject;
protected:
public:
  product_task(const processor_coordinate &d,kernel *k) : mpi_task(d,k) {
    fmt::print("Please consider making the embedded decomposition not a pointer\n");
    node_queue = std::shared_ptr<algorithm>
      ( new omp_algorithm( k->get_out_object()->get_embedded_decomposition() ) );
    node_queue->set_name(fmt::format("omp-queue-in-task:{}",this->get_name()));
    if (get_out_object()->get_decomposition().get_split_execution()) {
      node_queue->set_sync_tests
	(
	 [] (std::shared_ptr<task> t) -> int {
	   return dynamic_cast<omp_task*>(t.get())->get_local_executability()
	     ==task_local_executability::YES; },
	 [] (std::shared_ptr<task> t) -> int {
	   return dynamic_cast<omp_task*>(t.get())->get_local_executability()
	     !=task_local_executability::YES; }
	 );
    }
  };

  /*
   * Embedded OpenMP queue
   */
public:
  // local execute calls \ref algorithm::execute on the embedded queue
  virtual void local_execute
      (std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,void*,
       int(*)(std::shared_ptr<task>)) override;
  // local execute calls \ref algorithm::execute on the embedded queue
  virtual void local_execute
      (std::vector<std::shared_ptr<object>> &ins,std::shared_ptr<object> out,void *ctx)
    override {
    local_execute(ins,out,ctx,&task::task_test_true);
  };

  // See product_task::local_analysis
  std::shared_ptr<object> omp_object_from_mpi( std::shared_ptr<object> mpi_obj );
  // Used in omp_object_from_mpi
  std::shared_ptr<distribution> omp_distribution_from_mpi( std::shared_ptr<distribution> mpi_distr,processor_coordinate &mytid );

  //! Non-trivial override of the (no-op) virtual function
  virtual void local_analysis() override;

  virtual void declare_dependence_on_task( task_id *id ) override ; // pure virtual

  /*
   * Contexts for embedded tasks
   */
protected:
  std::function
  < void(kernel_function_types) > tasklocalexecutefn;
  void *tasklocalexecutectx{nullptr};
public:
  /*! Override the standard behaviour, because we already have a local function;
    instead we store this function to give as local function to the omp tasks */
  virtual void set_localexecutefn
  ( std::function< void(kernel_function_types) > f )
    override { tasklocalexecutefn = f; };
  /*! Similarly we store this context to give to the omp tasks */
  virtual void set_localexecutectx( void *ctx ) override { tasklocalexecutectx = ctx; };
};

/****
 **** Kernel
 ****/

/*!
  A product kernel is largely an mpi kernel, except that the \ref kernel::localexecutefn
  has a fixed value to execute the embedded \ref omp_algorithm.
 */
class product_kernel : public mpi_kernel {
private:
public:
  product_kernel() : kernel(),mpi_kernel() { 
    install_product_factory(); };
  product_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : mpi_kernel( in,out ),kernel(in,out) {
    install_product_factory(); };
  product_kernel( std::shared_ptr<object> out )
    : mpi_kernel( out ),kernel(out) {
    install_product_factory(); };

  void install_product_factory() {
    make_task_for_domain =
      [] (kernel *k,const processor_coordinate &dom) -> shared_ptr<task> 
      { auto t = shared_ptr<task>( new product_task(dom,k) );
	return t;
      };
  };
};

class product_origin_kernel : public product_kernel, public origin_kernel {
public:
  product_origin_kernel( std::shared_ptr<object> out )
    : kernel(out),product_kernel(out),origin_kernel(out) {};
};

/****
 **** Queue
 ****/

class product_algorithm : public mpi_algorithm {
private:
public:
  product_algorithm(const decomposition &d) : mpi_algorithm(d) {};
};

#endif
