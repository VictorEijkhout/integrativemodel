/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** product_base.cxx: Implementations of the product class
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <iostream>

#include <mpi.h>

#include "product_base.h"
using fmt::format;
using fmt::print;

using std::shared_ptr;

/****
 **** Basics
 ****/

/*!
  A product environment constructs a product architecture. See elsewhere for its structure.
 */
product_environment::product_environment(int argc,char **argv) : mpi_environment(argc,argv) {
  type = environment_type::PRODUCT;

  // discover MPI architecture
  int mytid,ntids;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);
  // discover OpenMP architecture
  int nthreads;
#pragma omp parallel shared(nthreads)
#pragma omp master
  nthreads = omp_get_num_threads();

  try {
    arch = new product_architecture(mytid,ntids,nthreads);
    //arch->add_domain(mytid);
    {
      architecture *const omp_arch = arch->get_embedded_architecture();
      if (has_argument("split"))
	  omp_arch->set_split_execution();
    }
  } catch (std::string c) { fmt::print("Environment error <<{}>>\n",c); }

};

std::string product_decomposition::as_string() {
  fmt::MemoryWriter w;
  w.write("Product decomposition based on <<{}>> and <<{}>>",
	  mpi_decomposition::as_string(),get_embedded_decomposition().as_string());
  return w.str(); };

/****
 **** Distribution
 ****/

shared_ptr<distribution> product_distribution::get_omp_distribution() {
  if (embedded_distribution==nullptr) throw(std::string("null omp distro"));
  return embedded_distribution; };

void product_distribution::set_dist_factory() {
  //! Factory method for cloning objects. This is used in \ref task::allocate_halo_vector.
  new_object = [this] (shared_ptr<distribution> d) -> shared_ptr<object> {
    return shared_ptr<object>( new product_object(d) ); };
  //! Factory method used for cloning objects while reusing data.
  new_object_from_data = [this] ( shared_ptr<vector<double>> d ) -> shared_ptr<object>
    { return shared_ptr<object>( new product_object(this->shared_from_this(),d) ); };
};

//! \todo can the embedded_distribution creation go into the basic constructor?
product_block_distribution::product_block_distribution
    (const decomposition &d,int o,index_int l,index_int g)
  : product_distribution(d),mpi_block_distribution(d,o,l,g),
    mpi_distribution(d),distribution(d) {
  print("product block distribution on decomp {} from o={} l={} g={}\n",
	d.as_string(),o,l,g);
  const auto &pcoord = proc_coord();
  const auto &domain = get_processor_structure(pcoord);
  print("setting up omp on p={} domain={}\n",pcoord.as_string(),domain->as_string());
  try {
    const auto &omp_decomp = d.get_embedded_decomposition();
    parallel_structure parallel(omp_decomp);
    parallel.create_from_indexstruct(domain); // this sets type to BLOCKED
    embedded_distribution = shared_ptr<distribution>( new omp_distribution(parallel) );
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> creating embedded for {}\n",c,pcoord.as_string()));
  } catch (...) {
    throw(fmt::format("Unknown error creating embedded for {}\n",pcoord.as_string()));
  }
  embedded_distribution->set_enclosing_structure
    ( *(this->get_enclosing_structure().get()) );
  //embedded_distribution->set_enclosing_structure( this->get_enclosing_structure() );
};

/****
 **** Object: it's all in the .h file
 ****/

/****
 **** Message: nothing particular
 ****/

/****
 **** Task
 ****/

/*!
  Make an \ref omp_distribution from an \ref mpi_distribution on
  a \ref processor_coordinate.

  \todo make actually private
*/
shared_ptr<distribution> product_task::omp_distribution_from_mpi
    ( shared_ptr<distribution> mpi_distr,processor_coordinate &mytid ) {
  auto omp_decomp = mpi_distr->get_embedded_decomposition();
  distribution_type mpi_type = mpi_distr->get_type();

  shared_ptr<distribution> omp_distr; 
  parallel_structure omp_struct(omp_decomp);
  //snippet mpi2ompblock
  auto outidx = mpi_distr->get_processor_structure(mytid);
  if (mpi_type==distribution_type::REPLICATED) {
    omp_struct.create_from_replicated_indexstruct(outidx);
  } else {
    omp_struct.create_from_indexstruct(outidx);
  }
  omp_distr = shared_ptr<distribution>( new omp_distribution(omp_struct) );
  //  omp_distr->lock_global_structure( mpi_distr->get_global_structure() );
  omp_distr->set_name( fmt::format("<<{}>>-embedded-on-p{}",
				   mpi_distr->get_name(),mytid.as_string()) );
  //snippet end
  // fmt::print("Global structure on {} set to {},{}\n",
  // 	     get_domain()->as_string(),
  // 	     mpi_distr->get_global_structure()->as_string(),
  // 	     omp_distr->get_global_structure()->as_string()
  // 	     );
  return omp_distr;
};

/*!
  Private method used in \ref product_task::local_analysis
  \todo not actually private right now
*/
shared_ptr<object> product_task::omp_object_from_mpi( shared_ptr<object> mpi_obj ) {
  if (!mpi_obj->has_data_status_allocated())
    throw(std::string("Product object needs allocated mpi object"));
  auto mycoord = get_domain();
  auto omp_decomp = mpi_obj->get_embedded_decomposition();
  auto mpi_distr = mpi_obj->get_distribution();
  if (mpi_distr==nullptr)
    throw(std::string("Could not upcast to mpi distr"));
  distribution_type mpi_type = mpi_distr->get_type();

  shared_ptr<distribution> omp_distr; shared_ptr<object> omp_obj;
  omp_distr = omp_distribution_from_mpi(mpi_distr,mycoord);
  int locdom = mpi_distr->get_domain_local_number(mycoord);
  if (mpi_type==distribution_type::REPLICATED) {
    index_int s = mpi_distr->local_allocation_p(mycoord);
    { auto outidx = mpi_distr->get_processor_structure(mycoord);
      if (s!=outidx->volume())
	throw(std::string("size incompatibility")); }
    // VLE final `s' parameter missing here
    omp_obj = omp_distr->new_object_from_data(mpi_obj->get_numa_data_pointer());
      //(mpi_obj->get_data(locdom));
  } else {
    omp_obj = omp_distr->new_object_from_data(mpi_obj->get_numa_data_pointer());
    //omp_obj = omp_distr->new_object_from_data(mpi_obj->get_data(locdom));
  }
  omp_obj->set_name( fmt::format("<<{}>>-omp-alias",mpi_obj->get_name()) );
  // fmt::print("Create omp obj from mpi {} with global struct <<{}>>\n",
  // 	     omp_obj->get_name(),omp_obj->get_global_structure()->as_string());
  return omp_obj;
};

/*!
  This routine pertains to a task on the outer, MPI level. Since it contains
  a complete OpenMP task queue, the task requires local analysis.

  Since product is the simplest hybrid model this is is all fairly easy:
  - we already have an \ref omp_algorithm in the task,
  - we add a single kernel to it 
  - the output object of that kernel is a wrapping of the \ref outvector as an \ref omp_object
  - the input object a wrapping of either the invector or the halo
 */
void product_task::local_analysis() {
  auto omp_decomp = get_out_object()->get_distribution()->get_embedded_decomposition();
  auto mycoord = get_domain();

  omp_outobject = omp_object_from_mpi(get_out_object());
  shared_ptr<kernel> k;
  if (has_type_origin()) {
    omp_inobject = nullptr;
    k = shared_ptr<kernel>( new omp_origin_kernel(omp_outobject) );
    k->set_name("embedded-omp-origin");
  } else {
    k = shared_ptr<kernel>( new omp_kernel(omp_outobject) );
    k->set_name( fmt::format("<<{}>>-embedded-omp-kernel",get_name()) );
    if (get_dependencies().size()==0)
      throw(std::string("Suspiciously no dependencies for compute kernel"));
    for ( auto d : get_dependencies() ) {
      // origin kernel for the halo of this dependency
      auto inobject = omp_object_from_mpi( d.get_beta_object() );
      auto ko = shared_ptr<kernel>( new omp_origin_kernel(inobject) );
      ko->set_name("embedded-omp-origin");
      node_queue->add_kernel(ko);
      // compute kernel for this dependency
      k->add_in_object(inobject);
      if (d.has_type_explicit_beta()) {
	k->set_last_dependency().set_explicit_beta_distribution
	  ( omp_distribution_from_mpi(d.get_explicit_beta_distribution(),mycoord) );
      } else {
	k->set_last_dependency().copy_from(d); // this copies the signature function
      }
      k->set_last_dependency().set_name( fmt::format("{}-dependency",k->get_name()) );
    }
    if (k->get_dependencies().size()==0)
      throw("Suspiciously no dependencies for embedded compute kernel\n");
    k->set_name("embedded-omp-compute");
    k->set_localexecutectx( tasklocalexecutectx );
  }
  k->set_localexecutefn( tasklocalexecutefn );
  node_queue->add_kernel(k);

  node_queue->analyze_dependencies();
  if (omp_outobject->get_decomposition().get_split_execution())
    node_queue->set_outer_as_synchronization_points();
  node_queue->determine_locally_executable_tasks();
};

/*!
  Execute the embedded tasks of \ref product_task: the \ref ctx argument is really
  an \ref omp_algorithm.

  In a product task, the local execute is called twice with a test that is not
  all-or-nothing, so we pass the test to the queue execute.
  \todo can we make that cast more elegant?
 */
void product_task::local_execute
    (std::vector<shared_ptr<object>> &beta_objects,shared_ptr<object> outobj,void *ctx,
     int(*tasktest)(shared_ptr<task> t)) {
  //  omp_algorithm *queue = (omp_algorithm*)ctx;
  //  printf("start embedded node queue conditional execute for step %d\n",get_step());
  if (!node_queue->has_type(algorithm_type::OMP)) // sanity check on that cast.....
    throw("Embedded queue not of OMP type");
  node_queue->execute(tasktest);
};

/*!
  Create a dependence in the local task graph. Since the task can be on a different
  address space, we declare a dependence on the task from the same kernel,
  but on this domain.
 */
void product_task::declare_dependence_on_task( task_id *id ) {
  int step = id->get_step(); auto domain = this->get_domain();
  try {
    add_predecessor( find_other_task_by_coordinates(step,domain) );
  } catch (std::string c) {
    fmt::print("Task <<{}>> error <<{}>> locating <{},{}>.\n{}\n",
	       get_name(),c,step,domain.as_string(),this->as_string());
    throw(std::string("Could not find Product local predecessor"));
  };
};

/****
 **** Kernel
 ****/

//! Construct the right kind of task for the base class method \ref kernel::split_to_tasks.
// shared_ptr<task> product_kernel::make_task_for_domain
//     (kernel *k,const processor_coordinate &dom) {
//   return shared_ptr<task>( new product_task(dom,this) );
// };

/****
 **** Queue
 ****/

