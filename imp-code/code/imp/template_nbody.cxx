/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** template_nbody.cxx : 
 ****     mode-independent template for N-body problems
 ****
 ****************************************************************/

/*! \page nbody N-body problems

  This is based on the Salmon paper.

  Salmon, J. K.; Warren, M. S. & Winckelmans, G. S. 
  Fast Parallel Tree Codes for Gravitational and Fluid Dynamical N-Body Problems
  Int. J. Supercomputer Appl, 1986, 8, 129-142

  - We go up the tree with an \ref mpi_centerofmass_kernel;
  - we go down the tree with the \ref mpi_sidewaysdown_kernel.

  The interesting part is that the down kernel includes messages that come
  from the up kernels. Thus, an overlapping (see \ref architecture::can_message_overlap)
  run is possibly more MPI efficient.
*/

#include "template_common_header.h"

/****
 **** Main program
 ****/

//!  \test We have a test for a simple N-body problem. See \subpage nbody.
int main(int argc,char **argv) {

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("nbody");

  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
#endif
  int ntids = arch->nprocs();
  decomposition decomp = new mpi_decomposition(arch,layout);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  auto mycoord = decomp.coordinate_from_linear(mytid);
#endif

  int nlocal = env->iargument("nlocal",1024);
  auto level_dist = shared_ptr<distribution>( new IMP_block_distribution(decomp,nlocal,-1) );
    shared_ptr<distribution> new_dist;

  //snippet uptreelevels
  auto coarsen = new multi_sigma_operator( sigma_operator( ioperator("/2") ) );
  std::vector<std::shared_ptr<distribution>> distributions;
  distributions.push_back(level_dist);
  index_int g = level_dist->global_volume();
  for (int level=0; ; level++) {
    new_dist = level_dist->operate(/*level_dist,*/coarsen);
    //fmt::print("Level {} distribution {}\n",level,new_dist->as_string());
    distributions.push_back(new_dist);
    g /= 2;
    new_dist->set_name(fmt::format("level-{}-distribution-of-{}-pts",level,g));
    if (g==1) break;
    level_dist = new_dist;
  }
  //snippet end

  auto flevels = new std::vector<shared_ptr<object>>;
  auto glevels = new std::vector<shared_ptr<object>>;
  auto matrices = new std::vector<shared_ptr<sparse_matrix>>;
  int toplevel = distributions.size()-1;
  for (int ilevel=0; ilevel<=toplevel; ilevel++) {
    auto level=distributions.at(ilevel);

    shared_ptr<object> fobject = shared_ptr<object>( new IMP_object( level ) );
    fobject->set_name( fmt::format("Fobject-{}",ilevel) );
    flevels->push_back( fobject );

    shared_ptr<object> gobject = shared_ptr<object>( new IMP_object( level ) );
    gobject->set_name( fmt::format("Gobject-{}",ilevel) );
    glevels->push_back( gobject );
    auto mat = shared_ptr<sparse_matrix>( new IMP_toeplitz3_matrix(level,1,0,1) );
    mat->set_name(fmt::format("level-{}-matrix",ilevel));
    matrices->push_back(mat);
  }

  algorithm queue = IMP_algorithm(decomp);
  queue.set_name("Nbody tree code");
  
  queue.add_kernel( shared_ptr<kernel>( new IMP_origin_kernel( glevels->at(0) ) ) );
  for (int level=0; level<toplevel; level++) {
    shared_ptr<kernel> cm,xp;

    cm = shared_ptr<kernel>( new IMP_centerofmass_kernel(glevels->at(level),glevels->at(level+1)) );
    cm->set_name( fmt::format("centerofmass-{}",level) );
    queue.add_kernel(cm);

    xp = shared_ptr<kernel>
      ( new IMP_sidewaysdown_kernel
	( flevels->at(level+1),glevels->at(level),flevels->at(level),matrices->at(level)) );
    xp->set_name( fmt::format("prolongate-{}",level) );
    queue.add_kernel(xp);
  }
  queue.add_kernel( shared_ptr<kernel>( new IMP_copy_kernel(glevels->at(toplevel),flevels->at(toplevel)) ) );

  queue.analyze_dependencies();

  queue.execute();

  delete env;
}
