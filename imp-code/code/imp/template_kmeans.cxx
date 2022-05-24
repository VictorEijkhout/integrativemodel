/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** template_threepoint.cxx : 
 **** mode-independent template for threepoint averaging
 ****
 ****************************************************************/

/*! \page kmeans k-Means clustering

  This is incomplete.
*/

#include "template_common_header.h"
#include "kmeans_functions.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("Kmeans options:\n");
    printf("  -n nnn : global number of points\n");
    printf("  -k nnn : number of clusters\n");
    printf("  -s nnn : number of steps\n");
    printf(":\n");
  };
  
  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("kmeans");
  
  /* Print help information if the user specified "-h" argument */
  if (env->has_argument("h")) {
    printf("Usage: %s [-d] [-s nsteps] [-n size] [-k clusters]\n",argv[0]);
    return -1;
  }
      
  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
  decomposition decomp = IMP_decomposition(arch);

  int
    dim = 2,
    nsteps = env->iargument("s",1),
    ncluster = env->iargument("k",5),
    globalsize = env->iargument("n",env->get_architecture()->nprocs());

  /*
   * Define data
   */

  // centers are 2k replicated reals

  //snippet kmeanscenter
  auto kreplicated = shared_ptr<distribution>( new IMP_replicated_distribution(decomp,dim,ncluster) );
  auto centers = shared_ptr<object>( new IMP_object( kreplicated ) );
  //snippet end

  // coordinates are Nx2 with the N distributed
  //snippet kmeanscoord
  auto twoblocked = shared_ptr<distribution>( new IMP_block_distribution(decomp,dim,-1,globalsize) );
  auto coordinates = shared_ptr<object>( new IMP_object( twoblocked ) );
  //snippet end

  // calculate Nxk distances, with the N distributed
  auto kblocked = shared_ptr<distribution>( new IMP_block_distribution(decomp,ncluster,-1,globalsize) );
  auto distances = shared_ptr<object>( new IMP_object( kblocked ) );

  // grouping should be N integers, just use reals
  auto blocked = shared_ptr<distribution>( new IMP_block_distribution(decomp,-1,globalsize) );
  auto grouping = shared_ptr<object>( new IMP_object( blocked ) );

  auto kdblocked = shared_ptr<distribution>
    ( new IMP_block_distribution(decomp,ncluster*(dim+1),-1,globalsize) );
  auto masked_coordinates = shared_ptr<object>( new IMP_object( kdblocked ) );

  /*
   * The kmeans algorithm
   */
  auto  kmeans = shared_ptr<algorithm>( new IMP_algorithm( decomp ) );
  kmeans->set_name("K-means clustering");
  
  auto initialize_centers = shared_ptr<kernel>( new IMP_kernel(centers) );
  initialize_centers->set_localexecutefn
    ( [] ( kernel_function_args ) -> void {
      set_initial_centers(outvector,p); } );
  kmeans->add_kernel( initialize_centers );

  auto set_initial_coordinates = shared_ptr<kernel>( new IMP_kernel( coordinates ) );
  set_initial_coordinates->set_name("set initial coordinates");
  set_initial_coordinates->set_localexecutefn(generate_random_coordinates);
  kmeans->add_kernel( set_initial_coordinates );

  auto calculate_distances = shared_ptr<kernel>( new IMP_kernel( coordinates,distances ) );
  calculate_distances->set_name("calculate distances");
  calculate_distances->set_localexecutefn(distance_calculation);
  calculate_distances->set_explicit_beta_distribution(coordinates->get_distribution());
  calculate_distances->add_in_object( centers);
  calculate_distances->set_explicit_beta_distribution( centers->get_distribution() );
  kmeans->add_kernel( calculate_distances );

  auto find_nearest_center = shared_ptr<kernel>( new IMP_kernel( distances,grouping ) );
  find_nearest_center->set_name("find nearest center");
  find_nearest_center->set_localexecutefn( &group_calculation );
  find_nearest_center->set_explicit_beta_distribution( blocked );
  kmeans->add_kernel( find_nearest_center );

  auto group_coordinates = shared_ptr<kernel>( new IMP_kernel( coordinates,masked_coordinates ) );
  group_coordinates->set_name("group coordinates");
  group_coordinates->set_localexecutefn( &coordinate_masking );
  group_coordinates->add_sigma_operator( ioperator("no_op") );
  group_coordinates->add_in_object(grouping);
  group_coordinates->set_explicit_beta_distribution(grouping->get_distribution());
  kmeans->add_kernel( group_coordinates );

  try {
    kmeans->analyze_dependencies();
  } catch (std::string c) {
    fmt::print("Analysis failed: {}\n",c);
  } catch (std::out_of_range c) {
    fmt::print("Analysis failed with out of range error\n");
  } catch (...) {
    fmt::print("Analysis failed with unknown error\n");
  }
  try {
    kmeans->execute();
  } catch (std::string c) {
    fmt::print("Execution failed: {}\n",c);
  } catch (std::out_of_range c) {
    fmt::print("Execution failed with out of range error\n");
  } catch (...) {
    fmt::print("Execution failed with unknown error\n");
  }

  return 0;
}

