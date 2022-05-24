/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** template_balance.cxx : 
 **** mode-independent template for load balancing
 ****
 ****************************************************************/

/*! \page balance Load balancing

  This is incomplete.
*/

#include "template_common_header.h"
#include "balance_functions.h"
using std::shared_ptr;
using std::vector;

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

#define NSTEPS 3
  environment::print_application_options =
    [] () {
    printf("Balance options:\n");
    printf("  -n nnn : local number of points\n");
    printf("  -s nnn : number of steps (default: %d\n",NSTEPS);
    printf("  -l nnn : step limit\n");
    printf("  -dim n : dimensionality\n");
    printf("  -trace : print norms\n");
    printf("  -1     : run single algorithm (unbalanced only)\n");
    printf("  -B     : do NOT load balance\n");
    printf(":\n");
  };
  
  try {
    /* The environment does initializations, argument parsing, and customized printf
     */
    IMP_environment env(argc,argv);
    env.set_name("balance");
  
    /* Print help information if the user specified "-h" argument */
    if ( env.has_argument("h") || env.has_argument("help") ) {
      printf("Usage: %s [ -h ] \n",argv[0]);
      printf("          [ -d ] [ -s nsteps ] [ -l limit ] [ -n size ] [ -B ] [ -1 ]\n");
      return -1;
    }
      
    IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env.get_architecture());
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    int mytid = arch->mytid();
    processor_coordinate mycoord( vector<int>{mytid} );
#endif
    int ntids = arch->nprocs();
    decomposition decomp = IMP_decomposition(arch);

    int
      maxstep = env.iargument("s",NSTEPS),
      localsize = env.iargument("n",10),
      dim = env.iargument("dim",1),
      trace = env.has_argument("trace"),
      balance = ! env.has_argument("B"),
      one = env.has_argument("1");
    int laststep = env.iargument("l",maxstep);
    if (laststep>maxstep) laststep = maxstep;
    if (one) balance = 0;

    auto
      block = shared_ptr<distribution>( new IMP_block_distribution(decomp,localsize,-1) ),
      load = shared_ptr<distribution>( new IMP_replicated_distribution(decomp,ntids) );
    block->set_name("block distribution");
    load->set_name("local distribution");

    domain_coordinate global_last = block->global_last_index()+1;

    // input and output vector are permanent
    auto
      input_vector = shared_ptr<object>( new IMP_object(block) ),
      output_vector = shared_ptr<object>( new IMP_object(block) );

    auto start_time = arch->unsynchronized_timer();
    algorithm onestep;
    for (int istep=0; istep<=laststep; istep++) {

      try {

	/*
	 * The work algorithm
	 */
	shared_ptr<object> tmp_vector;
	if (one) {
	  // start the big algorithm
	  if (istep==0) {
	    onestep = IMP_algorithm(decomp);
	    onestep.add_kernel( shared_ptr<kernel>( new IMP_origin_kernel(input_vector) ) );
	  }
	} else {
	  // new algorithm for one work step
	  onestep = IMP_algorithm(decomp);
	  onestep.set_kernel_zero(istep-1);
	  // we need to accept the input vector, whatever it was before
	  onestep.add_kernel( shared_ptr<kernel>( new IMP_origin_kernel(input_vector) ) );
	}

	// this is the work step; basically just a wait
	auto work = shared_ptr<kernel>( new IMP_copy_kernel(input_vector,output_vector) );
	work->set_localexecutefn
	  ( std::function< kernel_function_proto >{
	    [istep,maxstep] ( kernel_function_args ) -> void {
	      work_moving_weight( kernel_function_call , istep,maxstep );
	    } } );
	onestep.add_kernel(work);

	// get runtime statistics
	auto stats_vector = shared_ptr<object>( new IMP_object(load) );
	onestep.add_kernel( shared_ptr<kernel>( new IMP_stats_kernel(output_vector,stats_vector,summing) ) );

	if (trace) {
	  auto trace_partition  = shared_ptr<kernel>
	    ( new IMP_trace_kernel(output_vector,fmt::format("Partition at step {}:",istep)) );
	  trace_partition->add_in_object(stats_vector);
	  trace_partition->set_last_dependency().set_explicit_beta_distribution
	    (stats_vector->get_distribution());
	  trace_partition->set_localexecutefn
	    ( [istep] ( kernel_function_args ) -> void {
	      if (p.coord(0)==0) {
		fmt::print("Partition at step {}:\n",istep);
		report_partition( invectors.at(0)->partitioning_points(),
				  invectors.at(1)->get_raw_data() );
	      } } );
	  onestep.add_kernel(trace_partition);
	}

	// preserve output for the next loop
	if (istep<laststep) {
	  tmp_vector = shared_ptr<object>( new IMP_object(block) );
	  onestep.add_kernel( shared_ptr<kernel>( new IMP_copy_kernel(output_vector,tmp_vector) ) );
	}

	if (!one) {
	  onestep.analyze_dependencies();
	  onestep.execute();
	}

	if (istep<laststep) {
	  /*
	   * Load balance
	   */

	  if (balance) {
	    //snippet apply_diffuse
	    auto partition_points = block->partitioning_points();
	    int p = partition_points.size()-1;
	    MatrixXd adjacency;
	    if (dim==1) {
	      adjacency = AdjacencyMatrix1D(p);
	    } else if (dim==2) {
	      adjacency = AdjacencyMatrix1D(p);
	    } else {
	      fmt::print("Can not do dim={}\n",dim); return 1;
	    }
	    auto diffuse =
	      distribution_sigma_operator
	      ( [stats_vector,adjacency,trace,mytid]
		(shared_ptr<distribution> d) -> shared_ptr<distribution>  {
		try {
		  return transform_by_diffusion(d,stats_vector,adjacency);
		} catch (std::string c) {
		  throw(fmt::format("Error in averaging: {}",c));
		} } );
	    //snippet end
	    block = block->operate(diffuse);
	  }

	  input_vector = shared_ptr<object>( new IMP_object(block) );
	  input_vector->set_name(fmt::format("in{}",istep+1));
	  output_vector = shared_ptr<object>( new IMP_object(block) );
	  output_vector->set_name(fmt::format("out{}",istep+1));

	  if (one) {
	    onestep.add_kernel
	      ( shared_ptr<kernel>( new IMP_copy_kernel(tmp_vector,input_vector) ) );
	  } else {
	    algorithm load_balance = IMP_algorithm(decomp);
	    load_balance.add_kernel( shared_ptr<kernel>( new IMP_origin_kernel(tmp_vector) ) );
	    load_balance.add_kernel
	      ( shared_ptr<kernel>( new IMP_copy_kernel(tmp_vector,input_vector) ) );

	    load_balance.analyze_dependencies();
	    load_balance.execute();
	  }
	}

      } catch (std::string c) { fmt::print("Strange error in step {}: {}\n",istep,c);
      } catch (std::logic_error e) { fmt::print("Logic error: {}\n",e.what()); }
    }
    if (one) {
      onestep.analyze_dependencies();
      onestep.execute();
    }
    auto stop_time = arch->unsynchronized_timer();
    auto duration  = stop_time-start_time;
    auto millisec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(duration); 
    if (mytid==0)
      fmt::print("Total duration: {}\n",.001 * millisec_duration.count());

  } catch (string c) { fmt::print("Error in main: {}\n",c);
  } catch (...) { fmt::print("Strange error in main.\n");
  }

  return 0;
}
