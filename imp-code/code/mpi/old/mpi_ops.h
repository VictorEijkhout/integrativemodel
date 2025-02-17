// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** mpi_ops.h: Header file for the MPI operation kernels
 ****
 ****************************************************************/

#ifndef MPI_OPS_H
#define MPI_OPS_H 1

#include "mpi_base.h"
#include "imp_functions.h"
#include "imp_ops.h"

#ifdef VT
#include "VT.h"
#endif

/*! \page mpiops MPI Operations
  The basic mechanism of declaring distributions, objects, and kernels
  is powerful enough by itself. However, for convenience a number 
  of common operations have been declared.

  File mpi_ops.h contains:
  - \ref mpi_spmvp_kernel : the distributed sparse matrix vector product
  - \ref mpi_innerproduct : the distributed inner product
  - \ref mpi_outerproduct_kernel : combine distributed data with replicated data
*/

class mpi_setconstant_kernel : virtual public origin_kernel,public setconstant_kernel,virtual public mpi_kernel {
public:
  mpi_setconstant_kernel( std::shared_ptr<object> out,double v )
    : kernel(out),origin_kernel(out),
      //mpi_origin_kernel(out),
      setconstant_kernel(out,v) {
    install_mpi_factory(); //install_mpi_kernel_factory(this);
  };
};

//snippet mpicopy
/*!
  Copying is simple
*/
class mpi_copy_kernel : virtual public mpi_kernel,public copy_kernel {
public:
  mpi_copy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),copy_kernel(in,out) {
    install_mpi_factory(); //install_mpi_kernel_factory(this);
//snippet end
#ifdef VT
    VT_funcdef("copy kernel",VT_NOCLASS,&vt_kernel_class);
#endif
//snippet mpicopy
  };
};
//snippet end

//snippet mpibcast
/*!
  Broadcast is a trick. Read IMP-15 about how this is done.
*/
class mpi_bcast_kernel : public mpi_kernel,public bcast_kernel {
public:
  mpi_bcast_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),bcast_kernel(in,out),mpi_kernel(in,out) {
//snippet end
#ifdef VT
    VT_funcdef("bcast kernel",VT_NOCLASS,&vt_kernel_class);
#endif
//snippet mpibcast
  };
};
//snippet end

class mpi_gather_kernel : public mpi_kernel,public gather_kernel {
public:
  mpi_gather_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,std::function<kernel_function_proto> flocal  )
    : kernel(in,out),gather_kernel(in,out,flocal),mpi_kernel(in,out) {
  };
};

/*!
  Scale a vector by a scalar.
*/
class mpi_scale_kernel : virtual public mpi_kernel,public scale_kernel {
public:
  mpi_scale_kernel( double a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scale_kernel(a,in,out) {
    install_mpi_factory(); //install_mpi_kernel_factory(this);
#ifdef VT
    VT_funcdef("scale kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
  mpi_scale_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scale_kernel(a,in,out) {
    install_mpi_factory(); //install_mpi_kernel_factory(this);
#ifdef VT
    VT_funcdef("scale kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
};

/*!
  Scale a vector down by a scalar.
*/
class mpi_scaledown_kernel : public mpi_kernel,public scaledown_kernel {
public:
  mpi_scaledown_kernel( double a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scaledown_kernel(a,in,out),mpi_kernel(in,out) {
#ifdef VT
    VT_funcdef("scaledown kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
  mpi_scaledown_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scaledown_kernel(a,in,out),mpi_kernel(in,out) {
#ifdef VT
    VT_funcdef("scaledown kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
};

/*!
  AXPY is a local kernel; the scalar is passed as the context. The scalar
  comes in as double*, this leaves open the possibility of an array of scalars,
  also it puts my mind at ease re that casting to void*.
*/
class mpi_axpy_kernel : public mpi_kernel,public axpy_kernel {
public:
  mpi_axpy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,double *x )
    : kernel(in,out),axpy_kernel(in,out,x),mpi_kernel(in,out) {
#ifdef VT
    VT_funcdef("axpy kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
};

/*!
  Add two vectors together. For a more general version see \ref mpi_axbyz_kernel.
 */
class mpi_sum_kernel : public mpi_kernel,public sum_kernel {
public:
  mpi_sum_kernel( std::shared_ptr<object> in1,std::shared_ptr<object> in2,std::shared_ptr<object> out )
    : kernel(in1,out),sum_kernel(in1,in2,out),mpi_kernel(in1,out) {
#ifdef VT
    VT_funcdef("sum kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
};

/*!
  MPI version of the \ref axbyz_kernel.
*/
class mpi_axbyz_kernel : public mpi_kernel,public axbyz_kernel {
protected:
public:
  mpi_axbyz_kernel( char op1,std::shared_ptr<object> s1,std::shared_ptr<object> x1,
		    char op2,std::shared_ptr<object> s2,std::shared_ptr<object> x2,
		    std::shared_ptr<object> out )
    : kernel(x1,out),mpi_kernel(x1,out),axbyz_kernel(op1,s1,x1,op2,s2,x2,out) {
#ifdef VT
    VT_funcdef("axbyz kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
  //! Abbreviated creation leaving the scalars untouched
  mpi_axbyz_kernel( std::shared_ptr<object> s1,std::shared_ptr<object> x1,
		    std::shared_ptr<object> s2,std::shared_ptr<object> x2,
		    std::shared_ptr<object> out ) :
    mpi_axbyz_kernel( '+',s1,x1, '+',s2,x2, out ) {};
};

/*!
  A whole class for operations between replicated scalars.
  See \ref scalar_kernel and \ref char_scalar_op for the supported operations.
*/
class mpi_scalar_kernel : virtual public mpi_kernel,virtual public scalar_kernel {
protected:
public:
  mpi_scalar_kernel
  ( std::shared_ptr<object> in1,const std::string op,std::shared_ptr<object> in2,
    std::shared_ptr<object> out )
    : kernel(in1,out),scalar_kernel(in1,op,in2,out),mpi_kernel(in1,out) {
#ifdef VT
    VT_funcdef("scalar kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
};

/*! A sparse matrix-vector product is easily defined
  from the local product routine and using the 
  same sparse matrix as the index pattern of the beta distribution
*/
//snippet spmvpkernel
class mpi_spmvp_kernel : public mpi_kernel {
public:
  mpi_spmvp_kernel
      ( std::shared_ptr<object> in,std::shared_ptr<object> out,
	std::shared_ptr<sparse_matrix> mat)
	: //mpi_kernel(in,out),
	  kernel(in,out) {
    install_mpi_factory(); //install_mpi_kernel_factory(this);
    set_name(fmt::format("sparse-mvp{}",get_out_object()->get_object_number()));
    auto &d = set_last_dependency();
    d.set_index_pattern( mat );
//snippet end
    d.set_name( fmt::format("spmvp-into-{}",out->get_name()) );
#ifdef VT
    VT_funcdef("spmvp kernel",VT_NOCLASS,&vt_kernel_class);
#endif
//snippet spmvpkernel
    set_localexecutefn
      ( [mat] ( kernel_function_args ) -> void {
	return local_sparse_matrix_vector_multiply( kernel_function_call,mat ); } );
  };
  //! We perform the regular kernel analysis
  virtual void analyze_dependencies(bool trace=false) override {
    /*mpi*/kernel::analyze_dependencies(trace);
  };
};
//snippet end

//snippet sidewaysdownkernel
//! \todo get that set_name to work.
class mpi_sidewaysdown_kernel : virtual public mpi_kernel {
private:
  std::shared_ptr<distribution> level_dist,half_dist;
  std::shared_ptr<object> expanded,multiplied;
  std::shared_ptr<kernel> expand,multiply,sum;
public:
  mpi_sidewaysdown_kernel
      ( std::shared_ptr<object> top,std::shared_ptr<object> side,std::shared_ptr<object> out,
	std::shared_ptr<sparse_matrix> mat )
    : kernel(top,out),mpi_kernel(top,out) {
    //snippet end
	set_cookie(entity_cookie::SHELLKERNEL);
    int step = out->get_object_number();
    level_dist = out->get_distribution();
    if (level_dist==nullptr)
      throw(std::string("could not cast out object to dist"));
    half_dist = top->get_distribution();
    if (half_dist==nullptr)
      throw(std::string("could not cast top object to dist"));

    //expanded = new mpi_object(level_dist);
    expanded = level_dist->new_object(level_dist);
    expanded->set_name(fmt::format("parent-expanded-{}",step));
    expand = std::shared_ptr<kernel>( new mpi_kernel(top,expanded) );
    expand->set_localexecutefn( &scanexpand );
    expand->set_last_dependency().set_signature_function_function
      ( [] (index_int i) -> std::shared_ptr<indexstruct> {
	return halfinterval(i); } );
    //( &halfinterval );
    expand->set_name(fmt::format("parent-expand-{}",step));

    //multiplied = new mpi_object(level_dist);
    multiplied = level_dist->new_object(level_dist);
    multiply = std::shared_ptr<kernel>( new mpi_spmvp_kernel(side,multiplied,mat) );
    multiplied->set_name(fmt::format("cousin-product-{}",step));
    multiply->set_name(fmt::format("cousin-multiply-{}",step));
    
    sum = std::shared_ptr<kernel>( new mpi_sum_kernel(expanded,multiplied,out) );
    sum->set_name(fmt::format("sidewaysdown-sum{}",get_out_object()->get_object_number()));
#ifdef VT
    VT_funcdef("sidewaysdown kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
  virtual void set_name( std::string n ) override {
    expand->set_name(fmt::format("{}-{}",n,expand->get_name()));
    expanded->set_name(fmt::format("{}-{}",n,expanded->get_name()));
    multiplied->set_name(fmt::format("{}-{}",n,multiplied->get_name()));
    multiply->set_name(fmt::format("{}-{}",n,multiply->get_name()));
    sum->set_name(fmt::format("{}-{}",n,sum->get_name()));
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks(bool trace=false) override {
    split_contained_kernels(expand,multiply,sum);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies(bool trace=false) override {
    analyze_contained_kernels(expand,multiply,sum);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute(bool trace=false) override {
    expand->execute(trace); multiply->execute(trace); sum->execute(trace);
  };
  std::string as_string() override {
    fmt::memory_buffer w;
    format_to(w.end(),"K[{}]=\n",get_name());
    format_to(w.end(),"  <<{}>>\n",expand->as_string());
    format_to(w.end(),"  <<{}>>\n",multiply->as_string());
    format_to(w.end(),"  <<{}>>",sum->as_string());
    return to_string(w);
  };
};

//snippet centerofmass
class mpi_centerofmass_kernel : virtual public mpi_kernel {
public:
  mpi_centerofmass_kernel(std::shared_ptr<object> bot,std::shared_ptr<object> top,int k)
    : kernel(bot,top),mpi_kernel(bot,top) {
    set_localexecutefn
      ( [k] ( kernel_function_args ) -> void {
	scansumk( kernel_function_call,k ); } );
    set_last_dependency().set_signature_function_function
      ( [] (index_int i) -> std::shared_ptr<indexstruct> {
	return doubleinterval(i); } );
  };
  mpi_centerofmass_kernel(std::shared_ptr<object> bot,std::shared_ptr<object> top) :
    mpi_centerofmass_kernel(bot,top,1) {};
};
//snippet end

/*!
  For now the reduction is only a sum reduction.

  \todo parametrize this with the combination function.
  \todo write a separate page about composite kernels?

  An reduction is somewhat tricky. While it behaves as a single kernel,
  it is composed of two kernels: one local sum followed by a gather,
  followed by a redundantly computed local sum.

  This means we have to overwrite #analyze_dependencies and #execute
  to call the corresponding routines of the three enclosed kernels.

  \todo test  that "global_sum" is replicated_scalar et cetera
  \todo make constructor s/t top_summing replicated_distribution( new indexstruct(0,ngroups) )
 */
//snippet reductionkernel
class mpi_reduction_kernel : virtual public mpi_kernel {
private: // we need to keep them just to destroy them
  //  distribution *local_scalar;
public:
  mpi_reduction_kernel( std::shared_ptr<object> local_value,std::shared_ptr<object> global_sum)
    : kernel(local_value,global_sum),mpi_kernel(local_value,global_sum) {
    //fmt::print("Mpi reduction of type {}\n",global_sum->strategy_as_string());
    auto sumdistro = global_sum->get_distribution();
    if (!sumdistro->has_type_replicated())
      throw(fmt::format("Reduction output needs to be replicated, not {}",
			sumdistro->type_as_string()));
    //snippet end
    const auto &decomp = sumdistro->get_decomposition();
    set_name(fmt::format("scalar reduction{}",get_out_object()->get_object_number()));
    if (0) {
    } else if (decomp.has_collective_strategy(collective_strategy::RECURSIVE)) {
      set_cookie(entity_cookie::SHELLKERNEL);
      setup_recursive_reduction(local_value,global_sum);
    } else if (decomp.has_collective_strategy(collective_strategy::GROUP)) {
      set_cookie(entity_cookie::SHELLKERNEL);
      setup_grouped_reduction(local_value,global_sum);
    } else if (decomp.has_collective_strategy(collective_strategy::ALL_PTP)
	       || decomp.has_collective_strategy(collective_strategy::MPI)) {
      setup_ptp_reduction(decomp,local_value,global_sum);
    } else
      throw(fmt::format("Unknown collective strategy {}",env->get_collective_strategy()));
#ifdef VT
    VT_funcdef("reduction kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
protected:
  // stuff for the recursive implementation
  std::vector<std::shared_ptr<distribution>> levels; int nlevels;
  std::vector<std::shared_ptr<object>> objects;
  std::vector<std::shared_ptr<kernel>> kernels;
public:
  void setup_recursive_reduction(std::shared_ptr<object> local_value,std::shared_ptr<object> global_sum) {
    //auto cur_level = std::shared_ptr<distribution>
    //    ( new distribution(local_value->get_distribution().get()) );
    auto cur_level = local_value->get_distribution();
    std::shared_ptr<object> cur_object = local_value;
    ioperator div2("/2");
    //snippet recursivereduction
    for (nlevels=1; nlevels<100; nlevels++) {
      index_int lsize = cur_level->outer_size();
      //fmt::print("recursive level {}: {}\n",nlevels,cur_level->as_string());
      if (lsize==1) break; // we're done; discard the operate distribution and put the outobject
      levels.push_back( cur_level );
      objects.push_back( cur_object );
      try {
	cur_level = cur_level->operate(div2);
	cur_object = cur_level->new_object(cur_level);
      } catch (std::string c) {
	fmt::print("Error <<{}>> on dist <<{}>> at level {}\n",c,cur_level->as_string(),nlevels);
	  throw(std::string("Setup of recursive reduction failed")); }
    }
    if (levels.size()==0) { // this happens with one processor: we already have a scalar
      //fmt::print("recursive reduction, degenerate case\n");
      levels.push_back( cur_level );
      objects.push_back( cur_object );
    }
    levels.push_back( global_sum->get_distribution() );
    objects.push_back( global_sum );
    //snippet end
    if ( nlevels!=levels.size() )
      throw(fmt::format("level derivation failed {}-{}",nlevels,levels.size()));
  };
  void split_to_tasks_recursive() {
    if (nlevels==0)
      throw(fmt::format("No levels to split in <<{}>>",get_name()));
    else if (nlevels==1) {
      auto step = std::shared_ptr<kernel>( new mpi_copy_kernel(objects.at(0),objects.at(1)) );
      kernels.push_back(step);
      step->split_to_tasks();
      set_kernel_tasks( step->get_tasks() );
    } else {
      for (int level=0; level<nlevels-1; level++) {
	auto step = std::shared_ptr<kernel>( new mpi_kernel(objects[level],objects[level+1]) );
	step->set_name( fmt::format("allreduce-level-{}",level) );
	step->set_last_dependency().set_signature_function_function
	  ( [] (index_int i) -> std::shared_ptr<indexstruct> {
	    return doubleinterval(i); } );
	step->set_localexecutefn( &scansum );
	kernels.push_back(step);
	step->split_to_tasks();
	if (level==0)
	  set_kernel_tasks( step->get_tasks() );
	else
	  addto_kernel_tasks( step->get_tasks() );
      }
    }
  };
protected:
  // stuff for the grouping implementation
  std::shared_ptr<distribution> locally_grouped{nullptr},partially_reduced{nullptr};
  std::shared_ptr<object> partial_sums{nullptr};
  std::shared_ptr<kernel> sumkernel{nullptr},groupkernel{nullptr};
  std::shared_ptr<kernel> partial_summing,top_summing;
public:
  void setup_grouped_reduction(std::shared_ptr<object> local_value,std::shared_ptr<object> global_sum) {

    auto sumdistro = global_sum->get_distribution();
    const auto &decomp = sumdistro->get_decomposition();
    //snippet groupedreduction
    int
      P=decomp.domains_volume(), ntids=P, mytid=decomp.mytid(), g=P-1;
    int groupsize = 4*( (sqrt(P)+3)/4 );

    int
      //mygroupnum = mytid/groupsize,
      nfullgroups = ntids/groupsize,
      grouped_tids = nfullgroups*groupsize, // how many procs are in perfect groups?
      remainsize = P-grouped_tids, ngroups = nfullgroups+(remainsize>0);
    //snippet groupedreduction
    parallel_structure groups = parallel_structure(decomp);
    try {
      for (int p=0; p<P; p++) {
	auto pcoord = decomp.coordinate_from_linear(p);
	index_int groupnumber = p/groupsize,
	  f = groupsize*groupnumber,l=MIN(f+groupsize-1,g);
	groups.set_processor_structure
	  (pcoord,
	   std::shared_ptr<multi_indexstruct>
	   ( new multi_indexstruct
	     ( std::vector<std::shared_ptr<indexstruct>>{
	       std::shared_ptr<indexstruct>( new contiguous_indexstruct(f,l) ) } ) )
	   );
	// groups.set_processor_structure
	//   (p, std::shared_ptr<indexstruct>( new contiguous_indexstruct(f,l) ) );
      }
      groups.set_structure_type( groups.infer_distribution_type() );
      locally_grouped = std::shared_ptr<distribution>( new mpi_distribution(groups) );
    } catch (std::string c) {
      throw(fmt::format("Local grouping distribution failed: {}",c)); }

    parallel_structure partials = parallel_structure(decomp);
    try {
      for (int p=0; p<P; p++) {
	index_int groupnumber = p/groupsize;
	partials.set_processor_structure
	  (p, std::shared_ptr<indexstruct>( new contiguous_indexstruct(groupnumber) ) );
      }
      partials.set_structure_type( partials.infer_distribution_type() );
      partially_reduced = std::shared_ptr<distribution>( new mpi_distribution(partials) );
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> in partial reduction",c)); }
    //snippet end

    try {
      //partial_sums = new mpi_object(partially_reduced);
      partial_sums = partially_reduced->new_object(partially_reduced);
      partial_summing = std::shared_ptr<kernel>( new mpi_kernel(local_value,partial_sums) );
      partial_summing->set_name
	(fmt::format
	 ("reduction:partial-sum-local-to-group{}",get_out_object()->get_object_number()));
      partial_summing->set_explicit_beta_distribution(locally_grouped);
      partial_summing->set_localexecutefn( &summing );
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> in partial summing",c)); }

    top_summing = std::shared_ptr<kernel>( new mpi_kernel(partial_sums,global_sum) );
    top_summing->set_name
      (fmt::format("reduction:group-sum-to-global{}",get_out_object()->get_object_number()));
    parallel_structure top_beta = parallel_structure(decomp);
    for (int p=0; p<P; p++)
      top_beta.set_processor_structure
	(p, std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,ngroups-1)) );
    top_beta.set_structure_type( top_beta.infer_distribution_type() );    
    top_summing->set_explicit_beta_distribution( std::shared_ptr<distribution>( new mpi_distribution(top_beta) ) );
    top_summing->set_localexecutefn( &summing );
    //snippet end
  };
protected: // mpi & ptp reduction data
public: // mpi & ptp reduction routines
  void setup_ptp_reduction( const decomposition &decomp,std::shared_ptr<object> local_value,std::shared_ptr<object> global_sum) {
    //snippet reductionkernel
    sumkernel = std::shared_ptr<kernel>( new mpi_kernel(local_value,global_sum) );
    sumkernel->set_name
      (fmt::format("reduction:one-step-sum{}",get_out_object()->get_object_number()));
    sumkernel->set_last_dependency().set_explicit_beta_distribution
      ( std::shared_ptr<distribution>( new mpi_gathered_distribution(decomp) ) );
    // sumkernel->set_last_dependency().set_is_collective
    //   ( decomp.has_collective_strategy(collective_strategy::MPI) );
    sumkernel->set_localexecutefn( &summing );
    //snippet end
  };
public:
  virtual void split_to_tasks(bool trace=false) override {
    if (kernel_has_tasks()) return;
    const auto &decomp = get_out_object()->get_decomposition();
    if (0) {
    } else if (decomp.has_collective_strategy(collective_strategy::RECURSIVE)) {
      split_to_tasks_recursive();
    } else if (decomp.has_collective_strategy(collective_strategy::GROUP)) {
      partial_summing->split_to_tasks(); set_kernel_tasks( partial_summing->get_tasks() );
      top_summing->analyze_dependencies(); addto_kernel_tasks( top_summing->get_tasks() );
    } else if (decomp.has_collective_strategy(collective_strategy::ALL_PTP)
	       || decomp.has_collective_strategy(collective_strategy::MPI)) {
      sumkernel->split_to_tasks(); set_kernel_tasks( sumkernel->get_tasks() );
      // if (decomp.has_collective_strategy(collective_strategy::MPI)) {
      // 	auto tsks = sumkernel->get_tasks();
      // 	for ( auto t : *tsks )
      // 	  t->set_is_collective();
      // }
    } else throw(fmt::format("Strange collective strategy in <<{}>>",get_out_object()->get_name()));
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies(bool trace=false) override {
    const auto &decomp = get_out_object()->get_decomposition();
    try {
      this->split_to_tasks();
    } catch (std::string c) {
      throw(fmt::format("MPI reduct kernel analyze deps failed splitting: <<{}>>",c)); }
    try {
      if (0) {
      } else if (decomp.has_collective_strategy(collective_strategy::RECURSIVE)) {
	for ( auto k : kernels )
	  k->analyze_dependencies(trace);
      } else if (decomp.has_collective_strategy(collective_strategy::GROUP)) {
	partial_summing->analyze_dependencies(trace);
	top_summing->analyze_dependencies(trace);
      } else if (decomp.has_collective_strategy(collective_strategy::ALL_PTP)
		 || decomp.has_collective_strategy(collective_strategy::MPI)) {
	sumkernel->analyze_dependencies(trace);
      }
    } catch (std::string c) {
      throw(fmt::format("MPI reduct kernel analyze deps failed: <<{}>>",c)); }
  }
  void execute(bool trace=false) override {
    const auto &decomp = get_out_object()->get_decomposition();
    if (0) {
    } else if (decomp.has_collective_strategy(collective_strategy::RECURSIVE)) {
      for ( auto k : kernels )
	k->execute(trace);
    } else if (decomp.has_collective_strategy(collective_strategy::GROUP)) {
      partial_summing->execute(trace);
      top_summing->execute(trace);
    } else if (decomp.has_collective_strategy(collective_strategy::ALL_PTP)
	       || decomp.has_collective_strategy(collective_strategy::MPI)) {
      sumkernel->execute(trace);
    }
  };
  //  int get_groupsize() { return groupsize; };
};

//snippet innerproductkernel
/*!
  An inner product is somewhat tricky. While it behaves as a single kernel,
  it is composed of two kernels: one gather followed by a local summing.
  This means we have to overwrite #analyze_dependencies and #execute
  to call the corresponding routines of the two enclosed kernels.

  \todo add an option for an ortho parameter
  \todo test that "global_sum" is replicated_scalar
  \todo test that v1 v2 have equal distributions
*/
#if 1
class mpi_innerproduct_kernel : public mpi_kernel,public innerproduct_kernel {
public:
  mpi_innerproduct_kernel( std::shared_ptr<object> v1,std::shared_ptr<object> v2,std::shared_ptr<object> global_sum)
    : kernel(v1,global_sum),mpi_kernel(v1,global_sum),
      innerproduct_kernel(v1,v2,global_sum) {
  }
};
#else
class mpi_innerproduct_kernel : virtual public mpi_kernel {
private:
  std::shared_ptr<distribution> local_scalar; // we need to keep them just to destroy them
  std::shared_ptr<object> local_value;
protected:
  std::shared_ptr<kernel> prekernel,sumkernel;
public:
  mpi_innerproduct_kernel( std::shared_ptr<object> v1,std::shared_ptr<object> v2,std::shared_ptr<object> global_sum)
    : kernel(v1,global_sum),mpi_kernel(v1,global_sum) {
    //snippet end
    set_cookie(entity_cookie::SHELLKERNEL);
    auto decomp = global_sum->get_distribution();
    int step = v2->get_object_number();
    set_name(fmt::format("inner-product{}",get_out_object()->get_object_number()));

    // intermediate object for local sum:
    local_scalar = std::shared_ptr<distribution>( new mpi_scalar_distribution(decomp) );
    //local_value = new mpi_object(local_scalar);
    local_value = local_scalar->new_object(local_scalar);
    local_value->set_name(fmt::format("local-inprod-value{}",step));
    
    // local inner product kernel
    prekernel = std::shared_ptr<kernel>( new mpi_kernel(v1,local_value) );
    prekernel->set_name(fmt::format("local-innerproduct{}",get_out_object()->get_object_number()));
    // v1 is in place
    {
      auto d = prekernel->set_last_dependency();
      d.set_explicit_beta_distribution(v1);
      d.set_name("inprod wait for in vector");
    }
    // v2 is in place
    {
      prekernel->add_in_object(v2);
      auto d = prekernel->set_last_dependency();
      d.set_explicit_beta_distribution(v2);
      d.set_name("inprod wait for second vector");
    }
    // function
    prekernel->set_localexecutefn( &local_inner_product );

    // reduction kernel
    sumkernel = std::shared_ptr<kernel>( new mpi_reduction_kernel(local_value,global_sum) );
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks(bool trace=false) override {
    split_contained_kernels(prekernel,sumkernel);
  };
  /*! We override the default analysis by analyzing in sequence the contained kernels.
    This splits the kernels and analyzes task dependencies; then we take the tasks
    and add them to the innerproduct task list.
   */
  virtual void analyze_dependencies(bool trace=false) override {
    analyze_contained_kernels(prekernel,sumkernel);
    // for ( auto t : *get_tasks() )
    //   fmt::print("inprod task: {}\n",t->as_string());
  };
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute(bool trace=false) override { // synckernel->execute(trace);
    prekernel->execute(trace); sumkernel->execute(trace); };
  auto get_prekernel() { return prekernel; };
};
#endif

/*!
  A vector norm is like an inner product, but since it has only one input
  we don't need the synckernel
*/
#if 1
class mpi_norm_kernel : public mpi_kernel,public norm_kernel {
public:
  mpi_norm_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),mpi_kernel(in,out),norm_kernel(in,out) {};
};
#else
class mpi_norm_kernel : virtual public mpi_kernel {
private:
  mpi_distribution *local_scalar,*gathered_scalar;
  std::shared_ptr<object> local_value,*squared;
  int groupsize{4};
protected:
  std::shared_ptr<kernel> prekernel,sumkernel,rootkernel;
public:
  mpi_norm_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),mpi_kernel(in,out) {
    set_cookie(entity_cookie::SHELLKERNEL);
    int step = out->get_object_number();
    set_name(fmt::format("norm{}",get_out_object()->get_object_number()));

    // intermediate object for local sum:
    local_scalar = std::shared_ptr<distribution>( new mpi_scalar_distribution(out) );
    //local_value = new mpi_object(local_scalar);
    local_value = local_scalar->new_object(local_scalar);
    local_value->set_name(fmt::format("local-inprod-value{}",step));
    
    // local norm kernel
    prekernel = std::shared_ptr<kernel>( new mpi_kernel(in,local_value) );
    prekernel->set_name(fmt::format("local-norm-compute{}",get_out_object()->get_object_number()));
    prekernel->set_last_dependency().set_explicit_beta_distribution( in );
    prekernel->set_localexecutefn( &local_normsquared );

    // reduction kernel
    auto auto_dist = out->get_distribution();
    //squared = new mpi_object(out_dist);
    squared = out_dist->new_object(out_dist);
    squared->set_name(fmt::format("squared-inprod-value{}",step));
    sumkernel = std::shared_ptr<kernel>( new mpi_reduction_kernel(local_value,squared) );

    // we need to take a root
    rootkernel = std::shared_ptr<kernel>( new mpi_kernel(squared,out) );
    rootkernel->set_name(fmt::format("root of squares{}",get_out_object()->get_object_number()));
    rootkernel->set_explicit_beta_distribution( squared );
    rootkernel->set_localexecutefn( &vectorroot );
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks(bool trace=false) override {
    split_contained_kernels(prekernel,sumkernel,rootkernel);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies(bool trace=false) override {
    analyze_contained_kernels(prekernel,sumkernel,rootkernel);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute(bool trace=false) override {
    prekernel->execute(trace); sumkernel->execute(trace); rootkernel->execute(trace);
  };
  int get_groupsize() { return groupsize; };
};
#endif

/*!
  Norm squared is much like norm. 
*/
#if 1
class mpi_normsquared_kernel : public mpi_kernel,public normsquared_kernel {
public:
  mpi_normsquared_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),mpi_kernel(in,out),normsquared_kernel(in,out) {};
};
#else 
class mpi_normsquared_kernel : virtual public mpi_kernel {
private:
  std::shared_ptr<distribution> local_scalar,gathered_scalar;
  std::shared_ptr<object> local_value;
protected:
  std::shared_ptr<kernel> prekernel,sumkernel;
public:
  mpi_normsquared_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),mpi_kernel(in,out) {
    set_cookie(entity_cookie::SHELLKERNEL);
    set_name(fmt::format("norm squared{}",get_out_object()->get_object_number()));

    // intermediate object for local sum:
    local_scalar = out->get_distribution(); 
    local_value = local_scalar->new_object(local_scalar);
    local_value->set_name("local-inprod-value");
    
    // local norm kernel
    prekernel = std::shared_ptr<kernel>( new mpi_kernel(in,local_value) );
    prekernel->set_name(fmt::format("local-norm{}",get_out_object()->get_object_number()));
    prekernel->set_last_dependency().set_explicit_beta_distribution(in->get_distribution()); 
    prekernel->set_localexecutefn( &local_normsquared );

    // reduction kernel
    sumkernel = std::shared_ptr<kernel>( new mpi_reduction_kernel(local_value,out) );
  };
  //! We override the default split by splitting in sequence the contained kernels.
  virtual void split_to_tasks(bool trace=false) override {
    split_contained_kernels(prekernel,sumkernel,trace);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies(bool trace=false) override {
    analyze_contained_kernels(prekernel,sumkernel,trace);
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute(bool trace=false) override {
    prekernel->execute(trace);
    sumkernel->execute(trace);
  };
};
#endif

class mpi_stats_kernel : virtual public mpi_kernel {
private:
  std::shared_ptr<kernel> local_stats,global_stats;
public:
  mpi_stats_kernel( std::shared_ptr<object> in, std::shared_ptr<object> out, kernel_function f )
    : kernel(in,out),mpi_kernel(in,out) {
    auto decomp = in->get_distribution()->get_decomposition(); 
    
    auto
      scalar_structure = std::shared_ptr<distribution>( new mpi_block_distribution(decomp,1,-1) );
    auto local_value = scalar_structure->new_object(scalar_structure);
    // = new mpi_object(scalar_structure);
    local_stats = std::shared_ptr<kernel>( new mpi_kernel(in,local_value) );
    local_stats->set_localexecutefn(f);
    local_stats->set_explicit_beta_distribution(in->get_distribution());

    global_stats = std::shared_ptr<kernel>( new mpi_kernel(local_value,out) );
    global_stats->set_localexecutefn( &veccopy );
    global_stats->set_explicit_beta_distribution(out->get_distribution());
  };
  virtual void split_to_tasks(bool trace=false) override {
    split_contained_kernels(local_stats,global_stats);
  };
  virtual void analyze_dependencies(bool trace=false) override {
    analyze_contained_kernels(local_stats,global_stats);
  };
  virtual void execute(bool trace=false) override {
    local_stats->execute(trace); global_stats->execute(trace);
  }
};

/*!
  Preconditioning is an unexplored topic

  \todo add a context, passing a sparse matrix
*/
class mpi_preconditioning_kernel : public mpi_kernel,public preconditioning_kernel {
public:
  mpi_preconditioning_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),mpi_kernel(in,out),preconditioning_kernel(in,out) {};
};

/*!
  An outer product kernel is a data parallel application of a redundantly 
  distributed array of length k with a disjointly distributed array of 
  size N, giving Nk points. The redundant array is stored as context,
  and the explosion function is passed as parameter.
  \todo change that local dependency type to explicit
 */
class mpi_outerproduct_kernel : public mpi_kernel {
public:
  //snippet mpioutprodkernel
 mpi_outerproduct_kernel
     ( std::shared_ptr<object> in,std::shared_ptr<object> out,
       std::shared_ptr<object> replicated,void (*f) (kernel_function_types) )
       : mpi_kernel(in,out) {
   const auto indistro = in->get_distribution(),
     repldistro = replicated->get_distribution();
   set_last_dependency().set_explicit_beta_distribution(indistro);
   // add replicated object as another dependency
   add_in_object(replicated);
   set_last_dependency().set_explicit_beta_distribution(repldistro);
   
   set_name(fmt::format("outer product{}",get_out_object()->get_object_number()));
   set_localexecutefn(f);
 };
  //snippet end
  //! \todo why isn't this inherited from mpi_kernel
  task *make_task_for_domain(processor_coordinate &d,std::shared_ptr<object> in,std::shared_ptr<object> out) {
    return new mpi_task(d,in,out); };
};

//! Trace kernel only has an input object
//! \todo move that out_object to the imp_trace_kernel
class mpi_trace_kernel : virtual public mpi_kernel,public trace_kernel {
public:
  mpi_trace_kernel(std::shared_ptr<object> in,std::string c)
    : kernel(),mpi_kernel(),trace_kernel(in,c) {
    const auto indistro = in->get_distribution();
    out_object = indistro->new_object(indistro);
    out_object->set_name(fmt::format("trace-{}",in->get_name()));
  };
};

//! An MPI central differences kernel is exactly the same as \ref centraldifference_kernel
class mpi_centraldifference_kernel : public mpi_kernel,public centraldifference_kernel {
public:
  mpi_centraldifference_kernel(std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel(in,out),mpi_kernel(in,out),centraldifference_kernel(in,out) {
  };
};

//! \todo make an independent diffusion kernel class
class mpi_diffusion_kernel : virtual public mpi_kernel {
public:
  mpi_diffusion_kernel(std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel(in,out),mpi_kernel(in,out) {
    add_sigma_operator( ioperator("none") ); // we need the i index
    add_sigma_operator( ioperator(">=1") );  // we need the i+1 index
    add_sigma_operator( ioperator("<=1") );  // we need the i-1 index
    double damp = 1./6;
    set_localexecutefn
      ( [damp] ( kernel_function_args ) -> void {
	return central_difference_damp( kernel_function_call,damp ); } );
  };
};

#endif
