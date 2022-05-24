// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** omp_base.h: Headers for the OpenMP classes
 ****
 ****************************************************************/
#ifndef OMP_BASE_H
#define OMP_BASE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>
#include "imp_base.h"
#include "imp_functions.h"

/****
 **** Basics
 ****/

class omp_architecture;
class omp_environment : public environment {
private:
public:
  omp_environment(int,char**);
  // omp_environment( omp_environment& other ) : environment( other ) {
  //   arch = other.get_architecture(); };
  ~omp_environment();
  virtual architecture make_architecture() override; // pure virtual
  virtual void print_options() override;
  void print_stats() override;
  void record_task_executed();
};

//#endif

/****
 **** Architecture
 ****/

// index_int omp_allreduce(index_int contrib);
// double omp_allreduce_d(double contrib);
//int omp_allreduce_and(int contrib);
std::vector<index_int> *omp_gather(index_int contrib);

/*!
  In an OpenMP architecture we can only ask for \ref nprocs; 
  we do not override the exception when asking for 
  \ref architecture_data::mytid or
  \ref architecture_data::nthreads_per_node
 */
class omp_architecture : public architecture {
protected:
public :
  omp_architecture( int n )
    : architecture(n) {
    type = architecture_type::SHARED; protocol = protocol_type::OPENMP;
    beta_has_local_addres_space = 0;
    set_name(fmt::format("omp-architecture-on-{}",n));
  };

  virtual std::string as_string() override;
  virtual void set_power_mode() override { set_can_embed_in_beta(); };

  /*
   * Collectives
   */
  std::vector<index_int> *omp_gather(index_int contrib);
  int omp_reduce_scatter(int *senders,int root);
};

void omp_gather64(index_int contrib,std::vector<index_int> &gathered);

class omp_decomposition : public decomposition {
public:
  omp_decomposition() : decomposition() {};
  omp_decomposition( const architecture &arch,processor_coordinate &grid );
  omp_decomposition( const architecture &arch,processor_coordinate &&grid );
  omp_decomposition( const architecture &arch );
  virtual std::string as_string() override {
    return fmt::format("omp {}",decomposition::as_string());
  };
  void set_decomp_factory();
};

void make_omp_communicator(communicator *cator,int P);

/****
 **** Distribution
 ****/

index_int omp_location_of_first_index
    ( std::shared_ptr<distribution> d,const processor_coordinate &pcoord);
index_int omp_location_of_last_index
    ( std::shared_ptr<distribution> d,const processor_coordinate &pcoord);

//! OpenMP distribution differs from MPI in the use of shared memory.
class omp_distribution : virtual public distribution {
public:
  omp_distribution( const decomposition &d );
  omp_distribution( const parallel_structure& );
  omp_distribution( const parallel_structure&& );
  omp_distribution( const decomposition &d,index_int(*pf)(int,index_int),index_int nlocal );
  omp_distribution( std::shared_ptr<distribution> other );

  //  std::shared_ptr<distribution> omp_dist_operate_base( std::shared_ptr<ioperator> );
  std::shared_ptr<distribution> omp_dist_distr_union( distribution* );

  // NUMA and such
  void set_omp_routines();
  void set_numa();
};

class omp_block_distribution
  : public omp_distribution,public block_distribution,virtual public distribution {
public:
  // OpenMP block distribution from local / global
  omp_block_distribution(const decomposition &d,int o,index_int l,index_int g);
  //! Constructor with implicit ortho=1
  omp_block_distribution(const decomposition &d,index_int l,index_int g)
    : omp_block_distribution(d,1,l,g) {};
  //! Constructor with implicit ortho=1 and local is deduced
  omp_block_distribution(const decomposition &d,index_int g)
    : omp_block_distribution(d,-1,g) {};

  //! Constructor from vector of explicit block sizes. The gsize is ignored.
  omp_block_distribution(const decomposition &d,domain_coordinate &gsize )
    : distribution(d),omp_distribution(d),block_distribution(d,gsize) { memoize(); };
  //! Constructor from vector of explicit block sizes. The gsize is ignored.
  omp_block_distribution(const decomposition &d,domain_coordinate &&gsize )
    : distribution(d),omp_distribution(d),block_distribution(d,gsize) { memoize(); };
  //! Constructor from multi_indexstruct
  omp_block_distribution( const decomposition &d,std::shared_ptr<multi_indexstruct> idx )
    : distribution(d),omp_distribution(d),block_distribution(d,idx) { memoize(); };

  // 1D from array of local sizes
  omp_block_distribution(const decomposition &d,std::vector<index_int> sizes);

};

class omp_scalar_distribution : public omp_distribution,public scalar_distribution {
public:
  omp_scalar_distribution(const decomposition &d)
    : distribution(d),omp_distribution(d),scalar_distribution(d) { memoize(); };
};

class omp_cyclic_distribution : public omp_distribution,public cyclic_distribution {
public:
  omp_cyclic_distribution(const decomposition &d,int l,int g)
    : distribution(d),omp_distribution(d),cyclic_distribution(d,l,g) { memoize(); };
  omp_cyclic_distribution(const decomposition &d,int l) : omp_cyclic_distribution(d,l,-1) {};
};

class omp_replicated_distribution : public omp_distribution,public replicated_distribution {
public:
  omp_replicated_distribution(const decomposition &d,int ortho,index_int l)
    : distribution(d),omp_distribution(d),replicated_distribution(d,ortho,l) { memoize(); };
  omp_replicated_distribution(const decomposition &d,index_int l)
    : omp_replicated_distribution(d,1,l) {};
  omp_replicated_distribution(const decomposition &d)
    : omp_replicated_distribution(d,1) {};
  // //! Each domain has its own block of data
  // omp_replicated_distribution(const decomposition &d,int l)
  //   : distribution(d),omp_distribution(d),replicated_distribution(d,l)
  //     { memoize(); };
  // omp_replicated_distribution(const decomposition &d) : omp_replicated_distribution(d,1) {};
};

class omp_gathered_distribution : public omp_distribution,public gathered_distribution {
public:
  omp_gathered_distribution(const decomposition &d,int l)
    : distribution(d),omp_distribution(d),gathered_distribution(d,1,l) { memoize(); };
  omp_gathered_distribution(const decomposition &d) : omp_gathered_distribution(d,1) {};
};

/****
 **** Sparse matrix / index pattern
 ****/

class omp_sparse_matrix : public sparse_matrix {
 public:
  omp_sparse_matrix( std::shared_ptr<distribution> d )
    : sparse_matrix(d->get_dimension_structure(0)->get_enclosing_structure()) {
    globalsize = d->global_volume();
  };
  omp_sparse_matrix( std::shared_ptr<distribution> d,index_int g ) : omp_sparse_matrix(d) { globalsize = g; };
};

//! Make upper bidiagonal toeplitz matrix.
class omp_upperbidiagonal_matrix : public omp_sparse_matrix {
public:
  omp_upperbidiagonal_matrix( std::shared_ptr<distribution> dist, double d,double r )
    : omp_sparse_matrix(dist) {
    index_int g = dist->global_volume();
    for (index_int row=0; row<g; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row+1; if (col<g) add_element(row,col,r);
    }
  }
};

//! Make lower bidiagonal toeplitz matrix.
class omp_lowerbidiagonal_matrix : public omp_sparse_matrix {
public:
  omp_lowerbidiagonal_matrix( std::shared_ptr<distribution> dist, double l,double d )
    : omp_sparse_matrix(dist) {
    index_int g = dist->global_volume();
    for (index_int row=0; row<g; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
    }
  }
};

class omp_toeplitz3_matrix : public omp_sparse_matrix {
public:
  omp_toeplitz3_matrix( std::shared_ptr<distribution> dist, double l,double d,double r )
    : omp_sparse_matrix(dist) {
    index_int g = dist->global_volume();
    for (index_int row=0; row<g; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
      col = row+1; if (col<g) add_element(row,col,r);
    }
  }
};

/****
 **** Beta object
 ****/

/****
 **** Object
 ****/

/*!
  Data in an omp_object is allocated only once. The \ref get_data method
  returns this pointer, regardless on what `processor' is it requested.
  This is the right strategy, since each processor has its own first/last
  index.

  However, with a replicated object, each processor has the same first/last
  index, yet we still need to accomodate multiple instances of the data.
  Therefore, the constructor overallocated the data in this case,
  and \ref get_data(int) returns a unique pointer into the (single, overdimensioned)
  array.
 */
class omp_object : public object {
private:
protected:
public:
  omp_object( std::shared_ptr<distribution> d );
  //! Create from user data
  omp_object( std::shared_ptr<distribution> d, std::shared_ptr<std::vector<double>> dat );
  // Create an object from data of another object. \todo revisit registration
  omp_object( std::shared_ptr<distribution> d, std::shared_ptr<object> x );
  // this is just to make static casts possible
  void poly_object() { printf("this serves no purpose\n"); };

  /*
   * Data
   */
protected:
  //! The OpenMP object allocation has a special case for replicated objects
  void omp_allocate();
  //! Install all omp-specific data routines
  void set_data_handling() {
    allocate = [this] (void) -> void { omp_allocate(); };
  };
  void install_data_on_domains( data_pointer dat,index_int s);
    
public:
  std::string values_as_string();
  //! Mask shift compensates for the location of \ref object::first_index.
  virtual domain_coordinate mask_shift( processor_coordinate &p ) {
    if (!lives_on(p))
      throw(fmt::format("Should not ask mask shift for {} on {}",
			p.as_string(),get_name()));
    throw(fmt::format("mask shift not implemented"));
    // index_int s = 0;
    // for (int q=0; q<p; q++) if (!lives_on(q)) s += internal_local_size(q);
    // return s;
  };
  virtual void copy_data_from( std::shared_ptr<object> in,std::shared_ptr<message> smsg,std::shared_ptr<message> rmsg ) override;
};

/****
 **** Message
 ****/

class omp_message : public message {
public:
  omp_message(const decomposition &d,
	      const processor_coordinate &snd,const processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> &e,std::shared_ptr<multi_indexstruct> &g)
    : message(d,snd,rcv,e,g) {};
  omp_message(const decomposition &d,
	      const processor_coordinate &snd,const processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> &g)
    : message(d,snd,rcv,g) {};
};

/****
 **** Requests
 ****/

/*! 
  In OpenMP a request is a task on which we depend. 
  Fullfilling the wait involves copying data.
*/
class omp_request : public request {
public:
  std::shared_ptr<task> tsk{nullptr};
  std::shared_ptr<object> obj{nullptr};
  int closed{0};
public:
  omp_request( std::shared_ptr<task> t,std::shared_ptr<message> m,
	       std::shared_ptr<object> o,
	       request_type type=request_type::INCOMING )
    : request(m,request_protocol::OPENMP) { tsk = t; obj = o; this->type = type; };
  void poly() { return; }; //!< Very silly: just to make dynamic casts possible.
};

void omp_request_vector_wait( request_vector &v);

/****
 **** Kernel
 ****/

class omp_kernel : virtual public kernel {
private:
public:
  omp_kernel() : kernel() { install_omp_factory(); };
  omp_kernel( std::shared_ptr<object> out ) : kernel(out) {
    install_omp_factory(); };
  omp_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out) {
    install_omp_factory(); };
  omp_kernel( const kernel& k ) : kernel(k) { install_omp_factory(); };
  void install_omp_factory();
};

class omp_origin_kernel : public omp_kernel,public origin_kernel {
public:
  omp_origin_kernel( std::shared_ptr<object> out )
    : omp_kernel(out),kernel(out),origin_kernel(out) { install_omp_factory(); };
};

/****
 **** Task
 ****/

class omp_task : public task {
private:
protected:
  omp_lock_t dolock; //!< this locks the done variable of the base class
public: // methods
  // void make_requests_vector() {}; // requests = new omp_request_vector(); };
  // void delete_requests_vector() {}; // { delete requests; };
  omp_task( const processor_coordinate &d,kernel *k ) : task(d,k) { set_factories();
    omp_init_lock(&dolock); };
  //! Create origin kernel from anonymous kernel
  omp_task( const processor_coordinate &d,std::shared_ptr<object> out)
    : task(d,new omp_kernel(out)) { set_factories();
    omp_init_lock(&dolock); };
  //! Create compute kernel from anonymous kernel
  omp_task( const processor_coordinate &d,std::shared_ptr<object> in,std::shared_ptr<object> out)
    : task(d,new omp_kernel(in,out)) { set_factories();
    omp_init_lock(&dolock); };

  void set_factories() {
    requests_wait =
      [] (request_vector &v) -> void { omp_request_vector_wait(v); };
    // message_from_buffer = [this] (int step,std::string &buf) -> std::shared_ptr<message> {
    //   return omp_message_from_buffer(this->shared_from_this(),step,buf);
    // };
  };
  virtual void derive_send_messages(bool=false) override ;
  virtual void declare_dependence_on_task( task_id *id ) override ; // pure virtual

  // pure virtual synchronization functions
  //! In OpenMP there is no need to alert the sending party that we are ready to receive.
  virtual request_vector notifyReadyToSend( std::vector<std::shared_ptr<message>> &m )
    override { return request_vector(); };
  //! We make a dummy request for outgoing stuff. OpenMP doesn't need to send.  
  virtual std::shared_ptr<request> notifyReadyToSendMsg( std::shared_ptr<message> ) override ;
  virtual request_vector acceptReadyToSend( std::vector<std::shared_ptr<message>>&) override ;

  virtual void create_send_structure_for_dependency(int,int,dependency*) {};
  void make_infrastructure_for_sending( int n ) {}; // another pure virtual no-op

  /*
   * Execution stuff
   */
  void execute_as_root(bool=false) override; // same as base method, but with directives
protected:
  int done_on_thread{-1};
public:
  virtual void set_has_been_executed() override { task::set_has_been_executed();
    done_on_thread = omp_get_thread_num(); };
  int get_done_on_thread() { return done_on_thread; };
  virtual void check_local_executability() override;

};

class omp_origin_task : public omp_task {
public:
  omp_origin_task( processor_coordinate &d,std::shared_ptr<object> out )
    : omp_task(d,out) {
    containing_kernel->set_type( kernel_type::ORIGIN ); set_name("origin omp task");
  };
};

/****
 **** Queue
 ****/

class omp_algorithm : public algorithm {
private:
public:
  omp_algorithm() {};
  omp_algorithm(const decomposition &d)
    : algorithm(d) {
    type = algorithm_type::OMP;
    make_omp_communicator(this,this->domains_volume());
  }

  // the omp task execute has directives to make it parallel
  virtual void execute_tasks( int(*)(std::shared_ptr<task> t),bool=false ) override;

  //! Local analysis: 1. try to embed vectors in a halo. 2. find non-synchronizing tasks
  virtual void mode_analyze_dependencies() override {
    determine_locally_executable_tasks(); inherit_data_from_betas();
  };

  int find_task( int s,processor_coordinate *d );
};

#endif
