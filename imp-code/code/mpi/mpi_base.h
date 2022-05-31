// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** mpi_base.h: Header file for the MPI derived class
 ****
 ****************************************************************/

#ifndef MPI_BASE_H
#define MPI_BASE_H 1

#include <stdlib.h>
#include <cstdio>
#include <string>

#include <mpi.h>
#include <imp_base.h>
#include "imp_functions.h"

#ifdef VT
#include "VT.h"
#endif

/*! \page mpi MPI backend

  We realize an MPI backend by letting tasks synchronize through MPI messages.
  While our API still looks like sequential semantics, we now assume
  an SPMD mode of execution: each MPI process executes not just 
  the same executable, but it also never asks about information that distinguishes
  it from the other processors. This story springs a leak when we consider the
  \ref mpi_sparse_matrix class, where the matrix is constructed on each processor from a
  different local part.

  The MPI backend for IMP is characterized by the following
  - task synchronization has an active component of both sending and receiving;
  - redundant distributions are non-trivial, see \ref mpi_replicated_scalar and \ref mpi_gathered_scalar;
  - an \ref mpi_object has disjoint data allocated on each process;
  - we have some routines for packing and unpacking a message object.
 */

/****
 **** Basics
 ****/
//class mpi_architecture;
#ifdef VT
void vt_register_kernels();
#endif

/*!
  An mpi decomposition has one domain per processor by default,
  unless there is a global over-decomposition parameters.

  No one ever inherits from this, but mode-specific distributions are built from this
  because it contains a distribution factory.
*/
class mpi_decomposition : public decomposition {
public:
  mpi_decomposition() : decomposition() {};
  //! Multi-d decomposition from explicit processor grid layout
  mpi_decomposition( const architecture &arch,processor_coordinate &&grid)
    : decomposition(arch,grid) {
    int mytid = arch.mytid(); int over = arch.get_over_factor();
    for ( int local=0; local<over; local++) {
      auto mycoord = this->coordinate_from_linear(over*mytid+local);
      try {
	add_domain(mycoord);
      } catch (...) { fmt::print("trouble adding domain\n"); };
      //fmt::print("Tid {} translates to domain <<{}>>\n",mytid,mycoord->as_string());
    }
    set_decomp_factory(); 
  };
  mpi_decomposition( const architecture &arch,processor_coordinate &grid)
    : decomposition(arch,grid) {
    int mytid = arch.mytid(); int over = arch.get_over_factor();
    for ( int local=0; local<over; local++) {
      auto mycoord = this->coordinate_from_linear(over*mytid+local);
      try {
	add_domain(mycoord);
      } catch (...) { fmt::print("trouble adding domain\n"); };
      //fmt::print("Tid {} translates to domain <<{}>>\n",mytid,mycoord->as_string());
    }
    set_decomp_factory(); 
  };
  //! Default mpi constructor is one-d.
  mpi_decomposition( const architecture &arch )
    : mpi_decomposition(arch,arch.get_proc_layout(1)) {};

  void set_decomp_factory();
  virtual std::string as_string() const override;
};

void make_mpi_communicator(communicator *cator,const decomposition &d);

/****
 **** Distribution
 ****/

class mpi_object;
/*!
  An MPI distribution inherits from distribution, and therefore it also
  inherits the \ref architecture that the base distribution has.
  This class has no data of its own, just the "set_mpi" functionality.
*/
class mpi_distribution : virtual public distribution {
public:
  mpi_distribution( const decomposition &d ); // basic constructor
  mpi_distribution( const parallel_structure &struc ); // from structure
  mpi_distribution( const parallel_structure &&struc ); // from structure
  mpi_distribution( const decomposition &d,std::shared_ptr<parallel_indexstruct>);
  //! Constructor from function.
  mpi_distribution( const decomposition &d,index_int(*pf)(int,index_int),index_int nlocal);
  mpi_distribution( std::shared_ptr<distribution> other )
    : mpi_distribution( other->get_structure() ) {};

  void set_mpi_routines();

  // Resolve, then set mask
  //virtual void add_mask( processor_mask *m ) override;
};

class mpi_block_distribution
  : public mpi_distribution,public block_distribution,virtual public distribution {
public:
  // Explicit one-d constructor from ortho,local,global.
  mpi_block_distribution( const decomposition &d,int o,index_int l,index_int g);
  //! Constructor with implicit ortho=1.
  mpi_block_distribution( const decomposition &d,index_int l,index_int g)
    : mpi_block_distribution(d,1,l,g) {};
  //! Constructor with implicit ortho and local is decided.
  mpi_block_distribution( const decomposition &d,index_int g)
    : mpi_block_distribution(d,-1,g) {};

  // 1D from local sizes
  mpi_block_distribution(  const decomposition &d,const std::vector<index_int> lsizes );
  // multi-D from endpoint
  mpi_block_distribution( const decomposition &d,domain_coordinate &gsize);
  mpi_block_distribution( const decomposition &d,domain_coordinate &&gsize)
    : distribution(d),mpi_distribution(d),block_distribution(d,gsize) { memoize(); };
};

class mpi_scalar_distribution
  : public mpi_distribution,public scalar_distribution,virtual public distribution {
public:
  mpi_scalar_distribution( const decomposition &d)
    : distribution(d),mpi_distribution(d),scalar_distribution(d) { memoize(); };
};

class mpi_cyclic_distribution
  : public mpi_distribution,public cyclic_distribution,virtual public distribution  {
public:
  mpi_cyclic_distribution( const decomposition &d,index_int l,index_int g)
    : distribution(d),mpi_distribution(d),cyclic_distribution(d,l,g) { memoize(); };
  mpi_cyclic_distribution( const decomposition &d,index_int l)
    : mpi_cyclic_distribution(d,l,-1) {};
};

class mpi_blockcyclic_distribution
  : public mpi_distribution,public blockcyclic_distribution,virtual public distribution  {
public:
  mpi_blockcyclic_distribution( const decomposition &d,index_int bs,index_int nb,index_int g)
    : distribution(d),mpi_distribution(d),blockcyclic_distribution(d,bs,nb,g)  { memoize(); };
  mpi_blockcyclic_distribution( const decomposition &d,index_int bs,index_int nb)
    : mpi_blockcyclic_distribution(d,bs,nb,-1) {};
};

class mpi_replicated_distribution
  : public mpi_distribution,public replicated_distribution,virtual public distribution  {
public:
  mpi_replicated_distribution( const decomposition &d,int ortho,index_int l)
    : distribution(d),mpi_distribution(d),replicated_distribution(d,ortho,l) { memoize(); };
  mpi_replicated_distribution( const decomposition &d,index_int l)
    : mpi_replicated_distribution(d,1,l) {};
  mpi_replicated_distribution( const decomposition &d)
    : mpi_replicated_distribution(d,1) {};
};

class mpi_gathered_distribution
  : public mpi_distribution,public gathered_distribution,virtual public distribution  {
public:
  //! Create a gathered distribution with s per processor and k orthogonal.
  mpi_gathered_distribution( const decomposition &d,int k,index_int s)
    : distribution(d),mpi_distribution(d),gathered_distribution(d,k,s) { memoize(); };
  //! Creating from a single integer parameter assumes k=1
  mpi_gathered_distribution( const decomposition &d,index_int l)
    : mpi_gathered_distribution(d,1,l) {};
  //! Creating without integer parameters corresponds to one element per proc
  mpi_gathered_distribution( const decomposition &d)
    : mpi_gathered_distribution(d,1,1) {};
};

class mpi_binned_distribution : public mpi_distribution,public binned_distribution {
public:
  mpi_binned_distribution(decomposition &d,object *o)
    : distribution(d),mpi_distribution(d),binned_distribution(d,o) { memoize(); };
};

/****
 **** Sparse matrix / index pattern
 ****/

class mpi_sparse_matrix : public sparse_matrix {
protected:
  //! \todo can we add mycoord to decomposition or so?
  processor_coordinate mycoord;
  std::shared_ptr<multi_indexstruct> mystruct;
  index_int my_first,my_last,localsize; //!< these need to be domain_coordinate
  index_int maxrowlength{-1};
  std::vector< std::vector<sparse_element> > elements;
  // for one-sided creation:
  std::vector<sparse_element> element_storage;
  MPI_Win create_win;
public:
  mpi_sparse_matrix() : sparse_matrix() {};
  mpi_sparse_matrix( std::shared_ptr<distribution> d,index_int g=-1 );
  std::shared_ptr<indexstruct> all_columns() override {
    throw("can not ask all columns in MPI\n"); };
  virtual sparse_matrix *transpose() const override {
    throw(std::string("MPI matrix transpose not implemented"));
  };
};

class mpi_upperbidiagonal_matrix : public mpi_sparse_matrix {
public:
  mpi_upperbidiagonal_matrix( std::shared_ptr<distribution> dist, double d,double r )
    : mpi_sparse_matrix(dist) {
    auto globalsize = dist->global_volume();
    for (index_int row=my_first; row<=my_last; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row+1; if (col<globalsize) add_element(row,col,r);
    }
  }
};

class mpi_lowerbidiagonal_matrix : public mpi_sparse_matrix {
public:
  mpi_lowerbidiagonal_matrix( std::shared_ptr<distribution> dist, double l,double d )
    : mpi_sparse_matrix(dist) {
    for (index_int row=my_first; row<=my_last; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
    }
  }
};

class mpi_toeplitz3_matrix : public mpi_sparse_matrix {
public:
  mpi_toeplitz3_matrix( std::shared_ptr<distribution> dist, double l,double d,double r )
    : mpi_sparse_matrix(dist) {
    auto globalsize = dist->global_volume();
    for (index_int row=my_first; row<=my_last; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
      col = row+1; if (col<globalsize) add_element(row,col,r);
    }
  }
};

/****
 **** Object
 ****/

class mpi_object : public object {
private:
public:
  mpi_object( std::shared_ptr<distribution> d );
  mpi_object( std::shared_ptr<distribution> d,
	      std::shared_ptr<std::vector<double>> dat,
	      index_int offset=0 );
  mpi_object( std::shared_ptr<distribution> d,std::shared_ptr<object> x);
  // this is just to make static casts possible
  void poly_object() { printf("this serves no purpose\n"); };

  /*
   * Data
   */
private:
  void mpi_allocate();
protected:
  //! Install all mpi-specific data routines
  void set_data_handling() {
    allocate = [this] (void) -> void { mpi_allocate(); };
  };

public:
  //! MPI has no mask shift, because a processor is either there or not
  virtual domain_coordinate mask_shift( processor_coordinate &p ) {
  //  virtual index_int mask_shift(int np) {
    if (!get_distribution()->lives_on(p))
      throw(fmt::format("Should not ask mask shift for {} on {}",p.as_string(),get_name()));
    return 0;
  };
};

/****
 **** Message
 ****/

class mpi_message : public message {
public:
  mpi_message(const decomposition &d,
	      const processor_coordinate &snd,const processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> e,std::shared_ptr<multi_indexstruct> g)
    : message(d,snd,rcv,e,g) {};
  mpi_message(const decomposition &d,
	      const processor_coordinate &snd,const processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> g)
    : message(d,snd,rcv,g) {};
public: // figure out a way to access this
  MPI_Datatype embed_type{-1}; index_int lb{-1},extent{-1};
public:
  void compute_src_index() override; void compute_tar_index() override;
};

// utility routines for mpi messages
int mpi_message_buffer_length(int dim);

/****
 **** Kernel
 ****/

//void install_mpi_kernel_factory( kernel* );

/*!
  The MPI kernel class exists to provide a few MPI-specific allocators.

  - the task creator for a domain
  - the beta distribution
 */
class mpi_kernel : virtual public kernel {
protected:
#ifdef VT
  int vt_kernel_class;
#endif
private:
public:
  mpi_kernel();
  mpi_kernel( std::shared_ptr<object> out );
  mpi_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out );
  //! This copy constructur is used in task creation. 
  mpi_kernel( const kernel& k );

  // set the make_task_for_domain
  void install_mpi_factory();
};

/*!
  An origin kernel only has output data.
  After task queue optimization its tasks may receive post and xpct messages.

  By default we set the local function to no-op, this means that we can
  set the vector elements externally.
 */
class mpi_origin_kernel : public mpi_kernel,public origin_kernel,virtual public kernel {
public:
  mpi_origin_kernel( std::shared_ptr<object> out,std::string name )
    : mpi_kernel(out),origin_kernel(out,name),kernel(out) {
    install_mpi_factory();
  };
  mpi_origin_kernel( std::shared_ptr<object> out )
    : mpi_kernel(out),origin_kernel(out),kernel(out) {
    install_mpi_factory();
  };
};

/****
 **** Task
 ****/

class mpi_request : public request {
protected:
  MPI_Request rquest;
public:
  mpi_request(std::shared_ptr<message> m,MPI_Request r)
    : request(m,request_protocol::MPI) { rquest = r; };
  MPI_Request get_mpi_request() { return rquest; };
  void poly() { return; };
};

//class mpi_request_vector;
void mpi_request_vector_wait(request_vector&);

class mpi_task : public task {
private:
protected:
public:
  //! Make a task on a surrounding kernel.
  mpi_task(const processor_coordinate &d,kernel *k) : task(d,k) { set_factories(); };
  // make an origin task for standalone testing
  mpi_task(const processor_coordinate &d,std::shared_ptr<object> out)
    : task(d,new mpi_kernel(out)) { set_factories(); };
  // make a compute task for standalone testing
  mpi_task(const processor_coordinate &d,std::shared_ptr<object> in,std::shared_ptr<object> out)
    : mpi_task(d,out) { containing_kernel->add_in_object(in); };

  // Factories and such
  void set_factories() {
    requests_wait =
      [] (request_vector &v) -> void { mpi_request_vector_wait(v); };
    MPI_Comm comm = MPI_COMM_WORLD;
    message_from_buffer = [this,comm] (int step,std::string &buf) -> std::shared_ptr<message> {
      return mpi_message_from_buffer(comm,this->shared_from_this(),step,buf);
    };
  };
  virtual void declare_dependence_on_task( task_id *id ) override ; // pure virtual

  // pure virtual synchronization functions
  request_vector acceptReadyToSend( std::vector<std::shared_ptr<message>> &msgs ) override ;
  //??? request_vector notifyReadyToSend( std::vector<std::shared_ptr<message>> &msgs );
  std::shared_ptr<request> notifyReadyToSendMsg( std::shared_ptr<message> msg ) override;

  virtual void derive_send_messages(bool=false) override ;
  void make_infrastructure_for_sending( int n );

  /* Stuff */
  //  std::string as_string() const;
};

class mpi_collective_task : public mpi_task {
public:
  //! Make a task on a surrounding kernel.
  mpi_collective_task(const processor_coordinate &d,kernel *k) : mpi_task(d,k) {};

  // pure virtual synchronization functions
  request_vector acceptReadyToSend( std::vector<std::shared_ptr<message>>& ) override;
  request_vector notifyReadyToSend( std::vector<std::shared_ptr<message>> & ) override;
};

class mpi_origin_task : public mpi_task {
public:
  mpi_origin_task( processor_coordinate &d,std::shared_ptr<object> out )
    : mpi_task(d,out) {
    containing_kernel->set_type( kernel_type::ORIGIN ); set_name("origin mpi task");
  };
};

/****
 **** Queue
 ****/

/*!
  An MPI task queue is a processor-specific subset of the global queue,
  which we never assemble. The magic is done in routine
  \ref mpi_task::declare_dependence_on_task to make this local task queue
  conform to the global one.
*/
class mpi_algorithm : public algorithm {
private:
public:
  mpi_algorithm() {};
  mpi_algorithm( const decomposition &d )
    : algorithm(d) {
    type = algorithm_type::MPI; make_mpi_communicator(this,d);
  };
  //! MPI-only analysis: try to embed vectors in a halo.
  virtual void mode_analyze_dependencies() override { inherit_data_from_betas(); };
  std::string kernels_as_string() override {
    if (mytid()==0)
      return algorithm::kernels_as_string();
    else {
      auto s =  new std::string; return *s;
    }
  };
};

#endif
