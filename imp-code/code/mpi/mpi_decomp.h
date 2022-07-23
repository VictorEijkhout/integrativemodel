// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** mpi_decomp.h: Header file for MPI decompositions
 ****
 ****************************************************************/

#pragma once

#include "mpi.h"
#include "imp_decomp.h"

/*!
  An mpi decomposition has one domain per processor by default,
  unless there is a global over-decomposition parameters.

  No one ever inherits from this, but mode-specific distributions are built from this
  because it contains a distribution factory.
*/
template<int d>
class mpi_decomposition : public decomposition<d> {
public:
  mpi_decomposition() : decomposition<d>() {};
  //! Multi-d decomposition from explicit processor grid layout
  mpi_decomposition( const coordinate<int,d> &grid)
    : decomposition<d>(grid) {
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

#if 0

/****
 **** Architecture
 ****/

// prototypes of mpi collectives, defined in mpi_base.cxx
index_int mpi_allreduce(index_int contrib,MPI_Comm comm);
double mpi_allreduce_d(double contrib,MPI_Comm comm);
int mpi_allreduce_and(int contrib,MPI_Comm comm);
//! \todo lose that star
void mpi_gather32(int contrib,std::vector<int>&,MPI_Comm comm);
void mpi_gather64(index_int contrib,std::vector<index_int>&,MPI_Comm comm);
//! \todo lose that star
std::vector<index_int> *mpi_overgather(index_int contrib,int over,MPI_Comm comm);
int mpi_reduce_scatter(int *senders,int root,MPI_Comm comm);
std::vector<index_int> mpi_reduce_max(std::vector<index_int> local_values,MPI_Comm comm);
std::vector<index_int> mpi_reduce_min(std::vector<index_int> local_values,MPI_Comm comm);

void mpi_message_as_buffer
( architecture &arch,std::shared_ptr<message> msg,
  /* char *b,int l */ 
  std::string&
  );
std::shared_ptr<message> mpi_message_from_buffer
    ( MPI_Comm,std::shared_ptr<task>,int,std::string&);

void mpi_architecture( architecture&, int=-1, int=-1 );

// class mpi_architecture : public architecture {
// public :
//   //! default constructor because we need the global variable
//   mpi_architecture() {};
//   //! Constructor.
//   mpi_architecture( int tid,int ntids )
//     : architecture(tid,ntids) {
//     type = architecture_type::SPMD; protocol = protocol_type::MPI;
//     beta_has_local_addres_space = 1;
//     set_name(fmt::format("mpi-architecture-on-proc{}-out-of-{}",tid,ntids));

//     message_as_buffer =
//       [] (architecture &a,std::shared_ptr<message> m,
// 	  // char *b,int l) -> void {
// 	  std::string &b) -> void {
// 	mpi_message_as_buffer(a,m,b /* ,l */); };
//     // For MPI we can actually report a `mytid'
//     mytid = [this] (void) -> int { return arch_procid; };
//   };
//   mpi_architecture( int mytid,int ntids,int o ) : mpi_architecture(mytid,ntids) {
//     set_over_factor(o); };
//   //! Copy constructor
//   mpi_architecture( mpi_architecture *a )
//     : architecture(a) {
//     comm = a->comm;
//   };

// protected:
//   MPI_Comm comm;
// public:
//   void set_mpi_comm( MPI_Comm c ) { comm = c; };
//   MPI_Comm get_mpi_comm() { return comm; };

//   //! Enable all tricky optimizations
//   virtual void set_power_mode() override {
//     set_can_embed_in_beta(); set_can_message_overlap(); };

//   virtual std::string as_string() override;
// };

#endif
