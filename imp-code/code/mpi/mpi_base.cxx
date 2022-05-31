/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** mpi_base.cxx: Implementations of the MPI derived classes
 ****
 ****************************************************************/

#include <stdarg.h>
#include <unistd.h> // just for sync
#include <iostream>
using std::cout;

#include <mpi.h>
#include "mpi_base.h"
#include "imp_base.h"

using fmt::format;
using fmt::print;

using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

using gsl::span;

#include "mpi_env.h"

/****
 **** Basics
 ****/

index_int mpi_allreduce(index_int contrib,MPI_Comm comm) {
  index_int result;
  //  MPI_Comm_size(comm,&ntids);
  MPI_Allreduce(&contrib,&result,1,MPI_INDEX_INT,MPI_SUM,comm);
  return result;
};
double mpi_allreduce_d(double contrib,MPI_Comm comm) {
  int ntids; double result;
  MPI_Comm_size(comm,&ntids);
  MPI_Allreduce(&contrib,&result,1,MPI_DOUBLE,MPI_SUM,comm);
  return result;
};
int mpi_allreduce_and(int contrib,MPI_Comm comm) {
  int ntids, result;
  MPI_Comm_size(comm,&ntids);
  MPI_Allreduce(&contrib,&result,1,MPI_INT,MPI_PROD,comm);
  return result;
};

void mpi_gather32(int contrib,vector<int> &gathered,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  if (gathered.size()<ntids)
    throw(format("gather32 of {} elements into vector of size {}",ntids,gathered.size()));
  int P = ntids; int sendbuf = contrib;
  MPI_Allgather(&sendbuf,1,MPI_INT,gathered.data(),1,MPI_INT,comm);
};

void mpi_gather64(index_int contrib,vector<index_int> &gathered,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int P = ntids; index_int sendbuf = contrib;
  if (gathered.size()<P)
    throw(format("Vector too small ({}) for gather on {} procs",gathered.size(),P));
  MPI_Allgather(&sendbuf,1,MPI_INDEX_INT,gathered.data(),1,MPI_INDEX_INT,comm);
};

vector<index_int> *mpi_overgather(index_int contrib,int over,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int P = ntids*over;
  index_int sendbuf[over]; for (int i=0; i<over; i++) sendbuf[i] = contrib;
  auto gathered = new vector<index_int>; gathered->reserve(P);
  for (int i=0; i<P; i++)
    gathered->push_back(0);
  MPI_Allgather(sendbuf,over,MPI_INDEX_INT,gathered->data(),over,MPI_INDEX_INT,comm);
  return gathered;
};

//! The root is a no-op for MPI, but see OpenMP
int mpi_reduce_scatter(int *senders,int root,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int P = ntids;
  vector<int> recvcounts(P,1);
  //int *recvcounts = new int[P],
  //for (int i=0; i<P; ++i) recvcounts[i] = 1;
  int nsends;
  MPI_Reduce_scatter(senders,&nsends,recvcounts.data(),MPI_INT,MPI_SUM,comm);
  //delete recvcounts;
  return nsends;
};

vector<index_int> mpi_reduce_max(vector<index_int> local_values,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int nvalues = local_values.size();
  vector<index_int> global_values(nvalues);
  for (int n=0; n<nvalues; n++) global_values[n] = -1;
  MPI_Allreduce
    (local_values.data(),global_values.data(),nvalues,MPI_INDEX_INT,MPI_MAX,comm);
  return global_values;;
};

vector<index_int> mpi_reduce_min(vector<index_int> local_values,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int nvalues = local_values.size();
  vector<index_int> global_values(nvalues);
  for (int n=0; n<nvalues; n++) global_values[n] = -1;
  MPI_Allreduce
    (local_values.data(),global_values.data(),nvalues,MPI_INDEX_INT,MPI_MIN,comm);
  return global_values;;
};

#ifdef VT
#include "VT.h"
#include "imp_static_vars.h"
void vt_register_kernels() {
  //  VT_classdef("copy kernel",&vt_copy_kernel); // (const char * classname, int * classhandle)
};
#endif

/*!
  Document mpi-specific options.
*/
/****
 **** Decomposition
 ****/

/*!
  A factory for making new distributions from this decomposition
*/
void mpi_decomposition::set_decomp_factory() {
  new_block_distribution = [this] (index_int g) -> shared_ptr<distribution> {
    return shared_ptr<distribution>( make_shared<mpi_block_distribution>(*this,g) );
  };
};

std::string mpi_decomposition::as_string() const {
  return "mpidecomp"; // fmt::format("MPI decomposition <<{}>>",decomposition::as_string());
};

/****
 **** Distribution
 ****/

void make_mpi_communicator(communicator *cator,const decomposition &d) {
  cator->the_communicator_mode = communicator_mode::MPI;

  {
    MPI_Comm *mpicom = new MPI_Comm; *mpicom = MPI_COMM_WORLD;
    cator->communicator_context = (void*)mpicom;
  }

  MPI_Comm comm = MPI_COMM_WORLD;
  cator->procid =
    [comm] (void) -> int { int tid; MPI_Comm_rank(comm,&tid); return tid; };
  cator->proc_coord =
    [comm,d] (/*const decomposition &d*/) -> processor_coordinate
    { int tid; MPI_Comm_rank(comm,&tid); return processor_coordinate(tid,d); };
  cator->proc_coord_rv =
    [comm] (decomposition &&d) -> processor_coordinate
    { int tid; MPI_Comm_rank(comm,&tid); return processor_coordinate(tid,d); };
  cator->nprocs = [comm] (void) -> int { int np; MPI_Comm_size(comm,&np); return np; };

  cator->allreduce =
    [comm] (index_int contrib) -> index_int { return mpi_allreduce(contrib,comm); };
  cator->allreduce_d =
    [comm] (double contrib) -> double { return mpi_allreduce_d(contrib,comm); };
  cator->allreduce_and =
    [comm] (int contrib) -> int { return mpi_allreduce_and(contrib,comm); };
  cator->gather32 =
    [comm] (int contrib,vector<int> &gathered) -> void {
    mpi_gather32(contrib,gathered,comm); };
  cator->gather64 =
    [comm] (index_int contrib,vector<index_int> &gathered) -> void {
    return mpi_gather64(contrib,gathered,comm); };
  cator->overgather =
    [comm] (index_int contrib,int over) -> vector<index_int>* {
    return mpi_overgather(contrib,over,comm); };
  cator->reduce_scatter =
    [comm] (int *senders,int root) -> int { return mpi_reduce_scatter(senders,root,comm); };
  cator->reduce_max =
    [comm] (vector<index_int> local) -> vector<index_int> {
    return mpi_reduce_max(local,comm); };
  cator->reduce_min =
    [comm] (vector<index_int> local) -> vector<index_int> {
    return mpi_reduce_min(local,comm); };

};

//! Basic constructor
mpi_distribution::mpi_distribution(  const decomposition &d )
try : distribution(d) {
  try { make_mpi_communicator(this,d); } catch (...) { throw(format("comm fail")); };
  try { set_mpi_routines(); } catch (...) { throw(format("Failure to set mpi")); };
  } catch (...) { print("Basic distribution constructor failed\n"); };


/*! Constructor from parallel structure. See \ref dependency::ensure_beta_distribution
*/
mpi_distribution::mpi_distribution( const parallel_structure &struc )
  // VLE does not work:
  // : mpi_distribution( static_cast<decomposition>(struc) ) {
  // because we need to do distribution(struc) and thereby parallel_structure(struc)
  : distribution(struc) {
  try { make_mpi_communicator(this,get_decomposition());
  } catch (...) { throw(format("comm fail")); };
  try { set_mpi_routines(); } catch (...) { throw(format("Failure to set mpi")); };

  try { memoize();
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not memoizing in mpi distribution from struct {}",
		      struc.as_string()));
  }
};
mpi_distribution::mpi_distribution( const parallel_structure &&struc )
  : distribution(struc) {
  try { make_mpi_communicator(this,get_decomposition());
  } catch (...) { throw(format("comm fail")); };
  try { set_mpi_routines(); } catch (...) { throw(format("Failure to set mpi")); };

  try { memoize();
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not memoizing in mpi distribution from struct {}",
		      struc.as_string()));
  }
};

mpi_distribution::mpi_distribution
    ( const decomposition &d,shared_ptr<parallel_indexstruct> idx )
  : mpi_distribution( parallel_structure(d,idx) ) {
  //print("mpi dist from pidx, no action required\n");
};

mpi_distribution::mpi_distribution
    ( const decomposition &d,index_int(*pf)(int,index_int),index_int nlocal)
  : mpi_distribution(d) {
  if (!d.get_same_dimensionality(1))
    throw(format("Can only create from function in 1D"));

  //  print("set dim0\n");
  get_dimension_structure(0)->create_from_function( pf,nlocal );
  //  print("...set\n");

  try { memoize();
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not memoizing in mpi distribution from function"));
  }
};

void mpi_distribution::set_mpi_routines() {

  /*
   * Factories
   */
  // Factory for new distributions
  new_distribution_from_structure =
    [] (const parallel_structure &strct) -> shared_ptr<distribution> {
    auto new_dist =  shared_ptr<distribution>( new mpi_distribution(strct ) );
    //new_dist->compute_global_first_index( new parallel_structure );
    return new_dist; };
  // Factory for new scalar distributions
  new_scalar_distribution = [this] (void) -> shared_ptr<distribution> {
    auto decomp = dynamic_cast<decomposition*>(this);
    if (decomp==nullptr) throw(format("weird upcast in mpi new_scalar_distribution"));
    return shared_ptr<distribution>( new mpi_scalar_distribution( *decomp ) ); };
  // Factory for new objects
  new_object = [this] ( shared_ptr<distribution> d ) -> shared_ptr<object>
    { auto o = shared_ptr<object>( new mpi_object(d) ); return o; };
  // Factory for new objects from user-supplied data
  new_object_from_data = [this] ( shared_ptr<vector<double>> d ) -> shared_ptr<object>
    { return shared_ptr<object>( new mpi_object(this->shared_from_this(),d) ); };
  // Factory for making mode-dependent kernels
  new_kernel_from_object = [] ( shared_ptr<object> out ) -> shared_ptr<kernel>
    { return shared_ptr<kernel>( new mpi_kernel(out) ); };
  // Factory for making mode-dependent kernels, for the imp_ops kernels.
  kernel_from_objects =
    [] ( shared_ptr<object> in,shared_ptr<object> out ) -> shared_ptr<kernel>
    { return shared_ptr<kernel>( new mpi_kernel(in,out) ); };
  // Set the message factory.
  auto decomp = get_decomposition();
  new_message =
    [decomp] (const processor_coordinate &snd,const processor_coordinate &rcv,
	    shared_ptr<multi_indexstruct> g) -> shared_ptr<message> {
    return shared_ptr<message>( new mpi_message(decomp,snd,rcv,g) ); };
  new_embed_message =
    [decomp] (const processor_coordinate &snd,const processor_coordinate &rcv,
	    shared_ptr<multi_indexstruct> e,
	    shared_ptr<multi_indexstruct> g) -> shared_ptr<message> {
    return shared_ptr<message>( new mpi_message(decomp,snd,rcv,e,g) ); };

  /*
   * NUMA
   */
  //! The first index that this processor can see.
  auto me = first_local_domain();
  get_numa_structure =
    [this,me] () -> shared_ptr<multi_indexstruct> 
    { return get_processor_structure(me); };
  get_global_structure =
    [this,me] () -> shared_ptr<multi_indexstruct> 
    { return get_enclosing_structure(); };

  location_of_first_index =
    [] (  shared_ptr<distribution> d,const processor_coordinate &p) -> index_int {
    return 0; }; 
  location_of_last_index =
    [] ( shared_ptr<distribution> d,const processor_coordinate &p) -> index_int {
    return d->volume(p)-1; };

  local_allocation = [this] (void) -> index_int {
    auto p = proc_coord( /* *dynamic_cast<decomposition*>(this) */ );
    return distribution::local_allocation_p(p); };
  //! A processor can only see its own part of the structure
  //snippet mpivisibility
  get_visibility =
    [this] (processor_coordinate &p) -> shared_ptr<multi_indexstruct>
    { return get_processor_structure(p); };
  //snippet end

  /*
   * Memo computation
   */
   compute_global_first_index =
    [this,me] (parallel_structure *pstr) -> domain_coordinate {
    try {
      domain_coordinate gf;
      auto lf = pstr->first_index_r(me);
      gf = domain_coordinate( mpi_reduce_min( lf.data(),MPI_COMM_WORLD) );
      return gf;
    } catch (string c) {
      throw(format("Could not mpi global first on {}: <<{}>>",me.as_string(),c));
    }
  };

  compute_global_last_index =
    [this,me] (parallel_structure *pstr) -> domain_coordinate {
    auto ll = pstr->last_index_r(me);
    auto gl = domain_coordinate( mpi_reduce_max( ll.data(),MPI_COMM_WORLD) );
    return gl;
  };

  compute_offset_vector =
    [this,me] () -> domain_coordinate {
    return first_index_r(me)-global_first_index();
  };
};

void test_local_global_sanity(MPI_Comm comm,index_int ortho,index_int lsize,index_int gsize) {
  int minlocal,maxlocal,minglobal,maxglobal,minortho,maxortho;

  int i_ortho = (int)ortho,i_lsize = (int)lsize,i_gsize = (int) gsize;
  MPI_Allreduce(&i_ortho,&minortho,1,MPI_INT,MPI_MIN,comm);
  MPI_Allreduce(&i_ortho,&maxortho,1,MPI_INT,MPI_MAX,comm);
  if (minortho!=maxortho) {
    printf("orthogonal dimension needs to be uniform %d\n",i_ortho); throw(58);}

  MPI_Allreduce(&i_lsize,&minlocal,1,MPI_INT,MPI_MIN,comm);
  MPI_Allreduce(&i_lsize,&maxlocal,1,MPI_INT,MPI_MAX,comm);
  MPI_Allreduce(&i_gsize,&minglobal,1,MPI_INT,MPI_MIN,comm);
  MPI_Allreduce(&i_gsize,&maxglobal,1,MPI_INT,MPI_MAX,comm);

  if (minlocal<0 && minglobal<0) {
    printf("local/global insufficiently specified %d-%d\n",i_lsize,i_gsize); throw(55);}
  if (minglobal>=0) { // case: global specified
    if (minglobal!=maxglobal) {
      printf("global inconsistently specified %d\n",i_gsize); throw(56);}
    if ( maxlocal>=0 ) {
      printf("can not specify local (l:%d,max:%d) with global %d\n",
	     i_lsize,maxlocal,i_gsize); throw(57);}
  } else { // case: global unspecified
    if (minlocal<0) {
      printf("Can not leave local %d unspecified with global %d\n",
	     i_lsize,i_gsize); throw(57);}
  }
  return;
}

//! Set a mask, after detecting unresolved parallel changes.
// void mpi_distribution::add_mask( processor_mask *m ) {
//   MPI_Comm comm = MPI_COMM_WORLD;
//   vector<int> includes = m->get_includes();
//   int P = includes.size();
//   MPI_Allreduce( MPI_IN_PLACE,includes.data(), P,MPI_INTEGER, MPI_MAX, comm);
//   distribution::add_mask( new processor_mask(this,includes) );
// };

//! Explicit one-d constructor from ortho,local,global.
mpi_block_distribution::mpi_block_distribution
    ( const decomposition &d,int o,index_int l,index_int g)
  : distribution(d),block_distribution(d,o,l,g),mpi_distribution(d) {
  // the factories and structures are set in the other constructors
  try {
    memoize();
  } catch (string c) {
    throw(format("Failure to memoize block dist g={}: {}",g,c));
  } catch (...) { throw(format("Failure to memoize block dist g={}",g)); }
};

//! Constructor in one-d from local sizes
mpi_block_distribution::mpi_block_distribution
    ( const decomposition &d,const vector<index_int> lsizes )
try
  : distribution(d),mpi_distribution(d),block_distribution(d,lsizes) { memoize(); }
catch (string c) {
  print("MPI block distribution from vector: base constructor(s) failed <<{}>>",c);
  throw; }
catch (...) { print("MPI block distribution from vector: base constructor(s) failed");
  throw; };

//! Constructor from vector denoting multi-d global size
mpi_block_distribution::mpi_block_distribution
    ( const decomposition &d,domain_coordinate &gsize)
try
  : distribution(d),mpi_distribution(d),block_distribution(d,gsize) { memoize(); }
catch (string c) {
  print("MPI block distribution from endpoint: base constructor(s) failed <<{}>>",c);
  throw; }
catch (...) { print("MPI block distribution from endpoint: base constructor(s) failed");
  throw; };



/****
 **** Sparse matrix / index pattern
 ****/

mpi_sparse_matrix::mpi_sparse_matrix( shared_ptr<distribution> d,index_int g )
  : sparse_matrix( d->get_processor_structure(d->proc_coord()),g ) {
  print("create mpi matrix with g={}\n",globalsize);
  auto mycoord = d->proc_coord();
  index_int localsize = d->volume(mycoord);
  if (g>0) { // one-sided stuff
    element_storage.reserve(localsize*g);
    MPI_Aint winsize = localsize*g;
    MPI_Win_create( element_storage.data(),winsize,sizeof(sparse_element),
		    MPI_INFO_NULL,MPI_COMM_WORLD,&create_win);
  };

  matrix_distribution = d;
  // identify the current process & structure
  mycoord = matrix_distribution->proc_coord();
  mystruct = matrix_distribution->get_processor_structure(mycoord);
  if (mystruct->get_dimensionality()!=1)
    throw(format("Matrix dimensionality needs to be 1"));
  //
  my_first = mystruct->first_index_r()[0];
  my_last = mystruct->last_index_r()[0];
};

/****
 **** Object
 ****/

//! Create an object from a distribution, locally allocating the data.
//! Use this as initializer only if it's the right allocation strategy.f
mpi_object::mpi_object( std::shared_ptr<distribution> d )
  : object(d) {
  set_data_handling(); 
  // if objects can embed in a halo, we'll allocate later; otherwise now.
  if (!d->get_can_embed_in_beta())
    allocate();
};

//! Create an object from a distribution, using a pointer to external data.
mpi_object::mpi_object( std::shared_ptr<distribution> d,
	    std::shared_ptr<std::vector<double>> dat,
	    index_int offset )
  : object(d) {
  set_data_handling(); 
  if (get_distribution()->local_ndomains()>1)
    throw(std::string("Can not create from other object on >1 domain"));
  auto data_span = gsl::span<double>( dat->data(),dat->size() );
  register_data_on_domain_number(0,data_span,0,offset);
  data_status = object_data_status::USER;    
};

//! Create an object from the data of another object
mpi_object::mpi_object( std::shared_ptr<distribution> d,std::shared_ptr<object> x)
  : object(d) {
  set_data_handling(); 
  if (get_distribution()->local_ndomains()>1)
    throw(std::string("Can not create from other object on >1 domain"));
  if (!x->has_data_status_allocated()) {
    printf("warning: allocating vector so that it can be inherited\n");
    x->allocate(); }
  try { 
    auto dat = x->get_numa_data_pointer();
    auto data_span = gsl::span<double>( dat->data(),dat->size() );
    register_data_on_domain_number(0,data_span,0);
  } catch (std::string c) {
    fmt::print("Error <<{}>> in constructor from <<{}>>",c,x->get_name());
    throw(std::string("Could not create object from object"));
  }
  data_status = object_data_status::REUSED;
};

//! Allocating is an all-or-nothing activity for MPI processes. \todo fix for masks
void mpi_object::mpi_allocate() {
  if (has_data_status_allocated()) return;
  auto decomp = get_decomposition();
  auto domains = get_distribution()->get_domains();
  // if (domains.size()>1)
  //   throw(string("Can not allocate mpi more than one domain"));

  index_int s=0;
  for ( auto dom : domains ) s += get_distribution()->local_allocation_p(dom);
  auto info = format("mpi node storage of {} bytes for {}",s,get_name());
  auto dat = create_data(s,info);
  set_numa_data(dat,s);
  auto numa_ptr = get_numa_data_pointer()->data();
  auto numa_size = get_numa_data_pointer()->size();

  try {
    auto dom_begin = numa_ptr;
    auto dom_end = numa_ptr; 
    for ( auto dom : domains ) {
      index_int s;
      if (false && get_distribution()->lives_on(dom)) continue;
      s = get_distribution()->local_allocation_p(dom);
      if (s<0)
	throw(format("Negative allocation {} for dom=<<{}>>",s,dom.as_string()));
      dom_end += s;
      //print("Allocating domain {} with {}\n",dom.as_string(),s);
      int domnum = decomp.get_domain_local_number(dom);
      auto dat = create_data(s,get_name());
      span<double> offset_dat(dom_begin,dom_end);
      //print("domain {} has data at {}\n",domnum,(long)(offset_dat.data()));
      register_data_on_domain_number(domnum,offset_dat,s);
      dom_begin = dom_end; dom_end = dom_begin;
    }
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not allocate {}",as_string()));
  }
};

/****
 **** Message
 ****/

/*!
  MPI messages have to be sent over. This computes the buffer length;
  the packing is done in #mpi_message_as_buffer
*/
int mpi_message_buffer_length(int dim) {
  return
    (0
     + 1 // cookie
     + 1 // dimension
     + 1 // collective?
     //     + 1 // MPI collective  ?
     + 1 // MPI skippable ?
     + 2 // sender, receiver
     + 4 // tag contents
     + 3 // dep/in/out object number
     + 1 // trailing cookie
     )*sizeof(int) // 32 bytes
    +2*dim*sizeof(index_int) // src_index, size
    ;
};

/*!
  Return the content of a message as a character buffer,
  as constructed by MPI packing. This is used in
  mpi_task#create_send_structure_for_task.

  \todo figure out a way to supply the MPI communicator. as argument? make this class method?
*/
void mpi_message_as_buffer
    ( architecture &arch,shared_ptr<message> msg,
      string &buffer ) {
  const char *buf = buffer.data(); int buflen = buffer.size();
  MPI_Comm comm = MPI_COMM_WORLD;
  int dim = msg->get_global_struct()->get_dimensionality();

  const auto &tag = msg->get_tag();
  const auto in_object = msg->get_in_object();
  const auto indistro = in_object->get_distribution();
  int sender = msg->get_sender().linearize(indistro->get_decomposition()),
    receiver = msg->get_receiver().linearize(indistro->get_decomposition());
  if (receiver!=arch.mytid())
    throw(format("Packing msg {}->{} but I am {}",
		      sender,receiver,arch.mytid()));
  auto global_struct = msg->get_global_struct();

  int
    innumber = msg->get_in_object_number(),
    outnumber = msg->get_out_object_number(),
    depnumber = msg->get_dependency_number(),
    //    collective = msg->get_is_collective(),
    skippable = msg->is_skippable();

  int pos = 0; 
  int cookie = -37, long_int;
  MPI_Pack(&cookie,     1,MPI_INT,(void*)buf,buflen,&pos,comm);
  MPI_Pack(&dim,        1,MPI_INT,(void*)buf,buflen,&pos,comm);
  //  MPI_Pack(&collective, 1,MPI_INT,(void*)buf,buflen,&pos,comm);
  MPI_Pack(&skippable , 1,MPI_INT,(void*)buf,buflen,&pos,comm);
  MPI_Pack(&sender,     1,MPI_INT,(void*)buf,buflen,&pos,comm);
  MPI_Pack(&receiver,   1,MPI_INT,(void*)buf,buflen,&pos,comm);
  auto tag_data = *( tag.get_data() );
  MPI_Pack(&tag_data,tag.get_length(),MPI_INT,(void*)buf,buflen,&pos,comm);
  MPI_Pack(&depnumber,  1,MPI_INT,(void*)buf,buflen,&pos,comm);
  MPI_Pack(&innumber,   1,MPI_INT,(void*)buf,buflen,&pos,comm);
  MPI_Pack(&outnumber,  1,MPI_INT,(void*)buf,buflen,&pos,comm);

  if (global_struct->is_empty()) {
    int type = 1;
    MPI_Pack(&type,       1,MPI_INT,(void*)buf,buflen,&pos,comm);
    // nothing to do
  } else if (global_struct->is_contiguous()) {
    int type = 2;
    MPI_Pack(&type,       1,MPI_INT,(void*)buf,buflen,&pos,comm);
    auto glb = global_struct->first_index_r(), siz = global_struct->local_size_r();
    for (int id=0; id<dim; id++) {
      index_int src_index=glb[id], size=siz[id];
      MPI_Pack(&size,     1,MPI_INDEX_INT,(void*)buf,buflen,&pos,comm); long_int = pos-long_int;
      MPI_Pack(&src_index,1,MPI_INDEX_INT,(void*)buf,buflen,&pos,comm); long_int = pos;
    }
  } else {
    int type = global_struct->type_as_int();
    MPI_Pack(&type,       1,MPI_INT,(void*)buf,buflen,&pos,comm);
    if (dim>1) throw(format("Can not send struct in d={}",dim));
    auto component_struct = global_struct->get_component(0);
    index_int size = component_struct->local_size();
    long_int = pos;
    MPI_Pack(&size,     1,MPI_INDEX_INT,(void*)buf,buflen,&pos,comm);
    long_int = pos-long_int;
    for (index_int i=0; i<size; i++) {
      auto idx = component_struct->get_ith_element(i);
      MPI_Pack(&idx,1,MPI_INDEX_INT,(void*)buf,buflen,&pos,comm);
    }
  }
  MPI_Pack(&cookie,     1,MPI_INT,(void*)buf,buflen,&pos,comm);
  if (pos>buflen)
    throw(format("packing {} but anticipated {} |long|={}\n",
		      pos,buflen,long_int));
};

/*!
  Receive a buffered message and unpack it to a real message. 
  
  This used to be a method of the mpi_message class, but we abandoned that.
  \todo figure out a way to supply the MPI communicator. as argument? make this class method?
  \todo we need to lose the \ref message::in_object_number. make objects findable.
 */
shared_ptr<message> mpi_message_from_buffer
    ( MPI_Comm comm, shared_ptr<task> t,int step,string &buffer) {
  //  MPI_Comm comm = MPI_COMM_WORLD;
  int mytid = 314159; // only for CHK
  MPI_Status status;  shared_ptr<message> rmsg;
  int len = buffer.size(), got,err;
  //print("receive into buffer size {}\n",len);
  int sender,receiver;

  MPI_Probe(MPI_ANY_SOURCE,step,comm,&status);
  MPI_Get_count(&status,MPI_PACKED,&got);
  int source = status.MPI_SOURCE;
  //print("Probe says we got {} from {}\n",got,source);
  if (got<2*sizeof(int))
    throw(string("Message buffer way too short"));
  if (got>buffer.size())
    throw(format("Buffer incoming {}, space {}",got,buffer.size()));
  char *buf = (char*)(buffer.data());
  MPI_Recv(buf,len,MPI_PACKED,source,step,comm,&status);

  int pos=0;
  int cookie;
  err = MPI_Unpack(buf,len,&pos,&cookie,   1,MPI_INT,comm); CHK1(err);
  if (cookie!=-37)
    throw(format("unpacking something weird. before pos={}: cookie {} s/b -37",pos,cookie));

  int dim;
  err = MPI_Unpack(buf,len,&pos,&dim, 1,MPI_INT,comm); CHK1(err);

  // int iscollective;
  // err = MPI_Unpack(buf,len,&pos,&iscollective, 1,MPI_INT,comm); CHK1(err);
  int skippable;
  err = MPI_Unpack(buf,len,&pos,&skippable,    1,MPI_INT,comm); CHK1(err);

  err = MPI_Unpack(buf,len,&pos,&sender,   1,MPI_INT,comm); CHK1(err);
  err = MPI_Unpack(buf,len,&pos,&receiver, 1,MPI_INT,comm); CHK1(err);
  if (receiver!=source)
    throw(format("Unpacking message {}->{}, but coming from {}\n",
		      sender,receiver,source));

  message_tag mtag; //  int tagstuff[4]; // VLE replace that 4 with a class method?
  err = MPI_Unpack(buf,len,&pos,mtag.set_data(),mtag.get_length(),MPI_INT,comm); CHK1(err);

  int depnumber, innumber,outnumber;
  err = MPI_Unpack(buf,len,&pos,&depnumber,1,MPI_INT,comm); CHK1(err);
  err = MPI_Unpack(buf,len,&pos,&innumber, 1,MPI_INT,comm); CHK1(err);
  err = MPI_Unpack(buf,len,&pos,&outnumber,1,MPI_INT,comm); CHK1(err);
  
  auto dep = t->get_dependency(depnumber);
  auto out_obj = dep.get_beta_object(),
    in_obj = dep.get_in_object();

  shared_ptr<multi_indexstruct> global_struct;
  int type;
  err = MPI_Unpack(buf,len,&pos,&type,1,MPI_INT,comm); CHK1(err);
  if (type==1) { // empty
    global_struct = shared_ptr<multi_indexstruct>( new empty_multi_indexstruct(dim) );
  } else if (type==2) { // contiguous
    global_struct = shared_ptr<multi_indexstruct>( new multi_indexstruct(dim) );
    for (int id=0; id<dim; id++) {
      index_int src_index,size;
      err = MPI_Unpack(buf,len,&pos,&size,     1,MPI_INDEX_INT,comm); CHK1(err);
      err = MPI_Unpack(buf,len,&pos,&src_index,1,MPI_INDEX_INT,comm); CHK1(err);
      global_struct->set_component
	(id,shared_ptr<indexstruct>( new contiguous_indexstruct(src_index,src_index+size-1)) );
    }
  } else {
    if (dim>1) throw(format("How did we send a general message of d={}",dim));
    auto component = shared_ptr<indexstruct>( new empty_indexstruct() );
    index_int size;
    err = MPI_Unpack(buf,len,&pos,&size,1,MPI_INDEX_INT,comm); CHK1(err);
    for (index_int i=0; i<size; i++) {
      index_int element;
      err = MPI_Unpack(buf,len,&pos,&element,1,MPI_INDEX_INT,comm); CHK1(err);
      component = component->add_element(element);
    }
    global_struct->set_component(0,component);
  }

  err = MPI_Unpack(buf,len,&pos,&cookie,   1,MPI_INT,comm); CHK1(err);
  // print("found trailing cookie\n");
  if (cookie!=-37)
    throw(format("corruption check failed {}",cookie));

  try {
    auto 
      snd = in_obj->get_distribution()->get_decomposition().coordinate_from_linear(sender),
      rcv = out_obj->get_distribution()->get_decomposition().coordinate_from_linear(receiver);
    rmsg = out_obj->get_distribution()->new_message(snd,rcv,global_struct);
  } catch (string c) { print("Error <<{}>> creating message from buffer\n",c); };
  rmsg->set_tag(mtag); // message_tag(vector<int>(tagstuff)) );
  rmsg->set_in_object(in_obj); rmsg->set_out_object(out_obj);
  rmsg->compute_src_index(); rmsg->set_send_type();
  //  rmsg->set_is_collective(iscollective);
  rmsg->set_skippable(skippable);
  
  return rmsg;
};

/*!
  Compute the location from where we start sending,
  both as an index, and as an MPI subarray type.
*/
//snippet mpisrcindex
void mpi_message::compute_src_index() {
  message::compute_src_index();
  const auto indistro = get_in_object()->get_distribution();
  int
    ortho = indistro->get_orthogonal_dimension(),
    dim = indistro->get_dimensionality();
  int err;
  err = MPI_Type_create_subarray(dim+1,numa_sizes,struct_sizes,struct_starts,MPI_ORDER_C,MPI_DOUBLE,&embed_type);
  if (err!=0) throw(format("Type create subarray failed for src: err={}",err));
  MPI_Type_commit(&embed_type);
  if (dim==1) {
    lb = struct_starts[0]; extent = struct_sizes[0];
  } else if (dim==2) {
    lb = struct_starts[0]*numa_sizes[1]+struct_starts[1];
    extent = (struct_starts[0]+struct_sizes[0]-1)*numa_sizes[1] + struct_starts[1]+struct_sizes[1] - lb;
  } else
    throw(format("Can not compute lb/extent in dim>2"));
  if (extent<=0)
    throw(format("Zero extent in message <<{}>>",message::as_string()));
};
//snippet end

//snippet mpitarindex
//! Where does the message land in the beta structure?
void mpi_message::compute_tar_index() {
  message::compute_tar_index();
  const auto outdistro = get_out_object()->get_distribution();
  int
    ortho = outdistro->get_orthogonal_dimension(),
    dim = outdistro->get_dimensionality();
  int err;
  err = MPI_Type_create_subarray(dim+1,numa_sizes,struct_sizes,struct_starts,MPI_ORDER_C,MPI_DOUBLE,&embed_type);
  if (err!=0) throw(format("Type create subarray failed for src: err={}",err));
  MPI_Type_commit(&embed_type);
  lb = struct_starts[dim]; // compute analytical lower bound and extent
  extent = struct_sizes[dim];
  for (int id=dim-1; id>=0; id--) {
    lb += struct_starts[id]*numa_sizes[id+1];
    extent += (struct_sizes[id]-1)*numa_sizes[id+1];
  }
  {
    MPI_Aint true_lb,true_extent;
    MPI_Type_get_true_extent(embed_type,&true_lb,&true_extent);
    if (true_lb!=lb*sizeof(double))
      throw(format("Computed lb {} mismatch true lb {}",lb,true_lb));
    if (true_extent!=extent*sizeof(double))
      throw(format("Computed extent {} mismatch true extent {}",extent,true_extent));
  }
  if (extent<=0)
    throw(format("Zero extent in message <<{}>>",message::as_string()));
};
//snippet end

/****
 **** Task
 ****/

/*!
  Build the send messages;
*/
void mpi_task::derive_send_messages(bool trace) {
  auto out = get_out_object();
  const auto outdistro = out->get_distribution();
  int dim = outdistro->get_dimensionality(), step = get_step();
  architecture arch = out->get_decomposition();
  if (trace) print("Deriving recv msgs for task {}\n",get_name());
  auto dom = get_domain(); int mytid = dom.linearize(outdistro->get_decomposition());
  MPI_Comm comm = MPI_COMM_WORLD;

  auto recv_messages = get_receive_messages();
  int nsends, nrecvs = recv_messages.size(),
    buflen = mpi_message_buffer_length(dim)+8;
  try {
    nsends = get_nsends();
  } catch (string c ) {
    print("Error <<{}>> determining nsends",c);
    throw(string("get_nsends fail")); };
  if (trace)
    print("[{}] #recv={}, #sends={}\n",dom.as_string(),nrecvs,nsends);

  vector< MPI_Request > mpi_requests(nrecvs);
  vector< MPI_Status >  mpi_statuses(nrecvs); 
  vector<string> buffers(nrecvs,string(buflen,' '));

  // turn each recv message into a buffer and asynchronously send it to the future sender
  for (int imsg=0; imsg<recv_messages.size(); imsg++) {
    auto msg = recv_messages.at(imsg);
    MPI_Request req;
    try {
      arch.message_as_buffer(arch,msg,buffers[imsg] /* ,buflen */);
    } catch (string c) {
      print("Error <<{}>> for buffering msg {}->{} in <<{}>>\n",
		 c,msg->get_sender().as_string(),msg->get_receiver().as_string(),get_name());
      throw(format("Could not convert msg to buffer"));
    }
    auto otherdom = msg->get_sender();
    int other = otherdom.linearize(outdistro->get_decomposition());
    if (trace)
      print("[{}] send msg to {}, tag={}\n",dom.as_string(),other,step);
    MPI_Isend(buffers.at(imsg).data(),buffers.at(imsg).size(),MPI_PACKED,other,step,comm,&req);
    mpi_requests.push_back(req);
  }

  // now receive the buffers that tell you what is requested of you
  {
    vector<shared_ptr<message>> my_send_messages;
    for (int i=0; i<nsends; ++i) {
      string buf(buflen+8,' ');
      try {
	auto msg = this->message_from_buffer(step,buf);
	msg->set_name( format("send-{}-obj:{}->{}",
	     msg->get_name(),msg->get_in_object_number(),msg->get_out_object_number()) );
	my_send_messages.push_back(msg);
      } catch (string c) {
	print("Error in message from buffer w obj <<{}>>: <<{}>>\n",out->get_name(),c);
	throw(format("Could not derive send msgs in <<{}>>",get_name()));
      }
    }
    if (trace) print(".. found {} send messages\n",my_send_messages.size());
    set_send_messages(my_send_messages);
  }

  int irequest = mpi_requests.size();
  int mpi_err = MPI_Waitall(irequest,mpi_requests.data(),MPI_STATUSES_IGNORE);
  // for (int irecv=0; irecv<nrecvs; irecv++)
  //   delete buffers[irecv];
  // delete buffers;

  //mpi_statuses.data());
  // if (mpi_err!=0) {
  //   for (int ireq=0; ireq<irequest; ireq++) {
  //     int errorcode = mpi_statuses[ireq].MPI_ERROR;
  //     if (errorcode!=0) {
  // 	char message[256]; int msglen;
  // 	MPI_Error_string(errorcode,message,&msglen);
  // 	print("Error [{}] in request # {}\n",message,ireq);
  //     }
  //   }
  // }
  //delete mpi_requests; delete mpi_statuses;
   
  // localize send structures
  for ( auto &msg : get_send_messages() ) {
    const auto indistro = msg->get_in_object()->get_distribution();
    msg->relativize_to( indistro->get_processor_structure(dom) );
  }
};

void mpi_request_vector_wait( request_vector &v ) {
  int s = v.size();
  vector<MPI_Request> reqs(s); // auto reqs = new MPI_Request[s];
  for (int i=0; i<s; i++) {
    if (v.at(i)->protocol!=request_protocol::MPI)
      throw(string("transmutated request?\n"));
    auto mpi_req = dynamic_cast<mpi_request*>(v.at(i).get());
    if (mpi_req==nullptr)
      throw(string("Could not upcast to mpi request"));
    reqs.at(i) = mpi_req->get_mpi_request();
    auto msg = mpi_req->get_message();
  }
  MPI_Waitall(s,reqs.data(),MPI_STATUSES_IGNORE);
  v.set_completed();
  // delete[] reqs;
};

/*!
  Post an MPI_Isend for a bunch of messages. The requests are passed back in
  an array that is in/out: this way a task can post messages that really
  belong to a much later task.
  \todo is there a way to get the MPI_Comm without that casting? note that by now none of our base classes are mpi_specific.
  \todo why do we make a new processor coordinate?
*/
shared_ptr<request> mpi_task::notifyReadyToSendMsg( shared_ptr<message> msg ) {
  const auto distro = get_out_object()->get_distribution();
  MPI_Comm comm = *(MPI_Comm*)( distro->get_communicator_context() );
  auto vec = msg->get_in_object(), halo = msg->get_out_object();
  const auto vecdistro = vec->get_distribution(),
    halodistro = halo->get_distribution();
  processor_coordinate dom(get_domain());
  int k = vecdistro->get_orthogonal_dimension(), ireq=0;
  auto
    numa_struct = vecdistro->get_processor_structure(dom),
    local_struct = msg->get_local_struct();
  index_int numa_size = numa_struct->volume();
  MPI_Request req;
  auto sender = msg->get_sender(),receiver = msg->get_receiver();
  if (sender==receiver && vec->has_data_status_inherited()
      && vec->get_data_parent()==halo->get_object_number()) { // we can skip certain messages
    msg->set_status( message_status::SKIPPED );
    return nullptr;
  }
  {
    index_int
      src_index = msg->get_src_index(),src_size = local_struct->volume();
    if (k*src_index+k*src_size>k*numa_size)
      throw(format("Message <<{}>>: send buffer overflow {}+{}>{} (k={})",
			msg->as_string(),k*src_index,k*src_size,numa_size,k));
  }
  if (halodistro->get_use_rma())
    throw(string("RMA not implemented"));
  mpi_message *mmsg = dynamic_cast<mpi_message*>(msg.get());
  if (mmsg==nullptr) throw(string("Could not convert snd msg to mpi"));
  int sender_no = sender.linearize(vecdistro->get_decomposition()),
    receiver_no = receiver.linearize(vecdistro->get_decomposition());
  {
    //snippet mpisend
    double *data = vec->get_data(dom).data();
    MPI_Isend( data,1,mmsg->embed_type,receiver_no,msg->get_tag_value(),comm,&req);
    //snippet end
  }
  return shared_ptr<request>( new mpi_request(msg,req) );
};

/*!
  Post an MPI_Irecv for a bunch of messages. The requests are passed back in
  an array that is in/out: this way a task can post messages that really
  belong to a much later task.

  \todo should we use get_numa_structure?
  \todo can we use the embed_type in the Iallgather case?
  \todo can we integrate this further with the OMP version?
*/
request_vector mpi_task::acceptReadyToSend( vector<shared_ptr<message>> &msgs ) {
  request_vector requests;
  MPI_Comm comm = MPI_COMM_WORLD;
  for ( auto msg : msgs ) {
    try {
      auto vec = msg->get_in_object(), halo = msg->get_out_object();
      const auto halodistro = halo->get_distribution(),
	vecdistro = vec->get_distribution();
      auto dom = get_domain(); 
      int k = halodistro->get_orthogonal_dimension();
      auto local_struct = msg->get_local_struct();
      auto sender = msg->get_sender(),receiver = msg->get_receiver();
      int sender_no = sender.linearize(vecdistro->get_decomposition()),
	receiver_no = receiver.linearize(vecdistro->get_decomposition());
      auto indistro = msg->get_in_object()->get_distribution();
      if (sender==receiver && vec->has_data_status_inherited()
	  && vec->get_data_parent()==halo->get_object_number())
	continue;
      MPI_Request req;
      if (msg->get_is_collective() // VLE need a simpler test here!
	  && halodistro->has_collective_strategy(collective_strategy::MPI)
	  && !msg->is_skippable() ) {
	throw(format("acceptReadyToSend for MPI collective use derived class"));
	auto sobject = msg->get_in_object();
	const auto sdistro = sobject->get_distribution();
	auto rdata = halo->get_raw_data();
	auto size = local_struct->volume();

	// decltype( sdistro->get_linear_offsets() ) offsets;
	// decltype( sdistro->get_linear_sizes() ) sizes;
	vector<int> offsets, sizes;
	try {
	  offsets = sdistro->get_linear_offsets(); sizes = sdistro->get_linear_sizes();
	} catch (string c) { print("Error: {}\n",c);
	  throw(format("Error getting linear offset/sizes for : {}",halo->as_string())); }
	int send_size = size * !sdistro->get_processor_skip(sender_no);
	{
	  auto sdata = sobject->get_data(dom).data();
	  MPI_Iallgatherv
	    (sdata,send_size,MPI_DOUBLE,
	     rdata,sizes.data(),offsets.data(),MPI_DOUBLE,
	     comm,&req);
	}
      } else {
	auto numa_size = halodistro->local_allocation_p(dom);
	mpi_message *rmsg = dynamic_cast<mpi_message*>(msg.get());
	if (rmsg==nullptr) throw(string("Could not convert recv msg to mpi"));
	if (rmsg->lb+rmsg->extent>numa_size)
	  throw(format("Irecv buffer overflow: lb={} extent={}, available={}",
			    rmsg->lb,rmsg->extent,numa_size));
	double *rdata = halo->get_data(dom).data();
        //snippet mpirecv
        MPI_Irecv( rdata,1,rmsg->embed_type,sender_no,msg->get_tag_value(),comm,&req);
        //snippet end
      }
      requests.add_request( shared_ptr<request>( new mpi_request(msg,req) ) );
    } catch (string c) { print("Error <<{}>> for msg <<{}>>\n",c,msg->as_string());
      throw(format("Could not post receive message for task <<{}>>",as_string()));
    }
  }
  return requests;
};

request_vector mpi_collective_task::acceptReadyToSend( vector<shared_ptr<message>> &msgs ) {
  request_vector requests;
  return requests;
};

request_vector mpi_collective_task::notifyReadyToSend( vector<shared_ptr<message>> &msgs ) {
  request_vector requests;
  MPI_Comm comm = MPI_COMM_WORLD;
  for ( auto msg : msgs ) {
    try {
      auto vec = msg->get_in_object(), halo = msg->get_out_object();
      const auto halodistro = halo->get_distribution(),
	vecdistro = vec->get_distribution();
      auto dom = get_domain(); 
      int k = halodistro->get_orthogonal_dimension();
      auto local_struct = msg->get_local_struct();
      auto sender = msg->get_sender(),receiver = msg->get_receiver();
      int sender_no = sender.linearize(vecdistro->get_decomposition()),
	receiver_no = receiver.linearize(vecdistro->get_decomposition());
      auto indistro = msg->get_in_object()->get_distribution();
      if (sender==receiver && vec->has_data_status_inherited()
	  && vec->get_data_parent()==halo->get_object_number())
	continue;
      MPI_Request req;
      {
	auto sobject = msg->get_in_object();
	const auto sdistro = sobject->get_distribution();
	auto rdata = halo->get_raw_data();
	auto size = local_struct->volume();
	vector<int> offsets, sizes;
	try {
	  offsets = sdistro->get_linear_offsets(); sizes = sdistro->get_linear_sizes();
	} catch (string c) { print("Error: {}\n",c);
	  throw(format("Error getting linear offset/sizes for : {}",halo->as_string())); }
	int send_size = size * !sdistro->get_processor_skip(sender_no);
	auto sdata = sobject->get_data(dom).data();
	MPI_Iallgatherv
	  (sdata,send_size,MPI_DOUBLE,
	   rdata,sizes.data(),offsets.data(),MPI_DOUBLE,
	   comm,&req);
      }
      requests.add_request( shared_ptr<request>( new mpi_request(msg,req) ) );
    } catch (string c) { print("Error <<{}>> for msg <<{}>>\n",c,msg->as_string());
      throw(format("Could not post receive message for task <<{}>>",as_string()));
    }
  }
  return requests;
};

//snippet mpi-depend
/*!
  Create a dependence in the local task graph. Since the task can be on a different
  address space, we declare a dependence on the task from the same kernel,
  but on this domain.
 */
void mpi_task::declare_dependence_on_task( task_id *id ) {
  int step = id->get_step(); auto domain = this->get_domain();
  try {
    add_predecessor( find_other_task_by_coordinates(step,domain) );
  } catch (string c) {
    print("Task <<{}>> error <<{}>> locating <{},{}>.\n{}\n",
	       get_name(),c,step,domain.as_string(),this->as_string());
    throw(string("Could not find MPI local predecessor"));
  };
};
//snippet end

/****
 **** Kernel
 ****/

mpi_kernel::mpi_kernel() : kernel() {};
mpi_kernel::mpi_kernel( shared_ptr<object> out )
try : kernel(out) {
  install_mpi_factory(); //install_mpi_kernel_factory(this);
      } catch (...) {
  throw(format("failed mpi_kernel constructor from out"));
 };
mpi_kernel::mpi_kernel( shared_ptr<object> in,shared_ptr<object> out )
try : kernel(in,out) {
  install_mpi_factory(); //install_mpi_kernel_factory(this);
      } catch (...) {
  throw(format("failed mpi_kernel constructor from in/out"));
 };
//! This copy constructur is used in task creation. 
mpi_kernel::mpi_kernel( const kernel& k )
try : kernel(k) {
  install_mpi_factory(); //install_mpi_kernel_factory(this);
      } catch (...) {
  throw(format("failed mpi_kernel constructor from kernel"));
 };

void mpi_kernel::install_mpi_factory() {
  try {
    auto b = is_collective();
  } catch (...) { print("kernel could not ask collective\n"); throw(string("error")); }
  if (is_collective()
      && get_decomposition().get_collective_strategy()==collective_strategy::MPI) {
    make_task_for_domain =
      [] ( kernel *k,const processor_coordinate &p ) -> shared_ptr<task> 
      { auto t = shared_ptr<task>( new mpi_collective_task(p,k) );
  	return t;
      };
  } else {
    make_task_for_domain =
      [] ( kernel *k,const processor_coordinate &p ) -> shared_ptr<task> 
      { auto t = shared_ptr<task>( new mpi_task(p,k) );
	return t;
      };
  }
};

/****
 **** Queue
 ****/
