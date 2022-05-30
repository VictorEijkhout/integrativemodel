/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** imp_base.cxx: Implementations of the base classes
 ****
 ****************************************************************/

#include "utils.h"
#include "imp_env.h"
#include "imp_base.h"

using fmt::format;
using fmt::print;
using fmt::memory_buffer;
using fmt::format_to;
using fmt::to_string;

using std::cerr;
using std::cout;
using std::endl;

using std::bad_alloc;
using std::string;

using std::move;
using std::make_shared;
using std::shared_ptr;

using std::vector;

using std::get;
using std::make_tuple;
using std::tuple;
using std::tie;


#include <climits>

/****
 **** Basics
 ****/

/****
 **** Architecture
 ****/

vector<int> make_endpoint(int d,int s) {
  vector<int> endpoint;
  for (int id=0; id<d; id++)
    endpoint.push_back(1);
  endpoint[0] = s;
  for (int id=0; id<d-1; id++) { // find the largest factor of endpoint[id] and put in id+1
    for (int f=(int)(sqrt(endpoint[id])); f>=2; f--) {
      if (endpoint[id]%f==0) {
	endpoint[id] /= f; endpoint[id+1] = f;
	break; // end factor finding loop, go to next dimension
      }
    }
  }
  for (int id=0; id<d; id++)
    endpoint[id]--;
  return endpoint;
};

/*
 * Architectures
 */
architecture::architecture()
  : entity(entity_cookie::ARCHITECTURE) {};

/*!
  Return the original processor; for now that's always zero. 
  See \ref architecture::get_proc_endpoint
  \todo this one is never used. Do away?
*/
processor_coordinate architecture::get_proc_origin(int d) const {
  return processor_coordinate( vector<int>(d,0) );
};

//! Multi-d descriptor of number of processes. This is actually the highest proc number.
//! \todo find a much better way of deducing or setting a processor grid
processor_coordinate architecture::get_proc_endpoint(int d) const {
  int P = nprocs()*get_over_factor();
  return processor_coordinate( make_endpoint(d,P) );
 //  processor_coordinate *coord;
 //  if (d==1) {
 //    coord = new processor_coordinate(1); coord->set(0,P-1);
 //  } else if (d==2) {
 //    coord = new processor_coordinate(2);
 //    int ntids_i,ntids_j;
 //    for (int n=sqrt(P+1); n>=1; n--)
 //      if (P%n==0) { // real grid otherwise
 // 	ntids_i = P/n; ntids_j = n;
 // 	coord->set(0,ntids_i-1); coord->set(1,ntids_j-1); goto found;
 //      }
 //    coord->set(0,P-1); coord->set(1,0); // pencil by default
 //  } else throw(format("Can not make proc endpoint of d={}",d));
 // found:
 //  return coord;
};

//! The size layout is one more than the endpoint \todo return as non-reference copy
processor_coordinate architecture::get_proc_layout(int dim) const {
  auto layout = get_proc_endpoint(dim);
  for (int id=0; id<dim; id++)
    layout.set(id, layout.coord(id)+1 );
  return layout;
};

//! Set the collective strategy by value.
void architecture::set_collective_strategy(collective_strategy s) {
  strategy = s;
};

//! Set the collective strategy by number; great for commandline arguments.
void architecture::set_collective_strategy(int s) {
  switch (s) {
  case 1 : set_collective_strategy( collective_strategy::ALL_PTP ); break;
  case 2 : set_collective_strategy( collective_strategy::GROUP ); break;
  case 3 : set_collective_strategy( collective_strategy::RECURSIVE ); break;
  case 4 : set_collective_strategy( collective_strategy::MPI ); break;
  case 0 :
  default :
    set_collective_strategy( collective_strategy::ALL_PTP ); break;
  }
};

string architecture_type_as_string( architecture_type type ) {
  switch (type) {
  case architecture_type::UNDEFINED : return string("Undefined architecture"); break;
  case architecture_type::SHARED : return string("Shared memory"); break;
  case architecture_type::SPMD : return string("MPI architecture"); break;
  case architecture_type::ISLANDS : return string("MPI+OpenMP") ; break;
  default : throw(string("Very undefined architecture")); break;
  }
};

string protocol_as_string(protocol_type p) {
  if (p==protocol_type::UNDEFINED) return string("UNDEFINED");
  else if (p==protocol_type::OPENMP) return string("OPENMP");
  else if (p==protocol_type::MPI) return string("MPI");
  else if (p==protocol_type::MPIplusOMP) return string("MPIplusOMP");
  else if (p==protocol_type::MPItimesOMP) return string("MPItimesOMP");
  else return string("UnKnown???");
};

//! Base case for architecture summary.
string architecture::summary() { fmt::memory_buffer w;
  format_to(w.end(),"{}",get_name());
  format_to(w.end(),", protocol: {}",protocol_as_string(protocol));
  format_to(w.end(),", collectives: {}",strategy_as_string());
  if (get_split_execution()) format_to(w.end(),", split execution");
  if (has_random_sourcing()) format_to(w.end(),", source randomization");
  if (get_can_embed_in_beta()) format_to(w.end(),", object embedding");
  if (get_can_message_overlap()) format_to(w.end(),", messages post early");
  if (algorithm::do_optimize) format_to(w.end(),", task graph optimized");
  return to_string(w);
};

std::string architecture::as_string() const {
  return format
    ("Type: {}, #procs: {}",architecture_type_as_string(type),arch_nprocs);
};


/****
 **** Decomposition
 ****/

//! Default decomposition uses all procs of the architecture in one-d manner.
decomposition::decomposition( const architecture &arch )
  : decomposition( arch, processor_coordinate
		   ( vector<int>{arch.nprocs()*arch.get_over_factor()} ) ) {
};

/*!
  In multi-d the user needs to indicate how the domains are laid out.
  The processor coordinate is a size specification, to make it compatible with nprocs
*/
//snippet decompfromcoord
decomposition::decomposition( const architecture &arch,processor_coordinate &sizes )
  : architecture(arch) {
  int dim = sizes.get_dimensionality();
  if (dim<=0)
    throw(string("Non-positive decomposition dimensionality"));
  domain_layout = sizes; //new processor_coordinate(sizes);
  set_corners();
};
//snippet end
decomposition::decomposition( const architecture &arch,processor_coordinate &&sizes )
  : architecture(arch) {
  int dim = sizes.get_dimensionality();
  if (dim<=0)
    throw(string("Non-positive decomposition dimensionality"));
  domain_layout = sizes; //new processor_coordinate(sizes);
  set_corners();
};

//! Get dimensionality.
int decomposition::get_dimensionality() const {
  int dim = domain_layout.get_dimensionality();
  return dim;
};

//! Number of domains
int decomposition::domains_volume() const {
  int p = domain_layout.volume();
  return p;
};

const processor_coordinate &decomposition::first_local_domain() const {
  if (mdomains.size()==0)
    throw(format("Decomposition has no domains"));
  return mdomains.at(0);
};
const processor_coordinate &decomposition::last_local_domain() const {
  if (mdomains.size()==0)
    throw(format("Decomposition has no domains"));
  return mdomains.at(mdomains.size()-1);
};

//! Get the local number where this domain is stored. Domains are multi-dimensionally numbered.
int decomposition::get_domain_local_number( const processor_coordinate &d ) const {
  // print("get domain local number of {} in decomp: <<{}>>\n",
  // 	d.as_string(),this->as_string());
  for ( int i=0; i<mdomains.size(); i++) {
    // print("compare domain {}: {} to match coordinate {}\n",
    // 	  i,mdomains.at(i).as_string(),d.as_string());
    if (mdomains.at(i)==d) return i;
  }
  throw(fmt::format("Domain has no localization"));
};

//! Get dimensionality, which has to be the same as something else.
int decomposition::get_same_dimensionality( int d ) const {
  if (d!=domain_layout.get_dimensionality())
    throw(format("Coordinate dimensionality mismatch decomposition:{} vs layout{}",
		 d,domain_layout.as_string()));
  return d;
};

void decomposition::copy_embedded_decomposition( decomposition &other ) {
  embedded_decomposition = other.embedded_decomposition;
};

const decomposition &decomposition::get_embedded_decomposition() const {
  if (embedded_decomposition==nullptr)
    throw(string("Null embedded decomposition"));
  return *(embedded_decomposition.get());
};

/*! Add a bunch of 1d local domains
  \todo test for 1d-ness
  \todo use std_ptr to indexstruct*
*/
void decomposition::add_domains( indexstruct *d ) {
  //print("adding domains from indexstruct <<{}>>\n",d->as_string());
  for ( auto i=d->first_index(); i<=d->last_index(); i++ ) {
    int ii = (int)i; // std::static_cast<int>(i);
    add_domain( processor_coordinate( vector<int>{ ii } ),false );
  }
  set_corners();
};

//! Add a multi-d local domain.
void decomposition::add_domain( processor_coordinate d,bool recompute ) {
  //print("adding domain at coordinate <<{}>>\n",d.as_string());
  int dim = d.get_same_dimensionality( domain_layout.get_dimensionality() );
  mdomains.push_back(d);
  if (recompute) set_corners();
};

//! Get the multi-dimensional coordinate of a linear one. \todo check this calculation
//! http://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
processor_coordinate decomposition::coordinate_from_linear(int p) const {
  int dim = domain_layout.get_dimensionality();
  if (dim<=0) throw(string("Zero dim layout"));
  processor_coordinate pp(dim);
  for (int id=dim-1; id>=0; id--) {
    int dsize = domain_layout.coord(id);
    if (dsize==0)
      throw(format("weird layout <<{}>>",domain_layout.as_string()));
    pp.set(id,p%dsize); p = p/dsize;
  };
  return pp;
};

void decomposition::set_corners() {
  // origin; we set this to all zeros, which may not be enough.
  int d = get_dimensionality();
  closecorner = processor_coordinate( vector<int>(d,0) );
  // farcorner; explicit endpoint.
  int P = domains_volume();
  farcorner = processor_coordinate( make_endpoint(d,P) );
};

//! \todo why can't we declare this const?
const processor_coordinate &decomposition::get_origin_processor() const {
  return closecorner;
};
const processor_coordinate &decomposition::get_farpoint_processor() const {
  return farcorner;
};

string decomposition::as_string() const {
  return "decomp";
  // format
  //   ("{}; dim={}, #domains: {}",
  //    architecture::as_string(),get_dimensionality(),domains_volume());
};

/*!
  We begin iteration by giving the first coordinate.
  We store the current iterate as a private processor_coordinate: `cur_coord'.
  Iteration is done C-style: the last coordinate varies quickest.

  Note: iterating is only defined for bricks.
*/
decomposition &decomposition::begin() {
  iterate_count = 0;
  if (start_coord.get_dimensionality()>0) {
    start_id = linearize(start_coord);
  } else {
    try { start_id = mytid();
    } catch (...) { print("should have been able to get mytid\n");
      throw("can not begin iterate over decomposition");
    }
    start_coord = coordinate_from_linear(start_id);
  }
  cur_coord = start_coord;
  //  print("begin {} @{}+{}",cur_coord.as_string(),start_id,iterate_count);
  return *this;
};

decomposition &decomposition::end() {
  // we end when we've counted all the tids
  iterate_count = domains_volume();
  return *this;
};

/*!
  Here's how to iterate: 
  - from last to first dimensions, find the dimension where you are not at the far edge
  - increase the coordinate in that dimension
  - all higher dimensions are reset to the first coordinate.
*/
void decomposition::operator++() {
  start_id = linearize(start_coord); // !!! VLE should not be necessary
  int ntids = domains_volume();
  int range_id{-1};
  //  print("increment {} @{}+{}",cur_coord.as_string(),start_id,iterate_count);
  iterate_count++;
  if (range_linear || (range_twoside && iterate_count>=0) ) {
    range_id = (start_id+iterate_count) % ntids;
  } else {
    range_id = (start_id-iterate_count+ntids) % ntids;
  }
  cur_coord = coordinate_from_linear(range_id);
  // print(".. to {} @{}+{}\n",cur_coord.as_string(),start_id,iterate_count);

  return;
};

bool decomposition::operator!=( const decomposition &other ) const {
  return iterate_count!=other.iterate_count;
};

bool decomposition::operator==( const decomposition &other ) const {
  return iterate_count==other.iterate_count;
};

processor_coordinate &decomposition::operator*() {
  //print("decomp::deref: {}\n",cur_coord.as_string());
  return cur_coord;
};

/****
 **** Parallel indexstructs
 ****/

//! Basic constructor with nd domains.
parallel_indexstruct::parallel_indexstruct(int nd) {
  processor_structures.reserve(nd);
  for (int is=0; is<nd; is++)
    processor_structures.push_back
      ( shared_ptr<indexstruct>( new unknown_indexstruct() ) );
};

//!< This assumes creation is complete.
int parallel_indexstruct::size() const {
  return processor_structures.size();
};

//! This sounds collective but it's really a local function.
bool parallel_indexstruct::is_known_globally() {
  for ( auto p : processor_structures )
    if (!p->is_known()) return false;
  return true;
}

/*! Query the number of domains in this parallel indexstruct.
  (In multi-d the parallel indexstruct is defined on fewer than global ndomains.)
  Note that this can be used before the actual structures are set.
*/
int parallel_indexstruct::pidx_domains_volume() const {
  int p = processor_structures.size();
  return p;
};

/*! Copy constructor. Only the type is copied literally; the structures are recreated.
  \todo we're not handling the local_structure case yet.
  \todo we really should not copy undefined structures
  \todo can we use the copy constructor of vector for the structures?
*/
parallel_indexstruct::parallel_indexstruct( const parallel_indexstruct *other ) {
  int P = other->pidx_domains_volume();
  // processor_structures = new vector< shared_ptr<indexstruct> >;
  processor_structures.reserve(P);

  for (int p=0; p<P; p++) {
    try {
      shared_ptr<indexstruct> otherstruct = other->get_processor_structure(p);
      processor_structures.push_back( otherstruct ); //->make_clone() );
    }
    catch (string c) {print("Error <<{}>> in get_processor_structure for copy\n",c);
      throw( format("Could not get pstruct {} from <<{}>>",p,other->as_string()) ); }
    catch (...) { print("Unknown error setting proc struct for p={}\n",p); }
  }
  const auto old_t = other->get_type();
  if (old_t==distribution_type::UNDEFINED)
    throw(format("Pidx to copy from has undefined distribution type: {}",other->as_string()));
  set_type(old_t);
};

//! Equality test by comparing pstructs \todo this does not account for localstruct
int parallel_indexstruct::equals(parallel_indexstruct *s) const {
  int t = 1;
  for (int p=0; p<pidx_domains_volume(); p++) {
    t = t && this->get_processor_structure(p)->equals(s->get_processor_structure(p));
    if (!t) return 0;
  }
  return t;
};

void psizes_from_global( vector<index_int> &sizes,int P,index_int gsize) {
  index_int blocksize = (index_int)(gsize/P);
  // first set all sizes equal
  for (int i=0; i<P; ++i)
    sizes[i] = blocksize;
  // then spread the remainder
  for (int i=0; i<gsize-P*blocksize; ++i)
    sizes[i]++;
}

/*!
  Create from an indexstruct. This can for instance be used to make an
  OpenMP parallel structure on an MPI numa domain.

  \todo make sure that the indexstruct is the same everywhere
*/
void parallel_indexstruct::create_from_indexstruct( shared_ptr<indexstruct> ind ) {
  if (!ind->is_strided())
    throw(format("Can not create pidx from indexstruct: {}",ind->as_string()));
  int P = pidx_domains_volume(); int stride = ind->stride();
  if (P==0)
    throw(string("parallel indexstruct does not have #domains set"));
  int Nglobal = ind->last_index()-ind->first_index()+1;
  index_int usize = Nglobal/P; vector<index_int> sizes(P);
  psizes_from_global(sizes,P,Nglobal);
  // now set first/last
  index_int sum = ind->first_index();
  for (int p=0; p<P; p++) {
    index_int first=sum, next=sum+sizes[p];
    if (next<first) throw(string("suspicious local segment derived"));
    if (stride==1)
      processor_structures.at(p) = 
	shared_ptr<indexstruct>( new contiguous_indexstruct(first,next-1) );
    else
      processor_structures.at(p) = 
	shared_ptr<indexstruct>( new strided_indexstruct(first,next-1,stride) );
    sum = next;
  }
  set_type(distribution_type::GENERAL);
};

/*!
  Usually we create parallel structures with indices 0..N-1
*/
void parallel_indexstruct::create_from_global_size(index_int gsize) {
  try {
    create_from_indexstruct
      ( shared_ptr<indexstruct>( new contiguous_indexstruct(0,gsize-1) ) );
  } catch (string c) { 
    throw(format("Error in create from global_size: <<{}>>",c)); 
  }
  set_type(distribution_type::CONTIGUOUS);
};

/*!
  Create a parallel index structure where every processor has the same
  number of points, allocated consecutively.

  \todo how do we do the case of mpi localsize?
*/
void parallel_indexstruct::create_from_uniform_local_size(index_int lsize) {
  int P = pidx_domains_volume();
  // now set first/last
  index_int sum = 0;
  for (int p=0; p<P; p++) {
    index_int first=sum, next=sum+lsize;
    processor_structures.at(p) =
      shared_ptr<indexstruct>( new contiguous_indexstruct(first,next-1) );
    sum = next;
  }
  set_type(distribution_type::CONTIGUOUS);
};

void parallel_indexstruct::create_from_local_sizes( vector<index_int> lsize) {
  // now set first/last
  int P = pidx_domains_volume();
  if (P!=lsize.size())
    throw(format("Global ndomains={} vs supplied vector of sizes: {}",P,lsize.size()));
  index_int sum = 0;
  for (int p=0; p<P; p++) {
    index_int s = lsize.at(p), first = sum, next = sum+s;
    processor_structures.at(p) =
      shared_ptr<indexstruct>( new contiguous_indexstruct(first,next-1) );
    sum = next;
  }
  set_type(distribution_type::CONTIGUOUS);
};

void parallel_indexstruct::create_from_replicated_local_size(index_int lsize) {
  create_from_replicated_indexstruct
    ( shared_ptr<indexstruct>( new contiguous_indexstruct(0,lsize-1) ) );
};

//! \todo can we use the same pointer everywhere?
void parallel_indexstruct::create_from_replicated_indexstruct(shared_ptr<indexstruct> idx)
{
  int P = pidx_domains_volume();
  for (int p=0; p<P; p++) {
    processor_structures.at(p) = idx;
  }
  set_type(distribution_type::REPLICATED);
};

//! Why is that neg-neg exception not perculating up in unittest_distribution:[40]?
void parallel_indexstruct::create_cyclic(index_int lsize,index_int gsize) {
  int P = pidx_domains_volume();
  if (gsize<0) {
    if (lsize<0) {
      print("Need lsize or gsize for cyclic, proceeding with lsize=1\n");
      lsize = 1;
    }
    gsize = P*lsize;
  } else if (lsize<0) lsize = gsize/P;

  if (P*lsize!=gsize) {
    print("Incompatible lsize {} vs gsize {}",lsize,gsize);
    gsize = P*lsize;
  }
  for (int p=0; p<P; p++) {
    processor_structures.at(p) =
      shared_ptr<indexstruct>( new strided_indexstruct(p,gsize-1,P) );
  }
  set_type(distribution_type::CYCLIC);
};

//! \todo get rid of that clone
void parallel_indexstruct::create_blockcyclic(index_int bs,index_int nb,index_int gsize) {
  if (nb==1) {
    create_cyclic(bs,gsize); return; }

  int P = pidx_domains_volume();
  if (gsize>=0) { printf("gsize ignored in blockcyclic\n");
  }
  index_int lsize = bs*nb;
  gsize = P*lsize;

  if (P*lsize!=gsize) {
    print("Incompatible lsize {} vs gsize {}",lsize,gsize);
    gsize = P*lsize;
  }
  for (int p=0; p<P; p++) {
    index_int proc_first = p*bs;
    composite_indexstruct *local = new composite_indexstruct();
    for (index_int ib=0; ib<nb; ib++) {
      index_int block_first = proc_first+ib*P*bs;
      local->push_back
	( shared_ptr<indexstruct>
	  ( new contiguous_indexstruct(block_first,block_first+bs-1) ) );
    }
    processor_structures.at(p) = shared_ptr<indexstruct>( local->make_clone() );
  }
  set_type(distribution_type::CYCLIC);
};

void parallel_indexstruct::create_from_explicit_indices(index_int *lens,index_int **sizes) {
  // this is the most general case. for now only for replicated scalars
  index_int nmax = 0;
  int P = pidx_domains_volume();
  for (int p=0; p<P; p++) {
    processor_structures.at(p) =
      shared_ptr<indexstruct>( new indexed_indexstruct(lens[p],sizes[p]) );
    for (index_int i=0; i<lens[p]; i++)
      nmax = MAX(nmax,sizes[p][i]);
  }
  set_type(distribution_type::GENERAL);
};

void parallel_indexstruct::create_from_function
        ( index_int(*pf)(int p,index_int i),index_int nlocal ) {
  index_int nmax = 0;
  int P = pidx_domains_volume();
  for (int p=0; p<P; p++) {
    index_int *ind = new index_int[nlocal];
    for (index_int i=0; i<nlocal; i++) {
      index_int n = (*pf)(p,i); nmax = MAX(nmax,n);
      ind[i] = n;
    }
    processor_structures.at(p) =
      shared_ptr<indexstruct>( new indexed_indexstruct( nlocal,ind ) );
  }
  set_type(distribution_type::GENERAL);
};

//! \todo instead of passing in an object, pass object data?
void parallel_indexstruct::create_by_binning( object *o, double mn, double mx, int id ) {
  throw(string("fix the binning routine"));
  set_type(distribution_type::GENERAL);
};

/*!
  Get the structure of processor p.
  In many cases, every processor has global knowledge, so regardless the address space
  you can indeed ask about any processor.
  However, it is allowed to return nullptr; see the copy constructor.
  \todo ugly. we should really fix the copy constructor.
*/
shared_ptr<indexstruct> parallel_indexstruct::get_processor_structure(int p) const {
  if (local_structure!=nullptr) return local_structure;
  int P = pidx_domains_volume();
  if (p<0 || p>=P)
    throw(format("Requested processor {} out of range 0-{}",p,P));
  else if (p>processor_structures.size())
    throw(format("No processor {} in structure of size {}",p,processor_structures.size()));
  shared_ptr<indexstruct> pstruct = processor_structures.at(p);
  if (pstruct==nullptr)
    throw(format("Found null processor structure at linear {}",p));
  return pstruct;
};

/*! Set a processor structure; this redefines the type as fully general
  \todo I don't like this push back stuff. Create the whole vector and just set
*/
void parallel_indexstruct::set_processor_structure(int p,shared_ptr<indexstruct> pstruct) {
  if (p<0 || p>processor_structures.size())
    throw(format("Setting pstruct #{} outside structures bound 0-{}",
		      p,processor_structures.size()));
  if (p<processor_structures.size()) {
    processor_structures.at(p) = pstruct;
  } else {
    throw(string("I don' like this pushback stuff"));
    processor_structures.push_back( pstruct );
  }
  uncompute_internal_quantities();
};

//! \todo should we throw an exception if there is a local struct?
index_int parallel_indexstruct::first_index( int p ) const {
  return get_processor_structure(p)->first_index();
};
//! \todo should we throw an exception if there is a local struct?
index_int parallel_indexstruct::last_index( int p ) const {
  return get_processor_structure(p)->last_index();
};
/*!
  Size of the pth indexstruct in a parallel structure,
  unless there is a local structure.
 */
index_int parallel_indexstruct::local_size( int p ) {
  try {
    shared_ptr<indexstruct> localstruct;
    if (local_structure!=nullptr)
      localstruct = local_structure;
    else
      localstruct = get_processor_structure(p);
    if (localstruct==nullptr)
      throw(string("null local struct"));
    if (!localstruct->is_known())
      throw(format("Should not ask local size of unknown struct"));
    else 
      return localstruct->local_size();
  } catch (string c) {
    throw(format("Trouble parallel indexstruct local size: {}",c));
  }
};

//! The very first index in this parallel indexstruct. This assumes the processors are ordered.
index_int parallel_indexstruct::global_first_index() const {
  return first_index(0);
};

//! The very last index in this parallel indexstruct. This assumes the processors are ordered.
index_int parallel_indexstruct::global_last_index() const {
  index_int g;
  try { g = last_index(pidx_domains_volume()-1);
  } catch (string c) {
    throw(format("Error <<{}>> in global_last_index",c));
  };
  return g;
};

/*!
  Return a single indexstruct that summarizes the structure in one dimension.
  \todo we should really test for gaps
*/
shared_ptr<indexstruct> parallel_indexstruct::get_enclosing_structure() const {
  try {
    index_int f = global_first_index(), l = global_last_index();
    //print("enclosing structure {}-{}\n",f,l);
    return shared_ptr<indexstruct>( new contiguous_indexstruct(f,l) );
  } catch (string c) {
    throw(format("Error <<{}>> getting pidx enclosing",c));
  }
};

//! From very first to very last index.
index_int parallel_indexstruct::outer_size() const {
  return global_last_index()-global_first_index()+1;
};

/*!
  Return if a processor contains a certain index. This routine throws an
  exception if the index is globally invalid.
 */
int parallel_indexstruct::contains_element(int p,index_int i) const {
  // this catches out of bound indices
  if (!is_valid_index(i)) {
    throw(string("Index globally out of bounds"));
  }
  return get_processor_structure(p)->contains_element( i );
};

/*!
  Return the number of any processor that contains the requested index, 
  or throw an exception if not found. Searching starts with processor p0.
*/
int parallel_indexstruct::find_index(index_int ind,int p0) {
  int P = pidx_domains_volume();
  for (int pp=0; pp<P; pp++) {
    int p = (p0+pp)%P;
    if (contains_element(p,ind))
      return p;
  }
  throw(string("Index not found on any process"));
};

/*!
  Same as parallel_indexstruct::find_index but start searching with the first processor
*/
int parallel_indexstruct::find_index(index_int ind) {
  return find_index(ind,0);
};

/*!
  Try to detect the type of a parallel structure.
  Some of the components can be nullptr, in which case we return false.
 */
bool parallel_indexstruct::can_detect_type(distribution_type t) const {
  int P = pidx_domains_volume();
  if (t==distribution_type::CONTIGUOUS) {
    index_int prev_last = LONG_MAX;
    bool first{true};
    for ( auto struc : processor_structures ) {
      if (!struc->is_contiguous()) return false;
      if (!first && struc->first_index()!=prev_last+1) return false;
      prev_last = struc->last_index();
      first = false;
    }
    return true;
  } else if (t==distribution_type::BLOCKED) {
    for ( auto struc : processor_structures ) {
      if (!struc->is_contiguous()) return false;
    }
    return true;
  } else return false;
};

//! \todo rewrite this for multi_structures
bool parallel_structure::can_detect_type(distribution_type t) const {
  if (is_orthogonal) {
    for (auto s : dimension_structures() )
      if (s==nullptr || !s->can_detect_type(t)) return false;
    return true;
  } else return false;
};

/*!
  Detect contiguous and blocked (locally contiguous) distribution types
  and set the type parameter accordingly.
  Otherwise we keep whatever is there.
*/
distribution_type parallel_indexstruct::infer_distribution_type() const {
  auto t = get_type();
  if (has_type_replicated()) return t;
  if (t!=distribution_type::CONTIGUOUS && can_detect_type(distribution_type::CONTIGUOUS)) {
    t = distribution_type::CONTIGUOUS;
  } else if (t!=distribution_type::BLOCKED && can_detect_type(distribution_type::BLOCKED)) {
    t = distribution_type::BLOCKED;
  } else
    t = distribution_type::GENERAL;
  return t;
}

//! \todo rewrite this for multi_structures
distribution_type parallel_structure::infer_distribution_type() const {
  if (is_orthogonal) {
    auto t = distribution_type::UNDEFINED;
    for ( auto s : dimension_structures() )
      t = max_type(t,s->infer_distribution_type());
    if (t==distribution_type::UNDEFINED)
      throw(format("Inferring undefined for orthogonal parallel structure {}",
		   as_string()));
    return t;
  } else
    return distribution_type::GENERAL;
};

//! Compute the more general of two types. Weed out the undefined case.
distribution_type max_type(distribution_type t1,distribution_type t2) {
  if (t1==distribution_type::UNDEFINED)
    return t2;
  else if (t2==distribution_type::UNDEFINED)
    return t1;
  else if ((int)t1<(int)t2)
    return t1;
  else return t2;
};

/*!
  Create a new parallel index structure by operating on this one.
  We do this by operating on each processor structure independently.

  \todo can we be more intelligently about how to set the type of the new structure?
  \todo write "operate_by" for indexstruct, then use copy constructor & op_by for this
 */
shared_ptr<parallel_indexstruct> parallel_indexstruct::operate( const ioperator &op) const {
  auto newstruct = shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = get_processor_structure(p),
      out = in->operate(op)->force_simplify();
    newstruct->set_processor_structure(p,out);
  }
  newstruct->set_type( newstruct->infer_distribution_type() );
  return newstruct;
};

shared_ptr<parallel_indexstruct> parallel_indexstruct::operate( const ioperator &&op) const {
  auto newstruct = shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = get_processor_structure(p),
      out = in->operate(op)->force_simplify();
    newstruct->set_processor_structure(p,out);
  }
  newstruct->set_type( newstruct->infer_distribution_type() );
  return newstruct;
};

shared_ptr<parallel_indexstruct> parallel_indexstruct::operate
    ( const ioperator &op,shared_ptr<indexstruct> trunc ) const {
  auto newstruct = shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = processor_structures.at(p),
      out = in->operate(op,trunc)->force_simplify();
    newstruct->set_processor_structure(p,out);    
  }
  newstruct->set_type(newstruct->infer_distribution_type());
  return newstruct;
};

//! \todo this should become the default one.
// shared_ptr<parallel_indexstruct> parallel_indexstruct::operate
//     ( shared_ptr<sigma_operator> op ) {
//   return operate(op.get());
// };

//! \todo we can really lose that star
//! \todo make domain_volume const, and then make this routine const
shared_ptr<parallel_indexstruct> parallel_indexstruct::operate
    ( const sigma_operator &op ) const {
  auto newstruct = shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = get_processor_structure(p),
      out = in->operate(op)->force_simplify();
    newstruct->set_processor_structure(p,out);
  }
  newstruct->set_type(newstruct->infer_distribution_type());
  return newstruct;
};

//! Operate, but keep the first index of each processor in place.
//! This probably doesn't make sense for shifting, but it's cool for multiplying and such.
shared_ptr<parallel_indexstruct> parallel_indexstruct::operate_base( const ioperator &op ) {
  auto newstruct = shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(string("Can not operate base on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    if (processor_structures.at(p)==nullptr) {
      throw(string("No index struct?")); }
    index_int newfirst = op.operate( processor_structures.at(p)->first_index() );
    ioperator back("shift",-newfirst), forth("shift",newfirst);
    auto
      t1struct = processor_structures.at(p)->operate( back ),
      t2struct = t1struct->operate(op)->force_simplify();
    newstruct->set_processor_structure( p, t2struct->operate( forth ) );
  }
  newstruct->set_type(this->type);
  return newstruct;
}

shared_ptr<parallel_indexstruct> parallel_indexstruct::operate_base( const ioperator &&op ) {
  auto newstruct = shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(string("Can not operate base on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    if (processor_structures.at(p)==nullptr) {
      throw(string("No index struct?")); }
    index_int newfirst = op.operate( processor_structures.at(p)->first_index() );
    ioperator back("shift",-newfirst), forth("shift",newfirst);
    auto
      t1struct = processor_structures.at(p)->operate( back ),
      t2struct = t1struct->operate(op)->force_simplify();
    newstruct->set_processor_structure( p, t2struct->operate( forth ) );
  }
  newstruct->set_type(this->type);
  return newstruct;
}

shared_ptr<parallel_indexstruct> parallel_indexstruct::struct_union
( shared_ptr<parallel_indexstruct> merge ) {
  auto newstruct =  shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  for (int p=0; p<pidx_domains_volume(); p++) {
    newstruct->processor_structures.at(p)
      = shared_ptr<indexstruct>
      ( this->processor_structures.at(p)->struct_union
	(merge->processor_structures.at(p).get()) );
  }
  return newstruct;  
}

void parallel_indexstruct::extend_pstruct(int p,indexstruct *i) {
  shared_ptr<indexstruct>
    bef = processor_structures.at(p),
    ext = shared_ptr<indexstruct>( bef->struct_union(i) );
  //print("Extended <<{}>> to <<{}>>\n",bef->as_string(),ext->as_string());
  set_processor_structure(p,ext);
};

/****************
 **** Parallel structure
 ****************/

/*!
  Parallel structure & distribution & object all derive from decomposition.
  Let's have a safe way of retrieving the decomposition.
*/
const decomposition &parallel_structure::get_decomposition() const {
  auto d = dynamic_cast< const decomposition* >(this);
  if (d==nullptr)
    throw(format("Could not upcast to decomposition: {}",as_string()));
  return *d;
};

vector<index_int> parallel_structure::partitioning_points() const {
  int dim = get_dimensionality();
  if (dim!=1) { print("d={}\n",dim);
    throw(format("partitioning points only defined in dim=1")); }
  if (!has_type_contiguous()) { print("t={}\n",type_as_string()); 
    throw(format("partitioning points only defined for block structures")); }
  const auto &dim1 = get_dimension_structure(0);
  auto nprocs = dim1->size();
  vector<index_int> points(nprocs+1);
  for (int ip=0; ip<nprocs; ip++)
    points.at(ip) = dim1->first_index(ip);
  points.at(nprocs) = dim1->last_index(nprocs-1)+1;
  return points;
};

parallel_structure parallel_structure::operate( const ioperator &op ) {
  auto op_copy = op;
  return operate(move(op_copy));
};

parallel_structure parallel_structure::operate( const ioperator &&op ) {
  parallel_structure rstruct(this->get_decomposition());

  auto orth = get_is_orthogonal();
  rstruct.set_is_orthogonal(orth);

  if (orth) {
    for (int is=0; is<get_dimensionality(); is++) {
      auto base_structure = get_dimension_structure(is);
      auto operated_structure = base_structure->operate(op);
      rstruct.set_dimension_structure(is,operated_structure);
    }
    rstruct.set_is_converted(false);
  } else {
    rstruct.set_is_converted(false);
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      auto base_structure = get_processor_structure(dcoord);
      auto operated_structure = base_structure->operate(op);
      auto simplified_structure = operated_structure->force_simplify();
      rstruct.set_processor_structure(dcoord,simplified_structure);
    }
  }
  
  if (op.is_shift_op())
    rstruct.set_structure_type( get_type() );
  else
    rstruct.set_structure_type( rstruct.infer_distribution_type() );

  return rstruct;
};

//! \todo unify with the non-truncating version
parallel_structure parallel_structure::operate
    ( const ioperator &op,shared_ptr<multi_indexstruct> trunc ) const {
  return operate(op,*(trunc.get()));
};
parallel_structure parallel_structure::operate
    ( const ioperator &op,const multi_indexstruct &trunc ) const {
  parallel_structure rstruct(this->get_decomposition());

  if (get_is_orthogonal()) {
    for (int id=0; id<get_dimensionality(); id++) {
      rstruct.set_dimension_structure
	(id,get_dimension_structure(id)->operate(op,trunc.get_component(id)));
    }
    rstruct.set_is_orthogonal();
    rstruct.set_is_converted(false);
  } else {
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      rstruct.set_processor_structure
	(dcoord,get_processor_structure(dcoord)->operate(op,trunc)->force_simplify());
    }
    rstruct.set_is_orthogonal(false);
    rstruct.set_is_converted(false);
  }
  
  if (op.is_shift_op())
    rstruct.set_structure_type( get_type() );
  else
    rstruct.set_structure_type( rstruct.infer_distribution_type() );

  return rstruct;
};

/*!
  Apply a multi operator to a multi structure by applying the components
  of the operator to the components of the structure.
  \todo can we collapse this with the single ioperator case?
*/
parallel_structure parallel_structure::operate( multi_ioperator *op ) {
  int dim = get_same_dimensionality(op->get_dimensionality());
  parallel_structure rstruct(this->get_decomposition());

  if (get_is_orthogonal()) {
    for (int is=0; is<get_dimensionality(); is++) {
      rstruct.set_dimension_structure
	(is,get_dimension_structure(is)->operate(op->get_operator(is)));
    }
    rstruct.set_is_orthogonal();
  } else {
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      rstruct.set_processor_structure
	(dcoord,
	 get_processor_structure(dcoord)->operate(op)->force_simplify());
    }
    rstruct.set_is_orthogonal(false);
    rstruct.set_is_converted(false);
  }
  
  if (op->is_shift_op())
    rstruct.set_structure_type( get_type() );
  else
    rstruct.set_structure_type( infer_distribution_type() );

  return rstruct;
};

//! \todo write unit test
parallel_structure parallel_structure::operate(const multi_sigma_operator &op) {
  int dim = get_same_dimensionality(op.get_dimensionality());
  parallel_structure rstruct(this->get_decomposition());

  if (0) {
  } else if (get_is_orthogonal()) {
    for (int is=0; is<dim; is++) {
      auto oldstruct = get_dimension_structure(is);
      auto newstruct = oldstruct->operate(op.get_operator(is));
      //print("Op {} -> {}\n",oldstruct->as_string(),newstruct->as_string());
      rstruct.set_dimension_structure(is,newstruct);
    }
    rstruct.set_is_orthogonal();
  } else {
    if (get_is_orthogonal())
      print("Warning: parallel_structure::operate on orthogonal\n");
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      rstruct.set_processor_structure
	(dcoord,
	 get_processor_structure(dcoord)->operate(op)->force_simplify());
    }
    rstruct.set_is_orthogonal(false);
    rstruct.set_is_converted(false);
  }
  
  if (op.is_shift_op())
    rstruct.set_structure_type( get_type() );
  else
    rstruct.set_structure_type( infer_distribution_type() );

  return rstruct;
};

//! Apply \ref parallel_indexstruct::operate_base to each dimension
parallel_structure parallel_structure::operate_base( const ioperator &op ) {
  parallel_structure rstruct(this->get_decomposition());

  if (get_is_orthogonal()) {
    for (int is=0; is<get_dimensionality(); is++) {
      rstruct.set_dimension_structure
	(is,get_dimension_structure(is)->operate_base(op));
    }
    rstruct.set_is_orthogonal();
  } else
    throw(string("Can not operate base unless orthogonal"));

  if (op.is_shift_op())
    rstruct.set_structure_type( get_type() );
  else
    rstruct.set_structure_type( infer_distribution_type() );

  return rstruct;
};

parallel_structure parallel_structure::operate_base( const ioperator &&op ) {
  parallel_structure rstruct(this->get_decomposition());

  if (get_is_orthogonal()) {
    for (int is=0; is<get_dimensionality(); is++) {
      rstruct.set_dimension_structure
	(is,get_dimension_structure(is)->operate_base(op));
    }
    rstruct.set_is_orthogonal();
  } else
    throw(string("Can not operate base unless orthogonal"));

  if (op.is_shift_op())
    rstruct.set_structure_type( get_type() );
  else
    rstruct.set_structure_type( infer_distribution_type() );

  return rstruct;
};

/*! Merge two parallel structures by doing \ref parallel_indexstruct::struct_union
  on the dimension components.
  \todo this actually gives the convex hull of the union. good? bad?
*/
parallel_structure parallel_structure::struct_union( const parallel_structure &merge) {
  int dim = get_dimensionality();
  parallel_structure rstruct(this->get_decomposition());

  for (int id=0; id<dim; id++)
    rstruct.set_dimension_structure
      ( id,get_dimension_structure(id)->struct_union( merge.get_dimension_structure(id) ) );
  return rstruct;
};

bool parallel_indexstruct::is_valid_index(index_int i) const {
  return (i>=first_index(0)) && (i<=last_index(pidx_domains_volume()-1));
};

bool parallel_structure::is_valid_index( const domain_coordinate &i) {
  try {
    return (i>=global_first_index()) && (i<=global_last_index());
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not test valid_index of {}",i.as_string()));
  }
};

bool parallel_structure::is_valid_index( const domain_coordinate &&i) {
  try {
    return (i>=global_first_index()) && (i<=global_last_index());
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not test valid_index of {}",i.as_string()));
  }
};

string parallel_structure::header_as_string() const {
  return format("#doms={}, type={}, globally known={}",
		     domains_volume(),type_as_string(),is_known_globally());
};

string parallel_indexstruct::as_string() const {
  memory_buffer w;
  for (int p=0; p<pidx_domains_volume(); p++)
    format_to(w.end()," {}:{}",
	    p,get_processor_structure(p)->as_string());
  format_to(w.end()," ]");
  return to_string(w);
};

string parallel_structure::as_string() const {
  try {
    memory_buffer w;
    format_to(w.end(),"{}: [",header_as_string() );
    if (get_is_converted()) {
      format_to(w.end(),"converted, #P={} : ",multi_structures.size());
      for (int is=0; is<multi_structures.size(); is++)
      	format_to(w.end(),"<<p={}: {}>>,",is,multi_structures[is]->as_string());
    } else {
      format_to(w.end(),"unconverted, ");
      for (int d=0; d<get_dimensionality(); d++)
      	format_to(w.end(),"d={}: {}, ",d,get_dimension_structure(d)->as_string());
    }
    format_to(w.end()," ]");
    return to_string(w);
  } catch (string c) { print("Error: {}\n",c);
    throw(format("Could not parallel_structure as_string"));
  }
};

/****
 **** Mask & coordinate
 ****/

//! Create an empty processor_coordinate object of given dimension
//snippet pcoorddim
processor_coordinate::processor_coordinate(int dim) {
  for (int id=0; id<dim; id++)
    coordinates.push_back(-1);
};
//snippet end

//! Create coordinate from linearized number, against decomposition \todo unnecessary?
processor_coordinate::processor_coordinate(int p,const decomposition &dec)
  : processor_coordinate(dec.get_dimensionality()) {
  int dim = dec.get_dimensionality();
  auto pcoord = dec.coordinate_from_linear(p);
  for (int d=0; d<dim; d++)
    set(d,pcoord.coord(d));
};

//! Constructor from explicit vector of sizes
processor_coordinate::processor_coordinate( vector<int> dims )
  : processor_coordinate(dims.size()) {
  for (int id=0; id<dims.size(); id++)
    set(id,dims[id]);
};
processor_coordinate::processor_coordinate( vector<index_int> dims )
  : processor_coordinate(dims.size()) {
  for (int id=0; id<dims.size(); id++)
    set(id,dims[id]);
};

//! Copy constructor
processor_coordinate::processor_coordinate( processor_coordinate *other ) {
  for (int id=0; id<other->coordinates.size(); id++)
    coordinates.push_back( other->coordinates[id] ); };

//! Create from the dimensionality of a decomposition.
processor_coordinate_zero::processor_coordinate_zero( const decomposition &d)
  :  processor_coordinate_zero(d.get_dimensionality()) {};

/*!
  Get the dimensionality by the size of the coordinate vector.
  The dimension zero case corresponds to the default constructor,
  which is used for processor coordinate objects stored in a decomposition object.
*/
int processor_coordinate::get_dimensionality() const {
  int s = coordinates.size();
  if (s<0)
    throw(string("Non-positive processor-coordinate dimensionality"));
  return s;
};

//! Get the dimensionality, and it should be the same as someone else's.
int processor_coordinate::get_same_dimensionality( int d ) const {
  int rd = get_dimensionality();
  if (rd!=d)
    throw(format("Non-conforming dimensionalities {} vs {}",rd,d));
  return rd;
};

void processor_coordinate::set(int d,int v) {
  if (d<0 || d>=get_dimensionality())
    throw(format("Can not set dimension {}",d));
  coordinates.at(d) = v;
};

//! Get one component of the coordinate.
int processor_coordinate::coord(int d) const {
  if (d<0 || d>=coordinates.size() )
    throw(format("dimension {} out of range for coordinate <<{}>>",d,as_string()));
  return coordinates.at(d);
};

/*! Compute volume by multiplying all coordinates
  \todo is this a bad name? what do we use it for?
*/
int processor_coordinate::volume() const {
  int r = 1;
  for (int id=0; id<coordinates.size(); id++) r *= coordinates[id]; return r;
};

//! Equality test
bool processor_coordinate::operator==( const processor_coordinate &&other ) const {
  int dim = get_same_dimensionality(other.get_dimensionality());
  for (int id=0; id<dim; id++)
    if (coord(id)!=other.coord(id)) return false;
  return true;
};
bool processor_coordinate::operator==( const processor_coordinate &other ) const {
  auto other_copy = other;
  return operator==(std::move(other_copy));
};

bool processor_coordinate::operator!=( const processor_coordinate &&other ) const {
  return !(*this==other);
};
bool processor_coordinate::operator!=( const processor_coordinate &other ) const {
  return ! (*this==other);
};

//! Equality to zero
bool processor_coordinate::is_zero() { auto z = 1;
  for ( auto c : coordinates ) z = z && c==0;
  return z;
};

bool processor_coordinate::operator>( const processor_coordinate other ) const {
  int dim = get_same_dimensionality( other.get_dimensionality());
  for (int id=0; id<dim; id++)
    if (coord(id)<=other.coord(id)) return false;
  return true;
};

bool processor_coordinate::operator>( index_int other) const {
  int dim = get_dimensionality();
  for (int id=0; id<dim; id++)
    if (coord(id)<=other) return false;
  return true;
};

//! Operate plus with second coordinate an integer
processor_coordinate processor_coordinate::operator+( index_int iplus) const {
  int dim = get_dimensionality();
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)+iplus);
  return pls;
};

//! Operate plus with second coordinate a coordinate
processor_coordinate processor_coordinate::operator+( const processor_coordinate &cplus) const {
  int dim = get_same_dimensionality(cplus.get_dimensionality());
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)+cplus.coord(id));
  return pls;
};

//! Operate minus with second coordinate an integer
processor_coordinate processor_coordinate::operator-(index_int iminus) const {
  int dim = get_dimensionality();
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)-iminus);
  return pls;
};

//! Operate minus with second coordinate a coordinate
processor_coordinate processor_coordinate::operator-( const processor_coordinate &cminus) const {
  int dim = get_same_dimensionality(cminus.get_dimensionality());
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)-cminus.coord(id));
  return pls;
};

//! Module each component wrt a layout vector \todo needs unittest
processor_coordinate processor_coordinate::operator%( const processor_coordinate modvec) const {
  int dim = get_same_dimensionality(modvec.get_dimensionality());
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id, coord(id)%modvec.coord(id));
  return pls;
};

//! Rotate a processor coordinate in a grid \todo needs unittest
processor_coordinate processor_coordinate::rotate
    ( vector<int> v, const processor_coordinate &m) const {
  int dim = get_same_dimensionality(v.size());
  auto pv = processor_coordinate(v);
  return ( (*this)+pv )%m;
};

/*
 * operations
 */
domain_coordinate processor_coordinate::operate( const ioperator &op ) {
  int dim = get_dimensionality();
  domain_coordinate opped(dim); // = new processor_coordinate(dim);
  for (int id=0; id<dim; id++)
    opped.set(id, op.operate(coord(id)) );
  return opped;
};

domain_coordinate processor_coordinate::operate( const ioperator &&op ) {
  int dim = get_dimensionality();
  domain_coordinate opped(dim); // = new processor_coordinate(dim);
  for (int id=0; id<dim; id++)
    opped.set(id, op.operate(coord(id)) );
  return opped;
};

//! Unary minus
processor_coordinate processor_coordinate::negate() {
  int dim = get_dimensionality();
  processor_coordinate n(dim);
  for (int id=0; id<dim; id++)
    n.set(id,-coord(id));
  return n;
};

/*!
  Get a process linear number wrt a surrounding cube.
*/
int processor_coordinate::linearize( const processor_coordinate &layout ) const {
  int dim = get_same_dimensionality(layout.get_dimensionality());
  int s = coord(0);
  for (int id=1; id<dim; id++) {
    auto layout_dim = layout[id];
    s = s*layout_dim + coord(id);
  }
  return s;
};

/*!
  Get a process linear number wrt a surrounding cube.
*/
int processor_coordinate::linearize( const decomposition &procstruct ) const {
  return linearize( procstruct.get_domain_layout() );
};

/*! Construct processor coordinate that is identical to self,
  but zero in the indicated dimension.
  The `farcorner' argument is not used, but specified for symmetry 
  with \ref processor_coordinate::right_face_proc.
  \todo this gets called with farpoint, should be origin
*/
processor_coordinate processor_coordinate::left_face_proc
    (int d,processor_coordinate &&farcorner) const {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate left(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) left.set(id,coord(id));
    else left.set(id,0);
  return left;
};

processor_coordinate processor_coordinate::left_face_proc
    (int d,const processor_coordinate &farcorner) const {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate left(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) left.set(id,coord(id));
    else left.set(id,0);
  return left;
};

/*! Construct processor coordinate that is identical to self,
  but maximal in the indicated dimension.
  The maximality is given by the `farcorner' argument.
*/
processor_coordinate processor_coordinate::right_face_proc
    (int d,const processor_coordinate &farcorner) const {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate right(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) right.set(id,coord(id));
    else right.set(id,farcorner.coord(id));
  return right;
};
processor_coordinate processor_coordinate::right_face_proc
    (int d,processor_coordinate &&farcorner) const {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate right(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) right.set(id,coord(id));
    else right.set(id,farcorner.coord(id));
  return right;
};

//! Is this coordinate on any face of the processor brick? \todo farcorner by reference
bool processor_coordinate::is_on_left_face( const decomposition &procstruct ) const {
  const auto origin = procstruct.get_origin_processor();
  for (int id=0; id<get_dimensionality(); id++)
    if (coord(id)==origin[id]) return true;
  return false;
};

//! Is this coordinate on any face of the processor brick? \todo farcorner by reference
bool processor_coordinate::is_on_right_face( const decomposition &procstruct ) const {
  const auto farcorner = procstruct.get_farpoint_processor();
  for (int id=0; id<get_dimensionality(); id++)
    if (coord(id)==farcorner[id]) return true;
  return false;
};

//! Is this coordinate on any face of the processor brick? \todo farcorner by reference
bool processor_coordinate::is_on_face( const decomposition &procstruct ) const {
  return is_on_left_face(procstruct) || is_on_right_face(procstruct);
};
bool processor_coordinate::is_on_face( shared_ptr<object> proc ) const {
  return is_on_face( proc->get_distribution() );
};
bool processor_coordinate::is_on_face( const object &proc ) const {
  return is_on_face( proc.get_distribution() );
};

//! Is this coordinate the origin?
bool processor_coordinate::is_null() const {
  for (int id=0; id<get_dimensionality(); id++)
    if (coord(id)!=0) return false;
  return true;
};

string processor_coordinate::as_string() const {
  memory_buffer w;
  format_to(w.end(),"P[");
  for ( int i=0; i<coordinates.size(); i++ )
    format_to(w.end(),"{},",coordinates.at(i));
  format_to(w.end(),"]");
  return to_string(w);
};

//! Mask constructor. Right now only for 1d and 2d
processor_mask::processor_mask( const decomposition &d )
  : decomposition(d) {
  int dim = get_dimensionality(), np = domains_volume();
  included.reserve(np);
  for (int p=0; p<np; p++)
    included.push_back(Fuzz::NO);
};

//! Create a mask from a list of integers.
processor_mask::processor_mask( const decomposition &d, vector<int> procs )
  : processor_mask(d) {
  for ( auto p : procs )
    included[p] = Fuzz::YES;
};

//! Copy constructor.
processor_mask::processor_mask( processor_mask& other )
  : decomposition(other) {
  throw(string("no processor mask copy constructor"));
}
//   int dim = get_dimensionality(), np = domains_volume(); auto P = d->get_farcorner();
//   //  if (dim==1) { int np  = P->coord(0)+1;
//     include1d = new Fuzz[np];
//     for (int p=0; p<P->coord(0); p++) include1d[p] = other.include1d[p];
//   // } else
//   //   throw(string("Can not copy mask in multi-d"));
// };

// ! create a mask with the first P processors added
processor_mask::processor_mask( const decomposition &d,int P ) : processor_mask(d) {
  for (int p=0; p<P; p++)
    included[p] = Fuzz::YES;
};
//   if (get_dimensionality()!=1)
//     throw(string("Can not add linear procs to mask in multi-d"));
//   for (int p=0; p<P; p++) {
//     processor_coordinate *c = new processor_coordinate(1);
//     c->set(0,p); add(c);
//   };
// };

//! Add a processor to the mask
void processor_mask::add( const processor_coordinate &p) {
  throw("mask adding disabled");
  // int plin = p.linearize(this);
  // included[plin] = Fuzz::YES;
};

//! Render mask as list of integers. This is only used in \ref mpi_distribution::add_mask.
vector<int> processor_mask::get_includes() { vector<int> includes;
  throw(string("get includes is totally wrong"));
  // int dim = get_dimensionality();
  // if (dim==1) {
  //   for (int i=0; i<domains_volume(); i++)
  //     includes.push_back( include1d[i]==Fuzz::YES );
  //   return includes;
  // } else
  //   throw(string("Can not get includes in multi-d"));
};

//! Test alivesness of a process.
int processor_mask::lives_on( const processor_coordinate &p) const {
  int plin = p.linearize(*this);
  return included[plin]==Fuzz::YES;
};

//! Remove a processor from the mask; this only makes sense for the constructor from P.
void processor_mask::remove(int p) {
  included[p] = Fuzz::NO;
};

/****
 **** Parallel structure
 ****/

//! Constructor.
parallel_structure::parallel_structure( const decomposition &d )
  : decomposition(d) {
  allocate_structure();
};

//! Shortcut for one-dimensional structure
parallel_structure::parallel_structure
    ( const decomposition &d,shared_ptr<parallel_indexstruct> pidx )
  : parallel_structure(d) {
  if (d.get_dimensionality()>1)
    throw(string("One dimensional constructor only works in 1D"));
  set_dimension_structure(0,pidx);
};

//! Copy constructor.
//! \todo we shouldn't have to infer the pidx type: should have been set when p was created
//! \todo right now we reuse the pointers. that's dangerous
parallel_structure::parallel_structure(parallel_structure *p)
  : parallel_structure(*p) {
  if (p->has_type(distribution_type::UNDEFINED))
    throw(string("Can not copy undefined parallel structure"));

  int dim = get_dimensionality();

  if (p->is_orthogonal) {
    is_orthogonal = true;
    try {
      for (int id=0; id<dim; id++) {
	auto old_struct = p->get_dimension_structure(id);
	auto dim_struct =
	  shared_ptr<parallel_indexstruct>
	      ( new parallel_indexstruct( old_struct.get() ) );
	set_dimension_structure(id,dim_struct);
      }
    } catch (string c) {
      throw(format("Parallel structure by copying orthogonal: {}",c));
    }
  }
  if (p->is_converted) {
    is_converted = true;
    try {
      for (int is=0; is<domains_volume(); is++) {
	auto pcoord = p->coordinate_from_linear(is);
	auto p_structure = p->get_processor_structure(pcoord);
	set_processor_structure(pcoord,p_structure);
      }
    } catch (string c) {
      throw(format("Parallel structure by copying converted: {}",c));
    }
  }
  is_orthogonal = p->is_orthogonal; is_converted = p->is_converted;
  known_globally = p->known_globally;
  set_structure_type(p->get_type());
  compute_enclosing_structure();
};

/*!
  We create an array of \ref multi_indexstruct, one for each global domain.
  Translation from linear to multi-d goes through \ref decomposition::get_farcorner.
*/
void parallel_structure::allocate_structure() {
  decomposition *d = dynamic_cast<decomposition*>(this);
  if (d==nullptr) throw(string("Could not cast to decomposition"));
  int dim = d->get_dimensionality();

  // set unknown multi structures
  try {
    for (int id=0; id<d->domains_volume(); id++)
      multi_structures.push_back
	( shared_ptr<multi_indexstruct>( new unknown_multi_indexstruct(dim) ) );
  } catch (string c) {
    throw(format("Trouble creating multi structures: {}",c));
  }

  // set empty dimension structures
  try {
    for (int id=0; id<dim; id++)
      dimension_structures_ref().push_back(nullptr);
    for (int id=0; id<dim; id++) {
      auto dimsize = d->get_size_of_dimension(id);
      auto dimstruct =
	shared_ptr<parallel_indexstruct>( new parallel_indexstruct(dimsize) );
      set_dimension_structure( id,dimstruct );
    }
  } catch (string c) {
    throw(format("Trouble creating dimension structures: {}",c));
  }
};

//! Initially creating the dimension structures.
void parallel_structure::push_dimension_structure(shared_ptr<parallel_indexstruct> pidx) {
  dimension_structures_ref().push_back(pidx);
  unset_memoization();
};

/*!
  Multi-d parallel structure can be set;
  this assumes the location in the \ref structure vector is already created.
*/
void parallel_structure::set_dimension_structure
    (int d,shared_ptr<parallel_indexstruct> pidx) {
  if (d<0 || d>=get_dimensionality())
    throw(format("Invalid dimension <<{}>> to set structure",d));
  dimension_structures_ref().at(d) = pidx;
  unset_memoization();
};

//! Get the parallel structure in a specific dimension.
shared_ptr<parallel_indexstruct> parallel_structure::get_dimension_structure(int d) const {
  if (d<0 || d>=get_dimensionality())
    throw
      (format
       ("Invalid dimension <<{}>> to get_dimension_structure in parallel_structure of d={}",
	d,get_dimensionality()));
  auto rstruct = dimension_structures().at(d);
  if (rstruct==nullptr)
    throw(format("Parallel indexstruct in dimension <<{}>> uninitialized",d));
  return rstruct;
};

//! Get the multi_indexstruct of a multi-d processor coordinate
shared_ptr<multi_indexstruct> parallel_structure::get_processor_structure
    ( const processor_coordinate &p ) const {
  //print("Get p={} from {}\n",p.as_string(),as_string());
  if (!get_is_converted()) {
    //throw(format("We really need the conversion in get_processor_structure"));
    try {
      convert_to_multi_structures();
    } catch (string c) { print("Error: {}\n");
      throw(format("Trouble converting to multi: {}",as_string()));
    }      
  }

  try {
    auto d = get_decomposition();
    auto layout = d.get_domain_layout();
    int plinear = p.linearize(layout);
    if (plinear>=multi_structures.size())
      throw(format("Coordinate {} linear {} out of bound {} for <<{}>>",
			p.as_string(),plinear,multi_structures.size(),layout.as_string()));
    auto rstruct = multi_structures.at(plinear);
    if (rstruct==nullptr)
      throw(format("Found null processor structure at {}, linear {}",
			p.as_string(),plinear));
    if (!rstruct->is_known())
      throw(format("Trying to return unknown multi_indexstruct as {} in {}",
			p.as_string(),as_string()));
    return rstruct;
  } catch (string c) { print("Error {}\n",c);
    throw(format("Could not get_proc_structure for {}",p.as_string()));
  }
};

/*!
  Set a processor structure, general multi-d case.
  But if this is one-dimensional we work on the dimension structure
  so that we maintain orthogonality.

  \todo can we be more clever about setting taht structure type?
*/
void parallel_structure::set_processor_structure
    ( processor_coordinate &p,shared_ptr<multi_indexstruct> pstruct) {
  if (!get_is_converted())
    convert_to_multi_structures();
  decomposition *d = dynamic_cast<decomposition*>(this);
  auto layout = d->get_domain_layout();
  int plinear = p.linearize(layout);
  multi_structures.at(plinear) = pstruct;
  set_is_converted(); set_is_orthogonal(false); 

  set_structure_type( infer_distribution_type() );
  unset_memoization();
  return;
};

//! Set a processor structure, shortcut for one-d.
void parallel_structure::set_processor_structure(int p,shared_ptr<indexstruct> pstruct) {
  if (get_dimensionality()>1)
    throw(string("Can not set unqualified pstruct in multi-d pidx"));
  get_dimension_structure(0)->set_processor_structure(p,pstruct);
  multi_structures.at(p) = shared_ptr<multi_indexstruct>( new multi_indexstruct(pstruct) );
  set_structure_type( distribution_type::GENERAL );
  unset_memoization();
};

/*
 * Creation
 */
//! Create from global size
void parallel_structure::create_from_global_size(index_int gsize) {
  if (get_dimensionality()>1)
    throw(format("Trying to create 1D for structure of dim={}",get_dimensionality()));
  get_dimension_structure(0)->create_from_global_size(gsize);
  set_structure_type( distribution_type::CONTIGUOUS );
  set_is_known_globally(); set_is_orthogonal(); set_is_converted(false);
  unset_memoization();
};

//! Create from global size, multi-d
void parallel_structure::create_from_global_size(vector<index_int> gsizes) {
  for (int id=0; id<gsizes.size(); id++)
    get_dimension_structure(id)->create_from_global_size(gsizes[id]);
  set_structure_type( distribution_type::CONTIGUOUS );
  set_is_known_globally();
  unset_memoization();
};

//! Create from indexstruct
void parallel_structure::create_from_indexstruct( shared_ptr<indexstruct> idx) {
  get_dimension_structure(0)->create_from_indexstruct(idx);
  set_structure_type( distribution_type::BLOCKED );
  unset_memoization();
};

//! Create from indexstruct, multi-d
void parallel_structure::create_from_indexstruct(multi_indexstruct &&idx) {
  int dim = get_same_dimensionality( idx.get_dimensionality() );
  for (int id=0; id<dim; id++)
    get_dimension_structure(id)->create_from_indexstruct(idx.get_component(id));
  set_structure_type( distribution_type::BLOCKED );
  set_is_known_globally();
  unset_memoization();
};
void parallel_structure::create_from_indexstruct(multi_indexstruct &idx) {
  auto idx_copy = idx;
  create_from_indexstruct(idx_copy);
};

//! Create from indexstruct, multi-d
void parallel_structure::create_from_indexstruct( shared_ptr<multi_indexstruct> idx) {
  int dim = get_same_dimensionality( idx->get_dimensionality() );
  for (int id=0; id<dim; id++)
    get_dimension_structure(id)->create_from_indexstruct(idx->get_component(id));
  set_structure_type( distribution_type::BLOCKED );
  set_is_known_globally();
  unset_memoization();
};

//! Create from replicated indexstruct
void parallel_structure::create_from_replicated_indexstruct( shared_ptr<indexstruct> idx) {
  get_dimension_structure(0)->create_from_replicated_indexstruct(idx);
  set_structure_type( distribution_type::REPLICATED );
  set_is_known_globally();
  unset_memoization();
}

//! Create from replicated indexstruct, multi-d
void parallel_structure::create_from_replicated_indexstruct(shared_ptr<multi_indexstruct> idx) {
  int dim = get_same_dimensionality( idx->get_dimensionality() );
  for (int id=0; id<dim; id++)
    get_dimension_structure(id)->create_from_replicated_indexstruct
      (idx->get_component(id));
  set_structure_type( distribution_type::REPLICATED );
  set_is_known_globally();
  unset_memoization();
};

void parallel_structure::create_from_uniform_local_size(index_int lsize) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_from_uniform_local_size(lsize);
  set_structure_type( distribution_type::CONTIGUOUS );
  set_is_known_globally();
  unset_memoization();
};

void parallel_structure::create_from_local_sizes( vector<index_int> szs ) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_from_local_sizes(szs);
  set_structure_type( distribution_type::CONTIGUOUS );
  set_is_known_globally();
  unset_memoization();
};

void parallel_structure::create_from_replicated_local_size(index_int lsize) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_from_replicated_local_size(lsize);
  set_structure_type( distribution_type::REPLICATED );
  set_is_known_globally();
  unset_memoization();
}

void parallel_structure::create_cyclic(index_int lsize,index_int gsize) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_cyclic(lsize,gsize);
  set_structure_type( distribution_type::CYCLIC );
  //convert_to_multi_structures();
  set_is_known_globally();
  unset_memoization();
};

void parallel_structure::create_blockcyclic(index_int bs,index_int nb,index_int gsize) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_blockcyclic(bs,nb,gsize);
  set_structure_type( distribution_type::GENERAL);
  //convert_to_multi_structures();
  unset_memoization();
};

void parallel_structure::create_from_explicit_indices(index_int *nidx,index_int **idx) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_from_explicit_indices(nidx,idx);
  set_structure_type( distribution_type::GENERAL);
  //convert_to_multi_structures();
  unset_memoization();
};

void parallel_structure::create_from_function( index_int(*f)(int,index_int),index_int n) {
  get_same_dimensionality(1);
  // from p,i
  get_dimension_structure(0)->create_from_function(f,n);
  set_structure_type( distribution_type::GENERAL);
  //convert_to_multi_structures();
  unset_memoization();
};

//! \todo add explicit bins to the create call; min and max become kernels
void parallel_structure::create_by_binning(object *o) {
  get_same_dimensionality(1);
  double mn = o->get_min_value(),mx = o->get_max_value();
  get_dimension_structure(0)->create_by_binning(o,mn-0.5,mx+0.5,0);
  set_structure_type( distribution_type::GENERAL );
  convert_to_multi_structures();
  unset_memoization();
};

//! Make \ref multi_structures from an orthogonally specified structure.
//! We don't do this if the structure is no longer orthgonal
void parallel_structure::convert_to_multi_structures(bool trace) const {
  if (get_is_converted()) return;
  int dim = get_dimensionality();
  if (trace) print("Converting {}D\n",dim);
  try {
    if (dim==1) {
      auto istruct = get_dimension_structure(0);
      int iprocs = istruct->size();
      if (multi_structures.size()<iprocs)
	throw(format("Insufficient multi_structures {} for dimension size {}",
			  multi_structures.size(),iprocs));
      for (int ip=0; ip<iprocs; ip++) {
	auto pstruct = shared_ptr<multi_indexstruct>
	  ( new multi_indexstruct
	    ( istruct->get_processor_structure(ip) ) );
	if (trace) print("Proc {}: {}\n",ip,pstruct->as_string());
	multi_structures.at(ip) = pstruct;
      }
    } else if (dim==2) {
      auto 
	istruct = get_dimension_structure(0),
	jstruct = get_dimension_structure(1);
      int iprocs = istruct->size(), jprocs = jstruct->size();
      if (multi_structures.size()<iprocs*jprocs)
	throw(format("Insufficient multi_structures {} for dimension size {}",
			  multi_structures.size(),iprocs*jprocs));
      int is=0;
      for (int ip=0; ip<iprocs; ip++)
	for (int jp=0; jp<jprocs; jp++)
	  multi_structures.at(is++) = shared_ptr<multi_indexstruct>
	    ( new multi_indexstruct
	      ( vector<shared_ptr<indexstruct>>{
		istruct->get_processor_structure(ip),jstruct->get_processor_structure(jp)
		  } ) );
    } else if (dim==3) {
      auto
	istruct = get_dimension_structure(0),
	jstruct = get_dimension_structure(1),
	kstruct = get_dimension_structure(2);
      int iprocs = istruct->size(), jprocs = jstruct->size(), kprocs = kstruct->size();
      if (multi_structures.size()<iprocs*jprocs*kprocs)
	throw(format("Insufficient multi_structures {} for dimension size {}",
			  multi_structures.size(),iprocs*jprocs*kprocs));
      int is=0;
      for (int ip=0; ip<iprocs; ip++)
	for (int jp=0; jp<jprocs; jp++)
	  for (int kp=0; kp<kprocs; kp++)
	    multi_structures.at(is++) = shared_ptr<multi_indexstruct>
	      ( new multi_indexstruct
		( vector<shared_ptr<indexstruct>>{
		  istruct->get_processor_structure(ip),
		    jstruct->get_processor_structure(jp),
		    kstruct->get_processor_structure(kp)
		    } ) );
    } else
      throw(string("Can not convert to multi_structures in dim>3"));
  } catch (string c) { print("Error <<{}>> converting\n",c);
    throw(string("Could not convert to multi_structures"));
  } catch (...) { print("Unknown error converting\n");
    throw(string("Could not convert to multi_structures"));
  }
  set_is_converted(); is_orthogonal = true;
  unset_memoization();
};

//! The vector of local sizes of a processor
const domain_coordinate &parallel_structure::local_size_r(const processor_coordinate &p) {
  try {
    return get_processor_structure(p)->local_size_r();
  } catch (string c) {
    throw(format("Trouble parallel structure local size: {}",c));
  }
};

//! Global size as a \ref processor_coordinate
const domain_coordinate &parallel_structure::global_size() {
  try {
    return get_enclosing_structure()->local_size_r();
  } catch (string c) {
    throw(format("Trouble parallel structure global size: {}",c));
  }
};

/*!
  Local number of points in the structure
*/
index_int parallel_structure::volume( const processor_coordinate &&p ) {
  try {
    auto struc = get_processor_structure(p);
    index_int
      vol = struc->volume();
    if (vol<0)
      throw(format("Negative volume for p={}",p.as_string()));
    return vol;
  } catch ( string c ) {
    throw(format("Problem parallel structure volume: {}",c));
  }
};

index_int parallel_structure::volume( const processor_coordinate &p ) {
  try {
    auto struc = get_processor_structure(p);
    index_int
      vol = struc->volume();
    if (vol<0)
      throw(format("Negative volume for p={}",p.as_string()));
    return vol;
  } catch ( string c ) {
    throw(format("Problem parallel structure volume: {}",c));
  }
};

//! Total number of points in the structure
index_int parallel_structure::global_volume() {
  try {
    const auto &enclosing = get_enclosing_structure();
    return enclosing->volume();
  } catch (string c) { print("Error: {}\n",c);
    throw(format("Could not compute global volume of {}",as_string()));
  }
};

/****
 **** Distribution
 ****/

//snippet distributiondef
//! The default constructor does not set the parallel_indexstruct objects:
//! that's done in the derived distributions.
distribution::distribution( const decomposition &d )
  : parallel_structure(d) {
  //print("Base distribution constructor\n");
  init_flag = true; set_name("some-distribution"); set_cookie(entity_cookie::DISTRIBUTION);
  set_dist_factory(); setup_memoization();
};
//snippet end

void distribution::setup_memoization() {
  int np;
  try { np = domains_volume();
  } catch (string c) {
    throw(format("decomposition volume problem: {}",c));
  }
  linear_sizes = vector<int>(np);
  linear_starts = vector<int>(np);
  linear_offsets = vector<int>(np);
};

//! Constructor from parallel structure
distribution::distribution( const parallel_structure &struc )
  : parallel_structure(struc) {
  if (!struc.has_content()) { print("Distribution from empty struct\n"); throw(1); };
  init_flag = true; set_name(format("distribution-from-{}",struc.get_name()));
  set_dist_factory(); setup_memoization();
};
distribution::distribution( const parallel_structure &&struc )
  : parallel_structure(struc) {
  if (!struc.has_content()) { print("Distribution from empty struct\n"); throw(1); };
  init_flag = true; set_name(format("distribution-from-{}",struc.get_name()));
  set_dist_factory(); setup_memoization();
};

//! Constructor from explicitly specified parallel_indexstruct.
distribution::distribution( const decomposition &d,shared_ptr<parallel_indexstruct> struc)
  : distribution(d) {
  set_dimension_structure(0,struc);
};

//! Copy constructor \todo why doesn't copying the numa structure work?
distribution::distribution( shared_ptr<distribution> d )
  : parallel_structure(d->get_decomposition()) {
  // done in decomposition copy: copy_communicator(dynamic_cast<communicator*>(d));
  //print("called distribution copy from pointer\n");
  init_flag = true; add_mask( d->mask );
  has_linear_data = d->has_linear_data;
  linear_sizes = d->linear_sizes;
  linear_starts = d->linear_starts;
  linear_offsets = d->linear_offsets;
  //
  set_structure_type(d->get_type()); set_name(d->get_name());
  set_orthogonal_dimension( d->get_orthogonal_dimension() );
  copy_dist_factory(d);
  copy_communicator(d->get_communicator());
  compute_global_first_index = d->compute_global_first_index; // in copy_operate_routines?
  compute_global_last_index = d->compute_global_last_index;
};

/*
 * Block distributions
 */

//! Constructor from multiple explicit parameters
block_distribution::block_distribution
    ( const decomposition &d,int ortho,index_int lsize,index_int gsize)
  : distribution(d) {
  if (gsize>0) {
    // from global size
    if (lsize<0) {
      // gsize>0, lsize<0 means only global given
      try {
	create_from_global_size(gsize);
      } catch (string c) {
	throw(format("Block distribution from g={} failed: {}",gsize,c));
      } catch (...) {
	throw(format("Block distribution from g={} failed",gsize));
      }
    } else {
      // gsize>0, lsize>=0 means each process has a local size
      create_from_local_sizes( *(overgather(lsize,get_over_factor())) );
    }
  } else {
    // gsize<0 means uniform local size
    if (lsize<0)
      throw(format("gsize<0 requires lsize>=0"));
    //create_from_uniform_local_size(lsize);
    int nprocs = domains_volume();
    vector<index_int> lsizes(nprocs);
    gather64(lsize,lsizes);
    //cerr << "create from locals:"; for (auto s : lsizes) cerr << " " << s; cerr << "\n";
    create_from_local_sizes(lsizes);
  }
  set_orthogonal_dimension(ortho);
};

//! Constructor for multi-d: we supply a vector of global sizes
block_distribution::block_distribution(  const decomposition &d,domain_coordinate &&sizes )
  : distribution(d) {
  int dim;
  try {
     dim = d.get_same_dimensionality(sizes.get_dimensionality());
  } catch (string c) {
    print("Decomposition dim {}, vs endpoint {}\n",
	       d.get_dimensionality(),sizes.get_dimensionality());
    throw(format("block distro failed <<{}>>",c));
  }
  create_from_indexstruct( multi_indexstruct(domain_coordinate_zero(dim),sizes-1) );
};
block_distribution::block_distribution(  const decomposition &d,domain_coordinate &sizes )
  : distribution(d) {
  int dim;
  try {
     dim = d.get_same_dimensionality(sizes.get_dimensionality());
  } catch (string c) {
    print("Decomposition dim {}, vs endpoint {}\n",
	       d.get_dimensionality(),sizes.get_dimensionality());
    throw(format("block distro failed <<{}>>",c));
  }
  create_from_indexstruct( multi_indexstruct(domain_coordinate_zero(dim),sizes-1) );
};

//! Constructor for one-d: we supply a vector of local sizes
block_distribution::block_distribution(  const decomposition &d,const vector<index_int> lsizes )
  : distribution(d) {
  if (d.get_dimensionality()!=1)
    throw(format("Can only create block dist from local sizes in d=1, not d={}",
		 d.get_dimensionality()));
  create_from_local_sizes(lsizes);
};

void distribution::create_from_unique_local( shared_ptr<multi_indexstruct> strct) {
  index_int lsize;
  try {
    get_same_dimensionality(1);
    int P = domains_volume(); lsize = strct->local_size(0);
    vector<index_int> sizes(P); gather64(lsize,sizes);
    create_from_local_sizes(sizes);
  } catch (string c) {
    throw(format("Error creating from unique local size {}: {}",lsize,c));
  } catch (...) {
    throw(format("Unknown error creating from unique local size {}",lsize)); }    
};

//! Test whether a coordinate lives on a processor
bool distribution::contains_element
    (const processor_coordinate &p,const domain_coordinate &&i) {
  auto pstruct = get_processor_structure(p);
  return pstruct->contains_element(i);
};

//! Test whether a coordinate lives on a processor
bool distribution::contains_element
    ( const processor_coordinate &p,const domain_coordinate &i) {
  auto pstruct = get_processor_structure(p);
  return pstruct->contains_element(i);
};

//! Local allocation of a distribution is local size of the struct times orthogonal dimension
index_int distribution::local_allocation_p( processor_coordinate &p ) {
  auto v = volume(p); //print("[{}] has volume {}\n",p.as_string(),v);
  return get_orthogonal_dimension()*v;
};

void parallel_structure::compute_enclosing_structure() {
  if (is_memoized())
    return;
  int dim = get_dimensionality();
  try {
    auto first = global_first_index(), last = global_last_index();
    set_enclosing_structure(contiguous_multi_indexstruct(first,last));
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not compute enclosing structure for <<{}>>",as_string()));
  }
};

/*!
  Get the multi-dimensional enclosure of the parallel structure.
  This is memo-ized. \todo add a test to make sure the structure is set
*/
// shared_ptr<multi_indexstruct> parallel_structure::get_enclosing_structure() {
//   try {
//     if (!is_memoized())
//       memoize_structure();
//   } catch( string c) { print("Error: {}\n",c);
//     throw(format("Could not get memoize for enclosing structure of {}",as_string()));
//   }
//   if (enclosing_structure==nullptr)
//     throw(format("Memoized enclosing structure is null"));
//   return enclosing_structure;
// };
shared_ptr<multi_indexstruct> parallel_structure::get_enclosing_structure() {
  return enclosing_structure;
};

/*!
  We do something funky with this in the product case.
 */
// void parallel_structure::set_enclosing_structure(shared_ptr<multi_indexstruct> struc) {
//   enclosing_structure = struc;
// };
void parallel_structure::set_enclosing_structure(const multi_indexstruct &struc) {
  enclosing_structure = make_shared<multi_indexstruct>(struc);
};
void parallel_structure::set_enclosing_structure(const multi_indexstruct &&struc) {
  enclosing_structure = make_shared<multi_indexstruct>(struc);
  //  enclosing_structure = struc;
};

/*!
  Compute the starts and sizes of the processor blocks.
  See also \ref distribution::compute_linear_offsets.
*/
void distribution::compute_linear_sizes() {
  if (has_linear_data) return;
  if (the_communicator_mode==communicator_mode::OMP)
    throw(string("Can not yet compute linear sizes for OMP"));
  index_int myfirst;
  try {
    myfirst = linearize(first_index_r(proc_coord(/* *this */)));
  } catch (string c) {
    print("Error: {}\n",c);
    throw(string("Could not linearize first index")); }
  try {
    gather32(myfirst,linear_starts);
    gather32(volume(proc_coord(/* *this */)),linear_sizes);
  } catch (string c) {
    print("Error: {}\n",c);
    throw(string("Could not gather linear")); }
  has_linear_data = true;
};

vector<int> &distribution::get_linear_sizes() {
  compute_linear_sizes();
  return linear_sizes;
};

/*!
  Compute the offsets of the processor blocks.
  The call to \ref distribution::get_linear_sizes
  ensures that the starts and linear sizes have been computed.
*/
void distribution::compute_linear_offsets() {
  compute_linear_sizes();
  //linear_offsets = new vector<int>;
  int scan = 0;
  int
    remember_start = linear_starts.at(0)-1,
    remember_size = linear_sizes.at(0)-1;
  int nprocs = linear_sizes.size();
  for (int iproc=0; iproc<nprocs; iproc++) {
    int
      start = linear_starts.at(iproc),
      size = linear_sizes.at(iproc);
    linear_offsets.at(iproc) = scan; //->push_back(scan);
    if (start==remember_start && size==remember_size)
      processor_skip.push_back(true);
    else {
      processor_skip.push_back(false);
      scan += size;
    }
    remember_start = start; remember_size = size;
  }
};

vector<int> &distribution::get_linear_offsets() {
  compute_linear_offsets();
  return linear_offsets;
};

//! Give the linear location of a domain coordinate
index_int parallel_structure::linearize( const domain_coordinate &coord ) {
  try {
    return coord.linear_location_in( global_first_index(),global_last_index() );
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not linearize {}",coord.as_string()));
  }
};

/*!
  Most of the factory routines are completely mode-dependent, so there
  are no initial values.
  Here we only set the initial message factory.
  (The \ref mpi_message object contains some MPI-specific stuff.)
*/
void distribution::set_dist_factory() {
  new_message = 
    [this] (const processor_coordinate &snd,const processor_coordinate &rcv,
	    shared_ptr<multi_indexstruct> g) -> shared_ptr<message> {
    throw(string("No default new_message")); };
  new_embed_message = 
    [this] (const processor_coordinate &snd,const processor_coordinate &rcv,
	    shared_ptr<multi_indexstruct> e,
	    shared_ptr<multi_indexstruct> g) -> shared_ptr<message> {
    throw(string("No default new_embed_message")); };
  location_of_first_index =
    [] ( shared_ptr<distribution> d,const processor_coordinate &p) -> index_int {
        throw(string("imp_base.h local_of_first_index")); };
  location_of_last_index =
    [] ( shared_ptr<distribution> d,const processor_coordinate &p) -> index_int {
        throw(string("imp_base.h local_of_last_index")); };
};

shared_ptr<distribution> distribution::operate( const ioperator &op) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(string("Could not upcast to par structure"));
  auto operated_structure = base_structure->operate(op);
  auto operated = new_distribution_from_structure(operated_structure);
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( format("{}-operated(pi)",get_name()) );
  return operated;
};

shared_ptr<distribution> distribution::operate( const ioperator &&op) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(string("Could not upcast to par structure"));
  auto operated_structure = base_structure->operate(op);
  auto operated = new_distribution_from_structure(operated_structure);
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( format("{}-operated(pi)",get_name()) );
  return operated;
};

shared_ptr<distribution> distribution::operate( multi_ioperator *op ) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(string("Could not upcast to par structure"));
  auto operated_structure = base_structure->operate(op);
  auto operated = new_distribution_from_structure(operated_structure);
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( format("{}-operated(mi)",get_name()) );
  return operated;
};

//! Operate on distribution by lambda generation on operated structure
shared_ptr<distribution> distribution::operate( const multi_sigma_operator &op) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(string("Could not upcast to par structure"));
  auto operated_structure = base_structure->operate(op);
  auto operated = new_distribution_from_structure(operated_structure);
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( format("{}-operated(ms)",get_name()) );
  return operated;
};

/*!
  Operate on a distribution.
  - global operator is applied as such
  - every other type loops over processors, so this requires 
    a globally known distribution
 */
shared_ptr<distribution> distribution::operate(distribution_sigma_operator &op) {
  auto op_copy = op;
  return operate(std::move(op_copy));
};

shared_ptr<distribution> distribution::operate(distribution_sigma_operator &&op) {
  auto decomp = this->get_decomposition();

  if (op.is_global_based()) {
    //print("dist sigma global\n");
    try {
      return op.operate(this->shared_from_this());
    } catch (string c) { print("Error in global dist_sigma operate: {}\n",c);
      throw(format("Could not global operate on {}",as_string()));
    }
  } else {
    if (!is_known_globally())
      throw(format("Can not operate non-global dist_sigma_op if not globally known: {}",
			as_string()));
    try {
      parallel_structure structure(decomp);
      for ( auto me : decomp ) {
	shared_ptr<multi_indexstruct> new_pstruct;
	try {
	  new_pstruct = op.operate(this->shared_from_this(),me);
	} catch (string c) {
	  throw(format("Operate dist_sig_op failed: {}",c)); };
	try {
	  structure.set_processor_structure(me,new_pstruct);
	} catch( string c) {
	  throw(format("Trouble setting struct {} for {}\n",
			    new_pstruct->as_string(),me.as_string()));
	}
      }
      structure.set_structure_type( structure.infer_distribution_type() );
      structure.set_is_known_globally();

      try {
	auto operated_dist = new_distribution_from_structure(structure);
	return operated_dist;
      } catch (string c) {
	throw(format("Could not do distribution factory: {}",c));
      }
    } catch (string c) {
      throw(format("Error in non-global dist_sig_op: {}",c));
    }
  }
};

//! Operate on the base. This mostly differs for multiplication operations.
shared_ptr<distribution> distribution::operate_base( const ioperator &op ) {
  decomposition *base_decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr) throw(string("Could not upcast to par structure"));
  if (!base_structure->has_type( this->get_type() )) throw(string("Type got lost"));
  auto operated_structure = base_structure->operate_base(op);
  auto operated_distro = new_distribution_from_structure(operated_structure);
  operated_distro->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  return operated_distro;
};

shared_ptr<distribution> distribution::operate_base( const ioperator &&op ) {
  decomposition *base_decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr) throw(string("Could not upcast to par structure"));
  if (!base_structure->has_type( this->get_type() )) throw(string("Type got lost"));
  auto operated_structure = base_structure->operate_base(op);
  auto operated_distro = new_distribution_from_structure(operated_structure);
  operated_distro->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  return operated_distro;
};

//! Operate and truncate
shared_ptr<distribution> distribution::operate_trunc
    ( const ioperator &op,shared_ptr<multi_indexstruct> trunc ) const {
  return operate_trunc(op,*(trunc.get())); };

//! Operate and truncate
shared_ptr<distribution> distribution::operate_trunc
    ( const ioperator &op,const multi_indexstruct &trunc ) const {
  auto base_structure = dynamic_cast<const parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(string("Could not upcast to par structure"));
  if (!base_structure->has_type( this->get_type() ))
    throw(string("Type got lost"));
  auto operated =
    new_distribution_from_structure(base_structure->operate(op,trunc));
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( format("{}-operated(tr)",get_name()) );
  return operated;
};

/*!
 * Union of distributions; the other may get converted in the process
 */
shared_ptr<distribution> distribution::distr_union( shared_ptr<distribution> other ) {
  if (get_orthogonal_dimension()!=other->get_orthogonal_dimension())
    throw(format("Incompatible orthogonal dimensions: this={}, other={}",
		      get_orthogonal_dimension(),other->get_orthogonal_dimension()));
  int dim = get_dimensionality();
  auto union_struct = new parallel_structure(this);
  if (get_is_orthogonal() && other->get_is_orthogonal()) {
    for (int id=0; id<dim; id++) {
      auto new_pidx =
	get_dimension_structure(id)->struct_union(other->get_dimension_structure(id));
      if (new_pidx->outer_size()==0)
	throw(format("Made empty in dim {} from <<{}>> and <<{}>>",
			  id,get_dimension_structure(id)->as_string(),
			  get_dimension_structure(id)->as_string()));
      union_struct->set_dimension_structure(id,new_pidx);
    }
    union_struct->set_is_orthogonal(true); union_struct->set_is_converted(false);
  } else {
    //printf("Union of multi_structures\n");
    if (get_is_orthogonal()) convert_to_multi_structures();
    if (other->get_is_orthogonal()) other->convert_to_multi_structures();
    for (int is=0; is<domains_volume(); is++) {
      auto pcoord = coordinate_from_linear(is);
      union_struct->set_processor_structure
	( pcoord, get_processor_structure(pcoord)
	  ->struct_union(other->get_processor_structure(pcoord))->force_simplify() );
    }
    union_struct->set_is_orthogonal(false); union_struct->set_is_converted(false);
  }
  union_struct->set_structure_type(distribution_type::GENERAL);
  auto union_d = new_distribution_from_structure(union_struct);
  union_d->set_orthogonal_dimension( get_orthogonal_dimension() );
  return union_d;
};

//! Copy the factory routines.
void distribution::copy_dist_factory(shared_ptr<distribution> d) {
  new_distribution_from_structure = d->new_distribution_from_structure;
  new_distribution_from_unique_local = d->new_distribution_from_unique_local;
  new_scalar_distribution = d->new_scalar_distribution;
  new_object = d->new_object; new_object_from_data = d->new_object_from_data;
  new_kernel_from_object = d->new_kernel_from_object;
  kernel_from_objects = d->kernel_from_objects;

  location_of_first_index = d->location_of_first_index;
  location_of_last_index = d->location_of_last_index;

  local_allocation = d->local_allocation; get_visibility = d->get_visibility;
  new_message = d->new_message; new_embed_message = d->new_embed_message;
};

//! Test on the requested type, and throw an exception if not.
void parallel_structure::require_type(distribution_type t) const {
  if (get_type()!=t) {
    throw(format("Type is <<{}>> should be <<{}>>",
		      type_as_string(),distribution_type_as_string(t)));
  }
};

string distribution_type_as_string(distribution_type t) {
  if (t==distribution_type::UNDEFINED)       return string("undefined");
  else if (t==distribution_type::CONTIGUOUS) return string("contiguous");
  else if (t==distribution_type::BLOCKED)    return string("blocked");
  else if (t==distribution_type::REPLICATED) return string("replicated");
  else if (t==distribution_type::CYCLIC)     return string("cyclic");
  else if (t==distribution_type::GENERAL)    return string("general");
  else return string("unknown");
};

//! \todo make sure this is collective!
shared_ptr<distribution> distribution::extend
    ( processor_coordinate ep,shared_ptr<multi_indexstruct> i ) {
  //parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  //if (base_structure==nullptr)
  //throw(string("Could not upcast to parallel structure"));
  const auto &base_structure = this->get_structure();
  // auto extended_structure = new parallel_structure( base_structure );
  ioperator no_op("none");
  //auto extended_structure = base_structure->operate(no_op);
  auto extended_structure(base_structure);

  auto extended_proc_struct = get_processor_structure(ep)->struct_union(i)->force_simplify();
  extended_structure.set_processor_structure(ep,extended_proc_struct);

  auto edist = new_distribution_from_structure(extended_structure);
  edist->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  edist->unset_memoization();
  //print("extended distribution: {}\n",edist->as_string());
  return edist;
};

//! \todo is this ever used? it's largely empty now....
bool distribution::equals(shared_ptr<distribution> d) {
  if (!(domains_volume()==d->domains_volume()))
    printf("Different numbers of procs\n");
  throw(format("Obsolute equality testing"));
  return false;
};

/*
 * Distributions:
 * replicated, etc
 */
replicated_distribution::replicated_distribution
    ( const decomposition &d,int ortho,index_int lsize)
  : distribution(d) {
  //print("create replicated distribution on {} domains\n",get_domains().size());
  create_from_replicated_local_size(lsize);
  set_orthogonal_dimension(ortho);
  set_name("replicated-scalar");
};

//! Get first index by reference, from processor \todo this needs to go: it copies too much
const domain_coordinate &parallel_structure::first_index_r( const processor_coordinate &p) {
  try {
    return get_processor_structure(p)->first_index_r();
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not get first index of <<{}>> from <<{}>>",
		      p.as_string(),as_string()));
  }
};

//! Get last index by reference, from processor \todo this needs to go: it copies too much
const domain_coordinate &parallel_structure::last_index_r( const processor_coordinate &p) {
  try {
    return get_processor_structure(p)->last_index_r();
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not get last index of <<{}>> from <<{}>>",
		      p.as_string(),as_string()));
  }
};

//! Get first index by reference, from processor by rvalue reference
const domain_coordinate &parallel_structure::first_index_r( const processor_coordinate &&p) {
  try {
    return get_processor_structure(p)->first_index_r();
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not get first index of rvref <<{}>> from <<{}>>",
		      p.as_string(),as_string()));
  }
};

//! Get last index by reference, from processor by rvalue reference
const domain_coordinate &parallel_structure::last_index_r( const processor_coordinate &&p) {
  try {
    return get_processor_structure(p)->last_index_r();
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not get last index of rvref <<{}>> from <<{}>>",
		      p.as_string(),as_string()));
  }
};

/*! Get the global first index, which is supposed to have been set by 
  the distribution creation routines.
*/
const domain_coordinate &parallel_structure::global_first_index() {
  try {
    if (!is_memoized()) memoize_structure();
    return stored_global_first_index;
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not get global_first_index of {}",as_string()));
  }
};

/*! Get the global last index, which is supposed to have been set by 
  the distribution creation routines.
*/
const domain_coordinate &parallel_structure::global_last_index() {
  try {
    if (!is_memoized()) memoize_structure();
    return stored_global_last_index;
  } catch (string c) {
    print("Error: <<{}>>\n",c);
    throw(format("Could not get global_last_index of {}",as_string()));
  }
};

shared_ptr<indexstruct> distribution_sigma_operator::operate
    (int dim,shared_ptr<distribution> d,processor_coordinate &p) const {
  if (!sigma_based)
    throw(format("Need to be of sigma type for this operate"));
  return sigop.operate( d->get_processor_structure(p)->get_component(dim) );
};

//! \todo the processor coordinate needs ot be passed by reference and be const
shared_ptr<indexstruct> distribution_sigma_operator::operate
    (int dim,shared_ptr<distribution> d,processor_coordinate p) const {
  if (!sigma_based)
    throw(format("Need to be of sigma type for this operate"));
  return sigop.operate( d->get_processor_structure(p)->get_component(dim) );
};

shared_ptr<multi_indexstruct> distribution_sigma_operator::operate
    ( const shared_ptr<distribution> d,processor_coordinate &p) const {
  if (!is_coordinate_based())
    throw(format("Need to be of lambda type for this operate"));
  return dist_coordinate_f(d,p);
};

/*!
  This is the variant to use if the operate, eh, operation contains collectives
*/
shared_ptr<distribution> distribution_sigma_operator::operate
    ( const shared_ptr<distribution> d ) const {
  if (!global_based)
    throw(format("Need to be of global type for this operate"));
  if (dist_global_f==nullptr)
    throw(format("Dist global f is null"));
  try {
    return dist_global_f(d);
  } catch (string c) { //print("dist_sig_op global operate failed: {}\n",c);
    throw(format("Dist_sig_op::operate failed: {}",c));
  } catch (...) { //print("dist_sig_op global operate failed\n");
    throw(string("Dist_sig_op::operate failed"));
  }
};


/*
 * Numa stuff
 */

// const domain_coordinate distribution::offset_vector() {
//   auto nstruct = get_numa_structure();
//   auto gstruct = get_global_structure();
//   return nstruct->first_index_r() - gstruct->first_index_r();
// };

const domain_coordinate &distribution::numa_size_r() {
  return get_numa_structure()->local_size_r();
};

/*
 * Memoization of structural information
 */
//! Compute first/last/enclosing \todo how does this related to compute_enclosing?
void parallel_structure::memoize_structure() {
  // compute global first/last, store in parallel_structure member
  try {
    //print("global first compute, apply to {}\n",this->as_string());
    stored_global_first_index = compute_global_first_index(this);
    //print("global first computed as: {}\n",stored_global_first_index.as_string());
  } catch (string c) { print("Error in memoize_structure: {}\n",c);
    throw(format("Could not memo'ize global_first of {}",as_string()));
  }

  //print("global last\n");
  try  {
    stored_global_last_index = compute_global_last_index(this);
    //print("global last computed as: {}\n",stored_global_last_index.as_string());
  } catch (string c) {
    throw(format("Error memo'izing l of {}: <<{}>>",get_name(),c)); };

  set_enclosing_structure(multi_indexstruct(stored_global_first_index,stored_global_last_index));
  // enclosing_structure =
  //   shared_ptr<multi_indexstruct>
  //   ( new multi_indexstruct(stored_global_first_index,stored_global_last_index) );

  set_is_memoized();
};

//snippet analyzepatterndependence
/*!
  This routine is called on an alpha distribution.
  For a requested segment of the beta structure, belonging to "mytid",
  find messages to mytid.
  This constructs:
  - the global_struct in global coordinates with the right sender
  - the local_struct expressed relative to the beta structure

  We can deal with a beta that sticks out 0--gsize, but the resulting
  global_struct will still be properly limited.

 \todo why doesn't the containment test at the end succeed?
 \todo can the tmphalo be completely conditional to the local address space?
*/
vector<shared_ptr<message>> distribution::messages_for_objects
    ( int step,const processor_coordinate &mycoord,self_treatment doself,
      shared_ptr<object> invector,shared_ptr<object> halo,
      bool trace) {
  auto indistro = invector->get_distribution(),
    beta_dist = halo->get_distribution();
  auto beta_block = beta_dist->get_processor_structure(mycoord);
  auto numa_block = beta_dist->get_numa_structure(); // to relativize against
  auto msgs = messages_for_segment
    (step,mycoord,doself,beta_block,numa_block,trace);
  for ( auto &m : msgs ) {
    int ndomains = indistro->domains_volume();
    m->set_in_object(invector); m->set_out_object(halo);
    m->set_tag_from_kernel_step(step,ndomains);
    m->set_halo_struct( numa_block /* halo_struct */);
  }
  return msgs;
};
			      
vector<shared_ptr<message>> distribution::messages_for_segment
    ( int step,const processor_coordinate &mycoord,self_treatment doself,
      shared_ptr<multi_indexstruct> beta_block,shared_ptr<multi_indexstruct> halo_block,
      bool trace) {
  if (trace)
    print("{}: deriving msgs to cover {} from {}\n",
	  mycoord.as_string(),beta_block->as_string(),this->as_string());
  vector<shared_ptr<message>> messages;
  int dim = get_dimensionality();
  auto buildup = shared_ptr<multi_indexstruct>( new multi_indexstruct(dim) );
  // Wow! Iterating over decomposition
  for ( const auto &pcoord : decomposition(get_decomposition(),mycoord) ) { // start at me
    // Note: we need to generate recv message for all data, so that we get send msgs
    // however, if the message is collective we post only one.
    decltype( get_processor_structure(pcoord) ) pstruct;
    try {
      pstruct = get_processor_structure(pcoord);
      if (trace)
	print(".. consider {} gamma block: {}\n",pcoord.as_string(),pstruct->as_string());
      auto mintersect = beta_block->intersect(pstruct);
      if (trace)
	print(".. intersection: {}\n",mintersect->as_string());
      // compute the intersect of the beta struct and alpha[p] in global coordinates
      if (mintersect->is_empty()) {
	if (trace) print(".. intersection empty\n");
	continue; }
      if (doself==self_treatment::EXCLUDE && pcoord==mycoord) {
	if (trace) print("Intersection ignored with {}: {} because self treatment exclude\n",
			 pcoord.as_string(),mintersect->as_string());
	continue; }
      if (buildup->contains(mintersect)) {
	if (trace) print("Intersection ignored with {}: {} because already covered in {}\n",
			 pcoord.as_string(),mintersect->as_string(),buildup->as_string());
	continue; }

      shared_ptr<message> m;
      try {
	auto simstruct = mintersect->force_simplify();
	m = shared_ptr<message>( new_message(pcoord,mycoord,simstruct) );
	if ( doself==self_treatment::ONLY && !(mycoord==pcoord) )
	  m->set_skippable();
	m->set_receive_type();
	if (beta_has_local_addres_space)
	  m->relativize_to( halo_block /* numa_block */ /* .force_simplify() */);
	messages.push_back( m );
	buildup = buildup->struct_union(mintersect)->force_simplify();
      } catch (string e) { print("Error <<{}>>",e);
	throw(format("Could not create message for {}:{}",
		     pcoord.as_string(),pstruct->as_string()));
      }
      if (trace)
	print("New message: {}; now covering {}\n",m->as_string(),buildup->as_string());
      if (buildup->contains(beta_block)) {
	if (trace) print("Now covering beta block {} with {}\n",
			 beta_block->as_string(),buildup->as_string());
	goto covered;
      }
    } catch (string e) { print("Error <<{}>>",e);
      throw(format("messages_for_segment covering loop breakdown in {}: {}",
		   pcoord.as_string(),pstruct->as_string()));
    }
  }
  {
    auto farcorner = get_farpoint_processor();
    for (int id=0; id<dim; id++) { // does the halo stick out in dimension id?
      shared_ptr<multi_indexstruct> intersect;
      index_int g = global_size().coord(id);
      // is the halo sticking out to the right?
      //snippet mpiembedmessage
      auto pleft = mycoord.left_face_proc(id,farcorner);
      intersect = beta_block->intersect
	(get_processor_structure(pleft)->operate(shift_operator(id,g)));
      if ( !intersect->is_empty()
	   && !(doself==self_treatment::EXCLUDE && pleft==mycoord) ) {
	buildup = buildup->struct_union(intersect);
	auto m = new_embed_message
	  (pleft,mycoord,intersect,intersect->operate(shift_operator(id,-g)));
	if (beta_has_local_addres_space) {
	  auto tmphalo = /* numa_block */ /* halo_struct */
	    halo_block->operate(shift_operator(id,-g));
	  m->relativize_to(tmphalo);
	}
	messages.push_back(m);
      }
      //snippet end
      // is the halo sticking out to the left
      auto pright = mycoord.right_face_proc(id,farcorner);
      intersect = beta_block->intersect
	( get_processor_structure(pright)->operate(shift_operator(id,-g)) );
      if ( !intersect->is_empty()
	   && !(doself==self_treatment::EXCLUDE && pright==mycoord) ) {
	buildup = buildup->struct_union(intersect);
	auto m = new_embed_message
	  (pright,mycoord,intersect,intersect->operate(shift_operator(id,g)));
	if (beta_has_local_addres_space) {
	  auto tmphalo = /* numa_block */ /* halo_struct */
	    halo_block->operate(shift_operator(id,g));
	  m->relativize_to(tmphalo);
	}
	messages.push_back(m);
      }
    }
  }

  if (!buildup->contains(beta_block))
    throw(format("message buildup {} does not cover beta {}",
  		      buildup->as_string(),beta_block->as_string()));
 covered:
  // for ( auto m : messages )
  //   m->set_halo_struct( halo_block /* numa_block */ /* halo_struct */);

  return messages;
};

string distribution::as_string() const {
  auto pstruct = dynamic_cast<const parallel_structure*>(this);
  if (pstruct==nullptr)
    throw(string("Could not cast distribution to parallel_structure"));
  return format("{}:[{}]",get_name(),pstruct->parallel_structure::as_string());
};

/****
 **** Sparse stuff
 ****/

int sparse_element::operator<( const sparse_element &other ) const {
  auto other_copy = other;
  return *this < move(other_copy);
};
int sparse_element::operator<( const sparse_element &&other ) const {
  return get_index()<other.get_index();
};

void sparse_row::add_element( const sparse_element &e ) {
  auto e_copy = e;
  add_element(move(e_copy));
};

void sparse_row::add_element( const sparse_element &&e ) {
  auto pos = row.begin();
  for ( ; pos!=row.end(); ++pos) {
    if (e<*pos) break;
  }
  row.insert( pos, e );
};

/*!
  Extract the indices from a sparse row
  \todo this can make a really large indexed before it simplifies.
*/
shared_ptr<indexstruct> sparse_row::all_indices() {
  auto all = shared_ptr<indexstruct>( new indexed_indexstruct() ); // empty?
  for ( auto elt : row ) {
    auto idx = elt.get_index();
    //print("adding index {}, ",idx);
    all->addin_element(idx);
  }
  //print("indices found: <<{}>>\n",all->as_string());
  all = all->force_simplify();
  //print("indices simplified: <<{}>>\n",all->as_string());
  return all;
};
/*!
  Sparse inner product of a sparse row and a dense row from an object.
  \todo omp struct 4 somehow takes the general path
 */
double sparse_row::inprod( shared_ptr<object> inobj,const processor_coordinate &p ) {
  double s = 0.;
  const auto distro = inobj->get_distribution();
  int number_of_elements = row.size();

  if (distro->has_type_locally_contiguous()) {
    auto odata   = inobj->get_data(p);
    auto in_size = inobj->volume(p);
    index_int first_vec_elt_global_idx = distro->first_index_r(p).coord(0);

    for (index_int icol=0; icol<number_of_elements; icol++) {
      const auto elt = row.at(icol); index_int global_column = elt.get_index();
      double x1 = elt.get_value();
      index_int local_idx_of_global_col = global_column-first_vec_elt_global_idx;
      if (local_idx_of_global_col<0 || local_idx_of_global_col>=in_size) {
	throw(format("[{}] local col={} -> global col={}@{}: outside of in_obj data: {}",
		     p.as_string(),
		     icol,global_column,local_idx_of_global_col,
		     inobj->get_processor_structure(p)->as_string()));
      }
      double x2 = odata.at(local_idx_of_global_col);
      s += x1*x2;
    }
  } else {
    for (index_int icol=0; icol<number_of_elements; icol++) {
      auto elt = row[icol]; index_int global_column = elt.get_index();
      double x1 = elt.get_value();
      double x2 = inobj->get_element_by_index(global_column,p);
      s += x1*x2;
    }
  }
  return s;
};

/*! Create a single process matrix of a given size.
  We use the default index set of [0,n)
 */
sparse_matrix::sparse_matrix( index_int m,index_int n )
  : sparse_matrix( indexstructure(contiguous_indexstruct(0,m-1)),n ) {};

/*! Sparse matrix constructor by adding a \ref sparse_rowi for each 
  row in the owned range.
  \todo we should at some point handle non-contiguous owned ranges
*/
sparse_matrix::sparse_matrix( indexstructure &&idx )
  : sparse_matrix(idx,-1) {};
sparse_matrix::sparse_matrix( indexstructure &&idx,index_int globalsize )
  : globalsize(globalsize),entity(entity_cookie::OPERATOR) {
  translation = idx; index_int s = idx.local_size();
  m = vector<shared_ptr<sparse_rowi>>(s);
  for (index_int i=0; i<s; i++)
    m.at(i) = make_shared<sparse_rowi>(idx.get_ith_element(i));
  set_name("sparse-mat");
};

//! \todo can we integrate this with the above through std::move?
sparse_matrix::sparse_matrix( indexstructure &idx )
  : sparse_matrix(idx,-1) {};
sparse_matrix::sparse_matrix( indexstructure &idx,index_int globalsize )
  : globalsize(globalsize),entity(entity_cookie::OPERATOR) {
  translation = idx; index_int s = idx.local_size();
  m = vector<shared_ptr<sparse_rowi>>(s);
  for (index_int i=0; i<s; i++)
    m.at(i) = make_shared<sparse_rowi>(idx.get_ith_element(i));
  set_name("sparse-mat");
};

sparse_matrix::sparse_matrix( shared_ptr<indexstruct> idx,index_int globalsize )
  : globalsize(globalsize),entity(entity_cookie::OPERATOR) {
  translation = indexstructure(idx); index_int s = idx->local_size();
  m = vector<shared_ptr<sparse_rowi>>(s);
  for (index_int i=0; i<s; i++)
    m.at(i) = make_shared<sparse_rowi>(idx->get_ith_element(i));
  set_name("sparse-mat");
};

/*!
  Add a new row to a matrix, keeping the rows sorted by global number.
  \todo throw error if already found
*/
void sparse_matrix::insert_row( shared_ptr<sparse_rowi> row ) {
  auto loc = translation.find(row->get_row_number());
  m.at(loc) = row;
};

/*!
  Set an element. This will create a row if needed.
*/
void sparse_matrix::add_element( index_int i,index_int j,double v ) {
  if (globalmat) {
    throw(format("Can not yet add global elements"));
  } else {
    auto row = get_row_index_by_number(i);
    if (row<0)
      throw(format("Non-existing row {}",i));
    if (globalsize>=0 && (j<0 || j>=globalsize) )
      throw(format("Column {} falls outside globalsize {}",j,globalsize));
    m.at(row)->add_element(j,v);
  }
};

//! Test whether an (i,j) index is already present.
bool sparse_matrix::has_element( index_int i,index_int j ) const {
  //  bool has; int row; tie(has,row) = try_get_row_index_by_number(i);
  auto [has,row] = try_get_row_index_by_number(i);
  if (!has) { //print("could not find row {}\n",i);
    return false;
  } else { //print("found row {} at {}\n",i,row);
    if (global_jlast>=global_jfirst) { // there is a j test
      //print("applying j test {} <? {} <? {}\n",global_jfirst,j,global_jlast);
      return j>=global_jfirst && j<=global_jlast && m.at(row)->has_element(j);
    } else {
      return m.at(row)->has_element(j);
    }
  }
};

/*!
  Get a row index (in the stored sturcture) by its absolute number.
  Return negative if not found.
  \todo keep a seek pointer.
 */
int sparse_matrix::get_row_index_by_number(index_int idx) const {
  for ( int irow=0; irow<m.size(); irow++ ) {
    index_int row_no = m.at(irow)->get_row_number();
    //print("compare {} to {}\n",idx,row_no);
    if (idx==row_no)
      return irow;
  }
  return -1;
};

tuple<bool,int> sparse_matrix::try_get_row_index_by_number(index_int idx) const {
  for ( int irow=0; irow<m.size(); irow++ ) {
    index_int row_no = m.at(irow)->get_row_number();
    //print("compare {} to {}\n",idx,row_no);
    if (idx==row_no) {
      //print("found\n");
      return {true,irow};
    }
  }
  //  return {false,-1};
  return make_tuple(false,-1);
};

/*!
  Get a row (in the stored sturcture) by its absolute number.
  \todo keep a seek pointer.
 */
shared_ptr<sparse_rowi> sparse_matrix::get_row_by_global_number(index_int idx) const {
  for ( int irow=0; irow<m.size(); irow++ ) {
    auto rp = m.at(irow);
    if (rp==nullptr)
      throw(format("row with local number {} is null",irow));
    index_int row_no = rp->get_row_number();
    print("compare {} to {}\n",idx,row_no);
    if (idx==row_no)
      return rp;
  }
  print(format("Not found {}\n",idx));
  throw(format("Could not find global row number {} on this proc",idx));
};

/*!
  Get all the row indices. This routines should not be used computationally.
 */
shared_ptr<indexstruct> sparse_matrix::row_indices() {
  auto idx = shared_ptr<indexstruct>{ new indexed_indexstruct() };
  for (auto row : m )
    idx->addin_element( row->get_row_number() );
  return idx->force_simplify();
};

string sparse_element::as_string() {
  return format("{}:{}",get_index(),get_value());
};

string sparse_row::as_string() {
  memory_buffer w;
  for (auto e=row.begin(); e!=row.end(); ++e)
    format_to(w.end(),"{} ",(*e).as_string());
  return to_string(w);
};

string sparse_rowi::as_string() {
  return format("<{}>: {}",get_row_number(),get_row()->as_string());
};

/*!
  All columns from the local matrix.
*/
shared_ptr<indexstruct> sparse_matrix::all_columns() {
  auto all = shared_ptr<indexstruct>{ new empty_indexstruct() };
  for (auto row : m) {
    try {
      all = all->struct_union( row->all_indices() );
    } catch ( string e ) { print("ERROR: {}\n",e);
      throw(format("Could not union with row {}",row->get_row_number()));
    }
  }
  auto r_all = all->force_simplify();
  // print("Sparse matrix columns <<{}>> simplified to <<{}>>\n",
  // 	     all->as_string(),r_all->as_string());
  return r_all;
};

/*!
  All columns by request.
  \todo get that indexstruct iterator to work
*/
shared_ptr<indexstruct> sparse_matrix::all_columns_from
    ( shared_ptr<multi_indexstruct> multi_wanted ) {
  if (multi_wanted->get_dimensionality()>1)
    throw(string("Can not get all columns from in multi-d"));
  auto wanted = multi_wanted->get_component(0);
  if (wanted->is_empty())
    throw(format("Matrix seems to be empty on this proc: {}",as_string()));
  auto all = shared_ptr<indexstruct>{ new empty_indexstruct() };
  for (int iirow=0; iirow<wanted->local_size(); iirow++) {
    try {
      int irow = wanted->get_ith_element(iirow);
      auto row = get_row_index_by_number(irow);
      if (row<0)
	throw(format("Could not locally get row {}",irow));
      all = all->struct_union( m.at(row)->all_indices() );
    } catch (string c) {
      throw(format("Error processing row #{} of {}: {}",iirow,wanted->as_string(),c));
    }
  }
  return all->force_simplify();
};

sparse_matrix *sparse_matrix::transpose() const {
  throw(string("sparse matrix transpose not implemented"));
};

/*! 
  Product of a sparse matrix into a blocked vector.
  \todo write iterator over all rows, inprod returning sparse_element
*/
void sparse_matrix::multiply
    ( shared_ptr<object> in,shared_ptr<object> out,processor_coordinate &p) {
  const auto distro = out->get_distribution();
  if (distro->get_dimensionality()>1)
    throw(string("spmvp not in multi-d"));
  if (distro->has_type_locally_contiguous()) {
    auto data = out->get_data(p);
    index_int
      tar0 = distro->location_of_first_index(distro,p),
      len = distro->volume(p),
      first_row_num = distro->first_index_r(p).coord(0);
    //print("multiply coputes {} elements starting {}\n",len,tar0);
    for (index_int i=0; i<len; i++) {
      index_int rownum = first_row_num+i;
      auto sprow = get_row_index_by_number(rownum);
      double v = m.at(sprow)->inprod(in,p);
      data.at(tar0+i) = v;
    }
  } else {
    auto outstruct = distro->get_processor_structure(p)->get_component(0);
    throw(format("Can only multiply into blocked type, not <<{}>>",
		      outstruct->as_string()));
  }
};

string sparse_matrix::as_string() {
  return format("Sparse matrix on {}, nnzeros={}",
		     row_indices()->as_string(),nnzeros());
};

string sparse_matrix::contents_as_string() {
  memory_buffer w; format_to(w.end(),"{}:\n",as_string());
  for (auto row=m.begin(); row!=m.end(); ++row)
    format_to(w.end(),"{}\n",(*row)->as_string());
  return to_string(w);
};

/****
 **** Object data
 ****/

shared_data_pointer data_allocate(index_int s) {
  return shared_data_pointer( new std::vector<double>(s) );
};

void object_data::set_numa_data( shared_ptr<vector<double>> dat,index_int s) {
  if (numa_data_pointer!=nullptr)
    throw(format("Trying to set numa data pointer for the second time"));
  numa_data_pointer = dat; numa_data_size = s;
};

/*!
  Create data pointers/offset/size for each domain.
  Offsets are mostly useful for OpenMP.
  This is called in the object_data constructor
*/
void object_data::create_data_pointers(int ndom) {
  domain_data_pointers = vector<data_pointer>(ndom);
  data_offsets = vector<index_int>(ndom);
  data_sizes = vector<index_int>(ndom);
};

/*!
  \todo this should really do a move of an lvalue ref
  \todo remove the size parameter
*/
void object_data::set_domain_data_pointer( int n,data_pointer p,index_int s,index_int o ) {
  domain_data_pointers.at(n) = p;
  data_sizes.at(n) = s;
};
// //! \todo is this ever called?
// void object_data::set_domain_data_pointer
//     ( int n,data_pointer dat,index_int s,index_int o) {
//   set_domain_data_pointer( n,dat,s );
// };

data_pointer object_data::get_nth_data_pointer( int i ) const {
  return domain_data_pointers.at(i); };

//! Allocate data and return; the calling environment will store this as numa
shared_ptr<vector<double>> object_data::create_data(index_int s, string c) {
  if (s<0)
    throw(format("Negative {} malloc for <<{}>>",s,c));
  shared_ptr<vector<double>> dat;
  try {
    dat = make_shared<vector<double>>(s);
  } catch ( bad_alloc ) {
    print("Bad alloc in create_data for s={}, {}\n",s,c);
    throw(format("Could not create data"));
  }
  data_status = object_data_status::ALLOCATED;
  return dat;
};

//! Create unnamed data
shared_ptr<vector<double>> object_data::create_data(index_int s) {
  return create_data(s,string("Unknown object"));
};

shared_ptr<vector<double>> object_data::get_numa_data_pointer() const {
  if (numa_data_pointer==nullptr)
    throw(string("null numa data pointer"));
  return numa_data_pointer;
};

double *object_data::get_raw_data() const {
  return numa_data_pointer->data();
};

index_int object_data::get_raw_size() const {
  return numa_data_size;
};

//! Inherit someone's data; base addresses align. \\! \todo should we lose the processor?
void object_data::inherit_data
    ( const processor_coordinate &p, shared_ptr<object> o,index_int offset ) {
  auto odat = o->get_nth_data_pointer(0);
  set_domain_data_pointer(0,odat,o->get_distribution()->volume(p));
  data_status = object_data_status::INHERITED;
};

//! Get the data size by local number, computed by \ref get_domain_local_number.
index_int object_data::get_data_size( int n ) {
  return data_sizes.at(n);
};

/*! Set an object to a constant value.
  Why on earth do we need a shared_ptr<double> ?
  \todo domains \todo extend to orthogonal dimensions
 */
void object_data::set_value( shared_ptr<double> x ) {
  if (!has_data_status_allocated())
    throw(string("Can not set value for unallocated"));
  for (int idom=0; idom<domain_data_pointers.size(); idom++) {
    auto data = get_nth_data_pointer(idom);
    index_int siz = get_data_size(idom);
    for (index_int i=0; i<siz; i++)
      data.at(i) = *x;
  }
};

void object_data::set_value(double x) {
  if (!has_data_status_allocated())
    throw(string("Can not set value for unallocated"));
  for (int idom=0; idom<domain_data_pointers.size(); idom++) {
    auto data = get_nth_data_pointer(idom);
    index_int siz = get_data_size(idom);
    for (index_int i=0; i<siz; i++)
      data.at(i) = x;
  }
};

double object_data::get_max_value() {
  double mx = -1.e300;
  for (int p=0; p<domain_data_pointers.size(); p++) {
    auto data = domain_data_pointers.at(p); auto s = data.size(); //int s = data_sizes[p];
    for (int i=0; i<s; i++) {
      double v = data.at(i);
      if (v>mx) mx = v;
    }
  }
  return mx;
};

double object_data::get_min_value() {
  if (domain_data_pointers.size()==0)
    throw(string("Can not get min value from non-existing data"));
  double mn = 1.e300;
  for (int p=0; p<domain_data_pointers.size(); p++) {
    auto data = domain_data_pointers.at(p); auto s = data.size();
    for (int i=0; i<s; i++) {
      double v = data.at(i);
      if (v<mn) mn = v;
    }
  }
  return mn;
};

string object_data::data_status_as_string() {
  if (data_status==object_data_status::UNALLOCATED) return string("UNALLOCATED");
  if (data_status==object_data_status::ALLOCATED) return string("ALLOCATED");
  if (data_status==object_data_status::INHERITED) return string("INHERITED");
  if (data_status==object_data_status::REUSED) return string("REUSED");
  if (data_status==object_data_status::USER) return string("USER");
  throw(string("Invalid data status"));
};

/****
 **** Object
 ****/

object::object( std::shared_ptr<distribution> d )
  : object_data(d->local_ndomains()),
    entity(entity_cookie::OBJECT) {
  object_distribution = d;
  object_number = count++; //data_is_filled = new processor_mask(d.get());
  //set_name(fmt::format("object-{}",object_number));
};

//! Store a data pointer in the right location, multi-d domain number
void object::register_data_on_domain
    ( processor_coordinate &dom,data_pointer dat,
      index_int s, index_int offset ) {
  register_data_on_domain_number
    (get_distribution()->get_domain_local_number(dom),dat,s,offset);
};

//! Store a data pointer by local index; see \ref register_data_on_domain for global
void object::register_data_on_domain_number
    ( int loc,data_pointer dat,index_int s,index_int offset ) {
  //print("{} register data {}+{} size={}\n",loc,(long)dat.data(),offset,s);
  set_domain_data_pointer(loc,dat,s,offset); register_allocated_space(s);
};

data_pointer object::get_data( const processor_coordinate &p ) const {
  auto p_copy = p;
  return get_data(std::move(p_copy));
};

data_pointer object::get_data( const processor_coordinate &&p ) const {
  //  if (!lives_on(p)) throw(format("object does not live on {}",p->as_string()));
  if (has_data_status_unallocated())
    throw(format("Trying to get data from unallocated object <<{}>>",get_name()));
  int locdom = get_distribution()->get_domain_local_number(p);
  auto pointer = get_nth_data_pointer(locdom);
  return pointer;
};

//snippet getelement
/*!
  Find an element by index. 
  This uses \ref get_numa_structure since for OMP we can see everything, for MPI only local.
  This is so far only used in the sparse matrix product.
  \todo broken in multi-d
*/
double object::get_element_by_index(index_int i,const processor_coordinate &p) const {
  const auto distro = get_distribution();
  auto localstruct = distro->get_numa_structure();
  index_int locali = localstruct->linearfind(i);
  if (locali<0 || locali>=distro->numa_local_size())
    throw(format("Found {} at local index {}, which is out of bounds of <<{}>>",
		      i,locali,localstruct->as_string()));
  return get_data(p).at(locali);
};
//snippet end

/*! Render object information as string. See also \ref object::values_as_string.
  \todo this becomes a circular call */
string object::as_string() {
  return format("{}:Distribution",get_name());
};

string object::values_as_string(processor_coordinate &p) {
  memory_buffer w; format_to(w.end(),"{}:",get_name());
  const auto distro = get_distribution();
  if (distro->get_orthogonal_dimension()>1)
    throw(string("Can not handle k>1 for object values_as_string"));
  auto data = this->get_data(p);
  index_int f = distro->location_of_first_index(distro,p),
    s = distro->volume(p);
  for (index_int i=0; i<s; i++)
    format_to(w.end()," {}:{}",i+f,data.at(i));
  return to_string(w);
};

string object::values_as_string(processor_coordinate &&p) {
  memory_buffer w; format_to(w.end(),"{}:",get_name());
  const auto distro = get_distribution();
  if (distro->get_orthogonal_dimension()>1)
    throw(string("Can not handle k>1 for object values_as_string"));
  auto data = this->get_data(p);
  index_int f = distro->location_of_first_index(distro,p),
    s = distro->volume(p);
  for (index_int i=0; i<s; i++)
    format_to(w.end()," {}:{}",i+f,data.at(i));
  return to_string(w);
};

/****
 **** Message
 ****/

/*!
  Message creation. We set the local struct to global; this will be relativized in MPI.
  - global struct : global indexes of the message
  - embed struct : global but before twe wrap halos, so it can be -1:0 and such.

  \todo can be omit the cloning? the outer structs are temporary enough
  \todo I wish we could make the local_struct through relativizing, but there's too much variability in how that is done.
*/
message::message(const decomposition &d,
		 const processor_coordinate &snd,const processor_coordinate &rcv,
		 shared_ptr<multi_indexstruct> &e,shared_ptr<multi_indexstruct> &g)
  : decomposition(d) {
  sender = snd; receiver = rcv;
  local_struct = g->make_clone();
  global_struct = g->make_clone();
  embed_struct = e->make_clone();
  set_name(format("message-{}->{}",snd.as_string(),rcv.as_string()));
};

//! Return the message sender.
processor_coordinate &message::get_sender() {
  if (sender.get_dimensionality()<=0)
    throw(string("Invalid sender"));
  return sender; };

//! Return the message receiver.
processor_coordinate &message::get_receiver() {
  if (receiver.get_dimensionality()<=0)
    throw(string("Invalid receiver"));
  return receiver; };

shared_ptr<multi_indexstruct> message::get_global_struct() {
  if (global_struct==nullptr) throw(string("msg has no global struct"));
  return global_struct; };

shared_ptr<multi_indexstruct> message::get_embed_struct() {
  if (embed_struct==nullptr) throw(string("msg has no embed struct"));
  return embed_struct; };

shared_ptr<multi_indexstruct> message::get_local_struct() {
  if (local_struct==nullptr) throw(string("msg has no local struct"));
  return local_struct; };

shared_ptr<object> message::get_in_object( ) {
  if (in_object==nullptr) throw(string("Message has no in object"));
  return in_object; };

shared_ptr<object> message::get_out_object( ) {
  if (out_object==nullptr) throw(string("Message has no out object"));
  return out_object; };

const shared_ptr<object> message::view_out_object( ) const {
  if (out_object==nullptr) throw(string("Message has no out object"));
  return out_object; };

//! Set the outputobject of a message.
void message::set_out_object( shared_ptr<object> out ) { out_object = out; };

//! Set the input object of a message.
void message::set_in_object( shared_ptr<object> in ) { in_object = in; };

//snippet subarray
void message::compute_subarray
( shared_ptr<multi_indexstruct> outer,shared_ptr<multi_indexstruct> inner,int ortho) {
  int dim = outer->get_same_dimensionality(inner->get_dimensionality());
  numa_sizes = new int[dim+1]; struct_sizes = new int[dim+1]; struct_starts = new int[dim+1];
  //  annotation.write("tar subarray:");
  auto loc = outer->location_of(inner);
  for (int id=0; id<dim; id++) {
    numa_sizes[id] = outer->local_size_r().coord(id);
    struct_sizes[id] = inner->local_size_r().coord(id);
    struct_starts[id] = loc->at(id);
    format_to(annotation.end()," {}:{}@{}in{}",id,struct_sizes[id],struct_starts[id],numa_sizes[id]);
  }
  // if (ortho>1)
  //   annotation.write(" (k={})",ortho);
  numa_sizes[dim] = ortho; struct_sizes[dim] = ortho; struct_starts[dim] = 0;
};
//snippet end

/*!
  Where does the send buffer fit in the input object?
  This is called by \ref message::set_in_object.
  This will be extended by \ref mpi_message::compute_src_index.

  \todo the names outer and inner are reversed, right?
*/
//snippet impsrcindex
void message::compute_src_index() {
  if (src_index!=-1)
    throw(format("Can not recompute message src index in <<{}>>",get_name()));
  const auto indistro = get_in_object()->get_distribution();
  try {
    auto send_struct = get_global_struct(); // local ???
    auto proc_struct = indistro->get_processor_structure(get_sender());
    src_index = send_struct->linear_location_in(proc_struct);
  } catch (string c) { throw(format("Error <<{}>> setting src_index",c)); }

  auto outer = indistro->get_numa_structure();
  auto inner = get_global_struct();
  int
    ortho = indistro->get_orthogonal_dimension();
  try {
    compute_subarray(outer,inner,ortho);
  } catch (string c) {
    throw(format("Could not compute src subarray for <<{}>> in <<{}>>: {}",
		      inner->as_string(),outer->as_string(),c));
  }
};
//snippet end

/*!
  Where does the receive buffer fit in the output object?
  This is call by \ref message::set_out_object.
  This will be extended by \ref mpi_message::compute_tar_index.
*/
//snippet imptarindex
void message::compute_tar_index() {
  if (tar_index!=-1)
    throw(format("Can not recompute message tar index in <<{}>>",get_name()));

  // computing target index. is that actually used?
  const auto outdistro = get_out_object()->get_distribution();
  try {
    auto local_struct = get_local_struct();
    auto processor_struct = outdistro->get_processor_structure(get_receiver());
    tar_index = local_struct->linear_location_in(processor_struct);
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not compute target index"));
  }

  auto outer = outdistro->get_numa_structure();
  auto inner = get_embed_struct();
  try {
    int
      ortho = outdistro->get_orthogonal_dimension();
    compute_subarray(outer,inner,ortho);
  } catch (string c) {
    throw(format("Could not compute tar subarray for <<{}>> in <<{}>>: {}",
		      inner->as_string(),outer->as_string(),c));
  }
};
//snippet end

index_int message::get_src_index() const {
  if (src_index==-1)
    throw(string("Src index not yet computed"));
  return src_index;
};

index_int message::get_tar_index() const {
  if (tar_index==-1)
    throw(string("Tar index not yet computed"));
  return tar_index;
};


//snippet msgrelativize
/*!
  Express the local structure of a message as the global struct
  relative to a halo structure.
*/
void message::relativize_to(shared_ptr<multi_indexstruct> container) { 
  local_struct = global_struct->relativize_to(container);
};
//snippet end

//! \todo is this actually used? use linearize for the sender/receiver or use print
void message::as_char_buffer(char *buf,int *len) {
  if (*len<14) {
    throw(string("Insufficient buffer provided for message->as_string"));
  }
  int dim = global_struct->get_dimensionality();
  if (dim>1)
    throw(format("Can not convert msg to buffer in {} dim\n",dim));
  sprintf(buf,"%2d->%2d:[%2lld-%2lld]",
	  get_sender()[0],get_receiver()[0],
	  global_struct->first_index(0),global_struct->last_index(0));
  *len = 14;
};

void message::set_send_type() {
  sendrecv_type = message_type::SEND;
};
void message::set_receive_type() {
  sendrecv_type = message_type::RECEIVE;
};
message_type message::get_sendrecv_type() const {
  if (sendrecv_type==message_type::NONE)
    throw(string("Message has sendrecv type none"));
  return sendrecv_type;
};
string message::sendrecv_type_as_string() const {
  auto str = format("(X)");
  if (get_sendrecv_type()==message_type::SEND)
    str = format("(S:{})",get_tag_value());
  else if (get_sendrecv_type()==message_type::RECEIVE)
    str = format("(R:{})",get_tag_value());
  else
    str = format("(U:{})",get_tag_value());
  return str;
};

bool message::get_is_collective() const {
  return view_out_object()->has_type_replicated();
};

void message::set_tag_by_content
    (processor_coordinate &snd,processor_coordinate &rcv,int step,int np) {
  auto farcorner = get_farpoint_processor();
  tag = message_tag(snd.linearize(farcorner),rcv.linearize(farcorner),step,np);
};

string message::as_string() {
  memory_buffer w;
  try {
    format_to(w.end(),"{}:",sendrecv_type_as_string());
    format_to(w.end(),"{}->{}:{}",
	    get_sender().as_string(),get_receiver().as_string(),
	    global_struct->as_string());
  } catch ( string e ) { print("ERROR: {}\n",e);
    throw(format("Could not string message"));
  }
  return to_string(w);
};

/****
 **** Signature function
 ****/

/*! Set type if uninitialized or already of this type; otherwise complain.
  Exception: we set an explicit beta by default, so that is accepted too.
*/
void signature_function::set_type( signature_type t ) {
  if (t==signature_type::UNINITIALIZED)
    throw(string("Should not set signature type uninitialized"));
  else if (has_type_uninitialized() || has_type_explicit_beta()) {
    if ( t<signature_type::UNINITIALIZED || t>signature_type::FUNCTION )
      throw(format("Invalid signature type {}",(long)t));
    type = t;
  } else if (has_type_initialized(t)) {
    /* set to same type is allowed */ return;
  } else {
    throw(format("Can not change signature type from {} to {}\n",
		 type_to_string(type),type_to_string(t)));
  }
};

//
// sigma from operators
//

/*!
  Add an operator to a signature function.
 */
void signature_function::add_sigma_operator( multi_ioperator *op ) {
  set_type_operator_based();
  operators.push_back(op);
};

//
// signature from function
//

/*!
  Supply the int->indexstruct function; we wrap it in an \index ioperator,
  which we apply to whole domains.
*/
void signature_function::set_signature_function_function( const multi_sigma_operator &op ) {
  set_type_function(); func = op;
};
//! The one-d special case is wrapped in the general mechanism
void signature_function::set_signature_function_function( const sigma_operator &op ) {
  auto sop = multi_sigma_operator(op);
  set_signature_function_function(sop);
};
//! Define a signature function from a pointwise function \todo add multi_idx->multi_idx func
void signature_function::set_signature_function_function
    ( std::function< shared_ptr<indexstruct>(index_int) > f ) {
  set_signature_function_function( sigma_operator(f) );
};

//
// signature from explicit beta
//

//! \todo test that this does not contain multi blocks?
void signature_function::set_explicit_beta_distribution( shared_ptr<distribution> d ) {
  explicit_beta_distribution = d; set_type_explicit_beta();
};

shared_ptr<distribution> signature_function::get_explicit_beta_distribution() const {
  if (!has_type_initialized(signature_type::EXPLICIT))
    throw(string("is not of explicit beta type"));
  if (explicit_beta_distribution==nullptr) throw(string("has no explicit beta"));
  return explicit_beta_distribution;
};

/*!
  Create the parallel_indexstruct corresponding to the beta distribution.

  This outputs a new object, even if it is just copying a gamma distribution.

  \todo can we get away with passing the pointer if it's an explicit beta?
  \todo int he operator-based case, is the truncation properly multi-d?
  \todo we need to infer the distribution type
  \todo efficiency: do this only on local domains
*/
parallel_structure signature_function::derive_beta_structure
    (shared_ptr<distribution> gamma,shared_ptr<multi_indexstruct> truncation,bool trace) const {
  return derive_beta_structure(gamma,*(truncation.get()),trace);
};
parallel_structure signature_function::derive_beta_structure
    (shared_ptr<distribution> gamma,const multi_indexstruct &truncation,bool trace) const {
  parallel_structure beta_struct;
  auto decomp = gamma->get_decomposition();
  if (has_type_uninitialized()) {
    throw(string("Can not derive_beta_structure from uninitialized signature"));
  } else if (has_type_explicit_beta()) {
    throw(format("Derive beta for explicit should have been done in ensure_beta\n"));    
    const auto &x = get_explicit_beta_distribution();
    if (x->has_type(distribution_type::UNDEFINED))
      throw(string("Explicit beta should not have type undefined\n"));
    beta_struct = parallel_structure(x->get_decomposition()); // WRONG: this gives empty
    beta_struct.set_structure_type(x->get_type());
  } else { // case: pattern, operator, function
    beta_struct = parallel_structure( decomp );
    beta_struct.set_structure_type(distribution_type::GENERAL);
    if (has_type_operator_based()) { // case: operators
      if (trace) print(".. .. beta from operators\n");
      beta_struct = derive_beta_structure_operator_based(gamma,truncation);
    } else if (has_type_pattern()) { // case: sparsity pattern
      if (trace) print(".. .. beta from pattern\n");
      beta_struct = derive_beta_structure_pattern_based(gamma);
    } else if (has_type_function()) { // case: signature function explicit
      if (trace) print(".. .. beta from function\n");
      beta_struct.set_structure_type( gamma->get_type() );
      //parallel_structure *gamma_struct = dynamic_cast<parallel_structure*>(gamma);
      for (int p=0; p<gamma->domains_volume(); p++) {
	auto pcoord = gamma->coordinate_from_linear(p);
	auto gamma_p = gamma->get_processor_structure(pcoord);
	auto newstruct = gamma_p->operate(func);
	beta_struct.set_processor_structure( pcoord,newstruct );
      }
    } else throw(string("Can not derive beta for this type"));
  }
  //print("Derived beta structure: {}\n",beta_struct.as_string());
  return beta_struct;
};

parallel_structure signature_function::derive_beta_structure_operator_based
    (shared_ptr<distribution> gamma,shared_ptr<multi_indexstruct> truncation) const {
  return derive_beta_structure_operator_based(gamma,*(truncation.get())); };
parallel_structure signature_function::derive_beta_structure_operator_based
    (shared_ptr<distribution> gamma,const multi_indexstruct &truncation) const {
  const auto decomp = gamma->get_decomposition();
  parallel_structure beta_struct( decomp );
  beta_struct.set_structure_type( gamma->get_type() );
  if (gamma->is_known_globally()) {
    //print("Make globally known gamma\n");
    for (int p=0; p<gamma->domains_volume(); p++) {
      auto pcoord = gamma->coordinate_from_linear(p); // identical block below. hm.
      try {
	auto newstruct = make_beta_struct_from_ops
	  (pcoord,gamma->get_processor_structure(pcoord),get_operators(),truncation);
	if (newstruct->volume()==0)
	  throw(format("Somehow made empty beta struct for coord {}",pcoord.as_string()));
	if (newstruct->is_multi())
	  newstruct = newstruct->enclosing_structure();
	beta_struct.set_processor_structure( pcoord,newstruct );
      } catch (string c) { print("Error in beta_from_ops: {}\n",c);
	throw(format("Could not make_beta_from_ops for p={}\n",pcoord.as_string())) ; }
    }
    beta_struct.set_is_known_globally();
  } else {
    try {
      auto pcoord = gamma->proc_coord(/*decomp*/);
      //print("Make gamma only on {}\n",pcoord.as_string());
      auto newstruct = make_beta_struct_from_ops
	(pcoord,gamma->get_processor_structure(pcoord),get_operators(),truncation);
      if (newstruct->volume()==0)
	throw(format("Somehow made empty beta struct for coord {}",pcoord.as_string()));
      if (newstruct->is_multi())
	newstruct = newstruct->enclosing_structure();
      beta_struct.set_processor_structure( pcoord,newstruct );
    } catch (string c) {
      throw(format("Deriving beta struct operator based: {}",c));
    }
    beta_struct.set_is_known_globally(false);
  }
  //print("Made beta struct op_based: {}\n",beta_struct.as_string());
  return beta_struct;
};

//! \todo How much do we lose by taking that surrounding contiguous?
parallel_structure signature_function::derive_beta_structure_pattern_based
    (shared_ptr<distribution> gamma) const {
  const auto decomp = gamma->get_decomposition();
  parallel_structure beta_struct( decomp );
  if (gamma->get_dimensionality()>1)
    throw(string("Several bugs in beta from type pattern"));
  //snippet pstructfrompattern
  for (auto dom : gamma->get_domains()) {
    auto base = gamma->get_processor_structure(dom);
    shared_ptr<indexstruct> columns,simple_columns;
    try {
      columns = pattern->all_columns_from(base);
    } catch (string c) {
      throw(format("Could not get columns for domain {}: {}",dom.as_string(),c)); }
    try {
      simple_columns = columns->over_simplify();
    } catch (string c) {
      throw(format("Could not simplify columns for domain {}: {}",dom.as_string(),c)); }
    beta_struct.set_processor_structure
      ( dom,shared_ptr<multi_indexstruct>( new multi_indexstruct(simple_columns) ) );
  }
  //snippet end
  return beta_struct;
};

/*!
  For an operator-based \ref signature_function, create \f$ \beta(p) \f$ by union'ing
  the \ref indexstruct objects from applying the \ref ioperator objects contained 
  in #ops.

  For the case of modulo-based operators, the result is truncated. This 
  is a different kind of truncation than going on in distribution::messages_for_segment.

  An operator result is allowed to be empty, for instance from a shift followed by
  truncation.

 */
shared_ptr<multi_indexstruct> signature_function::make_beta_struct_from_ops
( processor_coordinate &pcoord, // this argument only for tracing
  shared_ptr<multi_indexstruct> gamma_struct,
  const vector<multi_ioperator*> &ops, shared_ptr<multi_indexstruct> truncation ) const {
  return make_beta_struct_from_ops(pcoord,gamma_struct,ops,*(truncation.get())); };
shared_ptr<multi_indexstruct> signature_function::make_beta_struct_from_ops
( processor_coordinate &pcoord, // this argument only for tracing
  shared_ptr<multi_indexstruct> gamma_struct,
  const vector<multi_ioperator*> &ops,
  const multi_indexstruct &truncation ) const {
  int dim = gamma_struct->get_dimensionality();
  if (ops.size()==0)
    throw(string("Somehow no operators"));
  if (gamma_struct->is_empty())
    throw(format("Finding empty processor structure"));

  auto halo_struct = shared_ptr<multi_indexstruct>( new empty_multi_indexstruct(dim) );
  for ( auto beta_op : ops ) {
    shared_ptr<multi_indexstruct> beta_struct;
    if (beta_op->is_modulo_op()) {
      beta_struct = gamma_struct->operate(beta_op);
    } else {
      beta_struct = gamma_struct->operate(beta_op,truncation);
    }
    if (!beta_struct->is_empty()) {
      halo_struct = halo_struct->struct_union(beta_struct);
    }
  }

  if (halo_struct->is_empty()) {
    memory_buffer w;
    format_to(w.end(),"Make empty beta struct from {} by applying:",gamma_struct->as_string());
    for ( auto o : ops )
      format_to(w.end()," {},",o->as_string());
    throw(to_string(w));
  }
  halo_struct = halo_struct->force_simplify();
  return halo_struct;
};

/****
 **** Dependency
 ****/

//! Most of the time we create a dependency and later set the signature function
dependency::dependency(shared_ptr<object> in) {
  in_object = in;
  set_name( format("dependency on object <<{}>>",in->get_name()) );
};

/*! Push a new object on to the dependencies list.
  This is extended in the kernel class to set the kernel type.
*/
void dependencies::add_in_object( shared_ptr<object> in ) {
  dependency d(in);
  the_dependencies.push_back(d);
};

shared_ptr<object> dependency::get_in_object() const {
  if (in_object==nullptr)
    throw(string("Dependency has no in object"));
  return in_object;
};

shared_ptr<object> dependencies::get_in_object(int b) const {
  return get_dependency(b).get_in_object();
};

bool dependency::has_beta_object() const {
  return beta_object!=nullptr;
};

void dependency::set_beta_object( shared_ptr<object> h ) {
  if (beta_object!=nullptr)
    throw(format("Can not override beta in dependency <<{}>>",get_name()));
  beta_object = h; };

shared_ptr<object> dependency::get_beta_object() const {
  if (!has_beta_object())
    throw(string("No beta object to be got"));
  return beta_object;
};

shared_ptr<object> dependencies::get_beta_object( int d ) {
  return get_dependency(d).get_beta_object();
};

bool dependency::get_is_collective() const {
  if (beta_object==nullptr)
    throw(format("Can not ask collective until beta object set"));
  return beta_object->get_distribution()->has_type_replicated();
};

/*!
  Allocate a halo based on a distribution.
  \todo should we immediately store this in the internal object?
  \todo delete the one with object argument

  We use the \ref distribution::new_object factory routine for creating
  the actual object because dependencies are mode-independent.
*/
// shared_ptr<object> dependency::create_beta_vector
//     (shared_ptr<object> out,shared_ptr<distribution> beta_distribution ) {
//   return create_beta_vector(beta_distribution);
// };
shared_ptr<object> dependency::create_beta_vector
    (const shared_ptr<distribution> &beta_distribution ) {
  auto halo = beta_distribution->new_object(beta_distribution); // gives a shared_ptr
  halo->allocate();
  halo->set_name(format("[{}]:halo",this->get_name()));
  return halo;
};

/*!
  Make sure that this dependency has a beta distribution.

  \todo is that mask on the beta actually used?
*/
shared_ptr<distribution> dependency::find_beta_distribution
    (shared_ptr<object> outvector,bool trace) const {
  if (0) {
  } else if (has_beta_object()) {
    return beta_object->get_distribution();
    // why throw?
    throw(format("Dependency already has beta object/distribution"));
  } else if (has_type_explicit_beta()) {
    auto return_beta = get_explicit_beta_distribution();
    if (trace)
      print(".. .. found beta from explicit: {}\n",return_beta->as_string());
    return return_beta;
  } else {
    auto invector = get_in_object();
    parallel_structure pstruct;
    try {
      auto alpha = invector->get_distribution(),
	gamma = outvector->get_distribution();
      pstruct = derive_beta_structure(gamma,alpha->get_enclosing_structure(),trace);
    } catch (string c) { print("Error in derive beta struct: {}\n",c);
      throw(format("Failed to derive beta struct for in={} [{}] out={} [{}]\n",
			invector->get_name(),invector->as_string(),
			outvector->get_name(),outvector->as_string()));
    } catch (...) { throw(string("derive beta struct: other error")); }
    pstruct.set_structure_type( pstruct.infer_distribution_type() );
    
    //snippet ensurebeta
    const auto indistro = invector->get_distribution(),
      outdistro = outvector->get_distribution();
    try {
      auto the_beta_distribution = outdistro->new_distribution_from_structure(pstruct);
      // if (outdistro->has_mask())
      // 	the_beta_distribution->add_mask(outdistro->get_mask());
      //snippet end
      the_beta_distribution->set_name(format("beta-for-<<{}>>",get_name()));
      //print("construct beta with k={}\n",indistro->get_orthogonal_dimension());
      the_beta_distribution->set_orthogonal_dimension
	( indistro->get_orthogonal_dimension() );
      return the_beta_distribution;
    } catch (string c) { print("Error in beta new_distro_from_str: <<{}>>\n",c);
      throw(format("Could not ensure_beta_distr for outvector {}",outvector->as_string()));
    }
  }
};

//! Test that all betas are defined on this domain
bool dependencies::all_betas_live_on( const processor_coordinate &d) const {
  bool live{true};
  for (auto dep : the_dependencies)
    live = live && dep.get_beta_object()->get_distribution()->lives_on(d);
  return live;
};

//! Test that all betas are filled on this domain
bool dependencies::all_betas_filled_on( const processor_coordinate &d) const {
  bool live{true};
  for (auto dep : the_dependencies)
    live = live && dep.get_beta_object()->get_data_is_filled(d);
  return live;
};

//! Get mutable reference to dependency
dependency &dependencies::set_dependency(int d) {
  if (d<0 || d>=the_dependencies.size())
    throw(string("Can not get that dependency"));
  return the_dependencies.at(d);
};

//! Get immutable reference to dependency
const dependency &dependencies::get_dependency(int d) const{
  if (d<0 || d>=the_dependencies.size())
    throw(string("Can not get that dependency"));
  return the_dependencies.at(d);
};

/*!
  Get a vector of beta objects to pass to the local execute function.
  We really a vector, but this is only constructed once (per task execution),
  so the waste is not too bad.
*/
vector<shared_ptr<object>> &dependencies::get_beta_objects() {
  return beta_objects_vector;
};

/****
 **** Task
 ****/

vector<shared_ptr<message>> &task::get_receive_messages() {
  return recv_messages;
};
vector<shared_ptr<message>> &task::get_send_messages()   {
  return send_messages;
};
void task::set_receive_messages( vector<shared_ptr<message>> &msgs) {
  auto msgs_copy = msgs;
  return set_receive_messages(std::move(msgs_copy));
};
void task::set_receive_messages( vector<shared_ptr<message>> &&msgs) {
  if (recv_messages.size()>0)
    throw(format("Task {} already has receive msgs when setting with {}",
		      get_name(),msgs.size()));
  recv_messages = msgs;
  //  has_recv_messages = true;
};
void task::set_send_messages( vector<shared_ptr<message>> &msgs) {
  if (send_messages.size()>0)
    throw(format("Task {} already has send msgs when setting with {}",
		      get_name(),msgs.size()));
  send_messages = msgs;
  //  has_send_messages = true;
};
//! Take the receive msgs so that you can really give them to another task
vector<shared_ptr<message>> task::lift_recv_messages() {
  auto msgs = recv_messages; recv_messages.clear(); //has_recv_messages = false;
  return msgs;
};
//! Take the send msgs so that you can really give them to another task
vector<shared_ptr<message>> task::lift_send_messages() {
  auto msgs = send_messages; send_messages.clear(); //has_send_messages = false;
  return msgs;
};

void task::add_post_messages( vector<shared_ptr<message>> &msgs ) {
  for ( auto m : msgs ) {
    post_messages.push_back(m);
  }
}

vector<shared_ptr<message>> &task::get_post_messages() {
  return post_messages;
};

void task::add_xpct_messages( vector<shared_ptr<message>> &msgs ) {
  for ( auto m : msgs ) 
    xpct_messages.push_back(m);
};

vector<shared_ptr<message>> &task::get_xpct_messages() {
  return xpct_messages;
};

vector<shared_ptr<task>> &task::get_node_tasks() {
  if (!node_queue->get_has_been_analyzed())
    throw("Can not get embedded tasks until analyzed\n");
  return node_queue->get_tasks();
};

/*!
  Find the corresponding send message for a receive message,
  otherwise return nullptr. This can only work for OpenMP:
  see \ref omp_request_vector::wait.
*/
shared_ptr<message> task::matching_send_message(shared_ptr<message> rmsg) {
  auto smsgs = get_send_messages();
  if (smsgs.size()==0)
    throw(format("Task <<{}>> somehow has no send messages",as_string()));
  for (auto smsg : smsgs) {
    //print("Comparing smsg=<<{}>> rmsg=<<{}>>\n",smsg->as_string(),rmsg->as_string());
    if ( smsg->get_sender()==rmsg->get_sender() &&
	 smsg->get_receiver()==rmsg->get_receiver() ) {
      return smsg;
    }
  }
  return nullptr;
};

/*!
  This does the following:
  - create the recv messages, locally by inspecting the beta structure
  - create the send message, which can be complicated
  - allocate the halo, for modes that need this
  - construct the list of predecessor tasks, given in step/domain coordinates

  \todo make a unit test for the predecessors
  \todo the send structure routine is pure virtual, so belongs to task; move to dependency?
  \todo see allocate_halo_vector: probably not store the halo in the dependency. but see kernel::analyze_dependencies
*/
void task::analyze_dependencies(bool trace) {
  int step = get_step(); auto dom = get_domain();
  if (trace) print("Analyzing task at step={}, domain={}\n",step,dom.as_string());
  if (get_has_been_analyzed()) {
    if (trace) print(".. was already analyzed\n");
    return; }
  if (!has_type_origin()) {
    if (trace)
      print(".. not origin: analyzing messages\n");
    auto out = get_out_object();

    // create receive messages, these will carry the in object number
    try {
      set_receive_messages( derive_receive_messages(trace) );
    } catch (string c) { print("Error <<{}>>\n",c);
      throw(format("Could not derive recv msgs for task {}",get_name())); }
    // attach in/out objects to messages
    for ( auto m : get_receive_messages() ) {
      int innum = m->get_in_object_number();
      // record predecessor relations
      predecessor_coordinates.push_back( new task_id(innum,m->get_sender()) );
    }
    // create the send message structure
    try { derive_send_messages(trace); } catch (string c) {
      print("Error <<{}>> during derive send messages\n",c);
      throw(format("Task analyze dependencies failed for <<{}>>",as_string()));
    }
  }
  try { local_analysis(); }
  catch (string c) { print("Error <<{}>> in <<{}>>\n",c,get_name());
    throw(format("Local analysis failed")); }
  set_has_been_analyzed();
};

vector<shared_ptr<message>> dependencies::derive_dependencies_receive_messages
    (const processor_coordinate &pid,int step,bool trace) {
  vector<shared_ptr<message>> messages;
  auto &deps = get_dependencies();
  for ( int id=0; id<deps.size(); id++) { // we need the dependency number in the message
    auto &d = deps.at(id);
    if (!d.has_beta_object()) { print("no beta at dependency {}\n",id);
      throw(string("dependency needs beta dist for create recv")); }
    auto invector = d.get_in_object(),
      halo = d.get_beta_object(); // pointers
    auto indistro = invector->get_distribution();
    //snippet msgsforbeta
    self_treatment doself;
    if ( indistro->has_collective_strategy(collective_strategy::MPI)
	 && d.get_is_collective() )
      doself = self_treatment::ONLY;
    else
      doself = self_treatment::INCLUDE;
    vector<shared_ptr<message>> msgs;
    try {
      msgs = indistro->messages_for_objects( step,pid,doself,invector,halo, trace );
    } catch (string c) { print("Error: {}\n",c);
      throw(format("Could not derive messages for beta {} on {}",
		   halo->as_string(),pid.as_string()));
    }
    if (trace) print("found {} recv messages\n",msgs.size());
    //snippet end
    for ( auto msg : msgs ) {
      try {
        msg->set_name( format("recv-msg-{}-{}",
				   invector->get_object_number(),halo->get_object_number()) );
	msg->set_dependency_number(id); msg->set_receive_type();
        try {
          msg->compute_tar_index();
        } catch ( string c ) {
          print("Error computing tar index <<{}>>\n",c);
          throw(format("Could not localize recv message {}->{}",
			    msg->get_sender().as_string(),msg->get_receiver().as_string()));
	}

        //msg->set_is_collective( d.get_is_collective() );
	//msg->add_trace_level( this->get_trace_level() );
        messages.push_back( msg );
      } catch(string c) { print("Error <<{}>>\n",c);
	throw(format("Could not process message {} for dep <<{}>> in task <<{}>>",
			  msg->as_string(),d.as_string(),this->as_string()));
      }
    }
  }
  if (trace) print(".. found {} receive messages\n",messages.size());
  return messages;  
};

/*! 
  The receive structure of a task is spread over its dependencies,
  so the global call simply loops over the dependencies.
*/
vector<shared_ptr<message>> task::derive_receive_messages(bool trace)
{
  int step = get_step(); auto pid = get_domain();
  if (trace) print("Deriving recv msgs for task at {}:{}\n",step,pid.as_string());
  return containing_kernel->derive_dependencies_receive_messages(pid,step,trace);
};

/*!
  Convert knowledge of what I receive into who is sending to me.
  This uses a collective of some sort; based on MPI we call this
  `reduce-scatter'.
  \todo this calls reduce_scatter once for each task. not right with domains
*/
int task::get_nsends() {
  auto out = get_out_object();
  // first get my list of senders
  auto outdistro = out->get_distribution();
  int ntids = outdistro->domains_volume(), nrecvs=0,nsends;
  vector<int> my_senders; my_senders.reserve(ntids);
  auto layout = outdistro->get_domain_layout();
  for (int i=0; i<ntids; ++i) my_senders.push_back(0);
  for ( auto msg : get_receive_messages() ) {
    int s = msg->get_sender().linearize(layout);
    if (s<0 || s>=ntids)
      throw(format("Invalid linear dom={} for layout <<{}>>",s,layout.as_string()));
    my_senders[s]++; nrecvs++;
  }

  // to invert that, first find out how many procs want my data
  try {
    int lineardomain = get_domain().linearize(layout);
    nsends = outdistro->reduce_scatter(my_senders.data(),lineardomain);
  } catch (string c) {
    print("Error <<{}>> doing reduce-scatter\n",c);
    throw(format("Task <<{}>> computing <<{}>>, failing in get_nsends",
		      get_name(),get_out_object()->as_string()));
  }

  return nsends;
};

dependency &task::find_dependency_for_object_number(int innum) {
  for ( auto &dep : get_dependencies() ) {
    auto obj = dep.get_in_object();
    if (obj->get_object_number()==innum) {
      return dep;
    }
  }
  throw(format("Could not find dependency {}",innum));
};

/*!
  Post all messages and store the requests.
  The \ref notifyReadyToSendMsg call is pure virtual and implemented in each mode.
*/
request_vector task::notifyReadyToSend( vector<shared_ptr<message>> &msgs ) {
  request_vector requests;
  for ( auto msg : msgs ) {
    if (msg->is_skippable()) { // collective & using MPI & not with self
      msg->set_status( message_status::SKIPPED );
      continue;
    } else {
      auto newreq = notifyReadyToSendMsg(msg);
      msg->set_status( message_status::POSTED );
      requests.add_request( newreq );
    }
  }
  return requests;
};

//snippet taskexecute
/*!
  Do task synchronization and local execution of a task.

  Masking is tricky.
  - Of course we skip execution if the output does not live on this task.
  - If we have output, we do sends and receives, but
  - we skip execution if any halos are missing.
 */
void task::execute(bool trace) {
  const auto d = get_domain();
  auto outvector = get_out_object();

  // allocate the output if not already; maybe it's embedded.
  try { outvector->allocate();
  } catch (string c) {
    throw(format("Error during outvector allocate: {}",c)); }

  if (get_has_been_executed()) {
    if (trace) print("{}: Task bypassing\n",get_name());
    return;
  }
  if (!outvector->get_distribution()->lives_on(d))
    return;

  if (trace) 
    print("[{}] Execute task {}: {}\n",
	  get_domain().as_string(),get_name(),as_string());

  if (!get_has_been_optimized()) {
    auto smsgs = get_send_messages();
    try { auto reqs = notifyReadyToSend(smsgs);
      requests.add_requests(reqs);
      if (trace)
	print("{}: |smsgs|={}, posting {}\n",get_name(),smsgs.size(),reqs.size());
    } catch (string c) { print("Error <<{}>> in notifyReadyToSend\n",c);
      throw(format("Send posting failed for <<{}>>",get_name()));
    }
    //    }
    auto rmsgs = get_receive_messages();
    try { auto reqs = acceptReadyToSend(rmsgs);
      requests.add_requests(reqs);
      if (trace)
	print("{}: |rmsgs|={}, posting {}\n",get_name(),rmsgs.size(),reqs.size());
    } catch (string c) { print("Error <<{}>> in acceptReadyToSend\n",c);
      throw(format("Recv posting failed for <<{}>>",get_name()));
    }
    //    }
  }
  if (trace) print("{}: #msgs posted={}\n",get_name(),requests.size());

  auto &objs = get_beta_objects();
  if (trace) print("{}: #beta is {}\n",get_name(),objs.size());
  if (!has_type_origin() && objs.size()==0)
    throw(format("Non-origin task <<{}>> has zero inputs",get_name()));
  if (true || all_betas_live_on(d)) {
    if (outvector->get_distribution()->get_split_execution()) {
      if (trace) print("{}: split execution 1\n",get_name());
      try { // the local part that can be done without communication.
	local_execute(objs,outvector,localexecutectx,unsynctest);
      } catch (string c) { print("Error <<{}>> in split exec1\n",c);
	throw(format("Task local execution failed before sync"));
      }
    }
  } else print("skip pre-exec for missing halo\n");

  if (trace)
    print("{}: Waiting for {} requests {}\n",get_name(),requests.size(),requests.as_string());
  try {
    requests_wait(requests);
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Task <<{}>> request wait failed",get_name()));
  } catch (...) {
    throw(format("Task <<{}>> request wait failed",get_name())); }
  record_nmessages_sent( requests.size() );

  if (true || (all_betas_live_on(get_domain()) && all_betas_filled_on(d))) {
    try { // for product this executes the embedded queue; otherwise just local stuff
      if (trace) print("{}: split execution 2\n",get_name());
      if (!outvector->get_distribution()->get_split_execution())
	local_execute(objs,outvector,localexecutectx);
      else
	local_execute(objs,outvector,localexecutectx,synctest);
    } catch (string c) { print("Error <<{}>> in split exec2\n",c);
      throw(format("Task <<{}>> local execution failed after sync",get_name()));
    } catch (std::out_of_range) {
      throw(format("Task <<{}>> local execution failed with out of range after sync",
			get_name()));
    } catch (...) {
      throw(format("Task <<{}>> local execution failed with unknown exception after sync",
			get_name()));
    }
  } // else print("skip pre-exec on {} for missing halo\n",d);
 
  // Post the sends and receives of the task that will receive our data.
  // The wait for these requests happens elsewhere
  {
    auto requests = notifyReadyToSend(get_post_messages());
    for ( auto r : requests.requests() )
      pre_request_add(r);
    requests = acceptReadyToSend(get_xpct_messages());
    for ( auto r : requests.requests() )
      pre_request_add(r);
  }

  result_monitor(this->shared_from_this());
  set_has_been_executed();

}
//snippet end

/*!
  Conditional execution with an all-or-nothing test. 
  In the product mode we override this with a fine-grained test.
*/
void task::local_execute
    (vector<shared_ptr<object>> &beta_objects,shared_ptr<object> outobj,void *ctx,
     int(*tasktest)(shared_ptr<task> t)) {
  if (localexecutefn==nullptr)
    throw(format("Hm. no local function...."));
  if (outobj->has_data_status_unallocated())
    throw(format("Object <<{}>> has no data",outobj->get_name()));
  if ( tasktest==nullptr || (*tasktest)(this->shared_from_this()) ) {
    int s = get_step_counter(); //get_step(); 
    auto p = get_domain();
    double flopcount = 0.;
    try {
      //print("{} call function pointer\n",as_string());
      localexecutefn(s,p,beta_objects,outobj,&flopcount);
    } catch (string c) {
      throw(format("Task tested local exec <<{}>> failed: {}",get_name(),c));
    }
    set_flop_count(flopcount);
  }
};

//! Unconditional local execution.
void task::local_execute
    (vector<shared_ptr<object>> &beta_objects,shared_ptr<object> outobj,void *ctx) {
  if (localexecutefn==nullptr)
    throw(format("Hm. no local function...."));
  {
    int s = get_step_counter(); 
    auto p = get_domain();
    double flopcount = 0.;
    if (!p.get_same_dimensionality(outobj->get_distribution()->get_dimensionality()))
      throw(format("Random sanity check: p={}, obj={}",
			p.as_string(),outobj->get_name()));
    try {
      //print("{} call function pointer\n",as_string());
      localexecutefn(s,p,beta_objects,outobj,&flopcount);
    } catch (string c) { 
      throw(format("Task untested local exec <<{}>> failed: {}",get_name(),c));
    }
    set_flop_count(flopcount);
  }
};

void task::clear_has_been_executed() {
  done = 0;
  for ( auto m : get_send_messages() )
    m->clear_was_sent();
  for ( auto m : get_receive_messages() )
    m->clear_was_sent();
  requests.requests().clear();
};

/*!
  Check that all tasks have been executed exactly one.
 */
int algorithm::get_all_tasks_executed() {
  int all = 1;
  for ( auto t : get_tasks() )
    if (!t->get_has_been_executed()) { all = 0; break; }
  return allreduce_and(all);
};
int algorithm::get_all_msgs_completed() {
  int all = 1;
  for ( auto t : get_tasks() )
    if (!t->all_requests_completed()) { all = 0; break; }
  return allreduce_and(all);
};

/*!
  If we want to re-execute an algorithm, we need to clear the executed status.
*/
void algorithm::clear_has_been_executed() {
  for ( auto t : get_tasks() ) {
    if (!t->has_type_origin())
      t->clear_has_been_executed();
  }
};

string task::as_string() {
  memory_buffer w;
  format_to(w.end(),"{}[s={},p={}",get_name(),get_step(),get_domain().as_string());
  if (get_is_synchronization_point()) format_to(w.end(),",sync");
  if (has_type_origin())              format_to(w.end(),",origin");
  format_to(w.end(),"]");
  if (!has_type_origin()) {
    auto preds = get_predecessors();
    format_to(w.end(),", #preds={}: [",preds.size());
    for (auto p : preds )
      format_to(w.end(),"{} ",p->get_name());
    format_to(w.end()," ]");
  }
  auto rmsgs = get_receive_messages();
  if (rmsgs.size()>0) {
    format_to(w.end(),"\nReceive msgs:");
    for ( auto m : rmsgs)
      format_to(w.end()," {}",m->as_string());
  }
  auto smsgs = get_send_messages();
  if (smsgs.size()>0) {
    format_to(w.end(),"\nSend msgs:");
    for ( auto m : smsgs)
      format_to(w.end()," {}",m->as_string());
  }
  return to_string(w);
};

/****
 **** Kernel
 ****/

//! Pre-basic constructor
kernel::kernel() : entity(entity_cookie::KERNEL) {};

//! Origin kernel only has an output object
kernel::kernel( std::shared_ptr<object> out ) : kernel() {
  out_object = out;
  type = kernel_type::ORIGIN; //set_name("origin kernel");
};

/*!
  We get a vector of \ref task objects from \ref kernel::split_to_tasks;
  here is where we store it. See also \ref kernel::addto_kernel_tasks.
  \todo make reference
*/
void kernel::set_kernel_tasks(vector< shared_ptr<task> > tt) {
  if (kernel_has_tasks())
    throw(format("Can not set tasks for already split kernel <<{}>>",get_name()));
  addto_kernel_tasks(tt); was_split_to_tasks = 1;
};

/*!
  With composite kernels each of the component kernels makes a vector of tasks;
  we gradually add them to the surrounding kernel.
  \todo make reference
*/
void kernel::addto_kernel_tasks(vector< shared_ptr<task> > tt) {
  for (auto t : tt )
    kernel_tasks.push_back(t);
  was_split_to_tasks = 1;
};

const vector< shared_ptr<task> > &kernel::get_tasks() const {
  if (!kernel_has_tasks())
    throw(format("Kernel <<{}>> was not yet split to tasks",get_name()));
  return kernel_tasks;
};

/*!
  Analyzing kernel dependencies is mostly delegating the analysis
  to the kernel tasks. By dependencies we mean dependency on other kernels
  that originate the input objects for this kernel. 

  Origin tasks have no dependencies, so we skip them. Note that 
  ultimately they may still have outgoing messages; however
  that is set after queue optimization.
  \todo the ensure_beta_distribution call is also in task::analyze_dependencies. lose this one?
*/
void kernel::analyze_dependencies(bool trace) {
  if (get_has_been_analyzed()) {
    if (trace) print(".. was already analyzed\n");
    return;
  }

  int step = get_step();
  if (trace) print("Analyzing kernel <<{}>> at step={}\n",get_name(),step);
  if (!kernel_has_tasks()) {
    if (trace) print(".. first splitting to tasks\n");
    this->split_to_tasks(trace);
  }
  auto outobject = get_out_object();
  if (!has_type_origin()) {
    if (trace)
      print(".. not origin: making beta for {} dependencies\n",get_dependencies().size());
    endow_beta_objects(outobject,trace);
    if (the_dependencies.size()!=beta_objects_vector.size())
      throw(format("Beta objects missing: {} s/b {}\n",
			beta_objects_vector.size(),the_dependencies.size()));
    for (auto d : get_dependencies())
      if (!d.has_beta_object())
	throw(format("dependency is missing beta object"));
  }
  auto &tsks = this->get_tasks();
  if (trace) print(".. analyzing {} tasks\n",tsks.size());
  for ( auto &t : tsks ) {
    t->analyze_dependencies(trace);
  }
  set_has_been_analyzed();
};

/*! 
  Create beta object if not already there.
  This is called by analyze_dependencies
*/
void dependency::endow_beta_object(shared_ptr<object> out,bool trace) {
  if (has_beta_object()) {
    if (trace) print(".. dependency <<{}>> already has beta object\n",as_string());
  } else {
    if (trace) print(".. dependency <<{}>> creating beta object\n",as_string());
    auto the_beta_distribution = find_beta_distribution(out,trace);
    auto halo = create_beta_vector(the_beta_distribution);
    set_beta_object(halo);
  }
};

void kernel::endow_beta_objects(shared_ptr<object> out,bool trace) {
  for (auto &d : get_dependencies() ) {
    try {
      d.endow_beta_object(out,trace);
      auto halo = d.get_beta_object();
      beta_objects_vector.push_back(halo);
    } catch (string c) { print("Error <<{}>>\n",c);
      throw(format("Could not ensure beta in kernel<<{}>> for dependency <<{}>>",
			get_name(),d.get_name()));
    }
  }
};

/*!
  We generate a vector of tasks, which inherit a bunch of things from the surrounding kernel
  - the beta definition
  - the local function
  - the function context
  - the name (plus a unique id)
  of the kernel.
*/
void kernel::split_to_tasks(bool trace) {
  if (kernel_has_tasks()) return;
  auto outvector = get_out_object();
  const auto outdistro = outvector->get_distribution();
  kernel_tasks.reserve( outdistro->domains_volume() );

  auto domains = outdistro->get_domains();
  if (domains.size()==0) printf("WARNING zero domains in kernel <<%s>>\n",get_name().data());
  if (tracing_progress())
    print("Kernel <<{}>> splitting to {} domains\n",get_name(),domains.size());
  for (auto dom : domains) {
    if (trace) print(".. make task for domain {}\n",dom.as_string());
    auto t = make_task_for_domain(this,dom);
    t->find_kernel_task_by_domain =
      [&,outvector] (processor_coordinate &d) -> shared_ptr<task> {
      return get_tasks().at( d.linearize(outvector->get_distribution()->get_decomposition()) );
    };
    t->set_name( format("T[{}]-{}-{}",this->get_name(),
			     this->get_step(),dom.as_string()) );
    if (trace) print(".. now pushing task\n");
    kernel_tasks.push_back(t);
  }
  if (kernel_tasks.size()==0)
    print("Suspiciously zero tasks\n");
  was_split_to_tasks = 1;
};

void kernel::analyze_contained_kernels( shared_ptr<kernel> k1,shared_ptr<kernel> k2,bool trace ) {
  this->split_to_tasks();
  k1->analyze_dependencies();
  k2->analyze_dependencies(trace);
};

void kernel::analyze_contained_kernels( shared_ptr<kernel> k1,shared_ptr<kernel> k2,shared_ptr<kernel> k3,bool trace ) {
  this->split_to_tasks();
  k1->analyze_dependencies(trace);
  k2->analyze_dependencies(trace);
  k3->analyze_dependencies(trace);
};

void kernel::split_contained_kernels( shared_ptr<kernel> k1,shared_ptr<kernel> k2,bool trace ) {
  if (kernel_has_tasks()) return;
  try {
    k1->split_to_tasks(trace);  set_kernel_tasks( k1->get_tasks() );
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Fail to split kernel <<{}>>",k1->get_name())); }
  try {
    k2->split_to_tasks(trace); addto_kernel_tasks( k2->get_tasks() );
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Fail to split kernel <<{}>>",k2->get_name())); }
};

void kernel::split_contained_kernels( shared_ptr<kernel> k1,shared_ptr<kernel> k2,shared_ptr<kernel> k3,bool trace ) {
  if (kernel_has_tasks()) return;
  try {
    k1->split_to_tasks(trace);  set_kernel_tasks( k1->get_tasks() );
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Fail to split kernel <<{}>>",k1->get_name())); }
  try {
    k2->split_to_tasks(trace); addto_kernel_tasks( k2->get_tasks() );
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Fail to split kernel <<{}>>",k2->get_name())); }
  try {
    k3->split_to_tasks(trace); addto_kernel_tasks( k3->get_tasks() );
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Fail to split kernel <<{}>>",k3->get_name())); }
};

/*!
  Kernel tasks are independent, so why have a kernel execute function that
  executes the tasks of that kernel in sequence. For MPI and BSP purposes
  this suffices; however, for general task models we need to execute
  tasks in the context of a task queue.
*/
void kernel::execute(bool trace) {
  if (kernel_tasks.size()==0)
    throw(string("kernel should have tasks"));
  auto tsks = get_tasks();
  if (trace)
    print("Kernel {} executing {} tasks\n",get_name(),tsks.size());
  for ( auto t : tsks ) {
    if (t==nullptr)
      throw(format("Finding nullptr to task in kernel: {}",as_string()));
    t->execute(trace);
  }
};

int kernel::get_all_msgs_completed() {
  int all = 1;
  for ( auto t : get_tasks() )
    if (!t->all_requests_completed()) { all = 0; break; }
  return get_out_object()->get_distribution()->allreduce_and(all);
};

string kernel::as_string() const { memory_buffer w;
  auto o = get_out_object();
  format_to(w.end(),"K[{}]-out:<<{}#{}>>",get_name(),o->get_name(),
	  o->get_distribution()->global_volume());
  const auto& deps = get_dependencies();
  for ( const auto& d : deps ) { //.begin(); d!=deps.end(); ++d) {
    const auto& o = d.get_in_object();
    format_to(w.end(),"-in:<<{}#{}>>",o->get_name(),o->get_distribution()->global_volume());
  }
  return to_string(w);
};

//! Count how many messages the tasks in this kernel receive.
int kernel::local_nmessages() {
  auto tsks = get_tasks();
  int nmessages = 0;
  for ( auto t : tsks ) {// (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    nmessages += t->get_receive_messages().size();
  }
  return nmessages;
}

/****
 **** Queue
 ****/

/*!
  Add a kernel to the internal list of kernels. This is a user
  level routine, and does not involve any sort of analysis, which
  waits until algorithm::analyze_dependencies.
 */
void algorithm::add_kernel(shared_ptr<kernel> k) {
  k->set_step_counter( get_kernel_counter_pp() );
  //  print("adding kernel as count={}\n",k->get_step_counter());
  all_kernels.push_back( k );
};

/*! 
  Push the tasks of a kernel onto this queue. This is an internal routine
  called from algorithm::analyze_dependencies.
  This uses an overridable kernel::get_tasks

  \todo is this the right way, or should it be kernel::addtasks( tasks_queue* ) ?
*/
void algorithm::add_kernel_tasks_to_queue( shared_ptr<kernel> k ) {
  for ( auto t : k->get_tasks() ) {
    this->add_task(t);
  }
};

void algorithm::add_task( shared_ptr<task> &t ) {
  tasks.push_back(t);
};

shared_ptr<task>& algorithm::get_task(int n) {
  if (n<0 || n>=tasks.size()) throw(string("Invalid task number; forgot origin kernel?"));
  return tasks.at(n); };

//! Queue split to tasks is splitting all kernels. Nothing deeper than that.
void algorithm::split_to_tasks() {
  if (get_has_been_split_to_tasks()) return;
  for ( auto k : get_kernels() ) {
    k->split_to_tasks();
    for ( auto t : k->get_tasks() ) {
      this->add_task(t);
      t->find_other_task_by_coordinates =
	[&] (int s,processor_coordinate &d) -> shared_ptr<task>{
	return find_task_by_coordinates(s,d); };
      t->record_flop_count =
	[&] (double c) { record_flop_count(c); };
    }
  }
  set_has_been_split_to_tasks();
};

/*!
  We analyze a task queue by analyzing all its kernels, which recursively
  analyzes their tasks. The tasks are inserted into the queue.

  \todo increase message_tag_admin_threshold by max kernel number
  \todo figure out how to do the kernels dot file here.
 */
void algorithm::analyze_dependencies(bool trace) {
  if (get_has_been_analyzed()) throw(string("Can not analyze twice"));
  analysis_event.begin();
  try {
    if (trace) print("Algorithm: splitting {} kernels to tasks\n",get_kernels().size());
    split_to_tasks();
    if (trace) print("Algorithm: analyzing dependencies for {} tasks\n",global_ntasks());
    analyze_kernel_dependencies(trace);
    find_predecessors(); // make task-task graph
    find_exit_tasks();
    mode_analyze_dependencies(); // OMP: split into locally executable & not
    if (get_can_message_overlap()) optimize();
    for ( auto t : get_tasks() ) {
      t->set_sync_tests(unsynctest,synctest); // Product: inherit tests on sync execution
      // immediately throw exception if circular?
      if (t->check_circularity(t)) {
	print("Task <<{}>> root of circular dependency path\n",t->get_name());
	set_circular();
      }
    }
  } catch (string c) { // at top level print and throw int
    print("Queue analysis failed: {}\n{}\n",c,this->contents_as_string()); throw(-1);
  }
  if (do_optimize) {
    if (trace) print("Algorithm: optimizing\n");
    optimize();
  }
  set_has_been_analyzed();
  analysis_event.end();
  if (get_trace_summary())
    print("Queue contents:\n{}\n",this->contents_as_string());
};

/*!
  We analyze the kernels, doing local analysis between kernel and predecessors.
  Recursive global analysis is done in \ref algorithm::analyze_dependencies.
*/
void algorithm::analyze_kernel_dependencies(bool trace) {
  for ( auto k : get_kernels() ) {
    k->analyze_dependencies(trace);
    for ( auto d : k->get_dependencies() ) {
      d.get_in_object()->set_has_successor();
    };
  }
};

/*! 
  Find the tasks that have no successor. Traversing all tasks twice is no big deal.
*/
void algorithm::find_exit_tasks() {
  for ( auto t : get_tasks() ) {
    for ( auto p : t->get_predecessors() ) {
      p->set_is_non_exit();
    }
  }
  for ( auto t : get_tasks() ) {
    if (!t->is_non_exit()) {
      exit_tasks.push_back(t);
    }
  }
};

/*!
  Depth-first checking if a root task appears in its tree of dependencies.
*/
int task::check_circularity( shared_ptr<task> root ) {
  if (has_type_origin() || is_not_circular()) {
    return 0;
  } else {
    for ( auto t : get_predecessors() ) {
      //(auto t=get_predecessors()->begin(); t!=get_predecessors()->end(); ++t) {
      if (t->get_step()==root->get_step() && t->get_domain()==root->get_domain()) {
	print("Task <<{}>> is tail of circular dependency path\n",t->get_name());
	return 1;
      } else {
	int c = t->check_circularity(root);
	if (c) {
	  print("Task <<{}>> is on circular dependency path\n",t->get_name());
	  return 1;
	}
      }
    }
    set_not_circular();
  }
  return 0;
};

/*!
  Find a task number in the queue given its step/domain coordinates.
*/
shared_ptr<task> algorithm::find_task_by_coordinates
    ( int s,const processor_coordinate &d ) const {
  for ( auto t : tasks ) {
    if (t->get_step()==s && t->get_domain()==d)
      return t;
  }
  throw(format("Could not find task by coordinates {},{}",s,d.as_string()));
};

/*!
  Find a task number in the queue given its step/domain coordinates.
*/
int algorithm::find_task_number_by_coordinates( int s,const processor_coordinate &d ) const {
  for (int it=0; it<tasks.size(); it++) {
    auto t = tasks.at(it);
    if (t->get_step()==s && t->get_domain()==d) //.equals(d))
      return it;
  }
  throw(string("Could not find task number by coordinate"));
};

/*!
  For each task find its predecessor ids.
  Every task already has the 
  predecessors by step/domain coordinate; here we convert that to linear
  coordinates. 

  Note that tasks may live on a different address space. Therefore the 
  \ref declare_dependence_on_task routine is pure virtual. The MPI version 
  is interesting.
 */
void algorithm::find_predecessors() {
  //  for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
  for ( auto t : tasks ) {
    auto task_ids = t->get_predecessor_coordinates();
    for ( auto id : task_ids ) {
      try {
	t->declare_dependence_on_task( id );
      } catch (string c) {
	print("Error: {}\nQueue: {}\n",c,this->header_as_string());
	throw(format("Queue problem"));
      }
    }
  };
};

/*!
  After the standard analysis
  we split tasks in ones that are a synchronization point
  or dependent on one, and ones that are not and therefore can be
  executed locally without synchronizing in hybrid context.
*/
void algorithm::determine_locally_executable_tasks() {
  int nsync = 0;
  for ( auto t : get_tasks() ) {
    t->check_local_executability();
    task_local_executability lsync = t->get_local_executability();
    nsync += lsync==task_local_executability::NO;
  } 
};

/*!
  We declare the left and right OpenMP task in the origin kernel 
  to be a synchronization point. It's a first order approximation.
 */
void algorithm::set_outer_as_synchronization_points() {
  if (!get_has_been_split_to_tasks())
    split_to_tasks();
  //auto tsks = get_tasks();
  int cnt=0, org=0;
  for ( auto t : get_tasks() ) {
    if (t->has_type_origin()) { org++;
      auto d = t->get_domain();
      if (d.is_on_face( t->get_out_object() )) {
        t->set_is_synchronization_point(); cnt++;
      }
    }
  } 
  set_has_synchronization_tasks(cnt);
};

/*!
  Count the number of synchronization tasks,
  or return the number if this has already been determined.
 */
int algorithm::get_has_synchronization_tasks() {
  if (has_synchronization_points<0) {
    int count = 0;
    for ( auto t : get_tasks() ) {
      count += t->get_is_synchronization_point();
    }
    set_has_synchronization_tasks(count);
  }
  return has_synchronization_points;
};

//snippet queueoptimize
/*!
  We perform an optimization on the queue to get messages posted as early as possible.
  This is every so slightly tricky.

  For now we don't do this automatically: it has to be called by the user.
  \todo Is there a design flaw that we assume that each output is used only once?
 */
void algorithm::optimize() {
  for ( auto t : tasks ) {
    if (!t->has_type_origin()) {
      auto domain = t->get_domain();
      auto deps = t->get_dependencies();
      for ( auto d : deps ) {
	int originkernel = d.get_in_object()->get_object_number();
	auto origintask = find_task_by_coordinates(originkernel,domain);
	{ auto lift = t->lift_send_messages();
	  origintask->add_post_messages(lift); }
	{ auto lift = t->lift_recv_messages();
	  origintask->add_xpct_messages(lift); }
	origintask->pre_request_add =
	  [origintask,t] (shared_ptr<request> r) -> void {
	  // print("Task [{}]] receiving additional request for data from {}\n",
	  // 	     t->get_name(),origintask->get_name());
	  t->add_request(r);
	};
	t->set_has_been_optimized();
      }
    }
  }
}
//snippet end

/*!
  For a given object, find the kernel that produces it, and all the kernels
  that consume it. We throw an exception if we find more than one source.

  \todo is there a use for this?
*/
void algorithm::get_data_relations 
    (string object_name,
     vector<shared_ptr<kernel>> &sources,
     vector<shared_ptr<kernel>> &targets
     ) const {
  for ( auto k : all_kernels ) {
    if ( object_name==k->get_out_object()->get_name() ) {
      sources.push_back(k);
    }
    if ( object_name==k->last_dependency().get_in_object()->get_name() ) {
      targets.push_back(k);
    }
  }
};

/*!
  The base method for executing a queue is only a timer around 
  a method that executes the tasks. The timer calls are virtual
  because they require different treatment between MPI/OpenMP.
 */
void algorithm::execute( int(*tasktest)(shared_ptr<task> t),bool trace ) {
  if (tasktest==nullptr) throw(format("Missing task test in queue::execute"));
  if (is_circular()) throw(format("Can not execute queue with circular dependencies\n"));
  execution_event.begin();
  try {
    execute_tasks(tasktest,trace);
  } catch (string c) { print("Error <<{}>> in queue <<{}>> execute\n",c,get_name());
    throw(format("Task execute failed"));
  }
  execution_event.end();
  //print("Algorithm runtime: {}\n",execution_event.get_duration());
};

/*!
  Execute all tasks in a queue. Since they may not be in the right order,
  we take each as root and go down their predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.

  This is the base method; it suffices for MPI, but OpenMP overrides it by
  setting some directives before calling the base method.

  \todo move the first two lines to algorithm::execute, argument to this should be vector; check OMP first!
*/
void algorithm::execute_tasks( int(*tasktest)(shared_ptr<task> t),bool trace ) {
  if (tasktest==nullptr)
    throw(format("Missing task test in queue::execute_tasks"));
  for ( auto t : get_exit_tasks() ) {
    try {
      if ((*tasktest)(t))
	t->execute_as_root();
    } catch ( string c ) {
      print("Error <<{}>> for task <<{}>> execute\n",c,t->as_string());
      throw(string("Task queue execute failed"));
    }
  }
};

/*!
  Execute a task by taking it as root and go down the predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.
*/
void task::execute_as_root(bool trace) {
  if (get_has_been_executed()) {
    if (trace) print("Task already executed: {}\n",as_string());
    return;
  }

  if (trace) print("Executing task: {}\n",as_string());
  for ( auto t : get_predecessors() ) {
    if (trace)
      print(".. executing predecessor: {}\n",t->as_string());
    t->execute_as_root();
  }
  if (trace) print(".. local execution: {}\n",as_string());
  try {
    execute();
  } catch (string c) { print("Task execute error: {}\n",c);
    throw(format("Task {} execute failed",get_name()));
  } catch (...) {
    throw(format("Task {} execute failed",get_name()));
  }
};

/*!
  We try to embed vectors in the halo built on them.
  For now, this works in MPI only, called from \ref mpi_algorithm::mode_analyze_dependencies. 
  In OpenMP we have to think much harder.

  \todo how does this relate to multiple domains?
*/
void algorithm::inherit_data_from_betas() {
  for ( auto t : get_tasks() ) {
    auto deps = t->get_dependencies(); auto p = t->get_domain();
    for ( auto d : deps ) {
      auto in = d.get_in_object(),
	halo = d.get_beta_object();
      if (!in->has_data_status_allocated()) {
	auto embeddable = in->get_distribution()->get_processor_structure(p),
	  embedder = halo->get_distribution()->get_processor_structure(p);
	if (embedder->contains(embeddable)) {
	  index_int offset = embeddable->linear_location_in(embedder);
	  in->inherit_data(p,halo,offset);
	  in->set_data_parent(halo->get_object_number());
	} else  {
	  in->allocate();
	}
      }
    }
  }
};

//! \todo find a way to make the #tasks a global statistic
string algorithm::header_as_string() {
  return format("Queue <<{}>> protocol: {}; #tasks={}",
		     get_name(),protocol_as_string(protocol),global_ntasks());
};

string algorithm::kernels_as_string() {
  memory_buffer w;
  format_to(w.end(),"Kernels:");
  auto kernels = get_kernels();
  for (auto k : kernels ) 
    format_to(w.end(),"<<{}>>\n",k->as_string());
  format_to(w.end(),"]");
  return to_string(w);
};

string algorithm::contents_as_string() {
  memory_buffer w;
  auto tsks = get_tasks();
  format_to(w.end(),"Tasks [ ");
  for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    format_to(w.end(),"<<{}>> ",(*t)->header_as_string());
  }
  format_to(w.end(),"]");
  for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    format_to(w.end(),"\n");
    format_to(w.end(),"{}: predecessors ",(*t)->header_as_string());
    auto preds = (*t)->get_predecessor_coordinates();
    for ( auto p : preds ) //->begin(); p!=preds->end(); ++p)
      format_to(w.end(),"{}, ",p->as_string());
  }
  return to_string(w);
};

  // vector<shared_ptr<message>> messages;
  // auto &deps = get_dependencies();
  // for ( int id=0; id<deps.size(); id++) { // we need the dependency number in the message
  //   auto &d = deps.at(id);
  //   if (!d.has_beta_object()) { print("no beta at dependency {}\n",id);
  //     throw(string("dependency needs beta dist for create recv")); }
  //   auto invector = d.get_in_object(),
  //     halo = d.get_beta_object(); // pointers
  //   auto indistro = invector->get_distribution(),
  //     beta_dist = halo->get_distribution();
  //   int ndomains = indistro->domains_volume();
  //   //snippet msgsforbeta
  //   auto beta_block = beta_dist->get_processor_structure(pid);
  //   auto numa_block = beta_dist->get_numa_structure(); // to relativize against
  //   self_treatment doself;
  //   if ( indistro->has_collective_strategy(collective_strategy::MPI)
  // 	 && d.get_is_collective() )
  //     doself = self_treatment::ONLY;
  //   else
  //     doself = self_treatment::INCLUDE;
  //   vector<shared_ptr<message>> msgs;
  //   try {
  //     msgs = indistro->messages_for_segment( pid,doself,beta_block,numa_block,trace );
  //   } catch (string c) { print("Error: {}\n",c);
  //     throw(format("Could not derive messages for segment {} on numa {}",
  // 			beta_block->as_string(),numa_block.as_string()));
  //   }
  //   if (trace) print("found {} recv messages\n",msgs.size());
  //   //snippet end
  //   for ( auto msg : msgs ) {
  //     try {
  //       msg->set_name( format("recv-msg-{}-{}",
  // 				   invector->get_object_number(),halo->get_object_number()) );
  // 	msg->set_in_object(invector); msg->set_out_object(halo);
  // 	msg->set_dependency_number(id); msg->set_receive_type();
  //       try {
  //         msg->compute_tar_index();
  //       } catch ( string c ) {
  //         print("Error computing tar index <<{}>>\n",c);
  //         throw(format("Could not localize recv message {}->{}",
  // 			    msg->get_sender().as_string(),msg->get_receiver().as_string()));
  // 	}
  //       msg->set_tag_from_kernel_step(step,ndomains);
  //       //msg->set_is_collective( d.get_is_collective() );
  // 	msg->add_trace_level( this->get_trace_level() );
  //       messages.push_back( msg );
  //     } catch(string c) { print("Error <<{}>>\n",c);
  // 	throw(format("Could not process message {} for dep <<{}>> in task <<{}>>",
  // 			  msg->as_string(),d.as_string(),this->as_string()));
  //     }
  //   }
  // }
  // if (trace) print(".. found {} receive messages\n",messages.size());
  // return messages;
