/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2022
 ****
 **** imp_decomp.cxx: Implementations of the decomposition base classes
 ****
 ****************************************************************/

#include "imp_decomp.h"
#include <cassert>

#include <iostream>
using std::cout;
#include <numeric>
using std::accumulate;
#include <functional>
using std::multiplies;

using std::array,std::vector,std::string;
using std::shared_ptr,std::make_shared;
using fmt::format,fmt::print;

/*
 * Parallel structure
 */
template<int d>
parallel_structure<d>::parallel_structure( const coordinate<int,d>& procs )
  : procs(procs),
    structs( vector<shared_ptr<indexstruct>>(procs.span(),nullptr) ) {
};

template<int d>
parallel_structure<d>& parallel_structure<d>::from_global
( const coordinate<index_int,d>& points) {
  int P = procs.span(), N = points.span();
  //print("Splitting {} over {}\n",N,P);
  for ( int ip=0; ip<P; ip++ ) {
    index_int lo = (ip*N)/P, hi = ((ip+1)*N)/P;
    auto contig = make_shared<contiguous_indexstruct>(lo,hi-1);
    assert( contig->volume()==hi-lo );
    structs.at(ip) = shared_ptr<indexstruct>(contig);
  }
  return *this;
};

template<>
shared_ptr<indexstruct> parallel_structure<1>::get_processor_structure(int p) const {
  try {
    return structs.at(p);
  } catch (...) {
    throw(format("Error returning proc {}",p));
  }
};

template<int d>
parallel_structure<d> parallel_structure<d>::operate( ioperator op ) const {
  auto return_structure(*this);
  for ( auto& s : return_structure.structs )
    s = s->operate(op);
  return return_structure;
};

template<int d>
coordinate<index_int,d> parallel_structure::enclosing_structure() const {
};

#if 0
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
coordinate architecture::get_proc_origin(int d) const {
  return coordinate( vector<int>(d,0) );
};

//! Multi-d descriptor of number of processes. This is actually the highest proc number.
//! \todo find a much better way of deducing or setting a processor grid
coordinate architecture::get_proc_endpoint(int d) const {
  int P = nprocs()*get_over_factor();
  return coordinate( make_endpoint(d,P) );
 //  coordinate *coord;
 //  if (d==1) {
 //    coord = new coordinate(1); coord->set(0,P-1);
 //  } else if (d==2) {
 //    coord = new coordinate(2);
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
coordinate architecture::get_proc_layout(int dim) const {
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
#if 0
  format_to(w.end(),"{}",get_name());
  format_to(w.end(),", protocol: {}",protocol_as_string(protocol));
  format_to(w.end(),", collectives: {}",strategy_as_string());
  if (get_split_execution()) format_to(w.end(),", split execution");
  if (has_random_sourcing()) format_to(w.end(),", source randomization");
  if (get_can_embed_in_beta()) format_to(w.end(),", object embedding");
  if (get_can_message_overlap()) format_to(w.end(),", messages post early");
  if (algorithm::do_optimize) format_to(w.end(),", task graph optimized");
  return to_string(w);
#endif
  return "architecture";
};

std::string architecture::as_string() const {
  return format
    ("Type: {}, #procs: {}",architecture_type_as_string(type),arch_nprocs);
};


/****
 **** Decomposition
 ****/

//! Default decomposition uses all procs of the architecture in one-d manner.
template<int d>
decomposition::decomposition( const environment& env )
  : decomposition( endpoint<int,d>(environment.nprocs()) ) {
};

/*!
  In multi-d the user needs to indicate how the domains are laid out.
  The processor coordinate is a size specification, to make it compatible with nprocs
*/
//snippet decompfromcoord
template<int d>
decomposition::decomposition( const architecture &arch,coordinate &sizes )
  : architecture(arch) {
  int dim = sizes.get_dimensionality();
  if (dim<=0)
    throw(string("Non-positive decomposition dimensionality"));
  domain_layout = sizes; //new coordinate(sizes);
  set_corners();
};
//snippet end

//! Get dimensionality.
template<int d>
int decomposition::get_dimensionality() const {
  return d;
};

//! Number of domains
template<int d>
int decomposition::domains_volume() const {
  int p = domain_layout.volume();
  return p;
};

template<int d>
const coordinate &decomposition::first_local_domain() const {
  if (mdomains.size()==0)
    throw(format("Decomposition has no domains"));
  return mdomains.at(0);
};
const coordinate &decomposition::last_local_domain() const {
  if (mdomains.size()==0)
    throw(format("Decomposition has no domains"));
  return mdomains.at(mdomains.size()-1);
};

//! Get the local number where this domain is stored. Domains are multi-dimensionally numbered.
template<int d>
int decomposition::get_domain_local_number( const coordinate &dcoord ) const {
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
template<int d>
int decomposition::get_same_dimensionality( int dd ) const {
  return d=dd;
};

template<int d>
void decomposition::copy_embedded_decomposition( decomposition &other ) {
  embedded_decomposition = other.embedded_decomposition;
};

template<int d>
const decomposition &decomposition::get_embedded_decomposition() const {
  if (embedded_decomposition==nullptr)
    throw(string("Null embedded decomposition"));
  return *(embedded_decomposition.get());
};

/*! Add a bunch of 1d local domains
  \todo test for 1d-ness
  \todo use std_ptr to indexstruct*
*/
template<int d>
void decomposition::add_domains( indexstruct<int,d> *dom ) {
  //print("adding domains from indexstruct <<{}>>\n",d->as_string());
  for ( auto i=dom->first_index(); i<=dom->last_index(); i++ ) {
    int ii = (int)i; // std::static_cast<int>(i);
    add_domain( coordinate( vector<int>{ ii } ),false );
  }
  set_corners();
};

//! Add a multi-d local domain.
template<int d>
void decomposition::add_domain( coordinate<int,d> dom,bool recompute ) {
  //print("adding domain at coordinate <<{}>>\n",d.as_string());
  int dim = dom.get_same_dimensionality( domain_layout.get_dimensionality() );
  mdomains.push_back(dd);
  if (recompute) set_corners();
};

//! Get the multi-dimensional coordinate of a linear one. \todo check this calculation
//! http://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
template<int d>
coordinate decomposition::coordinate_from_linear(int p) const {
  int dim = domain_layout.get_dimensionality();
  if (dim<=0) throw(string("Zero dim layout"));
  coordinate pp(dim);
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
  closecorner = coordinate( vector<int>(d,0) );
  // farcorner; explicit endpoint.
  int P = domains_volume();
  farcorner = coordinate( make_endpoint(d,P) );
};

//! \todo why can't we declare this const?
const coordinate &decomposition::get_origin_processor() const {
  return closecorner;
};
const coordinate &decomposition::get_farpoint_processor() const {
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
  We store the current iterate as a private coordinate: `cur_coord'.
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

coordinate &decomposition::operator*() {
  //print("decomp::deref: {}\n",cur_coord.as_string());
  return cur_coord;
};

#endif

template class parallel_structure<1>;
