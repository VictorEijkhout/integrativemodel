#pragma once

#include <array>
#include <vector>
#include <string>

#include "utils.h"
#include "imp_entity.h"
#include "indexstruct.hpp"

template<class I,int d>
std::array<I,d> endpoint(I s);

// template<class I,int d>
// std::array<I,d> farpoint(I s);

class decomposition;
class domain_coordinate;
class message;
class distribution;
class object;
class task;

//! Processor coordinates on a pretend grid. We order them by rows.
//! \todo write method to range over this
template<class I,int d>
class coordinate {
protected :
  std::array<I,d> coordinates;
public :
  coordinate();
  coordinate(I span);
  coordinate( std::array<I,d> );

  // basic manipulation
  constexpr int dimensionality() const { return d; }
  I &at(int i);
  const I &at(int i) const;
  I span() const;
  bool before( const coordinate<I,d>& ) const;
  I linear( const coordinate<I,d>& ) const;

  // operators
  coordinate operator+( index_int i) const;
  coordinate operator+( const coordinate& ) const;
  coordinate operator-(index_int i) const;
  coordinate operator-( const coordinate& ) const;
  coordinate operator%( const coordinate ) const;
  bool operator>(index_int i) const;
  bool operator>( const coordinate<I,d>& ) const;
  bool operator==( const coordinate&& ) const;
  bool operator==( const coordinate& ) const;
  bool operator!=( const coordinate&& ) const;
  bool operator!=( const coordinate& ) const;
  //  int operator[](int i) const { return coord(i); };
  coordinate rotate( std::vector<int> v,const coordinate &m) const;
  //  const std::vector<int> &data() { return coordinates; };

  // linearization
  int linearize( const coordinate&) const; // linear number wrt cube layout
  int linearize( const decomposition& ) const; // linear number wrt cube

  // equality operation
  coordinate negate();
  bool is_zero();
  // coordinate operate( const ioperator &op );
  // coordinate operate( const ioperator &&op );
  domain_coordinate operate( const ioperator &op );
  domain_coordinate operate( const ioperator &&op );
  std::string as_string() const;

#if 0
  bool is_on_left_face(const decomposition&) const;
  bool is_on_right_face(const decomposition&) const;
  bool is_on_face( const decomposition& ) const;
  bool is_on_face( const std::shared_ptr<object> ) const;
  bool is_on_face( const object& ) const;
  bool is_null() const;
  coordinate left_face_proc(int d,coordinate &&farcorner) const ;
  coordinate right_face_proc(int d,coordinate &&farcorner) const ;
  coordinate left_face_proc(int d,const coordinate &farcorner) const ;
  coordinate right_face_proc(int d,const coordinate &farcorner) const ;
#endif
  
  // operators
  //  domain_coordinate operator*(index_int i);

  // iterating
protected:
  int iterator{-1};
public:
  coordinate& begin() { iterator = 0; return *this; };
  coordinate& end() { return *this; };
  bool operator!=( coordinate ps ) { return iterator<coordinates.size()-1; };
  void operator++() { iterator++; };
  int operator*() const {
    if (iterator<0)
      throw(fmt::format("deref negative iterator {} in {}",iterator,as_string()));
    int v = coordinates[iterator];
    //printf("deref coord @%d to %d\n",iterator,v);
    return v; };
};

/****
 **** Architecture
 ****/

template<int d>
class architecture {
protected:
  coordinate<int,d> endpoint;
public:
  architecture( int np )
    : endpoint( coordinate<int,d>(np) ) {
  };
};

template<int d>
class parallel_structure {
protected:
  coordinate<int,d> procs;
  coordinate<index_int,d> points;
  std::vector< indexstruct > structs;
public:
  parallel_structure( coordinate<int,d> procs,coordinate<index_int,d> points )
    : procs(procs),points(points),
      structs( std::vector<indexstruct>(procs.span()) ) {
  };
};

#if 0

/*!
  Collective strategy

  - undefined : set at architecture creation
  - all_ptp : naive by all sends and received
  - group : two level grouping
  - tree : recursive grouping by two
  - MPI : (not implemented) using MPI collectives
 */
enum class collective_strategy{ UNDEFINED,ALL_PTP,GROUP,RECURSIVE,MPI };

/*!
  Definition of the synchronization protocol. 
  Overlap with \ref architecture_type needs to be figured out.
  Observe partial ordering: MPIplusCrap >= MPI.
 */
enum class protocol_type { UNDEFINED,OPENMP,OPENMPinMPI,MPI,MPIRMA,MPIplusOMP,MPItimesOMP };
std::string protocol_as_string(protocol_type p);

/*!
  Architecture types. We need to edit this list (and edit the definition of
  \ref parallel_indexstruct::parallel_indexstruct) every time we add one.

  - SHARED : OpenMP
  - SPMD : MPI
  - ISLANDS : Product?
  - ??? : Hybrid?
  - ??? : Accelerator?
 */
enum class architecture_type { UNDEFINED,SHARED,SPMD,ISLANDS };
std::string architecture_type_as_string( architecture_type type );

/*!  
  A base class for architecture data. All accessor routines throw
  an exception by default; they are overridden in the mode-specific
  classes. Interestingly, there is no data in this class; we leave
  that again to the modes.

  We probably need to add access functions here if we introduce new modes.

  \todo reinstate disabled collective strategy handling
  \todo install lambda for getting environment
 */
class message;
class architecture : public entity {
protected:
public: // data
  int beta_has_local_addres_space{-1}; //!< \todo why is this public data?
public: // methods
  //! Default constructor
  architecture();
  //! architecture constructor will mostly be called through derived architectures.
  architecture(int m,int n) : architecture(n) { arch_procid = m; };
  architecture(int n) { // : entity(entity_cookie::ARCHITECTURE) {
    arch_nprocs = n; }; //set_name("some architecture"); };
  //! A mode-dependent switch to turn on tricky optimizations
  virtual void set_power_mode() {};

  /*
   * Type handling
   */
public: //! make this protected again
  architecture_type type{architecture_type::UNDEFINED};
  architecture *embedded_architecture{nullptr};
public:
  const architecture_type &get_architecture_type() const { return type; };
  int has_type(architecture_type t) const { return get_architecture_type()==t; };
  architecture *get_embedded_architecture() const {
    if (embedded_architecture==nullptr)
      throw(std::string("embedded architecture is not set"));
    return embedded_architecture; };

  /*
   * Over decomposition
   */
protected:
  int over_factor{1};
public:
  void set_over_factor(int o) { over_factor = o; };
  int get_over_factor() const { return over_factor; };

  /*
   * Protocols
   */
public: //! make this protected again
  protocol_type protocol{protocol_type::UNDEFINED};
public:
  protocol_type get_protocol() const { return protocol; };
  int get_is_mpi_based() const { return protocol>=protocol_type::MPI; };
  void set_protocol_is_embedded() { protocol = protocol_type::OPENMPinMPI; };
  int get_protocol_is_embedded() const { return protocol==protocol_type::OPENMPinMPI; };
protected:
  bool split_execution{false};
public:
  void set_split_execution(bool split=true) { split_execution = split; };
  auto get_split_execution() const { return split_execution; };

  /*
   * Processor handling
   */
public: //! make this protected again
  int arch_procid{-1},arch_nprocs{-1};
public:
  int nprocs() const {
    if (arch_nprocs<0) throw(std::string("Number of processors uninitialized"));
    return arch_nprocs; };
  //! Processor count of the embedded architecture
  int embedded_nprocs() const { return get_embedded_architecture()->nprocs(); };
  //! Total number of processors over top level plus embedded
  int product_nprocs() const { return nprocs()*embedded_nprocs(); };
  //! MPI has local identification, OMP doesn't
  std::function< int(void) > mytid{
    [] (void) -> int { throw(std::string("no mytid in general")); } };
  virtual int is_first_proc(int p) const { return p==0; };
  virtual int is_last_proc(int p)  const { return p==arch_nprocs-1; };
protected:
  coordinate coordinates;
public:
  coordinate get_proc_origin(int d) const; // far corner of the processor grid
  coordinate get_proc_endpoint(int d) const; // far corner of the processor grid
  coordinate get_proc_layout(int d) const; // far corner of the processor grid
  coordinate get_coordinates() const {
    if (coordinates==nullptr) throw(std::string("No coordinates translation set"));
    return coordinates; };
protected:
  int random_sourcing{0}; //!< Random resolution of MPI redundancy.
  int rma{0}; //!< Use MPI one-sided.
public:
  void set_random_sourcing() { random_sourcing = 1; };
  int has_random_sourcing() { return random_sourcing; };
  void set_use_rma() { rma = 1; };
  int get_use_rma() { return rma; };

  /*
   * Various properties
   */
protected:
  int can_embed_in_beta{0},can_message_overlap{0};
public:
  //! Declare that betas are in global space.
  void set_can_embed_in_beta() { set_can_embed_in_beta(1); };
  //! Set the embedding explicitly. Good for the unittests.
  void set_can_embed_in_beta(int can) { can_embed_in_beta = can; };
  int get_can_embed_in_beta() const { return can_embed_in_beta; };
  void set_can_message_overlap() { can_message_overlap = 1; };
  int get_can_message_overlap() const { return can_message_overlap; };

  /*
   * Collectives
   */
protected:
  collective_strategy strategy{collective_strategy::GROUP};
public:
  collective_strategy get_collective_strategy() const { return strategy; };
  int has_collective_strategy(collective_strategy s) const { return strategy==s; };
  void set_collective_strategy(int s);
  void set_collective_strategy(collective_strategy s);
  void set_collective_strategy_ptp() { strategy = collective_strategy::ALL_PTP; };
  void set_collective_strategy_group() { strategy = collective_strategy::GROUP; };
  void set_collective_strategy_recursive() { strategy = collective_strategy::RECURSIVE; };
  std::string strategy_as_string() const {
    switch (strategy) {
    case collective_strategy::ALL_PTP : return std::string("point-to-point"); break;
    case collective_strategy::GROUP : return std::string("grouped"); break;
    case collective_strategy::RECURSIVE : return std::string("treewise"); break;
    case collective_strategy::MPI : return std::string("using mpi"); break;
    default : return std::string("undefined"); }
  };

  /*
   * Timing
   */
  std::chrono::system_clock::time_point unsynchronized_timer() {
    return std::chrono::system_clock::now(); };
  //! \todo reinstall the barrier
  std::chrono::system_clock::time_point synchronized_timer() {
    //barrier();
    return unsynchronized_timer(); };
  
  /*
   * Tracing and output
   */
protected:
public:
  std::function< void(std::string) > print_trace{ [] (std::string c) -> void { return; } };
  void set_stdio_print() { print_trace = [] (std::string c) { fmt::print("{}\n",c); }; };
  virtual std::string summary();

  /*
   * Distribution factory
   */
  virtual std::shared_ptr<distribution> new_scalar_distribution() {
    throw(std::string("Error: can not call base case")); };

  std::function< void(architecture&,std::shared_ptr<message>,
		      std::string& /* char*,int */ ) > message_as_buffer {
    [] (architecture &a,std::shared_ptr<message>m,
	// char *buf,int len) -> void {
	std::string &b) -> void {
      throw(std::string("message_as_buffer not set")); }
  };
  
  // I/O
  virtual std::string as_string() const;
};

#endif
