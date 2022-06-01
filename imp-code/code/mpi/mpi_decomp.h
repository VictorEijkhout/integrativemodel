#include "mpi.h"
#include "imp_decomp.h"

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
