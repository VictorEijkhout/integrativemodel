/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2023
 ****
 **** mpi_env.cxx : mpi environment management
 ****
 ****************************************************************/

#include <stdarg.h>
#include <unistd.h> // just for sync
#include <iostream>
using std::cout;

#include "mpi_env.h"

using fmt::format, fmt::print;
using std::string;
using std::vector;

using gsl::span;

/*!
  An MPI environment has all the components of a base environment, plus
  - store a communicator and a task id.
  - disable printing for task id not zero
 */
mpi_environment::mpi_environment()
  : environment() {
};

/*!
  - Initialize MPI
  - set the comm_size/rank functions
  - initialize the singleton instance
*/
void mpi_environment::init( int &argc,char **&argv ) {
  MPI_Init(&argc,&argv);
  comm = MPI_COMM_WORLD;
  nprocs = [this] () -> int {
    int np;
    MPI_Comm_size(comm,&np);
    return np;
  };
  procid = [this] () -> int {
    int np;
    MPI_Comm_rank(comm,&np);
    return np;
  };
  int procid=-1;
  MPI_Comm_rank(comm,&procid);
  if (procid==0)
    cout << "Successful init MPI\n";
  environment::instance().init(argc,argv,/* do this: */ procid==0);
};

//! See also the base destructor for trace output.
mpi_environment::~mpi_environment() {
  // if (has_argument("dot") ) //&& get_architecture().mytid()==0)
  //   kernels_to_dot_file();
  // if (has_argument("dot"))
  //   tasks_to_dot_file();
  int procid=-1;
  MPI_Comm_rank(comm,&procid);
  if (procid==0)
    printf("MPI finalize\n");
  MPI_Finalize();
};

#if 0
mpi_environment::mpi_environment(int argc,char **argv) : environment(argc,argv) {
  type = environment_type::MPI;
  MPI_Init(&nargs,&the_args);
  comm = MPI_COMM_WORLD;
  MPI_Comm_set_errhandler(comm,MPI_ERRORS_RETURN);
  
  delete_environment = [this] () -> void { mpi_delete_environment(); };
#ifdef VT
  vt_register_kernels();
  VT_initialize(&nargs,&the_args);
#endif
  // arch = make_architecture();
  // int mytid = arch.mytid();

  set_is_printing_environment ( mytid==0 );
  if (has_argument("help") || has_argument("h")) {
    if (get_is_printing_environment())
      print_options();
    MPI_Abort(comm,0);
  }
  if (mytid==0 && has_argument("trace")) {
    printf("set trace print\n");
    arch.set_stdio_print();
  }

  // default collectives in the environment
  allreduce =      [this] (index_int i) -> index_int { return mpi_allreduce(i,comm); };
  allreduce_d =    [this] (double i) -> double { return mpi_allreduce_d(i,comm); };
  allreduce_and =  [this] (int i) -> int { return mpi_allreduce_and(i,comm); };

  gather32 =         [this] (int contrib,vector<int> &gathered) -> void {
    mpi_gather32(contrib,gathered,comm); };
  gather64 =         [this] (index_int contrib,vector<index_int> &gathered) -> void {
    mpi_gather64(contrib,gathered,comm); };
  overgather =     [this] (index_int contrib,int over) -> vector<index_int>* {
    return mpi_overgather(contrib,over,comm); };
  reduce_scatter = [this] (int *senders,int root) -> int {
    int procid; MPI_Comm_rank(comm,&procid);
    return mpi_reduce_scatter(senders,procid,comm); };

  // we can not rely on commandline argument on other than proc 0
  MPI_Bcast(&debug_level,1,MPI_INT,0,comm);
  arch.set_collective_strategy( get_collective_strategy() );
  {
    int e = has_argument("embed");
    MPI_Bcast(&e,1,MPI_INT,0,comm);
    if (e)
      arch.set_can_embed_in_beta();
    int v = has_argument("overlap");
    MPI_Bcast(&v,1,MPI_INT,0,comm);
    if (v)
      arch.set_can_message_overlap();
    int r = has_argument("random_source");
    MPI_Bcast(&r,1,MPI_INT,0,comm);
    if (r)
      arch.set_random_sourcing();
    int o = has_argument("rma");
    MPI_Bcast(&o,1,MPI_INT,0,comm);
    if (o)
      arch.set_use_rma();
  }

  // mode-specific summary
  mode_summarize_entities =
    [this] (void) -> result_tuple* { return mpi_summarize_entities(); };
};
#endif

void mpi_environment::print_options() {
  //  print_application_options();
  printf("MPI-specific options:\n");
  printf("  -embed : try embedding objects in halos\n");
  printf("  -overlap : post isends and irecvs early\n");
  printf("  -random_source : random resolution of ambiguous origins");
  printf("  -ram : use one-sided routines");
  printf("\n");
  environment::print_options();
}

#if 0
//! \todo make the over quantity variable
architecture mpi_environment::make_architecture() {
  int mytid,ntids;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);
  architecture a;
  {
    int over = iargument("over",1);
    if (over>1)
      throw(string("Can not handle over>1 yet"));
    mpi_architecture(a,ntids,mytid);
    //a.set_context( (void*)comm ); // comm world is already used in make_arch.....
  }
  return a;
};
#endif

//! This destructor routine needs to be called after the base destructor.
#if 0
void mpi_environment::mpi_delete_environment() {
  int procid;
  MPI_Comm_rank(comm,&procid);
  // if (procid==0)
  //   printf("MPI finalize\n");
  MPI_Finalize();
#ifdef VT
  VT_finalize();
#endif
};
#endif

/*!
  Sum all the space, entity counts were global to begin with.
  \todo this does not seem to be called, at least in the templates?!
  \todo task count is kernel count?
*/
#if 0
result_tuple *mpi_environment::mpi_summarize_entities() {
  auto mpi_results = new result_tuple;
  auto seq_results = local_summarize_entities();
  const architecture &arch = get_architecture();
  std::get<RESULT_OBJECT>(*mpi_results)       = std::get<RESULT_OBJECT>(*seq_results);
  std::get<RESULT_KERNEL>(*mpi_results)       = std::get<RESULT_KERNEL>(*seq_results);
  std::get<RESULT_TASK>(*mpi_results)
    = allreduce( std::get<RESULT_TASK>(*seq_results) );
  std::get<RESULT_DISTRIBUTION>(*mpi_results) = std::get<RESULT_DISTRIBUTION>(*seq_results);
  std::get<RESULT_ALLOCATED>(*mpi_results)
    = allreduce( std::get<RESULT_ALLOCATED>(*seq_results) );
  std::get<RESULT_DURATION>(*mpi_results)     = std::get<RESULT_DURATION>(*seq_results);
  std::get<RESULT_ANALYSIS>(*mpi_results)     = std::get<RESULT_ANALYSIS>(*seq_results);
  {
    int my_nmessages = std::get<RESULT_MESSAGE>(*seq_results), all_nmessages;
    all_nmessages = allreduce( my_nmessages );
    std::get<RESULT_MESSAGE>(*mpi_results) = all_nmessages;
  }
  std::get<RESULT_WORDSENT>(*mpi_results)
    = allreduce_d( std::get<RESULT_WORDSENT>(*seq_results) );
  std::get<RESULT_FLOPS>(*mpi_results)
    = allreduce_d( std::get<RESULT_FLOPS>(*seq_results) );

  return mpi_results;
};
#endif

//! Open or append the tasks dot file.
void mpi_environment::tasks_to_dot_file() {
  int mytid, ntids;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);
  FILE *dotfile; string s;

  if (mytid>0) {
    int msgi;
    MPI_Recv(&msgi,1,MPI_INTEGER,mytid-1,17,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }

  const char *fname = "tasks.dot"; // format("{}-tasks.dot",get_name()).data();
  if (mytid==0) {
    dotfile = fopen(fname,"w");
    fprintf(dotfile,"digraph G {\n");
  } else
    dotfile = fopen(fname,"a");

#if 0
  printf("dot for %d\n",mytid);
  s = tasks_as_dot_string();
  fprintf(dotfile,"/* ==== data for proc %d ==== */\n",mytid);
  fprintf(dotfile,"%s\n",s.data());
  fprintf(dotfile,"/* .... end proc %d .... */\n",mytid);
#endif

  if (mytid==ntids-1) 
    fprintf(dotfile,"}\n");

  fflush(dotfile);
  fclose(dotfile);
  sync();
  if (mytid<ntids-1) {
    int msgi = 1;
    MPI_Ssend(&msgi,1,MPI_INTEGER,mytid+1,17,MPI_COMM_WORLD);
  }
  // just to make sure that no one deletes our communicator.
  MPI_Barrier(MPI_COMM_WORLD);
};

/*!
  For MPI we let the root print out everything
*/
void mpi_environment::print_all(string s) {
  int ntids,mytid,maxlen;
  MPI_Comm_rank(comm,&mytid);
  MPI_Comm_size(comm,&ntids);
  MPI_Request req;
  int siz = s.size();
  MPI_Reduce( (void*)&siz,&maxlen,1,MPI_INTEGER,MPI_MAX,0,comm);
  { // everyone, including zero, send to zero
    int siz = s.size();
    MPI_Isend( (void*)s.data(),siz,MPI_CHAR, 0,0,comm,&req);
  }
  if (mytid==0) {
    char *buffer = new char[maxlen+2];
    for (int id=0; id<ntids; id++) {
      MPI_Status stat; int n;
      MPI_Recv( buffer,maxlen,MPI_CHAR, id,0,comm,&stat);
      MPI_Get_count(&stat,MPI_CHAR,&n); buffer[n] = '\n'; buffer[n+1] = 0;
      print_to_file(id,buffer);
    }
  }
  MPI_Wait(&req,MPI_STATUS_IGNORE);
};

