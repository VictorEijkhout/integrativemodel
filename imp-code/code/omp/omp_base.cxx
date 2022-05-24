
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2020
 ****
 **** omp_base.cxx: Implementations of the OpenMP classes
 ****
 ****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <iostream>

#include <omp.h>
#include "omp_base.h"
#include "imp_base.h"

using fmt::format;
using fmt::format_to;
using fmt::to_string;
using fmt::print;

using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

using gsl::span;

/****
 **** Basics
 ****/

/*!
  Our OpenMP implementation does not have thread-local data yet, 
  so gathering will just be replication.
 */
vector<index_int> *omp_architecture::omp_gather(index_int contrib) {
  if (omp_in_parallel()) throw(string("Unsupported use case for gather"));
  int P = nprocs();
  auto gathered = new vector<index_int>; gathered->reserve(P);
  for (int i=0; i<P; i++)
    gathered->push_back(contrib);
  return gathered;
};

/*! \todo we need a more portable way of finding the number of processors */
void omp_gather64(index_int contrib,vector<index_int> &gathered) {
  if (omp_in_parallel()) throw(string("Unsupported use case for gather"));
  int P = gathered.size();
  if (gathered.size()<P) {
    throw(string("gather64 result vector too short: {}",gathered.size())); }
  for (int i=0; i<P; i++)
    gathered.at(i) = contrib;
};

/*!
  OMP implementation of the mode-independent reduce-scatter.
  \todo this gets called somewhere deep down where the arch is wrong ????
*/
int omp_architecture::omp_reduce_scatter(int *senders,int root) {
  printf("omp reduce scatter is wrong\n");
  return senders[root];
};

string omp_architecture::as_string() {
  fmt::memory_buffer w; format_to(w.end(),"OpenMP architecture on {} threads",nprocs());
  return to_string(w);
};

/*!
  Initialize environment and process commandline options.
  \todo think about these omp allreduce things.
 */
omp_environment::omp_environment(int argc,char **argv) : environment(argc,argv) {
  type = environment_type::OMP;
  arch = make_architecture();
  set_is_printing_environment();

  if (has_argument("help")) {
    print_options();
    abort();
  }
  arch.set_collective_strategy( iargument("collective",0) );

  if (has_argument("embed"))
    arch.set_can_embed_in_beta();

  ntasks_executed = 0; // lose this

  // default collectives in the environment
  allreduce =     [] (index_int i) -> index_int { return i; };
  allreduce_d =   [] (double i) -> double { return i; };
  allreduce_and = [] (int i) -> int { return i; };

  gather64 =      [this] (index_int contrib,vector<index_int> &gathered) -> void 
		  { omp_gather64(contrib,gathered); };
};

void omp_architecture( architecture &a, int ntids ) {
  a.arch_nprocs = ntids;
};

architecture omp_environment::make_architecture() {
  architecture arch; int nt; int over = iargument("over",1);
#pragma omp parallel shared(nt)
#pragma omp master
  {
    // nt becomes the nprocs value
    nt = omp_get_num_threads(); nt *= over;
    omp_architecture(arch,nt);
  }
  return arch;
};

//! This is largely identical to the MPI code.
omp_environment::~omp_environment() {
  if (has_argument("dot")) {
    kernels_to_dot_file();
    tasks_to_dot_file();
  }
};

void omp_environment::record_task_executed() {
  ntasks_executed++;
};

/*!
  Document omp-specific options.
*/
void omp_environment::print_options() {
  printf("OpenMP-specific options:\n");
  printf("  -embed : try embedding objects in halos\n");
  environment::print_options();
}

void omp_environment::print_stats() {
  double t_x=0.,tmax; int n_x=0;

  // find average execution time over multiple runs
  for ( auto t : execution_times ) {
    t_x = t_x + t; n_x++;
  } t_x /= n_x;

  tmax = t_x;
};

/****
 **** Decomposition
 ****/

//! \todo use std move to collapse these two
omp_decomposition::omp_decomposition( const architecture &arch,processor_coordinate &grid )
  : decomposition(arch,grid) {
  int ntids = arch.nprocs();
  for (int mytid=0; mytid<ntids; mytid++) {
    auto mycoord = this->coordinate_from_linear(mytid);
    add_domain(mycoord);
  }
  set_decomp_factory();
};

omp_decomposition::omp_decomposition( const architecture &arch,processor_coordinate &&grid )
  : decomposition(arch,grid) {
  int ntids = arch.nprocs();
  for (int mytid=0; mytid<ntids; mytid++) {
    auto mycoord = this->coordinate_from_linear(mytid);
    add_domain(mycoord);
  }
  set_decomp_factory();
};

//! Default constructor is one-d.
omp_decomposition::omp_decomposition( const architecture &arch )
  : omp_decomposition(arch,arch.get_proc_layout(1)) {};

//! A factory for making new distributions from this decomposition
void omp_decomposition::set_decomp_factory() {
  new_block_distribution = [this] (index_int g) -> shared_ptr<distribution> {
    return shared_ptr<distribution>( new omp_block_distribution(*this,g) ); };
};

/****
 **** Distribution
 ****/

void make_omp_communicator(communicator *cator,int P) {
  cator->the_communicator_mode = communicator_mode::OMP;
  
  cator->nprocs =        [P] (void) -> int { return P; };
  cator->allreduce =     [] (index_int contrib) -> index_int { return contrib; };
  cator->allreduce_d =   [] (double contrib) -> double { return contrib; };
  cator->allreduce_and = [] (int contrib) -> int { return contrib; };
  cator->overgather =    [P] (index_int contrib,int over) -> vector<index_int>* {
    auto v = new vector<index_int>; v->reserve(over*P);
    for (int i=0; i<over*P; i++) v->push_back(contrib);
    return v; };
  cator->gather64 =
    [] (index_int contrib,vector<index_int> &gathered) -> void {
    return omp_gather64(contrib,gathered); };
};

//! Basic constructor
omp_distribution::omp_distribution( const decomposition &d ) 
  : distribution(d) {
  make_omp_communicator(this,this->domains_volume());
  set_omp_routines();
  set_name("omp-distribution");
};

//! Constructor from parallel structure
omp_distribution::omp_distribution( const parallel_structure &struc )
  : distribution(struc) {
  make_omp_communicator(this,this->domains_volume());
  set_omp_routines();
  try {
    memoize();
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not memoizing in omp distribution from struct {}",
		      struc.as_string()));
  }
  // VLE this code does not appear in MPI. do we need it here?
  // for ( int is=0; is<struc->get_structures().size(); is++ ) {
  //   parallel_indexstruct *oldstruct = struc->get_dimension_structure(is);
  //   set_dimension_structure(is,oldstruct);
  // }
  // set_type( struc->get_type() );
};
omp_distribution::omp_distribution( const parallel_structure &&struc )
  : distribution(struc) {
  make_omp_communicator(this,this->domains_volume());
  set_omp_routines();
  try {
    memoize();
  } catch (string c) { print("Error <<{}>>\n",c);
    throw(format("Could not memoizing in omp distribution from struct {}",
		      struc.as_string()));
  }
};

//! Constructor from function
omp_distribution::omp_distribution
    ( const decomposition &d,index_int(*pf)(int,index_int),index_int nlocal )
  : omp_distribution(d) {
  get_dimension_structure(0)->create_from_function( pf,nlocal );
  memoize();
};

//! Copy constructor from other distribution
omp_distribution::omp_distribution( shared_ptr<distribution> other )
  : distribution(other->get_structure()) {
  set_structure_type(other->get_type());
  
};

//! OpenMP block distribution from local / global
omp_block_distribution::omp_block_distribution
    (const decomposition &d,int o,index_int l,index_int g)
  : distribution(d),omp_distribution(d),block_distribution(d,o,l,g) {
  try { memoize();
  } catch (string c) {
    throw(format("Failed to memoize omp block distr: {}",c));
  }
};

//! OpenMP block distribution 1D from array of local sizes
omp_block_distribution::omp_block_distribution
    (const decomposition &d,std::vector<index_int> sizes)
  : distribution(d),omp_distribution(d),block_distribution(d,sizes) {
  d.get_same_dimensionality(1);
  try { memoize();
  } catch (std::string c) {
    throw(format("Failed to memoize omp block distr: {}",c));
  }
};

void omp_distribution::set_omp_routines() {
  // Factory for new distributions
  new_distribution_from_structure =
    [] (const parallel_structure &strct) -> shared_ptr<distribution> {
    return shared_ptr<distribution>( new omp_distribution( strct ) ); };
  // Factory for new scalar distributions
  new_scalar_distribution = [this] (void) -> shared_ptr<distribution> {
    auto decomp = dynamic_cast<decomposition*>(this);
    return shared_ptr<distribution>( new omp_scalar_distribution(decomp) ); };
  new_object = [this] (shared_ptr<distribution> d) -> shared_ptr<object>
    { auto o = shared_ptr<object>( new omp_object(d) ); return o; };
  new_object_from_data = [this] ( shared_ptr<vector<double>> d ) -> shared_ptr<object>
    { return shared_ptr<object>( new omp_object(this->shared_from_this(),d) ); };
  // Factory for making mode-dependent kernels (this seems only for testing?)
  new_kernel_from_object = [] ( shared_ptr<object> out ) -> shared_ptr<kernel>
    { return shared_ptr<kernel>( new omp_kernel(out) ); };
  // Factory for making mode-dependent kernels, for the imp_ops kernels.
  kernel_from_objects =
    [] ( shared_ptr<object> in,shared_ptr<object> out ) -> shared_ptr<kernel>
    { return shared_ptr<kernel>( new omp_kernel(in,out) ); };

  auto decomp = get_decomposition();
  // Set the message factory.
  new_message =
    [decomp] (const processor_coordinate &snd,const processor_coordinate &rcv,
	shared_ptr<multi_indexstruct> g) -> shared_ptr<message> {
    return shared_ptr<message>( new omp_message(decomp,snd,rcv,g) );
  };
  new_embed_message =
    [decomp] (const processor_coordinate &snd,const processor_coordinate &rcv,
	      shared_ptr<multi_indexstruct> e,
	      shared_ptr<multi_indexstruct> g) -> shared_ptr<message> {
    return shared_ptr<message>( new omp_message(decomp,snd,rcv,e,g) );
  };

  /*
   * NUMA
   */
  //snippet ompnuma
  get_numa_structure =
    [this] () -> shared_ptr<multi_indexstruct>
    { return get_enclosing_structure(); };
  get_global_structure =
    [this] () -> shared_ptr<multi_indexstruct> //const multi_indexstruct& 
    { return get_enclosing_structure(); };
  //snippet end

  location_of_first_index =
    [] ( shared_ptr<distribution> d,const processor_coordinate &p) -> index_int
    { return omp_location_of_first_index(d,p); };
  location_of_last_index =
    [] ( shared_ptr<distribution> d,const processor_coordinate &p) -> index_int
    { return omp_location_of_last_index(d,p); };
  //snippet end

  local_allocation = [this] (void) -> index_int { return global_allocation(); };
  //snippet ompvisibility
  //! A processor can see all of the address space
  get_visibility = [this] (processor_coordinate &p) -> shared_ptr<multi_indexstruct> {
    return get_enclosing_structure(); };
  //snippet end

  auto me_ish = decomp.first_local_domain();
  compute_global_first_index =
    [this,me_ish] (parallel_structure *pstr) -> domain_coordinate {
    try {
      auto first = pstr->first_index_r(me_ish);
      //print("Computed global first as {}\n",first->as_string());
      return domain_coordinate( first );
    } catch (string c) {
      print("Error: {}\n",c);
      throw(string("Could not compute omp global first index"));
    }
  };
  compute_global_last_index =
    [this] (parallel_structure *pstr) -> domain_coordinate {
    auto last_struct = pstr->get_processor_structure( get_farpoint_processor() ); 
    return domain_coordinate( last_struct->last_index_r() );
  };
  compute_offset_vector =
    [this] () -> domain_coordinate
    { return numa_first_index()-global_first_index();
  };
};

/*! 
  Find the location of a processor structure in the allocated data.
  \todo not correct for masks
  \todo this could use a unit test. next one too
*/
index_int omp_location_of_first_index
    (shared_ptr<distribution> d,const processor_coordinate &pcoord) {
  int dim = d->get_same_dimensionality(pcoord.get_dimensionality());
  auto enc = d->get_enclosing_structure();
  domain_coordinate
    d_first = d->first_index_r(pcoord),
    enc_first = enc->first_index_r(),
    enc_last = enc->last_index_r();
  index_int loc = d_first.linear_location_in(enc); //(enc_first,enc_last);
  //print("{} first={}, location in {} is {}\n",pcoord->as_string(),d_first.as_string(),enc->as_string(),loc);
  return loc;
};

//! \todo this should be done through linear location in numa struct
index_int omp_location_of_last_index
    (shared_ptr<distribution> d,const processor_coordinate &pcoord) {
  return d->last_index_r(pcoord).linear_location_in( d->global_last_index() );
};

/****
 **** Object
 ****/

//! Create an object from a distribution, locally allocating the data
omp_object::omp_object( std::shared_ptr<distribution> d )
  //try
  : object(d) {
  set_data_handling();
  // if objects can embed in a halo, we'll allocate later; otherwise now.
  if (!d->get_can_embed_in_beta()
      || d->get_type()==distribution_type::REPLICATED) {
    omp_allocate();
  }
};

/*! Create an object from user data. This is dangerous for replicated data;
  see constructor with extra integer parameter 
  \todo can we delegate this? \todo the registration is wrong
*/
omp_object::omp_object( shared_ptr<distribution> d, shared_ptr<std::vector<double>> dat )
try : object(d) {
  set_data_handling();
  if (get_distribution()->get_type()==distribution_type::REPLICATED)
    throw(std::string("too dangerous to create omp replicated object from data"));
  auto data_span = gsl::span<double>( dat->data(),dat->size() );
  install_data_on_domains(data_span,0);
  data_status = object_data_status::INHERITED;
      } catch (...) {
  print("Omp_object creation from distribution and data failed\n");
 };

  //! Create an object from data of another object. \todo revisit registration
omp_object::omp_object( std::shared_ptr<distribution> d, std::shared_ptr<object> x )
  : object(d) {
  set_data_handling();
  for ( auto dom : get_distribution()->get_domains() ) {
    auto old_data = x->get_data(dom);
    register_data_on_domain(dom,old_data);
  }
};

void omp_object::omp_allocate() {
  if (has_data_status_allocated()) return;
  const auto &domains = get_decomposition().get_domains();

  index_int s;
  try {
    s = get_distribution()->global_allocation();
    auto info = format("unique storage {} words for {}",s,get_name());

    if (get_distribution()->get_type()==distribution_type::REPLICATED) {
      s *= domains.size();
      info = format("replicated storage {} bytes for {}",s,get_name());
    } 
    auto dat = create_data(s,info);
    set_numa_data(dat,s);
  } catch (...) { print("Could not create numa data");
    throw(string("omp_allocate failed"));
  }

  add_data_count(s);
  if (get_trace_create_data())
    print("Create {} for <<{}>>, reaching {}\n",s,get_name(),get_data_count());

  auto numa_ptr = get_numa_data_pointer()->data();
  auto numa_size = get_numa_data_pointer()->size();
  //print("numa data at {} has size {}\n",(long)numa_ptr,numa_size);
  
  if (get_distribution()->get_type()==distribution_type::REPLICATED) {
    auto d0 = get_decomposition().first_local_domain();
    index_int
      slocal = get_distribution()->volume(d0);
    print("registering replicated data of size {}\n",slocal);
    auto dom_begin = numa_ptr;
    auto dom_end = numa_ptr; dom_end += slocal;
    for (int idom=0; idom<domains.size(); idom++) {
      auto dom = domains.at(idom);
      // auto offset_dat = shared_ptr<vector<double>>
      // 	(new vector<double>(dom_begin,dom_end));
      span<double> offset_dat(dom_begin,dom_end);
      print("domain {} has data at {}\n",idom,(long)(offset_dat.data()));
      register_data_on_domain( dom, offset_dat,slocal );
      dom_begin += slocal; dom_end += slocal;
    }
  } else {
    //install_data_on_domains(numa_ptr,s);
    install_data_on_domains( span<double>(numa_ptr,s),s );
  }
};

void omp_object::install_data_on_domains( data_pointer dat,index_int s) {
  int idom=0; index_int offset{0};
  auto domains = get_decomposition().get_domains();
  for ( auto dom : domains ) {
    register_data_on_domain(dom,dat);
  }
};

/*!
  This routine is typically used to copy from an in object to a halo.
  In OpenMP the halo is the output vector; see \ref omp_task::acceptReadyToSend,
  so the indexing.....

  \todo deal with the case where global/local struct are not contiguous
*/
//snippet ompcopydata
void omp_object::copy_data_from
    ( shared_ptr<object> in,shared_ptr<message> smsg,shared_ptr<message> rmsg ) {
  if (in==nullptr) throw(string("Null input object in copy_data_from"));
  if (has_data_status_unallocated() || in->has_data_status_unallocated())
    throw(string("Objects should be allocated by now"));
  //  shared_ptr<object> out = this->shared_from_this();
  auto out = this;

  auto p = rmsg->get_receiver(), q = rmsg->get_sender();
  int dim = get_same_dimensionality( in->get_dimensionality() );
  auto
    src_data = in->get_data(q), tar_data = out->get_data(p);
  auto src_struct = smsg->get_local_struct(), tar_struct = rmsg->get_local_struct(),
    src_gstruct = smsg->get_global_struct(), tar_gstruct = rmsg->get_global_struct();
  auto struct_size = tar_struct->local_size_r();

  int k = in->get_orthogonal_dimension();
  if (k>1)
    throw(string("copy data too hard with k>1"));

  domain_coordinate
    pfirst = tar_gstruct->first_index_r(), plast = tar_gstruct->last_index_r(),
    qfirst = src_gstruct->first_index_r(), qlast = src_gstruct->last_index_r();
  // print("Copy {}:{} -> {}:{}\n",
  // 	     q->as_string(),qfirst.as_string(),p->as_string(),pfirst.as_string());

  auto in_nstruct = in->get_numa_structure(),
    out_nstruct = out->get_numa_structure(),
    in_gstruct = in->get_global_structure(),
    out_gstruct = out->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  if (dim==0) {
  } else if (dim==2) {
    if (src_struct->is_contiguous() && tar_struct->is_contiguous()) {
      int done = 0;
      for (index_int isrc=qfirst[0],itar=pfirst[0]; itar<=plast[0]; isrc++,itar++) {
	for (index_int jsrc=qfirst[1],jtar=pfirst[1]; jtar<=plast[1]; jsrc++,jtar++) {
	  index_int
	    Iout = INDEX2D(itar,jtar,out_offsets,out_nsize),
	    Iin = INDEX2D(isrc,jsrc,in_offsets,in_nsize);
	  if (!done) {
	    print("{}->{} copy data {} between {}->{}\n",
		  q.as_string(),p.as_string(),src_data.at(Iin),Iin,Iout);
	    done = 1; }
	  tar_data.at(Iout) = src_data.at(Iin);
	}
      }

    } else {
      throw(string("omp 2d copy requires cont-cont"));
    }
  } else if (dim==1) {
    auto local = rmsg->get_local_struct(), global = rmsg->get_global_struct();
    if (global->is_contiguous() && local->is_contiguous()) {

      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	tar_data.at( INDEX1D(i,out_offsets,out_nsize) ) = 
	  src_data.at( INDEX1D(i,in_offsets,in_nsize) );
      }

    } else {
      index_int localsize = local->volume();
      index_int len = localsize*k;

      auto
	src_struct = global->get_component(0), tar_struct = local->get_component(0);
      index_int
        src0 = in->get_enclosing_structure()->linear_location_of(local),
        tar0 = this->get_enclosing_structure()->linear_location_of(global);
      if (src_struct->is_contiguous() && tar_struct->is_contiguous()) {
        throw(string("This should have bee done above"));
      } else if (tar_struct->is_contiguous()) {
        index_int itar=tar0; 
        for ( auto isrc : *src_struct )
          tar_data.at(itar++) = src_data.at(isrc);
      } else {
        for (index_int i=0; i<len; i++)
          tar_data.at( tar_struct->get_ith_element(i) ) =
	    src_data.at( src_struct->get_ith_element(i) );
      }
    }
  } else 
    throw(string("Can not omp copy in other than 1-d or 2-d"));
};
//snippet end

string omp_object::values_as_string() {
  fmt::memory_buffer w; format_to(w.end(),"{}:",get_name());
  if (get_orthogonal_dimension()>1)
    throw(string("Can not handle k>1 for object values_as_string"));
  auto data = this->get_raw_data(); index_int s = get_distribution()->global_volume();
  for (index_int i=0; i<s; i++)
    format_to(w.end()," {}:{}",i,data[i]);
  return to_string(w);
};

/****
 **** Request & request vector
 ****/

//snippet ompwait
/*!
  Wait for some requests, presumably for a task. 
  Each incoming requests carries an output object of another task.
  Outgoing requests are ignored. OpenMP does not send.
 */
void omp_request_vector_wait( request_vector &v) {
  int outstanding = v.size();
  for ( ; outstanding>0 ; ) { // loop until all requests fullfilled
    //print("Requests outstanding: {}\n",outstanding);
    for ( auto r : v.requests() ) {
      if (r->has_type(request_type::UNKNOWN))
	  throw(format("Can not wait for request of unknown type"));
      omp_request *req = dynamic_cast<omp_request*>(r.get());
      if (req==nullptr) throw(string("could not upcast to omp request"));
      if (req->has_type(request_type::OUTGOING)) {
	//print("Outgoing request, closing\n");
        req->closed = 1;
        outstanding--;
	continue;
      }
      if (!req->closed) {
        // unclosed request: make sure depency task is executed,
        auto t = req->tsk;
	//print("unclosed request for task <<{}>>\n",t->as_string());
        if (!t->get_has_been_executed())
          try {
            t->execute();
          } catch (string c) { print("Error <<{}>>\n",c); 
            throw(format("Could not execute {} as dependency",t->get_name()));
	  }
	auto recv_msg = req->msg;
	auto send_msg = recv_msg->send_msg;
	auto task_object = t->get_out_object();
	req->obj->copy_data_from( task_object,send_msg,recv_msg );
        req->closed = 1;
        outstanding--;
      }
    };
  }
};
//snippet end

/****
 **** Task
 ****/

/*!
  Build the send messages;
  \todo how far we unify this with MPI?
*/
void omp_task::derive_send_messages(bool trace) {
  auto dom = get_domain();
  vector<shared_ptr<message>> my_send_messages;  
  for ( auto msg : get_receive_messages() ) {
    auto sender_domain = msg->get_sender();
    shared_ptr<task> sender_task;
    try { sender_task = find_kernel_task_by_domain(sender_domain);
    } catch (string c) {
      throw(format("Error <<{}>> finding sender for msg <<{}>>",c,msg->as_string())); }
    auto &dep = find_dependency_for_object_number(msg->get_in_object_number());
    auto in = dep.get_in_object(), out = dep.get_beta_object();
    auto send_struct = msg->get_global_struct();

    auto send_msg = shared_ptr<message>
      ( new omp_message
	(out->get_decomposition(),sender_domain,dom, send_struct) );
    send_msg->set_in_object(in); send_msg->set_out_object(out);
    send_msg->set_name
      ( format
	("send-{}-obj:{}->{}",
	 msg->get_name(),msg->get_in_object_number(),msg->get_out_object_number()) );
    send_msg->set_send_type();
    try {
      send_msg->compute_src_index();
    } catch (string c) {
      throw(format("Error <<{}>> computing src index in <<{}>> from <<{}>>",
		   c,send_msg->as_string(),send_msg->get_in_object()->get_name())); }
    try {
      my_send_messages.push_back(send_msg);
      msg->send_msg = send_msg;
    } catch (string c) {
      throw(format("Error <<{}>> in remaining send msg actions",c)); }
  }
  set_send_messages(my_send_messages);
};

shared_ptr<request> omp_task::notifyReadyToSendMsg( shared_ptr<message> msg ) {
  return shared_ptr<request>
    ( new omp_request(shared_from_this(),msg,msg->get_out_object(),request_type::OUTGOING) );
};

/*!
  Execute a task by taking it as root and go down the predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.

  This routine is identical to the base routine, except for the omp directives.
*/
void omp_task::execute_as_root(bool trace) {
  if (get_has_been_executed()) {
    if (trace) print("Task already executed: {}\n",as_string());
    return;
  }

  if (trace) print("Executing task: {}\n",as_string());
  auto preds = get_predecessors();
  for (int it=0; it<preds.size(); it++) {
    auto tsk = preds.at(it);
    if (trace)
      print(".. executing predecessor: {}\n",tsk->as_string());
#pragma omp task
    tsk->execute_as_root(trace);
  }
#pragma omp taskwait

//   { // not sure if this fragment does any good
//     while (1) {
//       int all = 1;
//       for ( auto tsk : preds ) {
//         all *= tsk->get_has_been_executed();
//       }
//       if (all==0) {
// #pragma omp taskyield
//         ;
//       } else break;
//     }
//   }
  omp_set_lock(&dolock);
  if (!get_has_been_executed()) {
    execute(trace);
    set_has_been_executed();
  }
  omp_unset_lock(&dolock);

};

//snippet ompaccept
/*!
  This is the routine that posts the request for other tasks to supply data.
  We implement this as just listing the predecessors tasks: we can not actually 
  execute them here, because this call happens potentially twice in task::execute.
  The actual execution happens in \ref omp_request::wait.
  \todo I have my doubts abou that set_data_is_filled call. this is the wrong place.
 */
request_vector omp_task::acceptReadyToSend( vector<shared_ptr<message>> &msgs ) {
  request_vector requests;
  if (find_other_task_by_coordinates==nullptr)
    throw(format("{}: Need a task finding function",get_name()));
  for ( auto m : msgs ) {
    auto sender = m->get_sender(), receiver = m->get_receiver();
    auto halo = m->get_out_object();
    //halo->set_data_is_filled(receiver);
    int instep = m->get_in_object()->get_object_number();
    requests.add_request
      ( shared_ptr<request>
	( new omp_request( find_other_task_by_coordinates(instep,sender),m,halo ) ) );
  }
  return requests;
};
//snippet end

void omp_task::declare_dependence_on_task( task_id *id ) {
  int step = id->get_step(); auto domain = id->get_domain();
  try { add_predecessor( find_other_task_by_coordinates(step,domain) );
  } catch ( const char *c ) { throw(string("Could not find OMP queue predecessor")); };
};

/****
 **** Kernel
 ****/

/*! Construct the right kind of task for the base class
  method \ref kernel::split_to_tasks.
*/
void omp_kernel::install_omp_factory() {
  make_task_for_domain =
    [] ( kernel *k,const processor_coordinate &p ) -> shared_ptr<task> {
    auto t = shared_ptr<task>( new omp_task(p,k) );
    return t;
  };
};

/****
 **** Queue
 ****/

//! Find a task number by step/domain coordinates. \todo why not return a task pointer?
int omp_algorithm::find_task( int s,processor_coordinate *d ) {
  int ret;
  for (int n=0; n<tasks.size(); n++) {
    if (tasks[n]->get_step()==s && tasks[n]->get_domain()==d) {
      ret = n; goto exit;
    }
  }
  ret = -1;
 exit:
  //  printf("found step %d domain %d as %d\n",s,d,ret);
  return ret;
};

/*!
  Find all tasks, starting at `this' one, that only depend on
  non-synchronization tasks.
*/
void omp_task::check_local_executability() {
  task_local_executability exec_val = this->get_local_executability();
  if (exec_val==task_local_executability::INVALID) throw(string("Invalid executability"));
  else if (exec_val==task_local_executability::UNKNOWN) {
    if (this->get_is_synchronization_point()) {
      exec_val = task_local_executability::NO;
    } else {
      exec_val = task_local_executability::YES;
      auto preds = this->get_predecessors();
      for  ( auto p : preds ) { //(auto p=preds->begin(); p!=preds->end(); ++p) {
        omp_task *op = dynamic_cast<omp_task*>(p.get());
	if (op==nullptr)
	  throw(format("Could not upcast to omp task"));
        op->check_local_executability();
        task_local_executability opx = op->get_local_executability();
        if (opx==task_local_executability::NO) {
          exec_val = task_local_executability::NO; break;
        }
      }
    }
    this->set_can_execute_locally(exec_val);
  }
};

/*!
  Execute all tasks in a queue. Since they may not be in the right order,
  we take each as root and go down their predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.
*/
void omp_algorithm::execute_tasks( int(*tasktest)(shared_ptr<task> t),bool trace ) {
  if (tasktest==nullptr) throw(format("Missing task test"));
#pragma omp parallel
  {
#pragma omp single
    {
      for ( auto &t : get_exit_tasks() ) {
#pragma omp task untied
	try {
	  if ((*tasktest)(t)) {
	    if (trace)
	      print("executing task as root: {}\n",t->as_string());
	    t->execute_as_root();
	  }
	} catch ( string c ) {
	  print("Error <<{}>> for task <<{}>> execute\n",c,t->as_string());
	  throw(string("Task queue execute failed"));
	}
      }
#pragma omp taskwait
      //#pragma omp barrier
    }
  }
};
