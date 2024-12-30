/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-2024
 ****
 **** imp_env.cxx : environment management
 ****
 ****************************************************************/

#include "imp_env.h"

using std::string;
using fmt::format_to;

/****
 **** Environment
 ****/

/*! For now, only store argc/argv */
void environment::init(int argc,char **argv,bool dothis) {
  //  entity::set_env(this);
  set_command_line(argc,argv);
  set_name("imp");

  debug_level = iargument("d",0);
  strategy = iargument("collective",0);
};

//! Reporting and cleanup
environment::~environment() {
  print_summary();
  close_ir_outputfile();
  delete_environment(); // left over stuff from derived environments
};

/*!
  Print general options.
  This routine will be augmented by the mode-specific calls such as 
  \ref mpi_environment::print_options. These will also typically abort.
*/
void environment::print_options() {
  printf("General options:\n");
  printf("  -optimize : optimize task graph\n");
  printf("  -queue_summary : summary task queue after execution\n");
  printf("  -progress/reduct : trace progress / reductions\n");
  printf("  -collective n where n=0 (ptp) 1 (ptp) 2 (group) 3 (recursive) 4 (MPI)\n");
};

bool environment::has_argument(const char *name) {
  string strname{name};
  bool has = false; // hasarg_from_argcv(name,nargs,the_args)
  //    || hasarg_from_internal(strname);
  // if (get_is_printing_environment())
  //   printf("arg <%s>:%d\n",name,has);
  return has;
};

int environment::iargument(const char *name,int vdef) {
  int r = true; // hasarg_from_argcv(name,nargs,the_args);
  if (r) {
    int v = 0; //iarg_from_argcv(name,vdef,nargs,the_args);
    // if (get_is_printing_environment())
    //   printf("arg <%s>:%d\n",name,v);
    // get_architecture()->print_trace
    //   (format("Argument <<{}>> supplied as <<{}>>",name,v));
    return v;
  } else return vdef;
};

#if 0
void environment::push_entity( entity *e ) {
  list_of_all_entities.push_back(e);
};

int environment::n_entities() const {
  return list_of_all_entities.size();
};

//! A long listing of the names of all defined entities.
void environment::list_all_entities() {
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::OBJECT) {
      object *o = dynamic_cast<object*>(ent);
      print("Object: {}\n",o->get_name());
    } else if (c==entity_cookie::KERNEL) {
      auto k = dynamic_cast<kernel*>(ent);
      fmt::print("Kernel: {}\n",k->get_name());
    } else if (c==entity_cookie::TASK) {
      auto k = dynamic_cast<kernel*>(ent);
      task *t = dynamic_cast<task*>(k);
      if (t!=nullptr)
      	fmt::print("Task: {}\n",t->get_name());
    } else if (c==entity_cookie::DISTRIBUTION) {
      // distribution *k = dynamic_cast<distribution*>(ent);
      // print("Distribution: {}\n",k->get_name());
    }
  };
};

//! Count the allocated space of all objects
double environment::get_allocated_space() {
  double allocated = 0.; int nobjects = 0;
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::OBJECT) {
      nobjects++;
      object *o = dynamic_cast<object*>(ent);
      if (o!=nullptr) {
	double s = (*e)->get_allocated_space();
	allocated += s;
      }
    }
  }
  return allreduce_d(allocated);
};

//! A quick summary of all defined entities.
result_tuple *environment::local_summarize_entities() {
  auto results = new result_tuple;
  int n_message = 0, n_object = 0, n_kernel = 0, n_task = 0, n_distribution = 0;
  double flopcount = 0., duration = 0., analysis = 0.,
    msg_volume = 0.;
  for (auto e : list_of_all_entities ) {
    entity_cookie c = e->get_cookie();
    entity *ent = (entity*)(e);
    try {
      if (0) {
      } else if (c==entity_cookie::DISTRIBUTION) {
	// distribution *k = dynamic_cast<distribution*>(ent);
	n_distribution++;
      } else if (c==entity_cookie::KERNEL) {
	//auto k = dynamic_cast<kernel*>(ent);
	n_kernel++;
      } else if (c==entity_cookie::TASK) {
	//auto k = dynamic_cast<kernel*>(ent);
	n_task++;
      } else if (c==entity_cookie::MESSAGE) {
	message *m = dynamic_cast<message*>(ent);
	if (m!=nullptr && m->get_sendrecv_type()==message_type::SEND) {
	  //print("Counting message <<{}>> times {}\n",m->as_string(),m->how_many_times);
	  n_message += m->how_many_times;
	  msg_volume += m->volume() * m->how_many_times;
	}
      } else if (c==entity_cookie::OBJECT) {
	object *o = dynamic_cast<object*>(ent);
	n_object++;
      } else if (c==entity_cookie::QUEUE) {
	algorithm *q = dynamic_cast<algorithm*>(ent);
	duration += q->execution_event.get_duration();
	analysis += q->analysis_event.get_duration();
	flopcount += q->get_flop_count();
      }
    } catch ( string s ) { print("ERROR: {}\n",s);
      throw( format("Could not summarize entity: {}",e->as_string()) );
    }
  };
  std::get<RESULT_OBJECT>(*results) = n_object;
  std::get<RESULT_KERNEL>(*results) = n_kernel;
  std::get<RESULT_TASK>(*results) = n_task; // no reduce!
  std::get<RESULT_DISTRIBUTION>(*results) = n_distribution;
  std::get<RESULT_ALLOCATED>(*results) = get_allocated_space();
  std::get<RESULT_DURATION>(*results) = duration;
  std::get<RESULT_ANALYSIS>(*results) = analysis;
  //  printf("found local nmessages %d\n",n_message);
  std::get<RESULT_MESSAGE>(*results) = n_message; // reduce
  std::get<RESULT_WORDSENT>(*results) = msg_volume; // reduce_d
  std::get<RESULT_FLOPS>(*results) = flopcount; // reduce_d
  return results;
};

/*!
  Convert the result of \ref environment::summarize_entities to a string.
 */
string environment::summary_as_string( result_tuple *results ) {
  fmt::memory_buffer w;
  format_to(w.end(),"Summary: ");
  format_to(w.end(),"#objects: {}",std::get<RESULT_OBJECT>(*results));
  format_to(w.end(),", #kernels: {}",std::get<RESULT_KERNEL>(*results));
  format_to(w.end(),", #tasks: {}",std::get<RESULT_TASK>(*results));
  format_to(w.end(),", |space|={}",(float)std::get<RESULT_ALLOCATED>(*results));
  format_to(w.end(),", analysis time={}",std::get<RESULT_ANALYSIS>(*results));
  format_to(w.end(),", runtime={}",std::get<RESULT_DURATION>(*results));
  // format_to(w.end(),", analysis time={:9.5e}",std::get<RESULT_ANALYSIS>(*results));
  // format_to(w.end(),", runtime={:9.5e}",std::get<RESULT_DURATION>(*results));
  format_to(w.end(),", #msg={}",std::get<RESULT_MESSAGE>(*results));
  format_to(w.end(),", #words sent={:7.2e}",std::get<RESULT_WORDSENT>(*results));
  format_to(w.end(),", flops={:7.2e}",std::get<RESULT_FLOPS>(*results));
  return to_string(w);
};

int environment::nmessages_sent( result_tuple *results ) {
  return std::get<RESULT_MESSAGE>(*results);
};

#endif

void environment::print_summary() {
#if 0
  if (has_argument("summary")) {
    fmt::memory_buffer w;
    format_to(w.end(),"Summary:\n");

    // summary architecture and settings
    //format_to(w.end(),"\n{}",arch.summary());

    // summary of entities
    auto summary = mode_summarize_entities();
    string summary_string = summary_as_string(summary);
    format_to(w.end(),"\n{}",summary_string);

    if (get_is_printing_environment())
      fmt::print("{}\n",summary_string);
  }
#endif
};
string environment::as_string() {
  return "env"; //  get_architecture().as_string();
};

void environment::kernels_to_dot_file() {
  // FILE *dotfile; string s;
  // dotfile = fopen(format("{}-kernels.dot",get_name()).data(),"w");
  // s = kernels_as_dot_string();
  // fprintf(dotfile,"%s\n",s.data());
  // fclose(dotfile);
};

/*!
  Make a really long string of all the kernels, with dependencies.
  \todo the way we get the algorithm name is not very elegant. may algorithm_as_dot_string, then call this?
 */
string environment::kernels_as_dot_string() {
#if 0
  fmt::memory_buffer w;
  format_to(w.end(),"digraph G {}\n",'{');
  for ( auto e : list_of_all_entities ) {
    entity_cookie c = e->get_cookie();
    if (c==entity_cookie::QUEUE) {
      algorithm *a = dynamic_cast<algorithm*>(e);
      if (a!=nullptr) {
	format_to(w.end(),"  label=\"{}\";\n",a->get_name());
	format_to(w.end(),"  labelloc=t;\n");
      }
    }
  }
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::KERNEL) {
      auto k = dynamic_cast<kernel*>(ent);
      string outname = k->get_out_object()->get_name();
      //format_to(w.end(),"  \"{}\" -> \"{}\";\n",k->get_name(),outname);
      auto deps = k->get_dependencies();
      for ( auto d : deps ) {
    	format_to(w.end(),"  \"{}\" -> \"{}\";\n",
    		d.get_in_object()->get_name(),outname
    		);
      }
    }
  }
  format_to(w.end(),"{}\n",'}');
  return to_string(w);
#endif
  return "kernels";
};

//! The basic case for single processor;
//! see \ref mpi_environment::tasks_to_dot_file
void environment::tasks_to_dot_file() {
#if 0
  fmt::memory_buffer w;
  {
    format_to(w.end(),"digraph G {}\n",'{');
    string s = tasks_as_dot_string();
    format_to(w.end(),"{}\n",s.data());
    format_to(w.end(),"{}\n",'}');
  }
  FILE *dotfile; 
  dotfile = fopen(format("{}-tasks.dot",get_name()).data(),"w");
  fprintf(dotfile,"%s\n",to_string(w).data());
  fclose(dotfile);
#endif
};

string environment::tasks_as_dot_string() {
#if 0
  fmt::memory_buffer w;
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::TASK) {
      auto k = dynamic_cast<kernel*>(ent);
      task *t = dynamic_cast<task*>(k);
      for ( auto d : t->get_predecessor_coordinates() ) { //=deps->begin(); d!=deps->end(); ++d) {
    	format_to(w.end(),"  \"{}-{}\" -> \"{}-{}\";\n",
    		d->get_step(),d->get_domain().as_string(),
    		t->get_step(),t->get_domain().as_string()
    		);
      }
    }
  }
  return to_string(w);
#endif
  return "tasks";
};

//! Print a line plus indentation to stdout or the output file.
void environment::print_line( string c ) {
  if (!get_is_printing_environment()) return;
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"%s%s\n",indentation->data(),c.data());
};

void environment::open_bracket() { this->print_line( (char*)"<<" ); };
void environment::close_bracket() { this->print_line( (char*)">>" ); };

void environment::print_to_file( const string& s ) {
  if (!get_is_printing_environment()) return;
  // stdout or the file
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"%s%s\n",indentation->data(),s.c_str());
};

void environment::print_to_file( int p,const string& s ) {
  if (!get_is_printing_environment()) return;
  // stdout or the file
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"[p%d] %s%s\n",p,indentation->data(),s.c_str());
};

//! Open a new output file, and close old one if needed.
void environment::set_ir_outputfile( const char *nam ) {
  if (!get_is_printing_environment()) return;
  if (ir_outputfile!=nullptr) fclose(ir_outputfile);
  fmt::memory_buffer w;
  format_to(w.end(),"{}.ir",nam);
  ir_outputfilename = to_string(w);
  ir_outputfile = fopen(ir_outputfilename.data(),"w");
};

void environment::register_execution_time(double t) {
  execution_times.push_back(t);
}

void environment::record_task_executed() {
  // the openmp version has the same, but with a critical section
  ntasks_executed++;
};

void environment::register_flops(double f) {
  flops += f;
}
