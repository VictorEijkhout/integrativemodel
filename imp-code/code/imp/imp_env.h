#pragma once

#include <stdlib.h>
#include <cstdio>
// #include <stdio.h>
#include <string.h>

#include "fmt/format.h"
#include "gsl/gsl-lite.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <vector>
#include <string>

#include "utils.h"

enum class environment_type {
  BASE,MPI,OMP,PRODUCT,HYBRID,IR};
#define result_tuple std::tuple<\
				/* 0: object count */ int,		\
				/* 1 : kernel count */ int,		\
				/* 2 : task count */ int,               \
				/* 3 : distribution count */ int,       \
				/* 4 : allocated space */ index_int,    \
				/* 5 : run duration */ double,          \
				/* 6 : analysis time */ double,         \
				/* 7 : message count */ int,            \
				/* 8 : message volume */ double,        \
				/* 9 : flop count */ double \
				>
#define RESULT_OBJECT 0
#define RESULT_KERNEL 1
#define RESULT_TASK 2
#define RESULT_DISTRIBUTION 3
#define RESULT_ALLOCATED 4
#define RESULT_DURATION 5
#define RESULT_ANALYSIS 6
#define RESULT_MESSAGE 7
#define RESULT_WORDSENT 8
#define RESULT_FLOPS 9

/*!
  An environment describes the processor structure. 

  There is a feeble attempt 
  to include profiling information in this object.

  \todo make \ref set_command_line virtual with default, override in mpi case
*/
class environment { // : public entity_name {
public:
  static environment& instance() {
    static environment the_instance;
    return the_instance;
  };
protected:
  environment() {};
  ~environment(); // print stuff, close files
public:
  environment(environment const&)    = delete;
  void operator=(environment const&) = delete;
  /*
   * Polymorphic functions
   */
public:
  std::function< int() > nprocs{
    [] () -> int { throw("Function undefined: nprocs"); return -1; } };
  std::function< int() > procid{
    [] () -> int { throw("Function undefined: procid"); return -1; } };

protected:
  int debug_level{0};
  // tracing
  std::vector<double> execution_times;
  double flops{0.};
  int ntasks_executed;
public:
  // Store the commandline and look for the debug parameter
  void init(int argc,char **argv,bool=true);
  // Reporting and cleanup
  std::function< void(void) > delete_environment{
    [] () -> void { return; } };
protected:
  std::string environment_name{""};
public:
  void set_name(std::string n) { environment_name = n; };
  auto get_name() const { return environment_name; };

protected:
  int strategy{0};
public:
  int get_collective_strategy() { return strategy; };

  /*
   * Type
   */
protected:
  environment_type type{environment_type::BASE};
public:
  int check_type_is( environment_type t ) { return type==t; };
  int is_type_mpi() { return type==environment_type::MPI; };
  int is_type_omp() { return type==environment_type::OMP; };

  /*
   * Collectives in the environment
   */
  std::function< index_int(index_int) > allreduce { [] (index_int i) -> index_int { return i; } };
  std::function< index_int(index_int) > allreduce_d { [] (double i) -> double { return i; } };
  std::function< int(int) > allreduce_and { [] (double i) -> double { return i; } };
    
  std::function< void(int contrib,std::vector<int>&) > gather32 {
    [] (int c,std::vector<int>&) -> void { throw(std::string("No default gather32")); } };
  std::function< void(index_int contrib,std::vector<index_int>&) > gather64 {
    [] (index_int c,std::vector<index_int>&) -> void { throw(std::string("No default gather64")); } };
  std::function < std::vector<index_int>*(index_int,int) > overgather {
    [] (index_int c,int o) -> std::vector<index_int>* { throw(std::string("No default gather")); } };
  std::function< int(int *senders,int root) > reduce_scatter{nullptr};
  
  /*
   * Commandline
   */
protected:
  int nargs{0}; char **the_args{nullptr};
public:
  void set_command_line(int argc,char **argv) { nargs = argc; the_args = argv; };
  bool has_argument(const char*);
  //! Return an integer commandline argument with default value
  virtual int iargument(const char*,int);
  //  static std::function< void(void) > print_application_options;
  virtual void print_options();
protected:
  std::vector< std::string > internal_args; // for now unused.
public:
  int hasarg_from_internal(std::string a) {
    for (auto arg : internal_args)
      if (arg==a) return 1;
    return 0;
  };

  // profiling
  void register_execution_time(double);
  void register_flops(double);
  void record_task_executed();
  virtual void print_stats() {}; // by default no-op
#define DEBUG_STATS 1
#define DEBUG_PROGRESS 2
#define DEBUG_VECTORS 4
#define DEBUG_MESSAGES 8
  int get_debug_level() { return debug_level; };

  /*
   * Printable output
   */
protected:
  int do_printing{1};
public:
  //! Test whether this environment does printing; this gets disabled on MPI for mytid>0.
  void set_is_printing_environment( int p=1 ) { do_printing = p; };
  //! Are we a printing environment?
  int get_is_printing_environment() { return do_printing; };
protected:
  std::string ir_outputfilename{"dag.ir"};
  FILE *ir_outputfile{nullptr};
  std::string *indentation{new std::string};
public:
  virtual void set_ir_outputfile( const char *n );
  void close_ir_outputfile() {
    if (ir_outputfile!=nullptr) { fclose(ir_outputfile); ir_outputfile = nullptr; };
  };
  //! By default, return a string counting the global number of processors
  virtual std::string as_string();
  //! By default print to standard out
  void increase_indent() { indentation->push_back(' '); indentation->push_back(' '); };
  void decrease_indent() { indentation->resize( indentation->size()-2 ); };
    //indentation->pop_back(); indentation->pop_back(); };
  void print_line( std::string );
  void open_bracket(); void close_bracket();
  //void print_object_line( char *s );
  virtual void print_to_file( std::string );
  virtual void print_to_file( const char* );
  virtual void print_to_file( int, std::string );
  virtual void print_to_file( int, const char* );
  //  virtual void print() { this->print_to_file( this->as_string() ); };
  virtual void print_single(std::string s) { print_to_file(s); };
  virtual void print_all(std::string s) { print_to_file(s); };

  /*
   * Entities
   */
#if 0
protected:
  //  static std::vector<entity*> list_of_all_entities;
public:
  void push_entity( entity *e ); int n_entities() const; void list_all_entities();
  // basic summary
  result_tuple *local_summarize_entities();
  // mode-specific summary is by default the basic one, but see MPI!
  std::function< result_tuple*(void) > mode_summarize_entities{
    [this] (void) -> result_tuple* { return local_summarize_entities(); } };
  double get_allocated_space(); int nmessages_sent(result_tuple*);
  std::string summary_as_string( result_tuple *results );
#endif
  void print_summary();
  std::string kernels_as_dot_string();
  void kernels_to_dot_file();
  std::string tasks_as_dot_string();
  virtual void tasks_to_dot_file(); // more complicated with MPI
};

