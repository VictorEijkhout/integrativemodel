#include "imp_entity.h"

entity::entity() { entity_number = entity_count++; };

/*
 * Name handling
 */
virtual void entity::set_name( const std::string& n ) { name = n; };
const std::string &entity::get_name() const { return name; };

/*!
  Just about anything created in IMP is an entity, meaning that it's
  going to be pushed in a a list in the environment.
 */
//! This is the constructor that everyone should use
entity::entity( entity_cookie c ) : entity() {
  //  print("entity {}\n",(int)c);
  // env->push_entity(this);
  set_cookie(c);
};

string entity::cookie_as_string() const {
  switch (typecookie) {
  case entity_cookie::UNKNOWN :      return string("unknown"); break;
  case entity_cookie::DISTRIBUTION : return string("distribution"); break;
  case entity_cookie::KERNEL :       return string("kernel"); break;
  case entity_cookie::TASK :         return string("task"); break;
  case entity_cookie::MESSAGE :      return string("message"); break;
  case entity_cookie::OBJECT :       return string("object"); break;
  case entity_cookie::OPERATOR :     return string("operator"); break;
  case entity_cookie::QUEUE :        return string("queue"); break;
  case entity_cookie::ARCHITECTURE : return string("arch"); break;
  case entity_cookie::COMMUNICATOR : return string("comm"); break;
  case entity_cookie::DECOMPOSITION: return string("decomp"); break;
  default : return string("other");
  };
};

