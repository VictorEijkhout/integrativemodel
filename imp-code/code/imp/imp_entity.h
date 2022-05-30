/*!
  A simple class for containing an entity name; 
  right now only architecture inherits from this; 
  everything else inherits from \ref entity.
  \todo roll this into \ref entity
 */

/****
 **** Entity
 ****/

/*!
  We use this to track globally what kind of entities we have.
  
  SHELLKERNEL : kernel, except that we do not want it to show up in
          dot files. Stuff like reduction.
 */
enum class entity_cookie { UNKNOWN,
    ARCHITECTURE, COMMUNICATOR, DECOMPOSITION, DISTRIBUTION, MASK,
    SHELLKERNEL, SIGNATURE,
    KERNEL, TASK, MESSAGE, OBJECT, OPERATOR, QUEUE };

enum class trace_level {
  NONE=0, CREATE=1, PROGRESS=2, MESSAGE=4, REDUCT=8
    };

class environment;
/*!
  We define a basic entity class that everyone inherits from.
  This is mostly to be able to keep track of everything:
  there is a static environment member which has a list of entities;
  every newly created entity goes on this list.
*/
class entity {

protected:
  int entity_number{0};
  static inline int entity_count{0};
public:
  //! Default constructor
  entity();
  //! General constructor
  entity( entity_cookie c );
  entity( entity *e,entity_cookie c );
  ~entity() {};

private:
  std::string name;
public:
  virtual void set_name( const std::string &n );
  const std::string &get_name() const;

  /*
   * What kind of entity is this?
   */
protected:
  entity_cookie typecookie{entity_cookie::UNKNOWN};
public:
  void set_cookie( entity_cookie c ) { typecookie = c; };
  entity_cookie get_cookie() const { return typecookie; };
  std::string cookie_as_string() const;

public:
  static trace_level tracing;
  //static bool trace_progress;
public:
  static void add_trace_level( trace_level lvl ) {
    tracing = (trace_level) ( (int)tracing | (int)lvl );
  };
  bool has_trace_level( trace_level lvl ) {
    return (int)tracing & (int)lvl;
  };
  trace_level get_trace_level() { return tracing; };
  bool tracing_progress() { return ((int)tracing & (int)trace_level::PROGRESS)>0; };
  
  /*
   * Statistics: allocation and timing
   */
protected:
  float allocated_space{0.};
public:
  void register_allocated_space( float s ) { allocated_space += s; };
  float get_allocated_space() { return allocated_space; };

  /*
   * Output
   */
  virtual std::string as_string() const { //!< Base class method for rendering as string
    return "entity"; // fmt::format("type: {}",cookie_as_string());
  };
};

