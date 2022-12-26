#if 0
template<typename I,int d>
class multi_indexstruct;
template<typename I,int d>
class multi_sigma_operator_implementation {
protected:
  int dim{-1};
public:
  multi_sigma_operator_implementation(int dim) : dim(dim) {};
  bool get_same_dimensionality(int d) const {
    if (dim!=d) throw(fmt::format("Not the same dimensionality: {}<>{}",dim,d));
    return dim;
  };
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const domain_coordinate &point ) const = 0;
  virtual std::shared_ptr<multi_indexstruct> operate
      ( const multi_indexstruct &idx ) const = 0;
  virtual bool is_single_operator() const { return false; };
  virtual bool is_point_operator() const { return false; };
  virtual bool is_coord_struct_operator() const { return false; };
  virtual sigma_operator get_operator(int id) const {
    throw(fmt::format("Can not get operator in dim {}",id)); };
};

template<typename I,int d>
class multi_ioperator;
/*!
  We have limited support for multi-dimensional indexstructs.
  \todo lose the dim parameter
*/
template<typename I,int d>
class multi_indexstruct : public std::enable_shared_from_this<multi_indexstruct> {
protected:
  int dim{-1};
  std::vector< std::shared_ptr<indexstruct<I,d>> > components;
public:
  std::vector< std::shared_ptr<multi_indexstruct> > multi;
public:
  multi_indexstruct() {}; 
  multi_indexstruct(int);
  multi_indexstruct(std::shared_ptr<indexstruct<I,d>>);
  multi_indexstruct(domain_coordinate);
  multi_indexstruct(domain_coordinate,domain_coordinate);
  multi_indexstruct( std::vector< std::shared_ptr<indexstruct<I,d>> > );
  multi_indexstruct( std::vector<I> sizes );

  bool is_uninitialized() const { return dim<0; };
  bool is_multi() const { return multi.size()>0; };
  int multi_size() const { return multi.size(); };
  std::shared_ptr<multi_indexstruct> make_clone() const;
  std::shared_ptr<multi_indexstruct> relativize_to( std::shared_ptr<multi_indexstruct> );
  bool is_strided() const;
  int type_as_int() const;
  std::string as_string() const;
      
  /*
   * Access
   */
  int get_dimensionality() const { return dim; }; int get_same_dimensionality(int) const;
  bool is_known() const; bool is_empty() const; bool is_contiguous() const;
  std::string type_as_string() const;
protected:
  void set_needs_recomputing() { stored_volume = -1;
    stored_first_index = nullptr; stored_last_index = nullptr;
  };
  domain_coordinate stored_local_size;

protected:
  //! This has to be a pointer otherwise we get infinite inclusion
  mutable std::shared_ptr<multi_indexstruct> stored_enclosing_structure{nullptr};
public:
  //  void compute_enclosing_structure();
  const multi_indexstruct &enclosing_structure_r() const;
  const std::shared_ptr<multi_indexstruct> enclosing_structure() const;

protected:
  mutable I stored_volume{-1};
public:
  I volume() const;

protected:
  mutable std::shared_ptr<domain_coordinate> stored_first_index{nullptr};
  mutable std::shared_ptr<domain_coordinate> stored_last_index{nullptr};
public:
  //  void compute_first_index(); void compute_last_index();
  const domain_coordinate &first_index_r() const;
  const domain_coordinate &last_index_r() const;
  const domain_coordinate &local_size_r() const;
  domain_coordinate *stride() const;
  // 1d cases
  const coordinate<I,d> first_index(int d) const { return components.at(d)->first_index(); };
  const coordinate<I,d> last_index(int d)  const { return components.at(d)->last_index(); };
  I volume(int d)  const { return components.at(d)->volume(); };
  std::vector<domain_coordinate> get_corners() const;

  /*
   * Data manipulation
   */
  void set_component( int, std::shared_ptr<indexstruct<I,d>> );
  std::shared_ptr<indexstruct<I,d>> get_component(int) const;
  //  bool contains_element( const domain_coordinate *i );
  bool contains_element( const domain_coordinate &i ) const;
  bool contains_element( const domain_coordinate &&i ) const;
  bool contains( std::shared_ptr<multi_indexstruct> ) const;
  bool contains( const multi_indexstruct& ) const;
  bool equals( std::shared_ptr<multi_indexstruct> ) const;
  bool operator==( const multi_indexstruct& ) const;

  I linear_location_of( std::shared_ptr<multi_indexstruct> ) const;
  I linear_location_in( std::shared_ptr<multi_indexstruct> ) const;
  I linear_location_in( const multi_indexstruct& ) const;
  I linear_location_of( domain_coordinate * ) const;
  domain_coordinate *location_of( std::shared_ptr<multi_indexstruct> inner ) const;
  I linearfind( I i );
  domain_coordinate linear_offsets(std::shared_ptr<multi_indexstruct> ) const;

  /*
   * Operations
   */
  std::shared_ptr<multi_indexstruct> operate( const ioperator<I,d> &op ) const;
  std::shared_ptr<multi_indexstruct> operate( const ioperator<I,d> &&op ) const;
  // VLE lose!
  std::shared_ptr<multi_indexstruct> operate( multi_ioperator* ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( multi_ioperator*,const multi_indexstruct& ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( multi_ioperator*,std::shared_ptr<multi_indexstruct> ) const;
  // end lose
  std::shared_ptr<multi_indexstruct> operate( const multi_sigma_operator& ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( const ioperator<I,d>&,std::shared_ptr<multi_indexstruct> ) const;
  std::shared_ptr<multi_indexstruct> operate
      ( const ioperator<I,d>&,const multi_indexstruct& ) const;
  void translate_by(int d,I amt);

  std::shared_ptr<multi_indexstruct> struct_union( std::shared_ptr<multi_indexstruct>,bool=false );
  bool can_union_in_place(std::shared_ptr<multi_indexstruct> other,int &diff) const;
  std::shared_ptr<multi_indexstruct> struct_union_in_place
      (std::shared_ptr<multi_indexstruct> other,int diff);
  std::shared_ptr<multi_indexstruct> minus( std::shared_ptr<multi_indexstruct>,bool=false ) const;
  std::shared_ptr<multi_indexstruct> split_along_dim(int,std::shared_ptr<indexstruct<I,d>>) const;
  std::shared_ptr<multi_indexstruct> intersect( std::shared_ptr<multi_indexstruct> );
  std::shared_ptr<multi_indexstruct> intersect( const multi_indexstruct& );
  std::shared_ptr<multi_indexstruct> force_simplify(bool=false) const;

  /*
   * Ranging
   */
protected:
  I cur_linear{0};
  domain_coordinate cur_coord;
public:
  multi_indexstruct &begin();
  multi_indexstruct &end();
  void operator++();
  bool operator!=( multi_indexstruct& );
  bool operator==( multi_indexstruct& );
  domain_coordinate &operator*();
};

//! A multi_indexstruct with all components unknown
template<typename I,int d>
class unknown_multi_indexstruct : public multi_indexstruct {
public:
  unknown_multi_indexstruct( int dim ) : multi_indexstruct(dim) {
    for (int id=0; id<dim; id++)
      set_component( id, std::shared_ptr<indexstruct<I,d>>( new unknown_indexstruct() ) );
  };
};

/*!
  A empty multi indexstruct is mostly an initializer shortcut. 
  For now there are no specific methods for this class.
  \todo write a unit test for being empty
*/
template<typename I,int d>
class empty_multi_indexstruct : public multi_indexstruct {
public:
  empty_multi_indexstruct( int dim ) : multi_indexstruct(dim) {
    for (int id=0; id<dim; id++)
      set_component( id, std::shared_ptr<indexstruct<I,d>>( new empty_indexstruct() ) );
  };
};

/*!
  A contiguous multi indexstruct is mostly an initializer shortcut. 
  For now there are no specific methods for this class.
  \todo write a unit test for being contiguous
*/
template<typename I,int d>
class contiguous_multi_indexstruct : public multi_indexstruct {
public:
  contiguous_multi_indexstruct( const domain_coordinate first ) :
    contiguous_multi_indexstruct(first,first) {};
  contiguous_multi_indexstruct
      ( const domain_coordinate first,const domain_coordinate last ) :
    multi_indexstruct(first.get_dimensionality()) {
    int dim = first.get_same_dimensionality(last.get_dimensionality());
    for (int id=0; id<dim; id++)
      set_component
	( id, std::shared_ptr<indexstruct<I,d>>
	  ( new contiguous_indexstruct(first.coord(id),last.coord(id)) ) );
  };
};

/*!
  A strided multi indexstruct is mostly an initializer shortcut. 
  For now there are no specific methods for this class.
  \todo write a unit test for being strided
*/
template<typename I,int d>
class strided_multi_indexstruct : public multi_indexstruct {
public:
  strided_multi_indexstruct( domain_coordinate first,domain_coordinate last,int stride ) :
    multi_indexstruct(first.get_dimensionality()) {
    int dim = first.get_same_dimensionality( last.get_dimensionality() );
    for (int id=0; id<dim; id++)
      set_component
	( id, std::shared_ptr<indexstruct<I,d>>
	  ( new strided_indexstruct(first.coord(id),last.coord(id),stride) ) );
  };
};

/*! A vector of \ref ioperator objects.
 */
template<typename I,int d>
class multi_ioperator {
protected:
  std::vector<ioperator<I,d> /*std::shared_ptr<ioperator<I,d>>*/ > operators;
  bool operator_based{false};
  std::function< domain_coordinate*(domain_coordinate*) > pointf; bool point_based{false};
public:
  //! Create the operator vector and set with no-ops
  multi_ioperator( int d ) {
    if (d<=0)
      throw(fmt::format("multi ioperator dimension {} s/b >=1",d));
    for (int dm=0; dm<d; dm++)
      operators.push_back( ioperator("none") );
    //operators.push_back( std::shared_ptr<ioperator>(new ioperator("none")) );
    operator_based = true;
  };
  //! Create from same ioperator in all dimensions
  multi_ioperator( int dim,ioperator &op ) : multi_ioperator(dim) {
    for (int idim=0; idim<dim; idim++)
      set_operator(idim,op);
    operator_based = true;
  };
  multi_ioperator( int dim,ioperator &&op ) : multi_ioperator(dim) {
    for (int idim=0; idim<dim; idim++)
      set_operator(idim,op);
    operator_based = true;
  };
  //! Create from a single ioperator: one-d case shortcut.
  multi_ioperator( ioperator &op ) : multi_ioperator(1,op) {};
  multi_ioperator( ioperator &&op ) : multi_ioperator(1,op) {};
  // Create from explicit function
  multi_ioperator( std::function< domain_coordinate*(domain_coordinate*) > f ) {
    pointf = f; point_based = true; };
  
protected:
  iop_type type{iop_type::INVALID};
public:
  bool is_shift_op()       const {
    return type==iop_type::SHIFT_REL || type==iop_type::SHIFT_ABS; };
  //! For functionally defined operators, the user can set the type.
  void set_is_shift_op() { type = iop_type::SHIFT_ABS; };

  int get_dimensionality() const { return operators.size(); };
  int get_same_dimensionality( int d ) const {
    int dim = get_dimensionality();
    if (d!=dim) throw(fmt::format("multi_iop differing dims: {} & {}",dim,d));
    return dim; };
  const ioperator &get_operator(int id) const { return operators.at(id); };
  //void set_operator(int id,std::shared_ptr<ioperator> op);
  void set_operator(int id,ioperator &op);
  void set_operator(int id,ioperator &&op);
  bool is_modulo_op(); bool is_shift_op(); bool is_restrict_op();

  // operate on coordinate & multi_indexstruct
  domain_coordinate *operate( domain_coordinate *c );
  std::shared_ptr<multi_indexstruct> operate( std::shared_ptr<multi_indexstruct> idx ) {
    return idx->operate(this); };

  std::string as_string() {
    fmt::memory_buffer w;
    fmt::format_to(w.end(),"["); int id=0; const char *sep="";
    for ( auto iop : operators ) {
      fmt::format_to(w.end(),"{}op{}:{}",sep,id++,iop.as_string());
      sep = ", ";
    }
    fmt::format_to(w.end(),"]");
    return fmt::to_string(w);
  };
};

template<typename I,int d>
class multi_shift_operator : public multi_ioperator {
public:
  multi_shift_operator(int d) : multi_ioperator(d) {};
  multi_shift_operator( std::vector<I> shifts )
    : multi_ioperator(shifts.size()) {
    for (int id=0; id<shifts.size(); id++) {
      set_operator(id,shift_operator(shifts[id]));
    }
  };
  multi_shift_operator( domain_coordinate d )
    : multi_shift_operator( d.data() ) {}
};

/*! A stencil is an array of \ref multi_shift_operator objects.
  This is mostly for notational convenience.
  \todo make this an array of std::shared_ptr's; has to wait for \ref signature_function::add_sigma_oper to accept shared pointers.
*/
template<typename I,int d>
class stencil_operator {
  std::vector<multi_shift_operator*> ops;
  int dim{0};
public:
  stencil_operator(int d) { dim = d; };
  std::vector<multi_shift_operator*> &get_operators() { return ops; };
  void add(int i,int j) {
    if (dim!=2)
      throw(fmt::format("Stencil operator dimensionality defined as {}",dim));
    ops.push_back( new multi_shift_operator(std::vector<I>{i,j}) );
  };
  void add(int i,int j,int k) {
    if (dim!=3)
      throw(fmt::format("Stencil operator dimensionality defined as {}",dim));
    ops.push_back( new multi_shift_operator(std::vector<I>{i,j,k}) );
  };
  void add( const std::vector<I> &offset ) {
    if(dim!=offset.size())
      throw(fmt::format("Offset dimensionality {} incompatible with {}",offset.size(),dim));
    ops.push_back( new multi_shift_operator(offset) );
  };
  void add( domain_coordinate offset ) {
    if(dim!=offset.get_dimensionality())
      throw(fmt::format("DC Offset dimensionality {} incompatible with {}",
			offset.get_dimensionality(),dim));
    ops.push_back( new multi_shift_operator(offset) );
  };
  int get_dimensionality() const { return dim; };
};

template<typename I,int d>
class multi_indexstructure {
private:
  std::shared_ptr<multi_indexstruct> strct{nullptr};
public:
  multi_indexstructure() {};
  multi_indexstructure(int dim) {
    strct = std::shared_ptr<multi_indexstruct>( new unknown_multi_indexstruct(dim) );
  };
};
  // VLE why do we have an explicit copy constructor?
  // indexstructure( const indexstructure &idx )
  //   : indexstructure( idx.strct->make_clone() ) {};
  // indexstructure( const indexstructure &&idx )
  //   : indexstructure( idx.strct->make_clone() ) {};
  // //! if we omit this one, all assignments complain about `use of deleted operator='
  // indexstructure operator=(const indexstructure &idx ) const {
  //   fmt::print("indexstructure assignment operator is broken\n");
  //   auto c = dynamic_cast<contiguous_indexstruct*>( idx.strct.get() );
  //   if (c!=nullptr) {
  //     fmt::print("indexstructure operator= with contiguous\n");
  //     return indexstructure( contiguous_indexstruct(*c) );
  //   } else
  //     throw(std::string("Operator= not for this type"));
  // };

#endif
