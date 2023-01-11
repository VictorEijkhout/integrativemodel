template<int d>
class distribution {
 protected:
  /*
   * Orthogonal index structs
   */
  std::array<
    std::vector< indexstructure<index_int,1> >
    ,d> patches;
 public:
  distribution( const decomposition<d>&, const coordinate<index_int,d>& );

  /*
   * Numa stuff
   */
 protected:
  indexstructure<index_int,d> numa_structure;
  auto get_numa_structure() { return numa_structure; };
  const auto& get_numa_structure() const { return numa_structure; };
 public:
  coordinate<index_int,d> numa_first_index() const {
    return get_numa_structure()->first_index(); };
#if 0
  auto numa_offset() const {
    auto numa_loc = get_numa_structure()->first_index_r()[0];
    auto global_loc = get_global_structure()->first_index_r()[0];
    return numa_loc -global_loc;
  };
  index_int numa_local_size() { return get_numa_structure()->volume(); }; //!< \todo remove
  index_int numa_size() { return get_numa_structure()->volume(); };
#endif
};
