template<int d>
distribution<d>::distribution
    ( const decomposition<d>& d, const coordinate<index_int>& c ) {
  for (int id=0; id<d; id++) {
    int length = d.size_of_dimension();
    vector< indexstructure<index_int> > segments(length);
    patches.at(id) = segments;
  }
};
