template<typename T>
class WithInt {
public:
  WithInt<T>() = default;
  template<typename U>
  WithInt<T>( WithInt<U>& );
};

// template<>
// WithInt<int>::WithInt( WithInt<short> ) {};

int main() {
  
  WithInt<int> withint;
  WithInt<short> withshort;

  WithInt<int> int_from_short( withshort );

  return 0;
}

template class WithInt<int>;
template class WithInt<short>
template WithInt<T>::WithInt<int>( WithInt<short>& ) {};
