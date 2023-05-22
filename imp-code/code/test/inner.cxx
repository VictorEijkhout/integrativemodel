class Outer {
private: int i0;
  Outer(int i) : i0(i) {};
public:
  int compute(int i) { return i+i0;};
  class Inner {
  private:
    int i;
    Outer &outer;
  public:
    Inner(Outer& o) : outer(o) {};
    int operator*() { return outer.compute(i); };
  };
  Inner begin() {
    return Inner(*this);
  };
};
