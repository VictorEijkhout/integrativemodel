#include <cmath>
#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::setw;

#include <vector>
using std::vector;

#include <Eigen/Dense>
using Eigen::Matrix;
using Eigen::MatrixXd;

class mat {
private:
  MatrixXd m;
public:
  mat(int n) { m = MatrixXd::Constant(n,n,0.); };
};

int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
