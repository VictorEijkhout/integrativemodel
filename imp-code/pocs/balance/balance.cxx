#include <cmath>
#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

void setmovingweight( vector<double> &x,int step,int laststep ) {
  int gsize = x.size();
  double center = gsize * step / (1.*laststep);
  for ( int i=0; i<gsize; i++) {
    double imst = i-center;
    x.at(i) = 1 + gsize * exp( -imst*imst );
  }
}

vector<double> report_work
    ( int step,vector<double> work,vector<int> partitioning ) {
  int npartitions = partitioning.size()-1;
  vector<double> timing(npartitions);

  cout << "Step " << step << ": ";
  for (int ipart=0; ipart<npartitions; ipart++) {
    int istart = partitioning.at(ipart), iend = partitioning.at(ipart+1);
    double total=0;
    for (int point=istart; point<iend; point++)
      total += work.at(point);
    timing.at(ipart) = total;
    cout << "[ " << ipart << " - " << total << " ] ";
  }
  cout << endl;
  return timing;
}

// 1: solving 2: adaptive
#define SCHEME 2
vector<int> move_partitions( vector<int> cur,vector<double> timing ) {
  int npartitions = cur.size()-1, globalsize = cur.at(npartitions);
  vector<int> nxt(npartitions+1);
  vector<int> partition_size(npartitions,0);
  vector<double> work_rate(npartitions);
  
  for (int ipart=0; ipart<npartitions; ipart++) {
    partition_size.at(ipart) = cur.at(ipart+1)-cur.at(ipart);
    work_rate.at(ipart) = timing.at(ipart) / partition_size.at(ipart);
  }

#if SCHEME == 2
  vector<int> displacement(npartitions+1,0),
    dleft(npartitions,0), dright(npartitions,0),
    newtimes(npartitions,0);
  for (int ipart=1; ipart<=npartitions; ipart++) {
    displacement[ipart] = (timing[ipart]-timing[ipart-1])
      / sqrt(work_rate(ipart)*work_rate(ipart-1));
    dleft[ipart] = -min(0,displacement[ipart]);
    dright[ipart] = max(0,displacement[ipart]);
  }
  nxt[0] = cur[0]; nxt[npartitions] = cur[npartitions];
  for (int ipart=1; ipart<npartitions; ipart++) {
    nxt[ipart] = cur[ipart]
      + dright[ipart]*(cur[ipart]-cur[ipart-1])
      - dleft[ipart]*cur[ipart]
      +
  }
#endif

#if SCHEME == 1
  vector<double> diag(npartitions),up(npartitions),lo(npartitions);
  for (int ipart=0; ipart<npartitions; ipart++) {
    diag.at(ipart) = work_rate.at(ipart);
    if (ipart<npartitions-1) {
      up.at(ipart) = work_rate.at(ipart+1);
      lo.at(ipart) = work_rate.at(ipart);
    }
  }
  diag.at(0) = 1./diag.at(0);
  for (int ipart=0; ipart<npartitions-1; ipart++) {
    cout << "pivot " << ipart+1 << " : " << diag.at(ipart+1) ;
    diag.at(ipart+1) -= up.at(ipart) * diag.at(ipart) * lo.at(ipart);
    cout << " => " << diag.at(ipart+1) << endl;
    diag.at(ipart+1) = 1./diag.at(ipart+1);
  }

  // forward solve, using normalized L
  vector<double> z(npartitions), nxt_size(npartitions);
  z.at(0) = partition_size.at(0);
  for (int ipart=1; ipart<npartitions; ipart++)
    z.at(ipart) = partition_size.at(ipart)
      - lo.at(ipart-1) * diag.at(ipart-1) * z.at(ipart-1);
  // backward solve
  nxt_size.at(npartitions-1) = z.at(npartitions-1);
  for (int ipart=npartitions-2; ipart>=0; ipart--)
    nxt_size.at(ipart) = z.at(ipart);

  // see how the new sizes relate
  int nxt_total = 0;
  for (int ipart=0; ipart<npartitions; ipart++)
    nxt_total += nxt_size.at(ipart);
  int excess = nxt_total - cur.at(npartitions);

  if (excess<0) { excess = -excess;
    cout << "losing undershoot of " << excess << endl;
    int add = excess / npartitions;
    for (int ipart=0; ipart<npartitions; ipart++)
      nxt_size.at(ipart) += add;
    excess -= npartitions * add;
    for (int ipart=0; ipart<excess; ipart++)
      nxt_size.at(ipart) += 1;    
  } else if (excess>0) {
    cout << "losing overshoot of " << excess << endl;
    int add = excess / npartitions;
    for (int ipart=0; ipart<npartitions; ipart++)
      nxt_size.at(ipart) -= add;
    excess -= npartitions * add;
    for (int ipart=0; ipart<excess; ipart++)
      nxt_size.at(ipart) -= 1;    
  }
  
  // set the partitions
  nxt.at(0) = 0;
  for (int ipart=0; ipart<npartitions; ipart++)
    nxt.at(ipart+1) = nxt.at(ipart) + nxt_size.at(ipart);

  return nxt;
#endif
}

int main() {

#define N 1000
#define P 10
#define STEPS 20

  vector<double> work(N);
  vector<double> timing(P);
  vector<int> partitioning(P+1);

  for (int ip=0; ip<P; ip++)
    partitioning.at(ip) = ip*N/(double)P;
  partitioning.at(P) = N;

  for (int step=0; step<STEPS; step++) {
    setmovingweight(work,step,STEPS);
    timing = report_work(step,work,partitioning);
    partitioning = move_partitions(partitioning,timing);
  }
  
  return 0;
}
