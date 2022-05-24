/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** template_gropp.cxx : 
 ****     mode-independent template for conjugate gradients according to Gropp
 ****
 ****************************************************************/

/*! \page gropp Conjugate Gradients Method by Bill Gropp

  This is based on a presentation by Bill Gropp. However, I think
  his algorithm is wrong, since it contains two matrix vector products
  per iteration.
*/

#include "template_common_header.h"

/****
 **** Main program
 ****/

//! \test There is a test for a CG method with overlap of computation/communication. See \subpage gropp.

int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("Pipelined CG options:\n");
    printf("  -nlocal nnn : set points per processor\n");
    printf("  -steps nnn : set number of iterations\n");
    printf("  -trace : print norms\n");
    printf("\n");
  };

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("gropp");
  
  architecture *arch = env->get_architecture();
  //  arch->set_collective_strategy_group();
  decomposition decomp = IMP_decomposition(arch);
  
  int nlocal = env->iargument("nlocal",100); // points per processor
  int n_iterations = env->iargument("steps",20);
  int trace = env->has_argument("trace");
  
  // a bunch of vectors, block distributed
  auto blocked = shared_ptr<distribution>( new IMP_block_distribution(decomp,nlocal,-1) );
  shared_ptr<object> xt,x0,b0,r0,z0,ax0;
  xt    = shared_ptr<object>( new IMP_object(blocked) ); xt->set_name(fmt::format("xtrue"));
  xt->allocate(); xt->set_value(1.);
  x0    = shared_ptr<object>( new IMP_object(blocked) ); x0->set_name(fmt::format("x0"));
  x0->allocate(); x0->set_value(0.);
  b0    = shared_ptr<object>( new IMP_object(blocked) ); b0->set_name(fmt::format("b0"));
  ax0   = shared_ptr<object>( new IMP_object(blocked) ); ax0->set_name(fmt::format("ax0"));
  r0    = shared_ptr<object>( new IMP_object(blocked) ); r0->set_name(fmt::format("r0"));
  z0    = shared_ptr<object>( new IMP_object(blocked) ); z0->set_name(fmt::format("z0"));

  // scalars, all redundantly replicated
  auto scalar = shared_ptr<distribution>( new IMP_replicated_distribution(decomp) );
  vector<shared_ptr<object>> rnorms(n_iterations);  
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = shared_ptr<object>( new IMP_object(scalar) ); rnorms[it]->set_name(fmt::format("rnorm{}",it));
    rnorms[it]->allocate();
  }

  shared_ptr<object> one = shared_ptr<object>( new IMP_object(scalar) ); one->set_name("one"); one->set_value(1.);
  shared_ptr<object> oNe = shared_ptr<object>( new IMP_object(scalar) ); oNe->set_name("oNe"); oNe->set_value(1.);
  
  // let's define the steps of the loop body
  auto pipe_cg = shared_ptr<algorithm>( new IMP_algorithm(decomp) );
  pipe_cg->set_name("Pipelined Conjugate Gradients");
  
  // initial setup
  { auto k = shared_ptr<kernel>( new IMP_origin_kernel(one) ); k->set_name("origin one");
    pipe_cg->add_kernel( k ); }
  { auto k = shared_ptr<kernel>( new IMP_origin_kernel(oNe) ); k->set_name("origin oNe");
    pipe_cg->add_kernel( k ); }
  { auto xorigin = shared_ptr<kernel>( new IMP_origin_kernel( xt ) ); xorigin->set_name("origin xtrue");
    pipe_cg->add_kernel(xorigin); }
  { auto xorigin = shared_ptr<kernel>( new IMP_origin_kernel( x0 ) ); xorigin->set_name("origin x0");
    pipe_cg->add_kernel(xorigin); }
  { auto borigin = shared_ptr<kernel>( new IMP_centraldifference_kernel( xt,b0 ) ); borigin->set_name("b0=A xtrue");
    pipe_cg->add_kernel(borigin); }
  { auto atimesx0 = shared_ptr<kernel>( new IMP_centraldifference_kernel( x0,ax0 ) ); atimesx0->set_name("A x0");
    pipe_cg->add_kernel(atimesx0); }
  { auto rorigin = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,ax0, '-',oNe,b0, r0 ) ); rorigin->set_name("r0=ax0-b");
    pipe_cg->add_kernel(rorigin); }
  if (trace) {
    shared_ptr<object> rr0 = shared_ptr<object>( new IMP_object(scalar) );
    auto r0inp = shared_ptr<kernel>( new IMP_norm_kernel( r0,rr0 ) ); pipe_cg->add_kernel( r0inp );
    pipe_cg->add_kernel( shared_ptr<kernel>( new IMP_trace_kernel(rr0,std::string("Initial residual norm") )) );
  }
  { auto precr0 = shared_ptr<kernel>( new IMP_preconditioning_kernel( r0,z0 ) ); precr0->set_name("z0=Mr0");
    pipe_cg->add_kernel(precr0); }

  // define objects that need to carry from one iteration to the next
  shared_ptr<object> xcarry,rcarry,zcarry, pcarry,qcarry, rrp;
  
  for (int it=0; it<n_iterations; it++) {
    
    shared_ptr<object> x,r,z,az,p,q,mq, /* xcarry,rcarry,pcarry, rrp have to persist */
      rr{nullptr},rrp{nullptr},pap{nullptr},alpha{nullptr},beta{nullptr};

    x = shared_ptr<object>( new IMP_object(blocked) ); x->set_name(fmt::format("x{}",it));
    r = shared_ptr<object>( new IMP_object(blocked) ); r->set_name(fmt::format("r{}",it));
    z = shared_ptr<object>( new IMP_object(blocked) ); z->set_name(fmt::format("z{}",it));
    az = shared_ptr<object>( new IMP_object(blocked) ); az->set_name(fmt::format("az{}",it));
    p = shared_ptr<object>( new IMP_object(blocked) ); p->set_name(fmt::format("p{}",it));
    q = shared_ptr<object>( new IMP_object(blocked) ); q->set_name(fmt::format("q{}",it));
    mq = shared_ptr<object>( new IMP_object(blocked) ); mq->set_name(fmt::format("mq{}",it));

    rr    = shared_ptr<object>( new IMP_object(scalar) ); rr->set_name(fmt::format("rr{}",it));
    pap   = shared_ptr<object>( new IMP_object(scalar) ); pap->set_name(fmt::format("pap{}",it));
    alpha = shared_ptr<object>( new IMP_object(scalar) ); alpha->set_name(fmt::format("alpha{}",it));
    beta  = shared_ptr<object>( new IMP_object(scalar) ); beta->set_name(fmt::format("beta{}",it));

    if (it==0) {
      { auto xcopy = shared_ptr<kernel>( new IMP_copy_kernel( x0,x ) );
	pipe_cg->add_kernel(xcopy); xcopy->set_name(fmt::format("start x-{}",it)); }
      { auto rcopy = shared_ptr<kernel>( new IMP_copy_kernel( r0,r ) );
	pipe_cg->add_kernel(rcopy); rcopy->set_name(fmt::format("start r-{}",it)); }
      { auto zcopy = shared_ptr<kernel>( new IMP_copy_kernel( z0,z ) );
	pipe_cg->add_kernel(zcopy); zcopy->set_name(fmt::format("start z-{}",it)); }
    } else {
      auto xcopy = shared_ptr<kernel>( new IMP_copy_kernel( xcarry,x ) );
      pipe_cg->add_kernel(xcopy); xcopy->set_name(fmt::format("copy x-{}",it));
      auto rcopy = shared_ptr<kernel>( new IMP_copy_kernel( rcarry,r ) );
      pipe_cg->add_kernel(rcopy); rcopy->set_name(fmt::format("copy r-{}",it));
      auto zcopy = shared_ptr<kernel>( new IMP_copy_kernel( zcarry,z ) );
      pipe_cg->add_kernel(zcopy); zcopy->set_name(fmt::format("copy z-{}",it));
    }

    xcarry = shared_ptr<object>( new IMP_object(blocked) ); xcarry->set_name(fmt::format("x{}p",it));
    rcarry = shared_ptr<object>( new IMP_object(blocked) ); rcarry->set_name(fmt::format("r{}p",it));
    zcarry = shared_ptr<object>( new IMP_object(blocked) ); zcarry->set_name(fmt::format("z{}p",it));

    //snippet gropptemplate
    auto rnorm = shared_ptr<kernel>( new IMP_norm_kernel( r,rnorms[it] ) );
    pipe_cg->add_kernel(rnorm); rnorm->set_name(fmt::format("rnorm-{}",it));
    if (trace) {
      auto trace = shared_ptr<kernel>( new IMP_trace_kernel(rnorms[it],fmt::format("Norm in iteration {}",it)) );
      pipe_cg->add_kernel(trace); trace->set_name(fmt::format("rnorm trace {}",it));
    }

    rrp = shared_ptr<object>( new IMP_object(scalar) ); rrp->set_name(fmt::format("rho{}",it));

    { auto rho_inprod = shared_ptr<kernel>( new IMP_innerproduct_kernel( r,z,rr ) );
      pipe_cg->add_kernel(rho_inprod); rho_inprod->set_name(fmt::format("compute rho-{}",it)); }

    { auto matvec = shared_ptr<kernel>( new IMP_centraldifference_kernel( z,az ) );
      pipe_cg->add_kernel(matvec); matvec->set_name(fmt::format("z matvec-{}",it)); }

    if (it==0) { // initialize z<-p, az<-s

      auto pisz = shared_ptr<kernel>( new IMP_copy_kernel( z,p ) );
      pipe_cg->add_kernel(pisz); pisz->set_name(fmt::format("copy z to p-{}",it));
      auto qisaz = shared_ptr<kernel>( new IMP_copy_kernel( az,q ) );
      pipe_cg->add_kernel(qisaz); qisaz->set_name(fmt::format("copy az to q-{}",it));

    } else { // update p,q and copy rr

      auto beta_calc = shared_ptr<kernel>( new IMP_scalar_kernel( rr,"/",rrp,beta ) );
      pipe_cg->add_kernel(beta_calc); beta_calc ->set_name(fmt::format("compute beta-{}",it));
      auto rrcopy = shared_ptr<kernel>( new IMP_copy_kernel( rr,rrp ) );
      pipe_cg->add_kernel(rrcopy); rrcopy->set_name(fmt::format("save rr value-{}",it));

      auto pupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,z, '+',beta,pcarry, p ) );
      pipe_cg->add_kernel(pupdate); pupdate   ->set_name(fmt::format("update p-{}",it));
      auto qupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,az, '+',beta,qcarry, q ) ); // s = Ap?
      pipe_cg->add_kernel(qupdate); qupdate->set_name(fmt::format("update q-{}",it));

    }
  
    { auto precon = shared_ptr<kernel>( new IMP_preconditioning_kernel( q,mq ) );
      pipe_cg->add_kernel(precon); precon->set_name(fmt::format("q precon-{}",it)); }

    { auto pap_inprod = shared_ptr<kernel>( new IMP_innerproduct_kernel( p,q,pap ) );
      pipe_cg->add_kernel(pap_inprod); pap_inprod->set_name(fmt::format("pap innprod-{}",it)); }

    { auto alpha_calc = shared_ptr<kernel>( new IMP_scalar_kernel( rr,"/",pap,alpha ) );
      pipe_cg->add_kernel(alpha_calc); alpha_calc->set_name(fmt::format("compute alpha-{}",it)); }

    { auto xupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,x, '-',alpha,p, xcarry ) );
      pipe_cg->add_kernel(xupdate); xupdate->set_name(fmt::format("update x-{}",it)); }

    { auto rupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,r, '-',alpha,q, rcarry ) );
      pipe_cg->add_kernel(rupdate); rupdate->set_name(fmt::format("update r-{}",it)); }

    { auto zupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,z, '-',alpha,mq, zcarry ) );
      pipe_cg->add_kernel(zupdate); zupdate->set_name(fmt::format("update z-{}",it)); }

    pcarry = shared_ptr<object>( new IMP_object(blocked) ); pcarry->set_name(fmt::format("p{}p",it));
    qcarry = shared_ptr<object>( new IMP_object(blocked) ); qcarry->set_name(fmt::format("s{}p",it));

    { auto pcopy = shared_ptr<kernel>( new IMP_copy_kernel( p,pcarry ) ); // copy in #1, pupdate later
      pipe_cg->add_kernel(pcopy); pcopy->set_name(fmt::format("copy p-{}",it)); }
    { auto qcopy = shared_ptr<kernel>( new IMP_copy_kernel( q,qcarry ) ); // copy in #1, qupdate later
      pipe_cg->add_kernel(qcopy); qcopy->set_name(fmt::format("copy q-{}",it)); }
    //snippet end
  }

  pipe_cg->analyze_dependencies();
  pipe_cg->execute();

  env->print_single( std::string("Norms:\n") );
  for (int it=0; it<n_iterations; it++) {
    double *data = rnorms[it]->get_raw_data();
    env->print_single(fmt::format("{}:{}, ",it,data[0]));
  }
  env->print_single(std::string("\n"));

  delete env;

  return 0;
}
