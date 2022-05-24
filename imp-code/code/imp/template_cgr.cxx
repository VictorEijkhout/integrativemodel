/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-8
 ****
 **** template_cg.cxx : 
 ****     mode-independent template for conjugate gradients
 ****
 ****************************************************************/

/*! \page cg Conjugate Gradients Method

  This is incomplete.
*/

#include "template_common_header.h"
//#include "cg_kernel.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("cg");

  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
  arch->set_can_embed_in_beta(0);
  //object_data::set_trace_create_data();
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
#endif
  int ntids = arch->nprocs();
  decomposition decomp = IMP_decomposition(arch);
  
  //  env->set_ir_outputfile("cg");
  int nlocal = env->iargument("nlocal",100); // points per processor
  int n_iterations = env->iargument("steps",20);
  int trace = env->has_argument("trace");
  env->print_single
    ( fmt::format("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal) );
  //fmt::print("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal);

  // a bunch of vectors, block distributed
  auto blocked = shared_ptr<distribution>( new IMP_block_distribution(decomp,nlocal,-1) );
  shared_ptr<object> xt,x0,b0,r0,ax0;
  xt    = shared_ptr<object>( new IMP_object(blocked) ); xt->set_name(fmt::format("xtrue"));
  xt->allocate(); xt->set_value(1.);
  x0    = shared_ptr<object>( new IMP_object(blocked) ); x0->set_name(fmt::format("x0"));
  x0->allocate(); x0->set_value(0.);
  b0    = shared_ptr<object>( new IMP_object(blocked) ); b0->set_name(fmt::format("b0"));
  ax0   = shared_ptr<object>( new IMP_object(blocked) ); ax0->set_name(fmt::format("ax0"));
  r0    = shared_ptr<object>( new IMP_object(blocked) ); r0->set_name(fmt::format("r0"));

  // scalars, all redundantly replicated
  auto scalar = shared_ptr<distribution>( new IMP_replicated_distribution(decomp) );
  vector<shared_ptr<object>> rnorms(n_iterations);  
  for (int it=0; it<n_iterations; it++) {
    rnorms.at(it) = shared_ptr<object>( new IMP_object(scalar) ); rnorms.at(it)->set_name(fmt::format("rnorm{}",it));
    rnorms.at(it)->allocate();
  }

  shared_ptr<object> one = shared_ptr<object>( new IMP_object(scalar) ); one->set_name("one"); one->set_value(1.);
  shared_ptr<object> oNe = shared_ptr<object>( new IMP_object(scalar) ); oNe->set_name("oNe"); oNe->set_value(1.);
  
  // let's define the steps of the loop body
  auto cg = shared_ptr<algorithm>( new IMP_algorithm(decomp) );
  cg->set_name("Conjugate Gradients Method");

  // initial setup
  { auto k = shared_ptr<kernel>( new IMP_origin_kernel(one) ); k->set_name("origin one");
    cg->add_kernel( k ); }
  { auto k = shared_ptr<kernel>( new IMP_origin_kernel(oNe) ); k->set_name("origin oNe");
    cg->add_kernel( k ); }
  { auto xorigin = shared_ptr<kernel>( new IMP_origin_kernel( xt ) ); xorigin->set_name("origin xtrue");
    cg->add_kernel(xorigin); }
  { auto xorigin = shared_ptr<kernel>( new IMP_origin_kernel( x0 ) ); xorigin->set_name("origin x0");
    cg->add_kernel(xorigin); }
  { auto borigin = shared_ptr<kernel>( new IMP_centraldifference_kernel( xt,b0 ) ); borigin->set_name("b0=A xtrue");
    cg->add_kernel(borigin); }
  { auto atimesx0 = shared_ptr<kernel>( new IMP_centraldifference_kernel( x0,ax0 ) ); atimesx0->set_name("A x0");
    cg->add_kernel(atimesx0); }
  { auto rorigin = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,ax0, '-',oNe,b0, r0 ) ); rorigin->set_name("r0=ax0-b");
    cg->add_kernel(rorigin); }
  if (trace) {
    auto rr0 = shared_ptr<object>( new IMP_object(scalar) );
    auto r0inp = shared_ptr<kernel>( new IMP_norm_kernel( r0,rr0 ) ); cg->add_kernel( r0inp );
    cg->add_kernel( shared_ptr<kernel>( new IMP_trace_kernel(rr0,std::string("Initial residual norm") )) );
  }

  // define objects that need to carry from one iteration to the next
  shared_ptr<object> xcarry,rcarry,pcarry,rrp;
  
  auto 
    xbase = shared_ptr<object>( new IMP_object(blocked) ),
    xcbase = shared_ptr<object>( new IMP_object(blocked) ),
    rbase = shared_ptr<object>( new IMP_object(blocked) ),
    rcbase = shared_ptr<object>( new IMP_object(blocked) ),
    zbase = shared_ptr<object>( new IMP_object(blocked) ),
    pbase = shared_ptr<object>( new IMP_object(blocked) ),
    pcbase = shared_ptr<object>( new IMP_object(blocked) ),
    qbase = shared_ptr<object>( new IMP_object(blocked) );


  for (int it=0; it<n_iterations; it++) {

    shared_ptr<object> x,r, z,p,q, /* xcarry,rcarry,rrp have to persist */
      rr,pap,alpha,beta;
    x = shared_ptr<object>( new IMP_object(blocked,xbase) ); x->set_name(fmt::format("x{}",it));
    r = shared_ptr<object>( new IMP_object(blocked,rbase) ); r->set_name(fmt::format("r{}",it));
    z = shared_ptr<object>( new IMP_object(blocked,zbase) ); z->set_name(fmt::format("z{}",it));
    p = shared_ptr<object>( new IMP_object(blocked,pbase) ); p->set_name(fmt::format("p{}",it));
    q = shared_ptr<object>( new IMP_object(blocked,qbase) ); q->set_name(fmt::format("q{}",it));
    rr    = shared_ptr<object>( new IMP_object(scalar) ); rr->set_name(fmt::format("rr{}",it));
    pap   = shared_ptr<object>( new IMP_object(scalar) ); pap->set_name(fmt::format("pap{}",it));
    alpha = shared_ptr<object>( new IMP_object(scalar) ); alpha->set_name(fmt::format("alpha{}",it));
    beta  = shared_ptr<object>( new IMP_object(scalar) ); beta->set_name(fmt::format("beta{}",it));

    if (it==0) {
      auto xcopy = shared_ptr<kernel>( new IMP_copy_kernel( x0,x ) );
      cg->add_kernel(xcopy); xcopy->set_name("start x");
      auto rcopy = shared_ptr<kernel>( new IMP_copy_kernel( r0,r ) );
      cg->add_kernel(rcopy); rcopy->set_name("start r");
    } else {
      auto xcopy = shared_ptr<kernel>( new IMP_copy_kernel( xcarry,x ) );
      cg->add_kernel(xcopy); xcopy->set_name(fmt::format("copy x{}",it));
      auto rcopy = shared_ptr<kernel>( new IMP_copy_kernel( rcarry,r ) );
      cg->add_kernel(rcopy); rcopy->set_name(fmt::format("copy r{}",it));
      auto pcopy = shared_ptr<kernel>( new IMP_copy_kernel( pcarry,p ) );
      cg->add_kernel(pcopy); pcopy->set_name(fmt::format("copy p{}",it));
    }

    xcarry = shared_ptr<object>( new IMP_object(blocked,xcbase) ); xcarry->set_name(fmt::format("xcarry{}",it));
    rcarry = shared_ptr<object>( new IMP_object(blocked,rcbase) ); rcarry->set_name(fmt::format("rcarry{}",it));
    pcarry = shared_ptr<object>( new IMP_object(blocked,pcbase) ); pcarry->set_name(fmt::format("pcarry{}",it));

    //snippet cgtemplate
    auto rnorm = shared_ptr<kernel>( new IMP_norm_kernel( r,rnorms.at(it) ) );
    cg->add_kernel(rnorm); rnorm->set_name(fmt::format("r norm{}",it));
    if (trace) {
      auto trace = shared_ptr<kernel>( new IMP_trace_kernel(rnorms.at(it),fmt::format("Norm in iteration {}",it)) );
      cg->add_kernel(trace); trace->set_name(fmt::format("rnorm trace {}",it));
    }

    auto precon = shared_ptr<kernel>( new IMP_preconditioning_kernel( r,z ) );
    cg->add_kernel(precon); precon->set_name(fmt::format("preconditioning{}",it));

    auto rho_inprod = shared_ptr<kernel>( new IMP_innerproduct_kernel( r,z,rr ) );
    cg->add_kernel(rho_inprod); rho_inprod->set_name(fmt::format("compute rho{}",it));
    if (trace) {
      auto trace = shared_ptr<kernel>( new IMP_trace_kernel(rr,fmt::format("rtz in iteration {}",it)) );
      cg->add_kernel(trace); trace->set_name(fmt::format("rtz trace {}",it));
    }

    if (it==0) {
      auto pisz = shared_ptr<kernel>( new IMP_copy_kernel( z,pcarry ) );
      cg->add_kernel(pisz); pisz->set_name("copy z to p");
    } else {
      // use rrp object from previous iteration
      auto beta_calc = shared_ptr<kernel>( new IMP_scalar_kernel( rr,"/",rrp,beta ) );
      cg->add_kernel(beta_calc); beta_calc ->set_name(fmt::format("compute beta{}",it));

      auto pupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,z, '+',beta,p, pcarry ) );
      cg->add_kernel(pupdate); pupdate->set_name(fmt::format("update p{}",it));
    }

    // create new rrp, and immediately copy rr into it
    rrp = shared_ptr<object>( new IMP_object(scalar) ); rrp->set_name(fmt::format("rho{}p",it));
    auto rrcopy = shared_ptr<kernel>( new IMP_copy_kernel( rr,rrp ) );
    cg->add_kernel(rrcopy); rrcopy->set_name(fmt::format("save rr value{}",it));

    // matvec for now through 1d central difference
    auto matvec = shared_ptr<kernel>( new IMP_centraldifference_kernel( pcarry,q ) );
    cg->add_kernel(matvec); matvec->set_name(fmt::format("spmvp{}",it));

    auto pap_inprod = shared_ptr<kernel>( new IMP_innerproduct_kernel( pcarry,q,pap ) );
    cg->add_kernel(pap_inprod); pap_inprod->set_name(fmt::format("pap inner product{}",it));

    auto alpha_calc = shared_ptr<kernel>( new IMP_scalar_kernel( rr,"/",pap,alpha ) );
    cg->add_kernel(alpha_calc); alpha_calc->set_name(fmt::format("compute alpha{}",it));

    auto xupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,x, '-',alpha,pcarry, xcarry ) );
    cg->add_kernel(xupdate); xupdate->set_name(fmt::format("update x{}",it));

    auto rupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,r, '-',alpha,q, rcarry ) );
    cg->add_kernel(rupdate); rupdate->set_name(fmt::format("update r{}",it));
    //snippet end
  }

  try {
    cg->analyze_dependencies();
    cg->execute();
  } catch (std::string c) { fmt::print("{}\n",c); return -1; }
  
  env->print_single( std::string("Norms:\n") );
  for (int it=0; it<n_iterations; it++) {
    double *data = rnorms.at(it)->get_raw_data();
    env->print_single(fmt::format("{}:{}, ",it,data[0]));
  }
  env->print_single(std::string("\n"));

  delete env;

  return 0;
}
