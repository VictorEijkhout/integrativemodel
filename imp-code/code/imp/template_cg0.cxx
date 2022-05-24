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
  //  arch->set_can_embed_in_beta(0);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
#endif
  int ntids = arch->nprocs();
  decomposition decomp = IMP_decomposition(arch);
  
  env->set_ir_outputfile("cg");
  int nlocal = env->iargument("nlocal",100); // points per processor
  int n_iterations = env->iargument("steps",20);
  int trace = env->has_argument("trace");
  env->print_single
    ( fmt::format("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal) );
  fmt::print("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal);

  // a bunch of vectors, block distributed
  auto blocked = shared_ptr<distribution>( new IMP_block_distribution(decomp,nlocal,-1) );
  shared_ptr<object> x0,r0;
  x0    = shared_ptr<object>( new IMP_object(blocked) ); x0->set_name(fmt::format("x0"));
  r0    = shared_ptr<object>( new IMP_object(blocked) ); r0->set_name(fmt::format("r0"));

  // scalars, all redundantly replicated
  auto scalar = shared_ptr<distribution>( new IMP_replicated_distribution(decomp) );
  vector<shared_ptr<object>> rnorms(n_iterations);  
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = shared_ptr<object>( new IMP_object(scalar) );
    rnorms[it]->set_name(fmt::format("rnorm{}",it));
    rnorms[it]->allocate();
  }

  // the linear system
  //  IMP_sparse_matrix *A = new IMP_toeplitz3_matrix(blocked,-1,2,-1);
  x0->set_value(0.); r0->set_value(1.);

  auto one = shared_ptr<object>( new IMP_object(scalar) ); one->set_name("one"); one->set_value(1.);
  
  // let's define the steps of the loop body
  auto queue = shared_ptr<algorithm>( new IMP_algorithm(decomp) );
  shared_ptr<kernel> k;
  k = shared_ptr<kernel>( new IMP_origin_kernel(one) ); k->set_name("origin one");
  queue->add_kernel( k );
  auto xorigin = shared_ptr<kernel>( new IMP_origin_kernel( x0 ) ); xorigin->set_name("origin x0");
  queue->add_kernel(xorigin);
  auto rorigin = shared_ptr<kernel>( new IMP_origin_kernel( r0 ) ); rorigin->set_name("origin r0");
  queue->add_kernel(rorigin);

  vector<shared_ptr<object>> x(n_iterations),r(n_iterations),z(n_iterations),
    p(n_iterations),q(n_iterations),
    rr(n_iterations),pap(n_iterations),alpha(n_iterations),beta(n_iterations);
  
  for (int it=0; it<n_iterations; it++) {
    x.at(it) = shared_ptr<object>( new IMP_object(blocked) ); x.at(it)->set_name(fmt::format("x{}",it)); }
  r = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    r.at(it) = shared_ptr<object>( new IMP_object(blocked) ); r.at(it)->set_name(fmt::format("r{}",it)); }
  z = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    z.at(it) = shared_ptr<object>( new IMP_object(blocked) ); z.at(it)->set_name(fmt::format("z{}",it)); }
  p = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    p.at(it) = shared_ptr<object>( new IMP_object(blocked) ); p.at(it)->set_name(fmt::format("p{}",it)); }
  q = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    q.at(it) = shared_ptr<object>( new IMP_object(blocked) ); q.at(it)->set_name(fmt::format("q{}",it)); }

  rr = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    rr.at(it) = shared_ptr<object>( new IMP_object(scalar) ); rr.at(it)->set_name(fmt::format("rr{}",it)); }
  pap = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    pap.at(it) = shared_ptr<object>( new IMP_object(scalar) ); pap.at(it)->set_name(fmt::format("pap{}",it)); }
  alpha = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    alpha.at(it) = shared_ptr<object>( new IMP_object(scalar) ); alpha.at(it)->set_name(fmt::format("alpha{}",it)); }
  beta = new shared_ptr<object>[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    beta.at(it) = shared_ptr<object>( new IMP_object(scalar) ); beta.at(it)->set_name(fmt::format("beta{}",it)); }

  try {

    for (int it=0; it<n_iterations; it++) {
      double *data;
      shared_ptr<kernel> precon,rho_inprod, xupdate,rupdate;

      //snippet cgtemplate
      if (it==0) {
	precon = shared_ptr<kernel>( new IMP_preconditioning_kernel( r0,z.at(it) ) );
	rho_inprod = shared_ptr<kernel>( new IMP_innerproduct_kernel( r0,z.at(it),rr.at(it) ) );
      } else {
	precon = shared_ptr<kernel>( new IMP_preconditioning_kernel( r[it-1],z.at(it) ) );
	rho_inprod = shared_ptr<kernel>( new IMP_innerproduct_kernel( r[it-1],z.at(it),rr.at(it) ) );
      }
      queue->add_kernel(precon); precon->set_name(fmt::format("preconditioning{}",it));
      queue->add_kernel(rho_inprod); rho_inprod->set_name(fmt::format("compute rho{}",it));

      if (it==0) {
	auto pisz = shared_ptr<kernel>( new IMP_copy_kernel( z.at(it),p.at(it) ) );
	queue->add_kernel(pisz); pisz->set_name("copy z to p");
      } else {
	// use rrp object from previous iteration
	auto beta_calc = shared_ptr<kernel>( new IMP_scalar_kernel( rr.at(it),"/",rr[it-1],beta.at(it) ) );
	queue->add_kernel(beta_calc); beta_calc ->set_name(fmt::format("compute beta{}",it));
	auto pupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,z.at(it), '+',beta.at(it),p[it-1], p.at(it) ) );
	queue->add_kernel(pupdate); pupdate->set_name(fmt::format("update p{}",it));
      }

      //    auto matvec = shared_ptr<kernel>( new IMP_spmvp_kernel( pnew,q,A ) );
      auto matvec = shared_ptr<kernel>( new IMP_centraldifference_kernel( p.at(it),q.at(it) ) );
      queue->add_kernel(matvec); matvec->set_name(fmt::format("spmvp{}",it));

      auto pap_inprod = shared_ptr<kernel>( new IMP_innerproduct_kernel( p.at(it),q.at(it),pap.at(it) ) );
      queue->add_kernel(pap_inprod); pap_inprod->set_name(fmt::format("pap inner product{}",it));

      auto alpha_calc = shared_ptr<kernel>( new IMP_scalar_kernel( rr.at(it),"/",pap.at(it),alpha.at(it) ) );
      queue->add_kernel(alpha_calc); alpha_calc->set_name(fmt::format("compute alpha{}",it));

      if (it==0) {
	xupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,x0, '-',alpha.at(it),p.at(it), x.at(it) ) );
	rupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,r0, '-',alpha.at(it),q.at(it), r.at(it) ) );
      } else {
	xupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,x[it-1], '-',alpha.at(it),p.at(it), x.at(it) ) );
	rupdate = shared_ptr<kernel>( new IMP_axbyz_kernel( '+',one,r[it-1], '-',alpha.at(it),q.at(it), r.at(it) ) );
      }
      xupdate->set_name(fmt::format("update x{}",it)); queue->add_kernel(xupdate);
      rupdate->set_name(fmt::format("update r{}",it)); queue->add_kernel(rupdate);
      //snippet end

      auto rnorm = shared_ptr<kernel>( new IMP_norm_kernel( r.at(it),rnorms.at(it) ) );
      queue->add_kernel(rnorm); rnorm->set_name(fmt::format("r norm{}",it));
      if (trace) {
	auto trace = shared_ptr<kernel>( new IMP_trace_kernel(rnorms.at(it),fmt::format("Norm in iteration {}",it)) );
	queue->add_kernel(trace); trace->set_name(fmt::format("rnorm trace {}",it));
      }
    }

  } catch (std::string c) { fmt::print("Error <<{}>> during cg construction\n",c);
  } catch (...) { fmt::print("Unknown error during cg iteration\n"); }

  try {
    //  env->kernels_to_dot_file();
    queue->analyze_dependencies();
    queue->execute();
  } catch (std::string c) { fmt::print("Error <<{}>> during cg execution\n",c);
  } catch (...) { fmt::print("Unknown error during cg execution\n"); }


  env->print_single( std::string("Norms:\n") );
  for (int it=0; it<n_iterations; it++) {
    double *data = rnorms.at(it)->get_raw_data();
    env->print_single(fmt::format("{}:{}, ",it,data[0]));
  }
  env->print_single(std::string("\n"));

  delete env;

  return 0;
}
