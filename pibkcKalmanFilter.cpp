#ifdef DEBUG
  #ifndef __SUNPRO_C
    #include <cfenv>
    #include <cstdlib>
  #endif
#endif
    #include <admodel.h>
    #undef REPORT
    #define write_R(object) os_results << "$" << #object "\n" << object << endl;
#ifdef DEBUG
  #include <chrono>
#endif
#include <admodel.h>
#ifdef USE_ADMB_CONTRIBS
#include <contrib.h>

#endif
#include <df1b2fun.h>

#include <adrndeff.h>

  extern "C"  {
    void ad_boundf(int i);
  }
#include <pibkcKalmanFilter.htp>

  df1b2_parameters * df1b2_parameters::df1b2_parameters_ptr=0;
  model_parameters * model_parameters::model_parameters_ptr=0;
model_data::model_data(int argc,char * argv[]) : ad_comm(argc,argv)
{
  adstring tmpstring;
  tmpstring=adprogram_name + adstring(".dat");
  if (argc > 1)
  {
    int on=0;
    if ( (on=option_match(argc,argv,"-ind"))>-1)
    {
      if (on>argc-2 || argv[on+1][0] == '-')
      {
        cerr << "Invalid input data command line option"
                " -- ignored" << endl;
      }
      else
      {
        tmpstring = adstring(argv[on+1]);
      }
    }
  }
  global_datafile = new cifstream(tmpstring);
  if (!global_datafile)
  {
    cerr << "Error: Unable to allocate global_datafile in model_data constructor.";
    ad_exit(1);
  }
  if (!(*global_datafile))
  {
    delete global_datafile;
    global_datafile=NULL;
  }
  styr.allocate("styr");
  endyr.allocate("endyr");
  yrs.allocate(styr,endyr);
 yrs.fill_seqadd(styr,1);
  nobs.allocate("nobs");
  uncType.allocate("uncType");
  obs.allocate(1,nobs,1,3,"obs");
  srv_yrs.allocate(1,nobs);
  srv_obs.allocate(1,nobs);
  srv_cvs.allocate(1,nobs);
  srv_sds.allocate(1,nobs);
    for (int i=1;i<=nobs;i++) {
        srv_yrs(i) = (int) obs(i,1);
        srv_obs(i) = obs(i,2);
        if (uncType==0){
            //unc = cv's
            srv_cvs(i) = obs(i,3);
            srv_sds(i) = sqrt(log(1.0+square(srv_cvs(i))));
        } else
        if (uncType==1){
            //unc = arithmetic std devs's
            srv_cvs(i) = obs(i,3)/srv_obs(i);
            srv_sds(i) = sqrt(log(1.0+square(srv_cvs(i))));
        }
    }
cout<<"srv_yrs = "<<srv_yrs<<endl;
cout<<"srv_obs = "<<srv_obs<<endl;
cout<<"srv_cvs = "<<srv_cvs<<endl;
cout<<"srv_sds = "<<srv_sds<<endl;
  srv_var.allocate(1,nobs);
  srv_cst.allocate(1,nobs);
 srv_var = square(srv_sds);
 srv_cst = log(2.0*M_PI*srv_var);
cout<<"srv_var = "<<srv_var<<endl;
cout<<"srv_cst = "<<srv_cst<<endl;
  if (global_datafile)
  {
    delete global_datafile;
    global_datafile = NULL;
  }
}

model_parameters::model_parameters(int sz,int argc,char * argv[]) : 
 model_data(argc,argv) , function_minimizer(sz)
{
  model_parameters_ptr=this;
  initializationfunction();
  logSdLam.allocate("logSdLam");
  predLnScl.allocate(styr,endyr,"predLnScl");
  prior_function_value.allocate("prior_function_value");
  likelihood_function_value.allocate("likelihood_function_value");
  jnll.allocate("jnll");  /* ADOBJECTIVEFUNCTION */
  sdrepSdLam.allocate("sdrepSdLam");
  sdrepLnPred.allocate(styr,endyr,"sdrepLnPred");
  sdrepPred.allocate(styr,endyr,"sdrepPred");
}
void model_parameters::userfunction(void)
{
  jnll =0.0;
    jnll=0.0;
    for(int i=styr+1; i<=endyr; ++i){
      step(predLnScl(i-1),predLnScl(i),logSdLam);
    }
    for(int i=1; i<=nobs; ++i){
      obs_mod(predLnScl(srv_yrs(i)),i);
    }
    if (sd_phase()){
      sdrepSdLam  = exp(logSdLam);
      sdrepLnPred = predLnScl;
      sdrepPred   = exp(predLnScl);
    }
}

void SEPFUN1  model_parameters::step(const dvariable& predLnScl1, const dvariable& predLnScl2, const dvariable& logSdLam)
{
  begin_df1b2_funnel();
    dvariable var=exp(2.0*logSdLam);
    jnll+=0.5*(log(2.0*M_PI*var)+square(predLnScl2-predLnScl1)/var);
  end_df1b2_funnel();
}

void SEPFUN1  model_parameters::obs_mod(const dvariable& predLnScl, int i)
{
  begin_df1b2_funnel();
    jnll+=0.5*(srv_cst(i) + square(predLnScl-log(srv_obs(i)))/srv_var(i));
  end_df1b2_funnel();
}

void model_parameters::report(const dvector& gradients)
{
 adstring ad_tmp=initial_params::get_reportfile_name();
  ofstream report((char*)(adprogram_name + ad_tmp));
  if (!report)
  {
    cerr << "error trying to open report file"  << adprogram_name << ".rep";
    return;
  }
    maxGrad = max(fabs(gradients));
    report << jnll    << "  #objective function value" << endl;
    report << maxGrad << "  #max gradient" << endl;
    report << exp(logSdLam) << " #estimated process error" <<endl;
    report << "#years" << endl;
    report << yrs <<endl;
    report << "#predicted values (ln-scale)" << endl;
    report << predLnScl <<endl;
    report << "#predicted values (arith-scale)" << endl;
    report << exp(predLnScl) <<endl;
}

void model_parameters::final_calcs()
{
    cout<<"in FINAL_SECTION"<<endl;
    dvar_vector est(styr,endyr); //estimated survey quantity styr:endyr
    est(styr,endyr) = sdrepPred;
    dvar_vector sd(styr,endyr);  //std dev for survey quantity styr:endyr
    sd(styr,endyr)  = sdrepPred.sd;
    dvar_vector cv  = elem_div(sd,est);//cv for estimated/predicted survey quantity
    dvar_vector uci(styr,endyr);
    uci = exp(sdrepLnPred+1.96*sdrepLnPred.sd);
    dvar_vector lci(styr,endyr);
    lci = exp(sdrepLnPred-1.96*sdrepLnPred.sd);
    dvar_vector upp90th(styr,endyr);
    upp90th = exp(sdrepLnPred+1.645*sdrepLnPred.sd);
    dvar_vector low90th(styr,endyr);
    low90th = exp(sdrepLnPred-1.645*sdrepLnPred.sd);
    ofstream os_results("rwout.rep");
    double objFun = value(jnll);
    write_R(objFun)
    write_R(maxGrad)
    write_R(sdrepSdLam)
    write_R(sdrepSdLam.sd)
    write_R(nobs)
    write_R(uncType)
    write_R(srv_yrs)
    write_R(srv_obs)
    write_R(srv_sds)
    write_R(yrs)
    write_R(est)
    write_R(cv)
    write_R(lci)
    write_R(uci)
    write_R(low90th)
    write_R(upp90th)
    write_R(sdrepLnPred)
    write_R(sdrepLnPred.sd)
    os_results.close();
}
  long int arrmblsize=0;

int main(int argc,char * argv[])
{
#ifdef DEBUG
  auto start = std::chrono::high_resolution_clock::now();
  #ifndef __SUNPRO_C
std::feclearexcept(FE_ALL_EXCEPT);
  #endif
#endif
  ad_set_new_handler();
  ad_exit=&ad_boundf;
    gradient_structure::set_MAX_NVAR_OFFSET(3000);
    gradient_structure::set_NO_DERIVATIVES();
    gradient_structure::set_YES_SAVE_VARIABLES_VALUES();
      if (!arrmblsize) arrmblsize=150000;
    df1b2variable::noallocate=1;
df1b2variable::pool = new adpool();
initial_df1b2params::varsptr = new P_INITIAL_DF1B2PARAMS[1000];
{
    df1b2_parameters mp(arrmblsize,argc,argv);
    mp.iprint=10;

    function_minimizer::random_effects_flag=1;
    df1b2variable::noallocate=0;
    mp.preliminary_calculations();
    initial_df1b2params::separable_flag=1;
    mp.computations(argc,argv);
}
delete [] init_df1b2variable::list;
init_df1b2variable::list = NULL;
delete [] initial_df1b2params::varsptr;
initial_df1b2params::varsptr = NULL;
delete df1b2variable::pool;
df1b2variable::pool = NULL;
#ifdef DEBUG
  std::cout << endl << argv[0] << " elapsed time is " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << " microseconds." << endl;
  #ifndef __SUNPRO_C
bool failedtest = false;
if (std::fetestexcept(FE_DIVBYZERO))
  { cerr << "Error: Detected division by zero." << endl; failedtest = true; }
if (std::fetestexcept(FE_INVALID))
  { cerr << "Error: Detected invalid argument." << endl; failedtest = true; }
if (std::fetestexcept(FE_OVERFLOW))
  { cerr << "Error: Detected overflow." << endl; failedtest = true; }
if (std::fetestexcept(FE_UNDERFLOW))
  { cerr << "Error: Detected underflow." << endl; }
if (failedtest) { std::abort(); } 
  #endif
#endif
    return 0;
}

extern "C"  {
  void ad_boundf(int i)
  {
    /* so we can stop here */
    exit(i);
  }
}

void model_parameters::preliminary_calculations(void){
  #if defined(USE_ADPVM)

  admaster_slave_variable_interface(*this);

  #endif

}

model_data::~model_data()
{}

model_parameters::~model_parameters()
{}

void model_parameters::set_runtime(void){}

#ifdef _BORLANDC_
  extern unsigned _stklen=10000U;
#endif


#ifdef __ZTC__
  extern unsigned int _stack=10000U;
#endif

void df1b2_parameters::user_function(void)
{
  jnll =0.0;
    jnll=0.0;
    for(int i=styr+1; i<=endyr; ++i){
      step(predLnScl(i-1),predLnScl(i),logSdLam);
    }
    for(int i=1; i<=nobs; ++i){
      obs_mod(predLnScl(srv_yrs(i)),i);
    }
    if (sd_phase()){
      sdrepSdLam  = exp(logSdLam);
      sdrepLnPred = predLnScl;
      sdrepPred   = exp(predLnScl);
    }
}

void   df1b2_pre_parameters::step(const funnel_init_df1b2variable& predLnScl1, const funnel_init_df1b2variable& predLnScl2, const funnel_init_df1b2variable& logSdLam)
{
  begin_df1b2_funnel();
    df1b2variable var=exp(2.0*logSdLam);
    jnll+=0.5*(log(2.0*M_PI*var)+square(predLnScl2-predLnScl1)/var);
  end_df1b2_funnel();
}

void   df1b2_pre_parameters::obs_mod(const funnel_init_df1b2variable& predLnScl, int i)
{
  begin_df1b2_funnel();
    jnll+=0.5*(srv_cst(i) + square(predLnScl-log(srv_obs(i)))/srv_var(i));
  end_df1b2_funnel();
}

void df1b2_pre_parameters::setup_quadprior_calcs(void) 
{ 
df1b2_gradlist::set_no_derivatives(); 
quadratic_prior::in_qp_calculations=1; 
}  

void df1b2_pre_parameters::begin_df1b2_funnel(void) 
{ 
(*re_objective_function_value::pobjfun)=0; 
other_separable_stuff_begin(); 
f1b2gradlist->reset();  
if (!quadratic_prior::in_qp_calculations) 
{ 
df1b2_gradlist::set_yes_derivatives();  
} 
funnel_init_var::allocate_all();  
}  

void df1b2_pre_parameters::end_df1b2_funnel(void) 
{  
lapprox->do_separable_stuff(); 
other_separable_stuff_end(); 
funnel_init_var::deallocate_all(); 
} 

void model_parameters::begin_df1b2_funnel(void) 
{ 
if (lapprox)  
{  
{  
begin_funnel_stuff();  
}  
}  
}  

void model_parameters::end_df1b2_funnel(void) 
{  
if (lapprox)  
{  
end_df1b2_funnel_stuff();  
}  
} 
void df1b2_parameters::deallocate() 
{
  logSdLam.deallocate();
  predLnScl.deallocate();
  prior_function_value.deallocate();
  likelihood_function_value.deallocate();
  jnll.deallocate();
  sdrepSdLam.deallocate();
  sdrepLnPred.deallocate();
  sdrepPred.deallocate();
} 
void df1b2_parameters::allocate(void) 
{
  logSdLam.allocate("logSdLam");
  predLnScl.allocate(styr,endyr,"predLnScl");
  prior_function_value.allocate("prior_function_value");
  likelihood_function_value.allocate("likelihood_function_value");
  jnll.allocate("jnll");  /* ADOBJECTIVEFUNCTION */
  sdrepSdLam.allocate("sdrepSdLam");
  sdrepLnPred.allocate(styr,endyr,"sdrepLnPred");
  sdrepPred.allocate(styr,endyr,"sdrepPred");
}
