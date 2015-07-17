 // Random walk model for survey averaging using log-scale process and observation errors
GLOBALS_SECTION
    #include <admodel.h>
    #undef REPORT
    #define write_R(object) mysum << "$" << #object "\n" << object << endl;

DATA_SECTION
    //data inputs
    init_int styr               //start year for interpolation
    init_int endyr              //end year for interpolation
    ivector yrs(styr,endyr);    //styr:endyr
    !! yrs.fill_seqadd(styr,1);
    init_int nobs                //number of observations
    init_int uncType             //uncertainty type (0=cv's, 1=arithmetic cv's)
    init_matrix obs(1,nobs,1,3)      //observations matrix (year, estimate, uncertainty)
    //end of input data
        
    //derived quantities
    ivector srv_yrs(1,nobs)  //years for observations
    vector srv_obs(1,nobs)   //survey observations
    vector srv_cvs(1,nobs)   //observed survey cv's
    vector srv_sds(1,nobs)   //survey std_devs (ln-scale or arithmetic scale)
 LOCAL_CALCS
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
 END_CALCS
        
    number meany
    vector srv_var(1,nobs)  //variance
    vector srv_cst(1,nobs)  //likelihood constants
    !! srv_var = elem_prod(srv_sds,srv_sds);
    !! srv_cst = log(2.0*M_PI*srv_var);
 
PARAMETER_SECTION
    init_number logSdLam
    sdreport_vector sdrepPredLnScl(styr,endyr);
    sdreport_vector sdrepPredArScl(styr,endyr);
    random_effects_vector predLnScl(styr,endyr);
    objective_function_value jnll;

PROCEDURE_SECTION
    jnll=0.0;
    for(int i=styr+1; i<=endyr; ++i){
      step(predLnScl(i-1),predLnScl(i),logSdLam);
    }

    for(int i=1; i<=nobs; ++i){
      obs_mod(predLnScl(srv_yrs(i)),i);
    }

    if (sd_phase()){
      sdrepPredArScl = exp(predLnScl);
      sdrepPredLnScl = predLnScl;
    }

SEPARABLE_FUNCTION void step(const dvariable& predLnScl1, const dvariable& predLnScl2, const dvariable& logSdLam)
    dvariable var=exp(2.0*logSdLam);
    jnll+=0.5*(log(2.0*M_PI*var)+square(predLnScl2-predLnScl1)/var);

SEPARABLE_FUNCTION void obs_mod(const dvariable& predLnScl, int i)
    jnll+=0.5*(srv_cst(i) + square(predLnScl-log(srv_obs(i)))/srv_var(i));

REPORT_SECTION
    sdrepPredLnScl = predLnScl;
    report << sdrepPredLnScl <<endl;
  
FINAL_SECTION
    cout<<"in FINAL_SECTION"<<endl;
    dvar_vector UCI = exp(sdrepPredLnScl+1.96*sdrepPredLnScl.sd);
    dvar_vector LCI = exp(sdrepPredLnScl-1.96*sdrepPredLnScl.sd);
    dvar_vector upp90th = exp(sdrepPredLnScl+1.645*sdrepPredLnScl.sd);
    dvar_vector low90th = exp(sdrepPredLnScl-1.645*sdrepPredLnScl.sd);

    ofstream mysum("rwout.rep");
    write_R(nobs);
    write_R(uncType);
    write_R(srv_yrs);
    write_R(srv_obs);
    write_R(srv_sds);
    write_R(yrs);
    write_R(LCI);
    write_R(sdrepPredArScl);
    write_R(UCI);
    write_R(low90th);
    write_R(upp90th);
    write_R(sdrepPredLnScl);
    write_R(sdrepPredLnScl.sd);

    mysum.close();

TOP_OF_MAIN_SECTION
    gradient_structure::set_MAX_NVAR_OFFSET(3000);

    