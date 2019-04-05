 // Random walk model for survey averaging using log-scale process and observation errors
//Changes:
//  2019-04-04: 1. Added 1-step prediction and associated output because PIBKC assessment
//                 timing was changed to May, prior to the survey, thus requiring
//                 a prediction of the assessment-year survey quantity to project
//                 the population for the assessment year to determine OFL.
//
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
    ivector est_yrs(styr,endyr+1); //styr:(endyr+1)
    !! est_yrs.fill_seqadd(styr,1);
    init_int nobs                //number of observations
    init_int uncType             //uncertainty type (0=cv's, 1=arithmetic cv's)
    init_matrix obs(1,nobs,1,3)  //observations matrix (year, estimate, uncertainty)
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
    !!cout<<"srv_yrs = "<<srv_yrs<<endl;
    !!cout<<"srv_obs = "<<srv_obs<<endl;
    !!cout<<"srv_cvs = "<<srv_cvs<<endl;
    !!cout<<"srv_sds = "<<srv_sds<<endl;
        
    vector srv_var(1,nobs)  //variance
    vector srv_cst(1,nobs)  //likelihood constants
    !! srv_var = square(srv_sds);
    !! srv_cst = log(2.0*M_PI*srv_var);
    !!cout<<"srv_var = "<<srv_var<<endl;
    !!cout<<"srv_cst = "<<srv_cst<<endl;
 
PARAMETER_SECTION
    init_number logSdLam
    random_effects_vector predLnScl(styr,endyr);
    objective_function_value jnll;
    
    sdreport_number sdrepPredLnSclNxtObs;
    sdreport_vector sdrepPredLnScl(styr,endyr);

PROCEDURE_SECTION
    jnll=0.0;
    for(int i=styr+1; i<=endyr; ++i){
      step(predLnScl(i-1),predLnScl(i),logSdLam);
    }

    for(int i=1; i<=nobs; ++i){
      obs_mod(predLnScl(srv_yrs(i)),i);
    }

    if (sd_phase()){
      sdrepPredLnSclNxtObs = predLnScl(endyr);
      sdrepPredLnScl       = predLnScl;
    }

SEPARABLE_FUNCTION void step(const dvariable& predLnScl1, const dvariable& predLnScl2, const dvariable& logSdLam)
    dvariable var=exp(2.0*logSdLam);
    jnll+=0.5*(log(2.0*M_PI*var)+square(predLnScl2-predLnScl1)/var);

SEPARABLE_FUNCTION void obs_mod(const dvariable& predLnScl, int i)
    jnll+=0.5*(srv_cst(i) + square(predLnScl-log(srv_obs(i)))/srv_var(i));

REPORT_SECTION
    report << predLnScl <<endl;
  
FINAL_SECTION
    cout<<"in FINAL_SECTION"<<endl;
    dvar_vector est(styr,endyr+1); //estimated survey quantity styr-endyr + 1-step prediction for endyr+1
    est(styr,endyr) = exp(sdrepPredLnScl); est(endyr+1) = exp(sdrepPredLnSclNxtObs);
    dvar_vector sd(styr,endyr+1);  //std dev for ln-scale survey quantity styr-endyr + 1-step prediction for endyr+1
    sd(styr,endyr)  = sdrepPredLnScl.sd;   sd(endyr+1)  = logSdLam;
    
    dvar_vector uci(styr,endyr+1);
    uci(styr,endyr) = exp(sdrepPredLnScl+1.96*sdrepPredLnScl.sd);      uci(endyr+1) = exp(sdrepPredLnSclNxtObs+1.96*logSdLam);
    dvar_vector lci(styr,endyr+1);
    lci(styr,endyr) = exp(sdrepPredLnScl-1.96*sdrepPredLnScl.sd);      lci(endyr+1) =  exp(sdrepPredLnSclNxtObs-1.96*logSdLam);
    dvar_vector upp90th(styr,endyr+1);
    upp90th(styr,endyr) = exp(sdrepPredLnScl+1.645*sdrepPredLnScl.sd); upp90th(endyr+1) = exp(sdrepPredLnSclNxtObs+1.645*logSdLam);
    dvar_vector low90th(styr,endyr+1);
    low90th(styr,endyr) = exp(sdrepPredLnScl-1.645*sdrepPredLnScl.sd); low90th(endyr+1) = exp(sdrepPredLnSclNxtObs-1.645*logSdLam);

    dvar_vector cv  = sqrt(exp(square(sd))-1.0);//cv for estimated/predicted survey quantity
   
    ofstream mysum("rwout.rep");
    write_R(nobs);
    write_R(uncType);
    write_R(srv_yrs);
    write_R(srv_obs);
    write_R(srv_sds);
    write_R(est_yrs);
    write_R(est); 
    write_R(cv);
    write_R(lci);
    write_R(uci);
    write_R(low90th);
    write_R(upp90th);
    write_R(sdrepPredLnScl);
    write_R(sdrepPredLnScl.sd);

    mysum.close();

TOP_OF_MAIN_SECTION
    gradient_structure::set_MAX_NVAR_OFFSET(3000);

    