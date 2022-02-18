%%%
%%% setParams.m
%%%
%%% Sets basic MITgcm parameters plus parameters for included packages, and
%%% writes out the appropriate input files.
%%% attention: simTime, nIter0, diag_freq_avg, diag_freq_inst, init_extern,
%%% tau_x0, Topographic parameters, tAlpha, beta, alpha gamma, data.pkg, useRBCsalt
%%% ivdc_kappa (0 for kpp or 100 for normal run), MXLDEPTH, Canyon,
%%% tauRelaxT, prograde,  Time step size, bottomDragQuadratic, Domain size in y
%%% GM_isopycK, GM_background_K, surface heat flux, pChkptFreq
function nTimeSteps = setParams (inputpath,codepath,listterm,Nx,Ny,Nr)  
  

  %%%%%%%%%%%%%%%%%%
  %%%%% SET-UP %%%%%
  %%%%%%%%%%%%%%%%%%
  
  tau_x0 = 0.05*(2); %%% Wind stress magnitude
  
  tAlpha = 1e-4*(1); %%% Linear thermal expansion coeff
  
  Ws = 50*1000*(1);  %%% Slope half-width

  hFacMin = 0.3;
 

  GM_NoParCel = false;
  
  %%% Tapering scheme
  GM_taper_scheme     = 'clipping'; % clipping dm95 linear gkw91 ldd97
  %%% Maximum isopycnal slope 
  GM_maxSlope         = 0.1; % Default:0.01  Mak18:0.005  Mak17:0.05  Andrew:0.025
 
  if strcmp(GM_taper_scheme,'dm95')
  %%% DM95 critical slope
  GM_Scrit            = GM_maxSlope;
  %%% DM95 tapering width
  GM_Sd               = 0.0025;
  end
  
  
  % smaller values (e.g., 0.01) will destroy the simulations with continental
  % slopes for most of tapering schemes
  
  %%% Set true if initialization files are set externally
  init_extern = false
  %%% Random noise amplitude, only works for   init_extern = false
  tNoise = 0;  
  
  %%% Horizontal viscosity  
  viscAh = 10;  
  %%% Grid-dependent biharmonic viscosity
  viscA4Grid = 0.1; 
  %%% non-dimensional Smagorinsky biharmonic viscosity factor
  viscC4smag = 0;
  %%% Vertical viscosity
  viscAr = 3e-4;     % Si22:3e-4  % 1e-4
  %%% Vertical temp diffusion
  diffKrT = 1e-5;    % Mak18, Si22
   
  
  Prograde=false;
  
  GM_useML = false;
  UpVp_useML = false;
  REDI_useML = false;
  
  GM_background_K   = 70;
  GM_MLEner  = false;
  GM_Burger  = false;
  SmoothKgm =  false;  
  
  SmoothUpVp = false;
  SmoothDUpVpDy = false;
  UpVp_ML_maxVal = 100; %means we don't limit upvp
  DUpVpDy_ML_maxVal = 1e-6;%5e-7
  UpVp_VertStruc = false;
%   GM_VertStruc = true;
  
  
  GM_ML_minN2 = 1.0e-8;
  GM_ML_minVal_K = 0;
  GM_ML_maxVal_K = 1000;
    
  Lx = 2*1000; %%% Domain size in x 

%   if(~GM_VertStruc)
% Bulk Kgm  
NNpath_GM = 'G:\TracerScripts\Prograde\For ML\table6_ANNGM_fig20b\weight_bias_normalization';
%   else
% Depth-varying Kgm  
% NNpath_GM = 'G:\TracerScripts\Prograde\For ML\table6_VANNGM_fig20c\weight_bias_normalization';
%   end
  
% 6 Input for E and u'v'  
 NNpath_E = 'G:\TracerScripts\Prograde\For ML\Reynolds\code_Reynolds_stress\ANN_input_output\z_input6_buoyancy1_Ld_mean_Lx_f0N0\tauQ_1_sqrt_50_9_regularization_1e_4'
 NNpath_Rey = 'G:\TracerScripts\Prograde\For ML\Reynolds\code_Reynolds_stress\ANN_input_output\z_input6_buoyancy1_Ld_mean_Lx_f0N0_stress\tauQ_1_sqrt_50_tau_xy_9_regularization_1e_4'


% 5 Input  (Uz is not included)
%  NNpath_E = 'G:\TracerScripts\Prograde\For ML\NoUz\z_input5_buoyancy1_Ld_mean_Lx_f0N0_EKE\tauQ_1_sqrt_50_9_regularization_1e_4'
%  NNpath_Rey = 'G:\TracerScripts\Prograde\For ML\NoUz\z_input5_buoyancy1_Ld_mean_Lx_f0N0_stress\tauQ_1_sqrt_50_tau_xy_9_regularization_1e_4'

  
  %%% If set true, plots of prescribed variables will be shown
  showplots = true;      
  fignum = 1;
  
  %%% Data format parameters
  ieee='b';
  prec='real*8';
  realdigits = 8;
  realfmt=['%.',num2str(realdigits),'e'];
  
  %%% Get parameter type definitions
  paramTypes;
  
  %%% To store parameter names and values
  parm01 = parmlist;
  parm02 = parmlist;
  parm03 = parmlist;
  parm04 = parmlist;
  parm05 = parmlist;
  PARM={parm01,parm02,parm03,parm04,parm05}; 
  
  %%% Seconds in one hour
  t1min = 60;
  %%% Seconds in one hour
  t1hour = 60*t1min;
  %%% Seconds in one day
  t1day = 24*t1hour;
  %%% Seconds in 1 year
  t1year = 365*t1day;  
  %%% Metres in one kilometre
  m1km = 1000;
  
  %%% Load EOS utilities
%   addpath ~/Caltech/Utilities/GSW;
%   addpath ~/Caltech/Utilities/GSW/html;
%   addpath ~/Caltech/Utilities/GSW/library;
%   addpath ~/Caltech/Utilities/GSW/pdf;
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% FIXED PARAMETER VALUES %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  simTime = 30*t1year + 1*t1day; % %%% Simulation time  
  diag_phase_avg = 100*t1year;    
  
  nIter0 = 0; %%% Initial iteration 
  Ly = 500*m1km; %%% Domain size in y  
%   Ly = 800*m1km; %%% Domain size in y 
  Ls = 50*m1km; %%% Southern sponge width
  Ln = 50*m1km; %%% Northern sponge width
  Lsurf = 10; %%% Surface sponge width Yan
  H = 4000; %%% Domain size in z 
  g = 9.81; %%% Gravity
  f0 = 1e-4; %%% Coriolis parameter
  beta = 0; %4e-11;  %%% Beta parameter      
  

  viscA4 = 0; %%% Biharmonic viscosity
  viscAhGrid = 0; %%% Grid-dependent harmonic viscosity
  diffKhT = 0; %%% Horizontal temp diffusion

  
  
  %%% Topographic parameters
  Zs = 2250;% %%% Vertical slope position
%   Zs = 4000; %%% Vertical slope position flatbottomed
  Hs = 3500;%; %%% Shelf height
%   Hs = 0; %%% Shelf height flatbottomed
  Ys = 200*m1km; %%% Meridional slope position
  
  
  %%% Stratification parameters
  
  %%% Small domain, delta > 0 wind
  use_wind = true;
  restore_south = false;
  neg_delta = 0;
  mintemp = 0;
  maxtemp = 10;
  
  

  %%% Parameters related to periodic wind forcing
  %%% REF: http://mailman.mitgcm.org/pipermail/mitgcm-support/2012-November/008068.html
  %%% REF: http://mailman.mitgcm.org/pipermail/mitgcm-support/2012-November/008064.html
  %%% REF: http://mitgcm.org/public/r2_manual/latest/online_documents/node104.html
  periodicExternalForcing = false;
  externForcingCycle = t1year;
  nForcingPeriods = 50;
  if (~periodicExternalForcing)
    nForcingPeriods = 1;   
  end
  externForcingPeriod = externForcingCycle/nForcingPeriods;  
  
  %%% PARM01
  %%% momentum scheme
  parm01.addParm('vectorInvariantMomentum',true,PARM_BOOL);
  %%% viscosity  
  parm01.addParm('viscAr',viscAr,PARM_REAL);
  parm01.addParm('viscA4',viscA4,PARM_REAL);
  parm01.addParm('viscAh',viscAh,PARM_REAL);
  parm01.addParm('viscA4Grid',viscA4Grid,PARM_REAL);
  parm01.addParm('viscAhGrid',viscAhGrid,PARM_REAL);
  parm01.addParm('viscA4GridMax',0.5,PARM_REAL);
  parm01.addParm('viscAhGridMax',1,PARM_REAL);
  parm01.addParm('useAreaViscLength',false,PARM_BOOL);
  parm01.addParm('useFullLeith',true,PARM_BOOL);
  parm01.addParm('viscC4leith',0,PARM_REAL);
  parm01.addParm('viscC4leithD',0,PARM_REAL); 
  parm01.addParm('viscC2leith',0,PARM_REAL);
  parm01.addParm('viscC2leithD',0,PARM_REAL); 
  parm01.addParm('viscC4smag',viscC4smag,PARM_REAL);  
  %%% diffusivity
  parm01.addParm('tempAdvScheme',80,PARM_INT);
  parm01.addParm('saltAdvScheme',80,PARM_INT);
  parm01.addParm('diffKrT',diffKrT,PARM_REAL);
  parm01.addParm('diffKhT',diffKhT,PARM_REAL);
  parm01.addParm('diffK4T',0,PARM_REAL);  
  parm01.addParm('tempStepping',true,PARM_BOOL);
  parm01.addParm('saltStepping',true,PARM_BOOL);
  parm01.addParm('staggerTimeStep',true,PARM_BOOL);
  %%% equation of state
  parm01.addParm('eosType','LINEAR',PARM_STR); 
  parm01.addParm('tAlpha',tAlpha,PARM_REAL); 
  parm01.addParm('sBeta',0,PARM_REAL); 
  %%% boundary conditions
  parm01.addParm('no_slip_sides',false,PARM_BOOL);
  parm01.addParm('no_slip_bottom',false,PARM_BOOL);
  parm01.addParm('bottomDragLinear',0,PARM_REAL);
  parm01.addParm('bottomDragQuadratic',2.5e-3,PARM_REAL); % 2.5e-3 for normal runs, 0 for spindown
  %%% physical parameters
  parm01.addParm('f0',f0,PARM_REAL);
  parm01.addParm('beta',beta,PARM_REAL);
  parm01.addParm('gravity',g,PARM_REAL);
  %%% full Coriolis force parameters
  parm01.addParm('quasiHydrostatic',false,PARM_BOOL);
  parm01.addParm('fPrime',0,PARM_REAL);
  %%% implicit diffusion and convective adjustment  
  parm01.addParm('ivdc_kappa',100,PARM_REAL);
  parm01.addParm('implicitDiffusion',true,PARM_BOOL);
  parm01.addParm('implicitViscosity',true,PARM_BOOL);
  %%% exact volume conservation
  parm01.addParm('exactConserv',true,PARM_BOOL);
  %%% C-V scheme for Coriolis term
  parm01.addParm('useCDscheme',false,PARM_BOOL);
  %%% partial cells for smooth topography
  parm01.addParm('hFacMin',hFacMin,PARM_REAL);  
  %%% file IO stuff
  parm01.addParm('readBinaryPrec',64,PARM_INT);
  parm01.addParm('useSingleCpuIO',true,PARM_BOOL);
  parm01.addParm('debugLevel',1,PARM_INT);
  %%% Wet-point method at boundaries - may improve boundary stability
  parm01.addParm('useJamartWetPoints',true,PARM_BOOL);
  parm01.addParm('useJamartMomAdv',true,PARM_BOOL);


 
  %%% PARM02
  parm02.addParm('useSRCGSolver',true,PARM_BOOL);  
  parm02.addParm('cg2dMaxIters',1000,PARM_INT);  
  parm02.addParm('cg2dTargetResidual',1e-12,PARM_REAL);
 
  %%% PARM03
  parm03.addParm('alph_AB',1/2,PARM_REAL);
  parm03.addParm('beta_AB',5/12,PARM_REAL);
  parm03.addParm('nIter0',nIter0,PARM_INT);
  parm03.addParm('abEps',0.1,PARM_REAL);
  parm03.addParm('chkptFreq',0.01*t1year,PARM_REAL);
  parm03.addParm('pChkptFreq',1*t1year,PARM_REAL);
  parm03.addParm('taveFreq',0,PARM_REAL);
  parm03.addParm('dumpFreq',0,PARM_REAL);
  parm03.addParm('monitorFreq',t1year,PARM_REAL);
  parm03.addParm('dumpInitAndLast',true,PARM_BOOL);
  parm03.addParm('pickupStrictlyMatch',false,PARM_BOOL);
  if (periodicExternalForcing)
    parm03.addParm('periodicExternalForcing',periodicExternalForcing,PARM_BOOL);
    parm03.addParm('externForcingPeriod',externForcingPeriod,PARM_REAL);
    parm03.addParm('externForcingCycle',externForcingCycle,PARM_REAL);
  end
  
  %%% PARM04
  parm04.addParm('usingCartesianGrid',true,PARM_BOOL);
  parm04.addParm('usingSphericalPolarGrid',false,PARM_BOOL);    
  
    
  %%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% GRID SPACING %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%    
    

  %%% Zonal grid
  dx = Lx/Nx;  
  xx = (1:Nx)*dx;
  xx = xx-mean(xx);
  
  %%% Uniform meridional grid   
  dy = (Ly/Ny)*ones(1,Ny);  
  yy = cumsum((dy + [0 dy(1:end-1)])/2);
 
  %%% Plotting mesh
  [Y,X] = meshgrid(yy,xx);
  
  %%% Grid spacing increases with depth, but spacings exactly sum to H
  zidx = 1:Nr;
%   gamma = 20;  %%% For Nr = 133
%   alpha = 5;
  gamma = 10;  %%% For Nr = 70
  alpha = 10;
  dz1 = 2*H/Nr/(alpha+1);
  dz2 = alpha*dz1;
  dz = dz1 + ((dz2-dz1)/2)*(1+tanh((zidx-((Nr+1)/2))/gamma));
  zz = -cumsum((dz+[0 dz(1:end-1)])/2);

  %%% Store grid spacings
  parm04.addParm('delX',dx*ones(1,Nx),PARM_REALS);
  parm04.addParm('delY',dy,PARM_REALS);
  parm04.addParm('delR',dz,PARM_REALS);      
  
  %%% Don't allow partial cell height to fall below min grid spacing
  parm01.addParm('hFacMinDr',min(dz),PARM_REAL);
  
  
  %%%%%%%%%%%%%%%%%%%%%%
  %%%%% BATHYMETRY %%%%%
  %%%%%%%%%%%%%%%%%%%%%%
   

  %%% tanh-shaped slope  
  h = - Zs - (Hs/2)*tanh((Y-Ys)/Ws);
%==========================================================================
  %%% Add Canyons by Yan
%   Ys_curve = Ys + (2)*(Ws/2)*cos(2*pi*5/Lx*xx);
%   for i = 1:Nx
%       h(i, :) = - Zs - (Hs/2)*tanh((Y(i, :)-Ys_curve(i))/Ws);
%   end

%   %%% Add troughs
%   np = 5;
%   ycs = (Ys - Ws/4) + (Ws/2)*cos(2*pi*np/Lx*xx);
%   yns = (Ys + Ws/4) + (Ws/2)*cos(2*pi*np/Lx*xx);
%   for i = 1:Nx
%       ind = and(yy > ycs(i), yy < yns(i));
%       h(i, ind) = -H;
%   end
%==========================================================================  
  h(:,1) = 0;   
  h(:,end) = 0;  
  

  
  %%% Plot the bathymetry
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
%     contour(X,Y,h,'EdgeColor','k');    
%     xlabel('x');
%     ylabel('y');
%     zlabel('hb','Rotation',0);
    plot(Y(1,:),h(1,:));
    title('Model bathymetry');
  end
  disp(['shelf depth ' num2str(h(1, 2))]) 
  
  %%% Save as a parameter
  writeDataset(h,fullfile(inputpath,'bathyFile.bin'),ieee,prec);
  parm05.addParm('bathyFile','bathyFile.bin',PARM_STR); 
 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% NORTHERN TEMPERATURE/SALINITY PROFILES %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Exponential background temperature  
  Hscale = 1000;  
  tNorth = mintemp + (maxtemp-mintemp)*(exp(zz/Hscale)-exp(-H/Hscale))/(1-exp(-H/Hscale));
%   tSouth = mintemp + (maxtemp+difftemp-mintemp)*(exp(zz/Hscale)-exp(-H/Hscale))/(1-exp(-H/Hscale));
%   tSouth = tNorth + difftemp*(1+zz/H);
%   tSouth = tNorth + difftemp;
  if (restore_south)
    if (neg_delta)
      tSouth = maxtemp + difftemp + zz*N2South/g/tAlpha;
      tSouth(zz<-(H-Hs)) = mintemp + (maxtemp + difftemp - (H-Hs)*N2South/g/tAlpha)*(zz(zz<-(H-Hs))+H)/Hs;
    else
      tSouth = mintemp + difftemp + (zz+(H-Hs))*N2South/g/tAlpha;
      tSouth(zz<-(H-Hs)) = mintemp + difftemp*(zz(zz<-(H-Hs))+H)/Hs;
    end
  else
    tSouth = tNorth;
  end
  
  %%% Constant background salinity (currently unused)
  sRef = 1;    
  sNorth = sRef*ones(1,Nr);
  sSouth = 0*ones(1,Nr);
  
  %%% Plot the northern temperature
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    plot(tNorth,zz);
    xlabel('\theta_n_o_r_t_h');
    ylabel('z','Rotation',0);
    title('Northern relaxation temperature');
  end
  
  %%% Plot the southern temperature
  if (restore_south)
    if (showplots)
      figure(fignum);
      fignum = fignum + 1;
      clf;
      plot(tSouth,zz);
      xlabel('T_s_o_u_t_h');
      ylabel('z','Rotation',0);
      title('Southern relaxation temperature');
    end
  end
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% INITIAL DATA %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Random noise amplitude
%   tNoise = 0;  
  
  %%% Weights for linear interpolation
  wtNorth = repmat(Y/Ly,[1 1 Nr]);
  wtSouth = 1 - wtNorth;
      
  %%% Linear variation between southern and northern reference temperatures
  hydroThNorth = repmat(reshape(tNorth,[1,1,Nr]),[Nx Ny]);
  hydroThSouth = repmat(reshape(tSouth,[1,1,Nr]),[Nx Ny]); 
  hydroTh = wtNorth.*hydroThNorth + wtSouth.*hydroThSouth;
  TT = squeeze(hydroTh(1,:,:));
  
  %%% Plot the initial condition
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    [ZZ,YY]=meshgrid(zz,yy);
    hydroTh_plot = squeeze(hydroTh(1,:,:));
    HH = repmat(reshape(h(1,:),[Ny,1]),[1 Nr]);
    hydroTh_plot(ZZ<HH) = NaN;
    contourf(YY,ZZ,hydroTh_plot,30);
    colorbar;
    xlabel('y');
    ylabel('z');
    title('Initial temperature section');
  end
  
  %%% Linear initial salinity
  hydroSaNorth = repmat(reshape(sNorth,[1,1,Nr]),[Nx Ny]);
  hydroSaSouth = repmat(reshape(sSouth,[1,1,Nr]),[Nx Ny]);   
  hydroSa = wtNorth.*hydroSaNorth + wtSouth.*hydroSaSouth; 
    
  %%% Add some random noise
  hydroTh = hydroTh + tNoise*(2*rand(Nx,Ny,Nr)-1);  
    
  %%% Write to data files
  %%% N.B. These will not be used because the files will be overwritten by
  %%% those in the DEFAULTS folder
  if (~init_extern)
    writeDataset(hydroTh,fullfile(inputpath,'hydrogThetaFile.bin'),ieee,prec); 
    parm05.addParm('hydrogThetaFile','hydrogThetaFile.bin',PARM_STR);  
    writeDataset(hydroSa,fullfile(inputpath,'hydrogSaltFile.bin'),ieee,prec); 
    parm05.addParm('hydrogSaltFile','hydrogSaltFile.bin',PARM_STR);  
  else
    parm05.addParm('hydrogThetaFile','hydrogThetaFile.bin',PARM_STR);
    parm05.addParm('hydrogSaltFile','hydrogSaltFile.bin',PARM_STR);  
    parm05.addParm('uVelInitFile','uVelInitFile.bin',PARM_STR);  
    parm05.addParm('vVelInitFile','vVelInitFile.bin',PARM_STR);  
    parm05.addParm('pSurfInitFile','pSurfInitFile.bin',PARM_STR);  
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% DEFORMATION RADIUS %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  
  %%% Temperature stratification   
  DZ = repmat(dz,[Ny 1]);
  Tz = zeros(size(TT));  
  Tz(:,1) = (TT(:,1)-TT(:,2)) ./ (0.5*(DZ(:,1)+DZ(:,2)));
  Tz(:,2:Nr-1) = (TT(:,1:Nr-2)-TT(:,3:Nr)) ./ (DZ(:,2:Nr-1) + 0.5*(DZ(:,1:Nr-2)+DZ(:,3:Nr)));
  Tz(:,Nr) = (TT(:,Nr-1)-TT(:,Nr)) ./ (0.5*(DZ(:,Nr-1)+DZ(:,Nr)));  
  N2 = tAlpha*g*Tz;

  %%% Calculate internal wave speed and first Rossby radius of deformation
  N = sqrt(N2);
  Cig = zeros(size(yy));
  for j=1:Ny    
    for k=1:Nr
      if (zz(k) > h(1,j))        
        Cig(j) = Cig(j) + N(j,k)*dz(k);
      end
    end
  end
  Rd = Cig./(pi*abs(f0+beta*Y(1,:)));

  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    semilogx(N2,zz/1000);
    xlabel('N^2 (km)');
    ylabel('z (km)','Rotation',0);
    title('Buoyancy frequency');
  end
  
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    plot(yy/1000,Rd/1000);
    xlabel('y (km)');
    ylabel('R_d (km)','Rotation',0);
    title('First baroclinic Rossby deformation radius');
  end
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% CALCULATE TIME STEP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
  
  
  %%% These estimates are in no way complete, but they give at least some
  %%% idea of the time step needed to keep things stable. In complicated 
  %%% simulations, preliminary tests may be required to estimate the
  %%% parameters used to calculate these time steps.        
  
  %%% Gravity wave CFL

  %%% Upper bound for absolute horizontal fluid velocity (m/s)
  %%% At the moment this is just an estimate
  Umax = 2.0;  
  %%% Max gravity wave speed 
  cmax = max(Cig);
  %%% Max gravity wave speed using total ocean depth
  cgmax = Umax + cmax;
  %%% Advective CFL
  deltaT_adv = min([0.5*dx/cmax,0.5*dy/cmax]);
  %%% Gravity wave CFL
  deltaT_gw = min([0.5*dx/Umax,0.5*dy/Umax]);
  %%% CFL time step based on full gravity wave speed
  deltaT_fgw = min([0.5*dx/cgmax,0.5*dy/cgmax]);
    
  %%% Other stability conditions
  
  %%% Inertial CFL time step (Sf0<=0.5)
  deltaT_itl = 0.5/abs(f0);
  %%% Time step constraint based on horizontal diffusion 
  deltaT_Ah = 0.5*min([dx dy])^2/(4*viscAh);    
  %%% Time step constraint based on vertical diffusion
  deltaT_Ar = 0.5*min(dz)^2 / (4*viscAr);  
  %%% Time step constraint based on biharmonic viscosity 
  deltaT_A4 = 0.5*min([dx dy])^4/(32*viscA4);
  %%% Time step constraint based on horizontal diffusion of temp 
  deltaT_KhT = 0.4*min([dx dy])^2/(4*diffKhT);    
  %%% Time step constraint based on vertical diffusion of temp 
  deltaT_KrT = 0.4*min(dz)^2 / (4*diffKrT);
  
  %%% Time step size  
  deltaT = min([deltaT_fgw deltaT_gw deltaT_adv deltaT_itl deltaT_Ah deltaT_Ar deltaT_KhT deltaT_KrT deltaT_A4]);
  deltaT = round(deltaT);
%   deltaT = round(deltaT*.5); % time step size refinement for prograde run
  nTimeSteps = ceil(simTime/deltaT);
  simTimeAct = nTimeSteps*deltaT;
  
  %%% Write end time time step size  
  parm03.addParm('endTime',nIter0*deltaT+simTimeAct,PARM_INT);
  parm03.addParm('deltaT',deltaT,PARM_REAL); 
    
  
  %%%%%%%%%%%%%%%%%%%%%%
  %%%%% ZONAL WIND %%%%%
  %%%%%%%%%%%%%%%%%%%%%%
  
    
  %%% Set up a wind stress profile    
  if (use_wind)
%     tau_x = tau_x0./cosh(((yy-Ys)/Ws).^2);
    Lwind = 400*m1km;
    tau_x = tau_x0.*sin(pi*yy/Lwind).^2;
    tau_x(yy>Lwind) = 0;
    if (~neg_delta)
      tau_x = -tau_x;
    end
  else
    tau_x = 0*yy;
  end
  
  %%% Create wind stress matrix  
  tau_mat = zeros(size(Y)); %%% Wind stress matrix  
  for j=1:Ny   
    tau_mat(:,j) = tau_x(j);          
  end 
  
  %%% Plot the wind stress 
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    plot(yy,squeeze(tau_mat(1,:)));
    xlabel('y');
    ylabel('\tau_w','Rotation',0);
    title('Wind stress profile');
  end
  
  %%% Save as a parameter  
  writeDataset(tau_mat,fullfile(inputpath,'zonalWindFile.bin'),ieee,prec); 
  parm05.addParm('zonalWindFile','zonalWindFile.bin',PARM_STR);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% SURFACE HEAT/SALT FLUXES %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  
  
  %%%% Heat flux profile
  heat_profile = 0*yy;
  %%% Add surface heat flux by Yan
  % heat_profile(yy<=50e3) = 10;
  %%% Done
  heat_flux = zeros(Nx,Ny,nForcingPeriods);
  for n=1:nForcingPeriods
    for j=1:Ny      
      heat_flux(:,j,n) = heat_profile(j);             
    end         
  end

  %%% Plot the surface heat flux
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    plot(yy,squeeze(heat_flux(1,:,:)));
    xlabel('y');
    ylabel('heat flux');
    title('Surface heat flux');
  end  
  
  %%% Save as parameters
  writeDataset(heat_flux,fullfile(inputpath,'surfQfile.bin'),ieee,prec);
  parm05.addParm('surfQfile','surfQfile.bin',PARM_STR);  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% WRITE THE 'data' FILE %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Creates the 'data' file
  write_data(inputpath,PARM,listterm,realfmt);
  
 
  
  
  %%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%
  %%%%% RBCS %%%%%
  %%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%
  
  
  
    
  %%%%%%%%%%%%%%%%%%%%%%%
  %%%%% RBCS SET-UP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% To store parameter names and values
  rbcs_parm01 = parmlist;
  rbcs_parm02 = parmlist;
  RBCS_PARM = {rbcs_parm01,rbcs_parm02};
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% RELAXATION PARAMETERS %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  useRBCtemp = true;
  useRBCsalt = false;
  useRBCuVel = false;
  useRBCvVel = false;
  tauRelaxT = 7*t1day; % 7d for normal runs, 14d for prograde run
  tauRelaxS = t1day;
  tauRelaxU = t1day;
  tauRelaxV = t1day;
  rbcs_parm01.addParm('useRBCtemp',useRBCtemp,PARM_BOOL);
  rbcs_parm01.addParm('useRBCsalt',useRBCsalt,PARM_BOOL);
  rbcs_parm01.addParm('useRBCuVel',useRBCuVel,PARM_BOOL);
  rbcs_parm01.addParm('useRBCvVel',useRBCvVel,PARM_BOOL);
  rbcs_parm01.addParm('tauRelaxT',tauRelaxT,PARM_REAL);
  rbcs_parm01.addParm('tauRelaxS',tauRelaxS,PARM_REAL);
  rbcs_parm01.addParm('tauRelaxU',tauRelaxU,PARM_REAL);
  rbcs_parm01.addParm('tauRelaxV',tauRelaxV,PARM_REAL);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% RELAXATION TEMPERATURE/SALINITY %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Relaxation T/S matrices
  temp_relax = ones(Nx,Ny,Nr);  
  salt_relax = ones(Nx,Ny,Nr);  
  Ny_south = round(Ny/2);
  Ny_north = Ny - Ny_south;
  temp_relax(:,1:Ny_south,:) = repmat(reshape(tSouth,[1 1 Nr]),[Nx Ny_south 1]);  
  temp_relax(:,Ny_south+1:Ny,:) = repmat(reshape(tNorth,[1 1 Nr]),[Nx Ny_north 1]);  
  salt_relax(:,1:Ny_south,:) = repmat(reshape(sSouth,[1 1 Nr]),[Nx Ny_south 1]);  
  salt_relax(:,Ny_south+1:Ny,:) = repmat(reshape(sNorth,[1 1 Nr]),[Nx Ny_north 1]);
  
  %%% Yan: relax SST linearly from 15(onshore) to 10(offshore) for prograde
  %%% run
%   indy = find(yy <= 4.5e5, 1, 'last');
%   Tsurf = linspace(15, 10, indy+1);
%   for j = 1:indy+1
%       temp_relax(:, j, 1:10) = Tsurf(j);
%   end
  
  %%% Plot the relaxation temperature
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    [ZZ,YY]=meshgrid(zz,yy);
    contourf(YY,ZZ,squeeze(temp_relax(1,:,:)),30);
    colorbar;
    xlabel('y');
    ylabel('z');
    title('Relaxation temperature');
  end
  
  %%% Plot the relaxation salinity
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    [ZZ,YY]=meshgrid(zz,yy);
    contourf(YY,ZZ,squeeze(salt_relax(1,:,:)),30);
    colorbar;
    xlabel('y');
    ylabel('z');
    title('Relaxation salinity');
  end
  
  %%% Save as parameters
  writeDataset(temp_relax,fullfile(inputpath,'sponge_temp.bin'),ieee,prec); 
  rbcs_parm01.addParm('relaxTFile','sponge_temp.bin',PARM_STR);  
  writeDataset(salt_relax,fullfile(inputpath,'sponge_salt.bin'),ieee,prec); 
  rbcs_parm01.addParm('relaxSFile','sponge_salt.bin',PARM_STR);  
  
  
  %%%%%%%%%%%%%%%%%%%%%  
  %%%%% RBCS MASK %%%%%
  %%%%%%%%%%%%%%%%%%%%%  
  
  
  %%% Mask is zero everywhere by default, i.e. no relaxation
  msk_temp = zeros(Nx,Ny,Nr);  
  msk_salt = zeros(Nx,Ny,Nr);  
  
  %%% Set relaxation timescales 
  northRelaxFac = 1;  
  southRelaxFac = 1;  
  surfaceRelaxFac = 1; 
  %%% If a sponge BC is required, gradually relax in the gridpoints 
  %%% approaching the wall (no relaxation at the wall)    
  for j=1:Ny
    for k=1:Nr        
      if ( (yy(j)<Ls) )
        if (restore_south)
          msk_temp(:,j,k) = (1/southRelaxFac) * (Ls-yy(j)) / Ls;          
        end
        msk_salt(:,j,k) = (1/southRelaxFac) * (Ls-yy(j)) / Ls;          
      end
      if ( (yy(j)>Ly-Ln) )
        msk_temp(:,j,k) = (1/northRelaxFac) * (yy(j)-(Ly-Ln)) / Ln;      % prograde/spindown comment to remove north sponge    
        msk_salt(:,j,k) = (1/northRelaxFac) * (yy(j)-(Ly-Ln)) / Ln;          
      end
    end
  end
%   for j = 1:indy+1
%       for k = 1:Nr
%            if zz(k) > 0-Lsurf % Yan: for prograde run
%               msk_temp(:,j,k) = (1/surfaceRelaxFac) * (zz(k)-(-Lsurf)) / Lsurf;
%            end
%       end
%   end
  
  %%% Plot the temperature relaxation timescale
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    [ZZ,YY]=meshgrid(zz,yy);
    contourf(YY,ZZ,squeeze(msk_temp(1,:,:)),30);
    colorbar;
    xlabel('y');
    ylabel('z');
    title('Temperature relaxation fraction');
  end  
  
  %%% Plot the salinity relaxation timescale
  if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    [ZZ,YY]=meshgrid(zz,yy);
    contourf(YY,ZZ,squeeze(msk_salt(1,:,:)),30);
    colorbar;
    xlabel('y');
    ylabel('z');
    title('Salinity relaxation fraction');
  end  
   
  %%% Save as an input parameters
  writeDataset(msk_temp,fullfile(inputpath,'rbcs_temp_mask.bin'),ieee,prec); 
  rbcs_parm01.addParm('relaxMaskFile(1)','rbcs_temp_mask.bin',PARM_STR);
  writeDataset(msk_salt,fullfile(inputpath,'rbcs_salt_mask.bin'),ieee,prec); 
  rbcs_parm01.addParm('relaxMaskFile(2)','rbcs_salt_mask.bin',PARM_STR);  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% WRITE THE 'data.rbcs' FILE %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
 
  %%% Creates the 'data.rbcs' file
  write_data_rbcs(inputpath,RBCS_PARM,listterm,realfmt);
  
  
    
  
  %%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%
  %%%%% OBCS %%%%%
  %%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%
  
  
  
    
  %%%%%%%%%%%%%%%%%%%%%%%
  %%%%% OBCS SET-UP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% To store parameter names and values
  obcs_parm01 = parmlist;
  obcs_parm02 = parmlist;
  OBCS_PARM = {obcs_parm01,obcs_parm02};  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% DEFINE OPEN BOUNDARY TYPES (OBCS_PARM01) %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Enables an Orlanski radiation condition at the northern boundary
  useOrlanskiNorth = true;
  OB_Jnorth = Ny*ones(1,Nx);     
  obcs_parm01.addParm('useOrlanskiNorth',useOrlanskiNorth,PARM_BOOL);
  obcs_parm01.addParm('OB_Jnorth',OB_Jnorth,PARM_INTS); 
  
  %%% Enforces mass conservation across the northern boundary by adding a
  %%% barotropic inflow/outflow
  useOBCSbalance = true;
  OBCS_balanceFacN = -1; 
  OBCS_balanceFacE = 0;
  OBCS_balanceFacS = 0;
  OBCS_balanceFacW = 0;
  obcs_parm01.addParm('useOBCSbalance',useOBCSbalance,PARM_BOOL);  
  obcs_parm01.addParm('OBCS_balanceFacN',OBCS_balanceFacN,PARM_REAL);  
  obcs_parm01.addParm('OBCS_balanceFacE',OBCS_balanceFacE,PARM_REAL);  
  obcs_parm01.addParm('OBCS_balanceFacS',OBCS_balanceFacS,PARM_REAL);  
  obcs_parm01.addParm('OBCS_balanceFacW',OBCS_balanceFacW,PARM_REAL);  
  

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% ORLANSKI OPTIONS (OBCS_PARM02) %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Velocity averaging time scale - must be larger than deltaT.
  %%% The Orlanski radiation condition computes the characteristic velocity
  %%% at the boundary by averaging the spatial derivative normal to the 
  %%% boundary divided by the time step over this period.
  %%% At the moment we're using the magic engineering factor of 3.
  cvelTimeScale = 3*deltaT;
  %%% Max dimensionless CFL for Adams-Basthforth 2nd-order method
  CMAX = 0.45; 
  
  obcs_parm02.addParm('cvelTimeScale',cvelTimeScale,PARM_REAL);
  obcs_parm02.addParm('CMAX',CMAX,PARM_REAL);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% WRITE THE 'data.obcs' FILE %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Creates the 'data.obcs' file
  write_data_obcs(inputpath,OBCS_PARM,listterm,realfmt);
  
  
  
  
  
  %%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%
  %%%%% LAYERS %%%%%
  %%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%
  
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% LAYERS SET-UP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% To store parameter names and values
  layers_parm01 = parmlist;
  LAYERS_PARM = {layers_parm01};
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% LAYERS PARAMETERS %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Define parameters for layers package %%%
  
  %%% Number of fields for which to calculate layer fluxes
  layers_maxNum = 1;
  
  %%% Specify potential temperature
  layers_name = 'TH';  
  
  %%% Potential temperature bounds for layers  
  if (neg_delta)
    layers_bounds = mintemp:0.5:maxtemp+difftemp;
  else
%     layers_bounds = mintemp:(maxtemp-mintemp)/100:maxtemp;
    layers_bounds = [mintemp flip(tNorth) maxtemp];
  end
  
  %%% Reference level for calculation of potential density
  layers_krho = 1;    
  
  %%% If set true, the GM bolus velocity is added to the calculation
  layers_bolus = false;  
   
  %%% Layers
  layers_parm01.addParm('layers_bounds',layers_bounds,PARM_REALS); 
  layers_parm01.addParm('layers_krho',layers_krho,PARM_INT); 
  layers_parm01.addParm('layers_name',layers_name,PARM_STR); 
  layers_parm01.addParm('layers_bolus',layers_bolus,PARM_BOOL); 

  %%z% Create the data.layers file
  write_data_layers(inputpath,LAYERS_PARM,listterm,realfmt);
  
  %%% Create the LAYERS_SIZE.h file
  createLAYERSSIZEh(codepath,length(layers_bounds)-1,layers_maxNum); 
  
  
  
  
    
  %%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%
  %%%%% GMREDI %%%%%
  %%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%
  
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% GMREDI SET-UP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% To store parameter names and values
  gmredi_parm01 = parmlist;
  GMREDI_PARM = {gmredi_parm01};
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% GMREDI PARAMETERS %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Define parameters for gmredi package %%%
  %%%http://mailman.mitgcm.org/pipermail/mitgcm-support/2016-April/010377.html%%%

  %%% Isopycnal diffusivity
  GM_isopycK          = 0;

  %%% Thickness diffusivity




  %%% DM95 critical slope
%   GM_Scrit            = 0.025;

  %%% DM95 tapering width
%   GM_Sd               = 0.0025;

  %%% Add parameters
  gmredi_parm01.addParm('GM_isopycK',GM_isopycK,PARM_REAL);
  gmredi_parm01.addParm('GM_background_K',GM_background_K,PARM_REAL);
  gmredi_parm01.addParm('GM_maxSlope',GM_maxSlope,PARM_REAL);
  gmredi_parm01.addParm('GM_taper_scheme',GM_taper_scheme,PARM_STR);
  
  if strcmp(GM_taper_scheme,'dm95')
  gmredi_parm01.addParm('GM_Scrit',GM_Scrit,PARM_REAL);
  gmredi_parm01.addParm('GM_Sd',GM_Sd,PARM_REAL);
  end
  %%% Add NN parameters 
  gmredi_parm01.addParm('Prograde',Prograde,PARM_BOOL);
  gmredi_parm01.addParm('GM_useML',GM_useML,PARM_BOOL);
  gmredi_parm01.addParm('GM_NoParCel',GM_NoParCel,PARM_BOOL);
  gmredi_parm01.addParm('REDI_useML',REDI_useML,PARM_BOOL);
  gmredi_parm01.addParm('GM_MLEner',GM_MLEner,PARM_BOOL);
   gmredi_parm01.addParm('GM_Burger',GM_Burger,PARM_BOOL);
  gmredi_parm01.addParm('GM_ML_minN2',GM_ML_minN2,PARM_REAL);
  gmredi_parm01.addParm('GM_ML_minVal_K',GM_ML_minVal_K,PARM_REAL);
  gmredi_parm01.addParm('GM_ML_maxVal_K',GM_ML_maxVal_K,PARM_REAL);
  
  
    gmredi_parm01.addParm('UpVp_useML',UpVp_useML,PARM_BOOL);
    gmredi_parm01.addParm('SmoothUpVp',SmoothUpVp,PARM_BOOL);
    gmredi_parm01.addParm('UpVp_ML_maxVal',UpVp_ML_maxVal,PARM_REAL);
    gmredi_parm01.addParm('DUpVpDy_ML_maxVal',DUpVpDy_ML_maxVal,PARM_REAL);
    gmredi_parm01.addParm('UpVp_VertStruc',UpVp_VertStruc,PARM_BOOL);
%     gmredi_parm01.addParm('GM_VertStruc',GM_VertStruc,PARM_BOOL);
    gmredi_parm01.addParm('SmoothDUpVpDy',SmoothDUpVpDy,PARM_BOOL);
    gmredi_parm01.addParm('SmoothKgm',SmoothKgm,PARM_BOOL);
    
if(GM_useML)
  if(~GM_MLEner)  
 copyfile(strcat(NNpath_GM,'\GM_w_0.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_GM,'\GM_w_1.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_GM,'\GM_w_2.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_GM,'\GM_b_0.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_GM,'\GM_b_1.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_GM,'\GM_b_2.txt'),fullfile(inputpath));  
 copyfile(strcat(NNpath_GM,'\GM_normal.txt'),fullfile(inputpath));  
  else
 copyfile(strcat(NNpath_E,' \E_w_0.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_E,'\E_w_1.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_E,'\E_w_2.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_E,'\E_b_0.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_E,'\E_b_1.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_E,'\E_b_2.txt'),fullfile(inputpath));  
 end
end
  
%%z% Create the data.gmredi file
  write_data_gmredi(inputpath,GMREDI_PARM,listterm,realfmt);
  
 if(UpVp_useML)
 copyfile(strcat(NNpath_Rey,'\UpVp_w_0.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_Rey,'\UpVp_w_1.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_Rey,'\UpVp_w_2.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_Rey,'\UpVp_b_0.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_Rey,'\UpVp_b_1.txt'),fullfile(inputpath));
 copyfile(strcat(NNpath_Rey,'\UpVp_b_2.txt'),fullfile(inputpath));  
 copyfile(strcat(NNpath_E,'\E_normal.txt'),fullfile(inputpath));  
 end
 
 
 
 
 
 
   if(REDI_useML)
% calculate Lw and store it as an input


%%% Lw
SouthWallpos = zeros(1,Nr);
for i = 1:Nr
 pos =   find(zz(i)>h,1,'first');     
 SouthWallpos(i) = yy(pos)-dy(1)/2;
end
[YYY, ~, ~] = meshgrid(yy, xx, zz);

SouthWallpos = repmat(reshape(SouthWallpos,[1,1,Nr]),[Nx,Ny,1]);
Lw = (YYY-SouthWallpos)./pi;
NorthWallpos = yy(Ny-1)+dy(1)/2;
Lw = min((NorthWallpos-YYY)./pi,Lw);
Lw(Lw<0)=0

    if (showplots)
    figure(fignum);
    fignum = fignum + 1;
    clf;
    [ZZ,YY]=meshgrid(zz,yy);
    pcolor(YY,ZZ,squeeze(mean(Lw,1)));
    shading flat
    colorbar;
    xlabel('y');
    ylabel('z');
    title('Lw');
    end  
  
    
% fid=fopen([fullfile(inputpath) '\Lw.txt'],'wt');
%   for j=1:Ny
%     for k=1:Nr
%      fprintf(fid,'%d ',Lw(j,k));
%     end
%      fprintf(fid,'\n');
%   end
% fclose(fid); 
    
  writeDataset(Lw,fullfile(inputpath,'Lw.bin'),ieee,prec); 
 
   end  
   
   
   
   copyfile('G:\OtherModels\gmredi_ML\Model\src\Modified\calc_eddy_stress.F',fullfile(codepath));  
   copyfile('G:\OtherModels\gmredi_ML\Modified\gmredi_calc_ML.F',fullfile(codepath));  
   copyfile('G:\OtherModels\gmredi_ML\Modified\GMREDI.h',fullfile(codepath));  
   copyfile('G:\OtherModels\gmredi_ML\Modified\gmredi_readparms.F',fullfile(codepath));  
   copyfile('G:\OtherModels\gmredi_ML\Modified\gmredi_calc_tensor.F',fullfile(codepath));
 
 
 
 
 
 
  %%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%
  %%%%% DIAGNOSTICS %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%
    
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% DIAGNOSTICS SET-UP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% To store parameter names and values
  diag_parm01 = parmlist;
  diag_parm02 = parmlist;
  DIAG_PARM = {diag_parm01,diag_parm02};
  diag_matlab_parm01 = parmlist;
  DIAG_MATLAB_PARM = {diag_matlab_parm01}; %%% Matlab parameters need to be different
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% DIAGNOSTICS PARAMETERS %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  
  %%% Stores total number of diagnostic quantities
  ndiags = 0;
  
  diag_fields_avg = ...
  {...
    'UVEL','VVEL','WVEL',... %%% Velocities
    'UVELSQ','VVELSQ','WVELSQ',... %%% For Kinetic Energy
    'UV_VEL_Z', 'UV_VEL_C','WU_VEL','WV_VEL'... %%% Momentum fluxes
    'UVELTH','VVELTH','WVELTH', ... %%% Temperature fluxes
    'UVELSLT','VVELSLT','WVELSLT', ... %%% Salt fluxes
    'THETA','THETASQ', ... %%% Temperature
    'SALT','SALTSQ', ... %%% Salinity
    'PHIHYD', ... %%% Pressure
    'UUU_VEL', 'UVV_VEL', 'UWW_VEL', ...
    'VUU_VEL', 'VVV_VEL', 'VWW_VEL', ...
    'WUU_VEL', 'WVV_VEL', 'WWW_VEL', ...
    'UVELPHI', 'VVELPHI', 'WVELPHI', 'MXLDEPTH', ...
    'momKE', 'momHDiv', 'momVort3', ...          %%% momentum diagnostics
    'botTauX','botTauY', 'USidDrag', 'VSidDrag', ...
    'Um_Diss', 'Vm_Diss', 'Um_Advec', 'Vm_Advec', ...
    'Um_Cori', 'Vm_Cori', 'Um_Ext', 'Vm_Ext', ...
    'Um_AdvZ3', 'Vm_AdvZ3', 'Um_AdvRe', 'Vm_AdvRe', ...
    'ADVx_Um', 'ADVy_Um', 'ADVrE_Um', 'ADVx_Vm', ...
    'ADVy_Vm', 'ADVrE_Vm', 'VISCx_Um', 'VISCy_Um', ...
    'VISrE_Um', 'VISrI_Um', 'VISCx_Vm', 'VISCy_Vm', ...
    'VISrE_Vm', 'VISrI_Vm'
  };
% diag_fields_avg = {}; %...
%   {'UUU_VEL', 'UVV_VEL', 'UWW_VEL', ...
%     'VUU_VEL', 'VVV_VEL', 'VWW_VEL', ...
%     'WUU_VEL', 'WVV_VEL', 'WWW_VEL', ...
%     'WVELPHI'};
  
  numdiags_avg = length(diag_fields_avg);  
  diag_freq_avg = 5*t1year;
%  diag_freq_avg = 5*t1year;

     
  diag_parm01.addParm('diag_mnc',false,PARM_BOOL);  
  for n=1:numdiags_avg    
    
    ndiags = ndiags + 1;
    
    diag_parm01.addParm(['fields(1,',num2str(n),')'],diag_fields_avg{n},PARM_STR);  
    diag_parm01.addParm(['fileName(',num2str(n),')'],diag_fields_avg{n},PARM_STR);  
    diag_parm01.addParm(['frequency(',num2str(n),')'],diag_freq_avg,PARM_REAL);  
    diag_parm01.addParm(['timePhase(',num2str(n),')'],diag_phase_avg,PARM_REAL); 
    diag_matlab_parm01.addParm(['diag_fields{1,',num2str(n),'}'],diag_fields_avg{n},PARM_STR);  
    diag_matlab_parm01.addParm(['diag_fileNames{',num2str(n),'}'],diag_fields_avg{n},PARM_STR);  
    diag_matlab_parm01.addParm(['diag_frequency(',num2str(n),')'],diag_freq_avg,PARM_REAL);  
    diag_matlab_parm01.addParm(['diag_timePhase(',num2str(n),')'],diag_phase_avg,PARM_REAL);  
    
  end

  diag_fields_inst = ...
 {...
   'UVEL','VVEL','WVEL','THETA', ...
   'GM_ANN',...
   'UpVp_ML','DupvpDy','DupvpDy2', ...
   'ReANN_I1','ReANN_I2','ReANN_I3', ...   
   'EKEANNI1','EKEANNI2','EKEANNI3', 'EKEANNI4',...
  };
%    'REDI_K0','REDI_A','REDI_S', 'REDI_C', 'Sdelta','REDI_ANN',...   
  diag_fields_inst = ...
 {...
   'UVEL','VVEL','WVEL','THETA', ...
  };



  numdiags_inst = length(diag_fields_inst);  
  diag_freq_inst = .1*t1year;
%   diag_freq_inst = 1310;
%   diag_freq_inst = 1*t1day;
  diag_phase_inst = 0;
  
  for n=1:numdiags_inst    
    ndiags = ndiags + 1;
    diag_parm01.addParm(['fields(1,',num2str(ndiags),')'],diag_fields_inst{n},PARM_STR);  
    diag_parm01.addParm(['fileName(',num2str(ndiags),')'],[diag_fields_inst{n},'_inst'],PARM_STR);  
    diag_parm01.addParm(['frequency(',num2str(ndiags),')'],-diag_freq_inst,PARM_REAL);  
    diag_parm01.addParm(['timePhase(',num2str(ndiags),')'],diag_phase_inst,PARM_REAL); 
    diag_matlab_parm01.addParm(['diag_fields(1,',num2str(ndiags),')'],diag_fields_inst{n},PARM_STR);  
    diag_matlab_parm01.addParm(['diag_fileNames(',num2str(ndiags),')'],[diag_fields_inst{n},'_inst'],PARM_STR);  
    diag_matlab_parm01.addParm(['diag_frequency(',num2str(ndiags),')'],-diag_freq_inst,PARM_REAL);  
    diag_matlab_parm01.addParm(['diag_timePhase(',num2str(ndiags),')'],diag_phase_inst,PARM_REAL);  
  end
  
  %%% Create the data.diagnostics file
  write_data_diagnostics(inputpath,DIAG_PARM,listterm,realfmt);
  
  %%% Create the DIAGNOSTICS_SIZE.h file
  createDIAGSIZEh(codepath,ndiags,max(Nr,length(layers_bounds)-1));
  
  
  
  
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% WRITE PARAMETERS TO A MATLAB FILE %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  %%% Creates a matlab file defining all input parameters
  write_matlab_params(inputpath,[PARM RBCS_PARM OBCS_PARM DIAG_MATLAB_PARM LAYERS_PARM],realfmt);
 
  
end
