#include "GMREDI_OPTIONS.h"

CBOP
C !ROUTINE: GMREDI_CALC_ML

C !INTERFACE: ==========================================================
      SUBROUTINE GMREDI_CALC_ML(
     I             sigmaX,sigmaY,sigmaR,
     I             bi, bj, myTime, myIter, myThid )

C     !DESCRIPTION:
C     *==========================================================*
C     | SUBROUTINE GMREDI_CALC_ML
C     | Parameterize Kgm and u'v' for non-eddying runs.
C     | More details in Xie, Wei and Wang (2023)
C     *==========================================================*
C     | Kgm is fomulated by physical scalings using 
C     | EKE (or E) predicted by Artificial Neural Network (ANN).
C     *==========================================================*
C     | u'v' is predicted directly by (another) ANN.
C     *==========================================================*
C     | Kgm and u'v' are located at cell centers.
C     *==========================================================*

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "GMREDI.h"

C !INPUT PARAMETERS: ===================================================
C     bi, bj    :: tile indices
C     myTime    :: Current time in simulation
C     myIter    :: Current iteration number in simulation
C     myThid    :: My Thread Id. number
C     sigmaXYR  :: Density gradients in x, y and z direction
      INTEGER bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
      _RL sigmaX(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL sigmaY(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL sigmaR(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)

      
C !OUTPUT PARAMETERS: ==================================================

#ifdef ALLOW_GM_ML

C !LOCAL VARIABLES: ====================================================
C     N            ::  Buoyancy frequency
C     N2           ::  Square of buoyancy frequency
C     uFld_C       ::  Zonal velocity
C     DuDy         ::  Meridional gradient of zonal velocity
C     DuDz         ::  Vertical gradient of zonal velocity
C     E_ML_temp    ::  Eddy kinetic energy (EKE) predicted by ANN
C     UpVp_ML_temp ::  u'v' predicted by ANN
C     GM_ML_temp   ::  Eddy diffusivity following GEOMETRIC or MLT     
      _RL N(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL N2(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL uFld_C(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL DuDy(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL DuDz(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr) 
      _RL E_ML(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL E_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL UpVp_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL GM_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)      
      
C     dSigmaDxZint :: Zonal gradient of potential density     
C     dSigmaDyZint :: Meridional gradient of potential density
C     dSigmaDrZint :: Vertical gradient of potential density     
C     BVint        :: Vertical integral of buoyancy frequency
C     Siso         :: Bulk isopycnal slope      
C     delta        :: Slope parameter
C     Ri           :: Bulk Richardson number
C     Ld           :: WKB first BC deformation radius    
C     Lrhines      :: Rhines scale
C     H            :: Absolute value of water depth (positive)
C     dHdy         :: Offshore bathymetric slope 
C     Beta_t       :: Topographic beta (f/H*H_y)
C     E_ML_zavg    :: Vertical integral of EKE predicted by ANN     
C     Ue           :: Bulk eddy velocity  
      _RL dSigmaDxZint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL dSigmaDyZint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL dSigmaDrZint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL BVint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)      
      _RL Siso(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL delta(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Ri(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Ld(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL Lrhines(1-OLx:sNx+OLx,1-OLy:sNy+OLy)      
      _RL H(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL dHdy(1-Olx:sNx+Olx,1-Oly:sNy+Oly)  
      _RL Beta_t(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Siso_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL E_ML_zavg(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Ue(1-Olx:sNx+Olx,1-Oly:sNy+Oly) 

      
C     Input_UpVp   :: Input matrix of the ANN that predicts u'v'     
C     Input_E      :: Input matrix of the ANN that predicts EKE            
      _RL Input_UpVp(6,1) 
      _RL ML0_UpVp(64,1)
      _RL ML1_UpVp(32,1)
      _RL ML2_UpVp(1,1)    
      _RL Input_E(6,1)       
      _RL ML0_E(64,1)
      _RL ML1_E(32,1)
      _RL ML2_E(1,1)        
      
C     kLow_C :: Local version of the index of deepest wet grid cell on tracer grid
      INTEGER kLow_C(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
C     i,j,k    :: Loop counters
      INTEGER i,j,k,kp1,km1
CEOP

      
      
      
C     ======================================
C     Initialise some variables
C     ======================================

      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx
        kLow_C(i,j) = kLowC(i,j,bi,bj)
       ENDDO
      ENDDO

C     initialise 3d variables
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx
         N2(i,j,k)= GM_ML_minN2
         N(i,j,k) = SQRT(GM_ML_minN2)
         GM_ML_temp(i,j,k) = GM_ML_minVal_K 
         GM_ML_K(i,j,k,bi,bj) = GM_ML_minVal_K        
        ENDDO
       ENDDO
      ENDDO
      
C     Zeroing some cumulative fields
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx
        BVint(i,j)  = 0.0 _d 0
        E_ML_zavg(i,j)  = 0.0 _d 0
        dSigmaDxZint(i,j)  = 0.0 _d 0
        dSigmaDyZint(i,j)  = 0.0 _d 0
        dSigmaDrZint(i,j)  = 0.0 _d 0
       ENDDO
      ENDDO
      
      

C     ---------- calculation begins -----------
C     Prepare input variables for the trained fully connected 
C     artificial neural networks which are used to predict EKE and 
C     u'v', and prepare variables that are needed in GEOMETRIC and MLT
         


C    Zonal velocity at cell center
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx-1
            uFld_C(i,j,k) = op5 *
     &               (uVel(i+1,j,k,bi,bj)+uVel(i,j,k,bi,bj))
        ENDDO
       ENDDO
      ENDDO

      
      
C    Find the water depth at cell center (positive)
C    R_low_orig is defined by ignoring hFacMin constraint,
C    which provides a much smoother dHdy field.
C    Notice that H defined below cannot be used in depth-avg calculations.
C    H is positive.      
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx
          H(i,j) = -R_low_orig(i,j,bi,bj)
       ENDDO
      ENDDO
      
      
C    Bathymetric slope at cell center
      DO j=1-Oly+1,sNy+Oly-1
       DO i=1-Olx,sNx+Olx
        dHdy(i,j) = -op5 *
     &            (H(i,j+1)-H(i,j-1))
     &            *_recip_dyF(i,j,bi,bj)*maskC(i,j,1,bi,bj)
     &            *maskC(i,j+1,1,bi,bj)*maskC(i,j-1,1,bi,bj)
       ENDDO
      ENDDO
 
      
      
C     Topographic beta at cell center   
      DO j=1-Oly+1,sNy+Oly-1
       DO i=1-Olx,sNx+Olx
          IF ( H(i,j).NE. 0.0 _d 0 ) THEN
          Beta_t(i,j) = abs(fCori(i,j,bi,bj)/H(i,j)*dHdy(i,j))
          ELSE
          Beta_t(i,j) = 0.0 _d 0
          ENDIF
       ENDDO
      ENDDO      

      
      
C     Square of the buoyancy frequency at the top of a grid cell
C     Enforce a minimum N2; Mask N2, so it is zero at bottom
C     Calculation of N2 and N follows gmredi_k3d.F
      DO k=2,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx
         N2(i,j,k) = -gravity*recip_rhoConst*sigmaR(i,j,k)  
         N2(i,j,k) = MAX(N2(i,j,k),GM_ML_minN2)*maskC(i,j,k,bi,bj)
         N(i,j,k)  = SQRT(N2(i,j,k))
        ENDDO
       ENDDO
      ENDDO
C     N2(k=1) is always zero
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx
        N2(i,j,1) = 0.0 _d 0
        sigmaR(i,j,1) = 0.0 _d 0
        N(i,j,1)  = 0.0 _d 0
       ENDDO
      ENDDO
      
C     Integrate the buoyancy frequency vertically
      DO k=1,Nr
       kp1 = min(k+1,Nr)
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx
           BVint(i,j) = BVint(i,j) + hFacC(i,j,k,bi,bj)*drF(k)
     &                         *op5*(MaskC(i,j,k,bi,bj)*N(i,j,k)
     &                         +MaskC(i,j,kp1,bi,bj)*N(i,j,kp1))
        ENDDO
       ENDDO
      ENDDO    
      
C       Calculate the WKB Rossby deformation Radius
C       Cannot apply to equatorial region
        DO j=1-Oly,sNy+Oly
         DO i=1-Olx,sNx+Olx
          IF ( fCori(i,j,bi,bj).NE. 0.0 _d 0 ) THEN
          Ld(i,j)  = BVint(i,j)/fCori(i,j,bi,bj)/pi
           ELSE
          Ld(i,j)  = 0.0 _d 0
          ENDIF
         ENDDO
      ENDDO
 

C     Integrate rho_x and rho_y vertically   
C     sigmaX and sigmaY locate at U and V points respectively
C     dSigmaDxZint and dSigmaDyZint locate at cell centers     
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1
         dSigmaDxZint(i,j) = dSigmaDxZint(i,j) + op5
     &        *drF(k)*hfacC(i,j,k,bi,bj)*  
     &        ( sigmaX(i,j,k) + sigmaX(i+1,j,k) )
         dSigmaDyZint(i,j) = dSigmaDyZint(i,j) + op5
     &        *drF(k)*hfacC(i,j,k,bi,bj)*  
     &        ( sigmaY(i,j,k) + sigmaY(i,j+1,k) )
        ENDDO
       ENDDO
      ENDDO   
C     Integrate rho_z vertically, same as BVint
      DO k=1,Nr
       kp1 = min(k+1,Nr)
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx
      dSigmaDrZint(i,j) = dSigmaDrZint(i,j) + hFacC(i,j,k,bi,bj)*drF(k)
     &                     *op5*(MaskC(i,j,k,bi,bj)*sigmaR(i,j,k)
     &                         +MaskC(i,j,kp1,bi,bj)*sigmaR(i,j,kp1))
        ENDDO
       ENDDO
      ENDDO

C     Calculate Bulk Richardson Number which is used in GEOMETIC
        DO j=1-Oly,sNy+Oly-1
         DO i=1-Olx,sNx+Olx-1
          IF ( dSigmaDrZint(i,j).NE. 0.0 _d 0 ) THEN  
          Ri(i,j)= -fCori(i,j,bi,bj)**2*r_Low(i,j,bi,bj)
     &              *abs(dSigmaDrZint(i,j)
     &              /(dSigmaDxZint(i,j)**2+dSigmaDyZint(i,j)**2
     &               *gravity*recip_rhoConst)) 
          ELSE
          Ri(i,j) = 0.0 _d 0
          ENDIF
         ENDDO
        ENDDO   

  
          
C     3D Meridional (offhsore) gradient of zonal velocity
C     DuDy is located at cell center          
      DO k=1,Nr
       DO j=1-OLy+1,sNy+OLy-1
        DO i=1-OLx,sNx+OLx-1
        DuDy(i,j,k)=op5*(MaskS(i,j,k,bi,bj)       
     &        *_recip_dyC(i,j,bi,bj)
     &        *(uFld_C(i,j,k)-uFld_C(i,j-1,k))
     &         +MaskS(i,j+1,k,bi,bj)
     &        *_recip_dyC(i,j+1,bi,bj)
     &        *(uFld_C(i,j+1,k)-uFld_C(i,j,k)))
        ENDDO
       ENDDO
      ENDDO
      
      
C     3D Vertical gradient of zonal velocity
C     DuDz is located at cell center
C     Follows the calculation of dTdz in gmredi_ytransport.F  
      DO k=1,Nr
       kp1 = min(k+1,Nr)
       km1 = max(k-1,1)
       DO j=1-OLy,sNy+OLy
        DO i=1-OLx,sNx+OLx-1      
         DuDz(i,j,k) =  op5*(
     &         recip_drC(k)*maskC(i,j,k,bi,bj)*
     &            (uFld_C(i,j,km1)-uFld_C(i,j,k))
     &        +recip_drC(kp1)*maskC(i,j,kp1,bi,bj)*
     &            (uFld_C(i,j,k)-uFld_C(i,j,kp1)))
        ENDDO
       ENDDO
      ENDDO
           
   
C     Bulk isopycnal slope at cell centers    
        DO j=1-Oly,sNy+Oly-1
         DO i=1-Olx,sNx+Olx-1
          IF ( dSigmaDrZint(i,j).NE. 0.0 _d 0 ) THEN                
C  Siso = -sign(dSigmaDyZint)(dSigmaDyZint^2+dSigmaDxZint^2)/dSigmaDrZint
C  Assuming dSigmaDrZint is negative
          Siso_temp(i,j)=  dSigmaDyZint(i,j)/abs(dSigmaDyZint(i,j))
     &                *sqrt(dSigmaDxZint(i,j)**2+dSigmaDyZint(i,j)**2)
     &                 /abs(dSigmaDrZint(i,j))
          ELSE
          Siso_temp(i,j) = 0.0 _d 0         
          ENDIF
         ENDDO
        ENDDO     
C     Enforce lower and upper bounds for isopycnal slope
       DO j=1-Oly,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1   
            Siso(i,j) = MIN(Siso_temp(i,j),ANN_maxSlope)
            Siso(i,j) = MAX(Siso(i,j),-ANN_maxSlope)
        ENDDO
       ENDDO     

           

#ifdef ALLOW_DIAGNOSTICS
C     Diagnostics
      IF ( useDiagnostics ) THEN
      CALL DIAGNOSTICS_FILL(DuDy,        'ReANN_I1',0,Nr,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(DuDz,        'ReANN_I2',0,Nr,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(Siso,        'ReANN_I3',0,1,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(beta_t,      'ReANN_I4',0,1,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(sigmaY,      'ReANN_I5',0,Nr,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(Ld,          'ReANN_I6',0,1,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(dSigmaDyZint,'ReANN_I7',0,1,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(dSigmaDrZint,'ReANN_I8',0,1,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(sigmaR,      'ReANN_I9',0,Nr,2,bi,bj,myThid)
      ENDIF
#endif




 
      
C     ---------- ANN-inferred eddy momentum flux (EMF) ------------
      IF ( UpVp_useML) THEN
C    Predict EMF (u'v') over continental slope by
C    trained fully connected artificial neural network

      DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1
         IF (k.LE.kLow_C(i,j)) THEN
C InputFlag             
             Input_UpVp(1,1) = DuDy(i,j,k)/E_ML_normal(1,1)
             Input_UpVp(2,1) = DuDz(i,j,k)/E_ML_normal(1,2)
             Input_UpVp(3,1) = Siso(i,j)/E_ML_normal(1,3)		   	   
             Input_UpVp(4,1) = beta_t(i,j)/E_ML_normal(1,4)
             Input_UpVp(5,1) = (4000-abs(rC(k)))/4000/E_ML_normal(1,5)
             Input_UpVp(6,1) = Ld(i,j)/E_ML_normal(1,6)
             ML0_UpVp = matmul(UpVp_ML_w0,Input_UpVp) + UpVp_ML_b0
             ML0_UpVp = ((1.+0.2)*ML0_UpVp+(1.0-0.2)*abs(ML0_UpVp))/2.0
             ML1_UpVp = matmul(UpVp_ML_w1,ML0_UpVp) + UpVp_ML_b1
             ML1_UpVp = ((1.+0.2)*ML1_UpVp+(1.0-0.2)*abs(ML1_UpVp))/2.0
             ML2_UpVp =  matmul(UpVp_ML_w2,ML1_UpVp) + UpVp_ML_b2
C InputFlag              
             UpVp_ML_temp(i,j,k) = ML2_UpVp(1,1)/abs(ML2_UpVp(1,1))
     &                           *abs(ML2_UpVp(1,1)*E_ML_normal(1,7))**2
C            Enforce lower and upper bounds 
             UpVp_ML_temp(i,j,k) = 
     &                      MIN(UpVp_ML_temp(i,j,k),UpVp_ML_maxVal)
             UpVp_ML_temp(i,j,k) = 
     &                      MAX(UpVp_ML_temp(i,j,k),-UpVp_ML_maxVal)
           ENDIF
        ENDDO
       ENDDO
       ENDDO
       
       
      IF ( SmoothUpVp ) THEN 
      DO k=1,Nr
      DO j=1-OLy+3,sNy+OLy-3
       DO i=1-OLx,sNx+OLx-1        
C     Smooth UpVp horizontally
C     5-point running smooth      
          UpVp_ML(i,j,k,bi,bj) = ( UpVp_ML_temp(i,j,k)*1. _d 0
     &               + UpVp_ML_temp(i,j-1,k)*maskC(i,j-1,k,bi,bj)
     &               + UpVp_ML_temp(i,j+1,k)*maskC(i,j+1,k,bi,bj)
     &               + UpVp_ML_temp(i,j-2,k)*maskC(i,j-2,k,bi,bj)
     &               + UpVp_ML_temp(i,j+2,k)*maskC(i,j+2,k,bi,bj))
     &               /( 1. _d 0
     &                 + maskC(i,j-1,k,bi,bj)
     &                 + maskC(i,j+1,k,bi,bj)
     &                 + maskC(i,j-2,k,bi,bj)          
     &                 + maskC(i,j+2,k,bi,bj))      
     &                 * maskC(i,j,k,bi,bj)
          ENDDO
       ENDDO 
      ENDDO 

      
      ELSE
          
      DO k=1,Nr
      DO j=1-OLy+1,sNy+OLy-1
       DO i=1-OLx,sNx+OLx-1    
          UpVp_ML(i,j,k,bi,bj) = UpVp_ML_temp(i,j,k)
        ENDDO
      ENDDO   
      ENDDO

C   ENDIF for IF ( SmoothUpVp )    
      ENDIF       
       

C   ENDIF for IF ( UpVp_useML )
      ENDIF
      
      
      
       
       
      
C     ----------     Kgm part     ------------
   
      IF ( .NOT. GM_useML) THEN
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx
             GM_ML_k(i,j,k,bi,bj) = 0.0 _d 0
        ENDDO
       ENDDO
      ENDDO       
      ENDIF       
      



      IF ( GM_useML) THEN
C    Energy-kgm scheme begins  

C    Predict depth-varying EKE (or E) over continental slope
C    by trained fully connected artificial neural network. 
C    EKE (or E) is later used to infer Kgm via GEOMETRIC or MLT     

      DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1
         IF (k.LE.kLow_C(i,j)) THEN              
             Input_E(1,1) = DuDy(i,j,k)/E_ML_normal(1,1)
             Input_E(2,1) = DuDz(i,j,k)/E_ML_normal(1,2)
             Input_E(3,1) = Siso(i,j)/E_ML_normal(1,3)	  		   
             Input_E(4,1) = beta_t(i,j)/E_ML_normal(1,4)
             Input_E(5,1) = (4000-abs(rC(k)))/4000/E_ML_normal(1,5)
             Input_E(6,1) = Ld(i,j)/E_ML_normal(1,6)
             ML0_E = matmul(E_ML_w0,Input_E) + E_ML_b0
             ML0_E = ((1.+0.2)*ML0_E+(1.0-0.2)*abs(ML0_E))/2.0
             ML1_E = matmul(E_ML_w1,ML0_E) + E_ML_b1
             ML1_E = ((1.+0.2)*ML1_E+(1.0-0.2)*abs(ML1_E))/2.0
             ML2_E(1:1,1:1) = matmul(E_ML_w2,ML1_E) + E_ML_b2   
             E_ML_temp(i,j,k) =    op5 *ML2_E(1,1)/abs(ML2_E(1,1))
     &                            *abs(ML2_E(1,1)*E_ML_normal(1,7))**2 
C     No bound is set for the predicted value of E             
           ENDIF
        ENDDO
       ENDDO
      ENDDO

      
        
      IF (SmoothEner)  THEN
      DO k=1,Nr
      DO j=1-OLy+3,sNy+OLy-3
       DO i=1-OLx,sNx+OLx-1        
C     Smooth UpVp horizontally
C     5-point running smooth      
          E_ML(i,j,k) = ( E_ML_temp(i,j,k)*1. _d 0
     &                  + E_ML_temp(i,j-1,k)*maskC(i,j-1,k,bi,bj)
     &                  + E_ML_temp(i,j+1,k)*maskC(i,j+1,k,bi,bj)
     &                  + E_ML_temp(i,j-2,k)*maskC(i,j-2,k,bi,bj)
     &                  + E_ML_temp(i,j+2,k)*maskC(i,j+2,k,bi,bj))
     &                 /( 1. _d 0
     &                  + maskC(i,j-1,k,bi,bj)
     &                  + maskC(i,j+1,k,bi,bj)
     &                  + maskC(i,j-2,k,bi,bj)          
     &                  + maskC(i,j+2,k,bi,bj))      
     &                                         * maskC(i,j,k,bi,bj)
          ENDDO
       ENDDO 
      ENDDO               
        
      ELSE
           
      DO k=1,Nr
      DO j=1-OLy+1,sNy+OLy-1
       DO i=1-OLx,sNx+OLx-1    
          E_ML(i,j,k) = E_ML_temp(i,j,k)
        ENDDO
      ENDDO   
      ENDDO
      
C   ENDIF for IF ( SmoothEner )         
      ENDIF  
              
        
        
C    Calculate depth-averaged EKE, and then Ue
      DO k=1,Nr
       DO j=1-Oly+3,sNy+Oly-3
        DO i=1-Olx,sNx+Olx-1
            E_ML_zavg(i,j) = E_ML_zavg(i,j) + 
     &               drF(k)*hfacC(i,j,k,bi,bj)*E_ML(i,j,k)      
        ENDDO
       ENDDO
      ENDDO
      
      DO j=1-Oly+3,sNy+Oly-3
       DO i=1-Olx,sNx+Olx-1
        IF (r_Low(i,j,bi,bj).NE. 0.0 _d 0) THEN
C         The minus sign is because r_Low<0
          E_ML_zavg(i,j) = -E_ML_zavg(i,j)/r_Low(i,j,bi,bj)
        ELSE
          E_ML_zavg(i,j) = 0.0 _d 0
        ENDIF
          Ue(i,j)=sqrt(2.0*E_ML_zavg(i,j)) 
       ENDDO
       ENDDO

  
       
C     Predict Kgm in retrograde flows     
       
      IF(GM_SlopeAware) THEN
C     Slope-aware form          
C     Compute slope parameter
        DO j=1-Oly+1,sNy+Oly-1
         DO i=1-Olx,sNx+Olx-1
          IF ( Siso(i,j) .NE. 0.0 _d 0 ) THEN  
          delta(i,j) = dHdy(i,j)/Siso(i,j)
          ELSE
          delta(i,j) = 0.0 _d 0
          ENDIF
         ENDDO
        ENDDO    
    
        
      IF(GM_UeLrh) THEN           
C  Compute Kgm using slope-aware mixing length theory
C  More details in Wang and Stewart (2020)          
      DO j=1-Oly+3,sNy+Oly-3
       DO i=1-Olx,sNx+Olx-1
        IF (Beta_t(i,j).NE. 0.0 _d 0) THEN
           Lrhines(i,j) = sqrt(Ue(i,j)/Beta_t(i,j))
        ELSE
           Lrhines(i,j) = 0.0 _d 0
        ENDIF
       ENDDO
       ENDDO  

      DO k=1,Nr
       DO j=1-Oly+3,sNy+Oly-3
        DO i=1-Olx,sNx+Olx-1   
         GM_ML_temp(i,j,k) 
     &        =     3.3 _d -3 *( abs(delta(i,j)) +
     &              1/(abs(delta(i,j))+0.01) )*
     &              Ue(i,j)*Lrhines(i,j)
C            Enforce lower and upper bounds on the diffusivity
             GM_ML_temp(i,j,k) 
     &                = MIN(GM_ML_temp(i,j,k),GM_ML_maxVal_K)
             GM_ML_temp(i,j,k) 
     &                = MAX(GM_ML_temp(i,j,k),GM_ML_minVal_K)        
          ENDDO
        ENDDO
       ENDDO   

       
      ELSE
C     Compute Kgm via slope-aware GEOMETRIC framework
C     More details in Wang and Stewart (2020)   
      DO k=1,Nr
       DO j=1-Oly+3,sNy+Oly-3
        DO i=1-Olx,sNx+Olx-1            
         GM_ML_temp(i,j,k) 
     &       = 8.0 _d -3
     &         *(1.35*tanh(0.1*abs(delta(i,j)))+1/(abs(delta(i,j))+0.1))
     &         *sqrt(Ri(i,j))/fCori(i,j,bi,bj)*3*E_ML_zavg(i,j);
C            Enforce lower and upper bounds on the diffusivity
             GM_ML_temp(i,j,k) 
     &                = MIN(GM_ML_temp(i,j,k),GM_ML_maxVal_K)
             GM_ML_temp(i,j,k) 
     &                = MAX(GM_ML_temp(i,j,k),GM_ML_minVal_K)        
           ENDDO
       ENDDO
      ENDDO 
        

C ENDIF IF(GM_UeLrh)
      ENDIF
       


      
      ELSE
C     Non slope-aware form      
          
       IF(GM_UeLrh) THEN    
C    Predict Kgm via MLT          
       DO j=1-Oly+3,sNy+Oly-3
       DO i=1-Olx,sNx+Olx-1
        IF (Beta_t(i,j).NE. 0.0 _d 0) THEN
           Lrhines(i,j) = sqrt(Ue(i,j)/Beta_t(i,j))
        ELSE
           Lrhines(i,j) = 0.0 _d 0
        ENDIF
       ENDDO
      ENDDO  
                 
      DO k=1,Nr
       DO j=1-Oly+3,sNy+Oly-3
        DO i=1-Olx,sNx+Olx-1   
         GM_ML_temp(i,j,k) 
     &        =   1.3 _d -2 * Ue(i,j)*Lrhines(i,j)
C            Enforce lower and upper bounds on the diffusivity
             GM_ML_temp(i,j,k) 
     &                = MIN(GM_ML_temp(i,j,k),GM_ML_maxVal_K)
             GM_ML_temp(i,j,k) 
     &                = MAX(GM_ML_temp(i,j,k),GM_ML_minVal_K)        
           ENDDO
       ENDDO
       ENDDO  
    
       
       ELSE
C     Predict Kgm via GEOMETRIC       
      DO k=1,Nr
       DO j=1-Oly+3,sNy+Oly-3
        DO i=1-Olx,sNx+Olx-1           
         GM_ML_temp(i,j,k) 
     &       = 3.16 _d -2
     &         *sqrt(Ri(i,j))/fCori(i,j,bi,bj)*E_ML_zavg(i,j);         
C            Enforce lower and upper bounds on the diffusivity
             GM_ML_temp(i,j,k) 
     &                = MIN(GM_ML_temp(i,j,k),GM_ML_maxVal_K)
             GM_ML_temp(i,j,k) 
     &                = MAX(GM_ML_temp(i,j,k),GM_ML_minVal_K)        
           ENDDO
       ENDDO
      ENDDO 
      
C   ENDIF for (GM_UeLrh)            
       ENDIF
       
C   ENDIF for (GM_SlopeAware)              
      ENDIF   




           
          DO k=1,Nr
           DO j=1-Oly,sNy+Oly
            DO i=1-Olx,sNx+Olx  
             GM_ML_k(i,j,k,bi,bj) =   GM_ML_temp(i,j,k)               
            ENDDO
           ENDDO
          ENDDO 

      
      
      
C ENDIF for (GM_useML)      
       ENDIF
      
#ifdef ALLOW_DIAGNOSTICS
C     Diagnostics
      IF ( useDiagnostics ) THEN
        CALL DIAGNOSTICS_FILL(Ue,      'EKEANNI1',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(Lrhines, 'EKEANNI3',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(delta,   'EKEANNI2',0,1,2,bi,bj,myThid) 
        CALL DIAGNOSTICS_FILL(E_ML,    'EKEANNI4',0,Nr,2,bi,bj,myThid)
      ENDIF
#endif
       

  
  
C     ----------    Kgm part ends    ------------


  
  
C     ----------   Kredi part begins  ------------
   
      IF ( .NOT. REDI_useML ) THEN      
        DO k=1,Nr
         DO j=1-Oly,sNy+Oly
          DO i=1-Olx,sNx+Olx
               REDI_ML_k(i,j,k,bi,bj) = 0.0 _d 0
          ENDDO
         ENDDO
        ENDDO        
      ENDIF 
      

      
      
#endif /* ALLOW_GM_ML */

      RETURN
      END
