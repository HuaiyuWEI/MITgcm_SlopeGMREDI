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
C     | Parameterizes Kgm and u'v' for non-eddy runs.
C     *==========================================================*
C     | Kgm can be predicted directly from Neural Network,
C     | or be fomulated by physical scalings using 
C     | EKE (or E) predicted from Neural Network.
C     *==========================================================*
C     | u'v' is predicted directly from Neural Network.
C     *==========================================================*
C     | Kgm and u'v' are located at the cell centers.
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
C     uFld_C   :: 3D zonal velocity at cell centers
C     E_ML     :: Energy predicted by ANN
C     BVint    :: The vertical integral of Buoyancy frequency
C     Siso     :: Bulk isopycnal slope
C     delta    :: Slope parameter
C     Uzavg    :: Depth-mean zonal velocity at cell centers
C     H        :: Water depth (positive) at cell centers
C     SBurger  :: Slope Burger Number  N/f*H_y
C     Beta_t   :: Topographic beta     f/H*H_y
      _RL N2(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL N(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL uFld_C(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL DuDy(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL DuDz(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL Uy_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr) 
      _RL Uz_temp1(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr) 
      _RL Uz_temp2(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr) 
      _RL E_ML(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL UpVp_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL GM_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL BVint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL BVzavg(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL dSigmaDxZint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL dSigmaDyZint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL dSigmaDrZint(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Siso(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL delta(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Ri(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Ld(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL Lrhines(1-OLx:sNx+OLx,1-OLy:sNy+OLy)      
      _RL Lmix(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL Uzavg(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL H(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL dHdy(1-Olx:sNx+Olx,1-Oly:sNy+Oly)  
      _RL SBurger(1-Olx:sNx+Olx,1-Oly:sNy+Oly) 
      _RL Beta_t(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL temp_2D(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL E_ML_zavg(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Ue(1-Olx:sNx+Olx,1-Oly:sNy+Oly) 
      
      _RL Input(4,1)
      _RL ML0(64,1)
      _RL ML1(32,1)
      _RL ML2(1,1)

      _RL ML0_UpVp(64,1)
      _RL ML1_UpVp(32,1)
      _RL ML2_UpVp(1,1)    

      _RL ML0_E(64,1)
      _RL ML1_E(32,1)
      _RL ML2_E(1,1)   
C InputFlag      
      _RL Input_E(6,1)  
      _RL Input_UpVp(6,1)   
      
      _RL temp_E 
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
        Uzavg(i,j)  = 0.0 _d 0
        E_ML_zavg(i,j)  = 0.0 _d 0
        dHdy(i,j)  = 0.0 _d 0
        dSigmaDxZint(i,j)  = 0.0 _d 0
        dSigmaDyZint(i,j)  = 0.0 _d 0
        dSigmaDrZint(i,j)  = 0.0 _d 0
       ENDDO
      ENDDO
      
      

C     ---------- calculation begin ------------
C     ---- calculate some basic quantities ----

C    Zonal velocity at cell center
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx-1
            uFld_C(i,j,k) = op5 *
     &               (uVel(i+1,j,k,bi,bj)+uVel(i,j,k,bi,bj))
        ENDDO
       ENDDO
      ENDDO

      
      
C    Water depth at cell center
C    Find the water depth at cell center (positive)
C    R_low_orig is defined by ignoring hFacMin constrain,
C    which can give us a much smoother dHdy
C    Notice that this H cannot be used in depth-avg calculations.
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx
          H(i,j) = -R_low_orig(i,j,bi,bj)
       ENDDO
      ENDDO

      
      
C    Bathymetric slope at cell center
C    H used here is positive  
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
          ENDIF
       ENDDO
      ENDDO      

      
      
C     Square of the buoyancy frequency at the top of a grid cell
C     Enforce a minimum N2
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
      
C       Calculate the Rossby deformation Radius
        DO j=1-Oly,sNy+Oly
         DO i=1-Olx,sNx+Olx
          IF ( fCori(i,j,bi,bj).NE. 0.0 _d 0 ) THEN
          Ld(i,j)  = BVint(i,j)/fCori(i,j,bi,bj)/pi
          ENDIF
         ENDDO
      ENDDO
      
C       Calculate depth-mean buoyancy frequency
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx-1
        IF (r_Low(i,j,bi,bj).NE. 0.0 _d 0) THEN
C         The minus sign is because r_Low<0
          BVzavg(i,j) = -BVint(i,j)/r_Low(i,j,bi,bj)
        ELSE
          BVzavg(i,j) = 0.0 _d 0
        ENDIF
       ENDDO
       ENDDO      

      
C     ----------   Reynolds stress part begins    ------------
      IF ( UpVp_useML .OR.  GM_MLEner) THEN
          
C     Input01: 3D Meridional gradient of zonal velocity
C     DuDy is located at cell center          
      DO k=1,Nr
       DO j=1-OLy+1,sNy+OLy-1
        DO i=1-OLx,sNx+OLx-1
        Uy_temp(i,j,k)=op5*(MaskS(i,j,k,bi,bj)
     &        *_recip_dyC(i,j,bi,bj)
     &        *(uFld_C(i,j,k)-uFld_C(i,j-1,k))
     &         +MaskS(i,j+1,k,bi,bj)
     &        *_recip_dyC(i,j+1,bi,bj)
     &        *(uFld_C(i,j+1,k)-uFld_C(i,j,k)))
        ENDDO
       ENDDO
      ENDDO
      
C      IF ( SmoothUyUz ) THEN     
C      DO k=1,Nr
C       DO j=1-Oly+2,sNy+Oly-2
C        DO i=1-Olx,sNx+Olx-1     
C        DuDy(i,j,k) = ( Uy_temp(i,j,k)*1. _d 0
C     &                + Uy_temp(i,j-1,k)*maskC(i,j-1,k,bi,bj)
C     &                + Uy_temp(i,j+1,k)*maskC(i,j+1,k,bi,bj))
C     &               /( 1. _d 0
C     &                 + maskC(i,j-1,k,bi,bj)
C     &                 + maskC(i,j+1,k,bi,bj))      
C     &                 * maskC(i,j,k,bi,bj)
C        ENDDO
C       ENDDO
C      ENDDO 
      
C      ElSE

      DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1  
         DuDy(i,j,k) = Uy_temp(i,j,k)
        ENDDO
       ENDDO
      ENDDO  
      
C      ENDIF      

C     Input02: 3D Vertical gradient of zonal velocity
C     DuDz is located at cell center
C     CHECK if recip_hFacC is needed      
      DO k=1,Nr
       kp1 = min(k+1,Nr)
       km1 = max(k-1,1)
       DO j=1-OLy,sNy+OLy
        DO i=1-OLx,sNx+OLx-1      
         Uz_temp1(i,j,k) =  op5*(
     &         recip_drC(k)*maskC(i,j,k,bi,bj)*
C     &         recip_hFacC(i,j,k,bi,bj)*
     &            (uFld_C(i,j,km1)-uFld_C(i,j,k))
     &        +recip_drC(kp1)*maskC(i,j,kp1,bi,bj)*
C     &         recip_hFacC(i,j,kp1,bi,bj)*
     &            (uFld_C(i,j,k)-uFld_C(i,j,kp1)))
        ENDDO
       ENDDO
      ENDDO
           
C      IF ( SmoothUyUz ) THEN  
CC Smooth Uz horizontally         
C      DO k=1,Nr
C       DO j=1-Oly+1,sNy+Oly-1
C        DO i=1-Olx,sNx+Olx-1     
C        Uz_temp2(i,j,k) = ( Uz_temp1(i,j,k)*1. _d 0
C     &                + Uz_temp1(i,j-1,k)*maskC(i,j-1,k,bi,bj)
C     &                + Uz_temp1(i,j+1,k)*maskC(i,j+1,k,bi,bj))
C     &               /( 1. _d 0
C     &                 + maskC(i,j-1,k,bi,bj)
C     &                 + maskC(i,j+1,k,bi,bj))      
C     &                 * maskC(i,j,k,bi,bj)
C        ENDDO
C       ENDDO
C      ENDDO  
CC Smooth Uz vertically       
C       DO k=1,Nr
C       kp1 = min(k+1,Nr)
C       km1 = max(k-1,1)
C       DO j=1-Oly+1,sNy+Oly-1
C        DO i=1-Olx,sNx+Olx-1     
C        DuDz(i,j,k) = ( Uz_temp2(i,j,k)*1. _d 0*drF(k)
C     &                + Uz_temp2(i,j,kp1)*maskC(i,j,kp1,bi,bj)*drF(kp1)
C     &                + Uz_temp2(i,j,km1)*maskC(i,j,km1,bi,bj)*drF(km1))
C     &               /( 1. _d 0*drF(k)
C     &                 + maskC(i,j,kp1,bi,bj)*drF(kp1)
C     &                 + maskC(i,j,km1,bi,bj)*drF(km1))      
C     &                 * maskC(i,j,k,bi,bj)
C        ENDDO
C       ENDDO
C      ENDDO           
      
C      ElSE
          
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx-1  
         DuDz(i,j,k) = Uz_temp1(i,j,k)
        ENDDO
       ENDDO
      ENDDO 
      
C      ENDIF   
      

     
      
C     Input03: Bulk isopycnal slope at cell centers
      
C     Integrate Tx and Ty vertically   
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
      
C     Integrate Tz vertically
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
      
C     Compute bulk isopycnal slope
        DO j=1-Oly,sNy+Oly-1
         DO i=1-Olx,sNx+Olx-1
          IF ( dSigmaDrZint(i,j).NE. 0.0 _d 0 ) THEN  
C  Assume dSigmaDrZint is negative
          temp_2D(i,j)=  dSigmaDyZint(i,j)/abs(dSigmaDyZint(i,j))
     &                *sqrt(dSigmaDxZint(i,j)**2+dSigmaDyZint(i,j)**2)
     &                 /abs(dSigmaDrZint(i,j))
          ENDIF
         ENDDO
        ENDDO    
        
C      IF ( SmoothSiso ) THEN      
C      DO j=1-Oly+1,sNy+Oly-2
C       DO i=1-Olx,sNx+Olx-1      
C          Siso(i,j) = ( temp_2D(i,j)*1. _d 0
C     &                + temp_2D(i,j-1)*maskC(i,j-1,1,bi,bj)
C     &                + temp_2D(i,j+1)*maskC(i,j+1,1,bi,bj))
C     &               /( 1. _d 0
C     &                 + maskC(i,j-1,1,bi,bj)
C     &                 + maskC(i,j+1,1,bi,bj))      
C     &                 * maskC(i,j,1,bi,bj)
C       ENDDO
C      ENDDO     
          
C      ELSE
       DO j=1-Oly,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1   
            Siso(i,j) = temp_2D(i,j)
        ENDDO
       ENDDO     
C      ENDIF
            
#ifdef ALLOW_DIAGNOSTICS
C     Diagnostics
      IF ( useDiagnostics ) THEN
        CALL DIAGNOSTICS_FILL(DuDy,   'ReANN_I1',0,Nr,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(DuDz,   'ReANN_I2',0,Nr,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(Siso,   'ReANN_I3',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(beta_t, 'ReANN_I4',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(Ld,     'ReANN_I6',0,1,2,bi,bj,myThid)
      ENDIF
#endif

      ENDIF

        
       IF ( UpVp_useML) THEN
C    Predict Reynolds stress over continental slope
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
             UpVp_ML(i,j,k,bi,bj) = 
     &                      MIN(UpVp_ML(i,j,k,bi,bj),UpVp_ML_maxVal)
             UpVp_ML(i,j,k,bi,bj) = 
     &                      MAX(UpVp_ML(i,j,k,bi,bj),-UpVp_ML_maxVal)
           ENDIF
        ENDDO
       ENDDO
       ENDDO
       
       
      IF ( SmoothUpVp ) THEN 
      DO k=3,Nr
      DO j=1-OLy+3,sNy+OLy-3
       DO i=1-OLx,sNx+OLx-1        
C     Smooth UpVp horizontally
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
C  Change surface u'v' to zero
      DO k=1,2
      DO j=1-OLy+3,sNy+OLy-3
       DO i=1-OLx,sNx+OLx-1    
          UpVp_ML(i,j,k,bi,bj) = 0.0 _d 0
        ENDDO
      ENDDO    
      ENDDO
      
      
      ELSE
          
      DO k=3,Nr
      DO j=1-OLy+1,sNy+OLy-1
       DO i=1-OLx,sNx+OLx-1    
          UpVp_ML(i,j,k,bi,bj) = UpVp_ML_temp(i,j,k)
        ENDDO
      ENDDO   
      ENDDO
C  Change surface u'v' to zero
      DO k=1,2
      DO j=1-OLy+1,sNy+OLy-1
       DO i=1-OLx,sNx+OLx-1    
          UpVp_ML(i,j,k,bi,bj) = 0.0 _d 0
        ENDDO
      ENDDO     
      ENDDO
      
      ENDIF       
      
      
 

 
      
      
       
       

C   ENDIF for IF ( UpVp_useML )
      ENDIF
      
      
      
      
C     ----------      Kgm part     ------------
   
      IF ( .NOT. GM_useML ) THEN
      
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx
             GM_ML_k(i,j,k,bi,bj) = 0.0 _d 0
        ENDDO
       ENDDO
      ENDDO  

      ELSE
     
          
C    slope Burger number
C    calculated here because it is used in 
C    both Energy-kgm scheme and Kgm-ANN scheme     

      DO j=1-Oly+1,sNy+Oly-1
       DO i=1-Olx,sNx+Olx
        IF (fCori(i,j,bi,bj).NE. 0.0 _d 0) THEN
        SBurger(i,j) = BVzavg(i,j)/fCori(i,j,bi,bj)*abs(dHdy(i,j))
        ENDIF
       ENDDO
      ENDDO
      
      
      IF ( GM_MLEner ) THEN
C    Energy-kgm scheme begin  
C    Predict EKE or E over continental slope
C    which is later used to infer Kgm via traditional scalings      
      DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1
         IF (k.LE.kLow_C(i,j)) THEN
C InputFlag              
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
C InputFlag   
             E_ML(i,j,k) =    op5 *ML2_E(1,1)/abs(ML2_E(1,1))
     &                            *abs(ML2_E(1,1)*E_ML_normal(1,7))**2 
           ENDIF
        ENDDO
       ENDDO
      ENDDO

C  Deal with extrem values in E_ML   
C      DO k=1,Nr
C       DO j=1-Oly+3,sNy+Oly-3
C        DO i=1-Olx,sNx+Olx-1
C        IF (k.LE.kLow_C(i,j)) THEN
C              kp1 = min(k+1,Nr)
C              km1 = max(k-1,1)
C              temp_E = (E_ML(i,j-1,k)*maskC(i,j-1,k,bi,bj)
C     &                + E_ML(i,j+1,k)*maskC(i,j+1,k,bi,bj)
C     &                + E_ML(i,j,kp1)*maskC(i,j,kp1,bi,bj)              
C     &                + E_ML(i,j,km1)*maskC(i,j,km1,bi,bj))
C     &               /(  maskC(i,j-1,k,bi,bj)
C     &                 + maskC(i,j+1,k,bi,bj)
C     &                 + maskC(i,j,kp1,bi,bj)
C     &                 + maskC(i,j,km1,bi,bj))      
C     &                 * maskC(i,j,k,bi,bj)
C              IF (abs(E_ML(i,j,k)) .GT. 2*abs(temp_E)
C    &         .AND. maskC(i,j,k,bi,bj) .NE. 0. _d 0) THEN
C                     E_ML(i,j,k)=temp_E
C              ENDIF
C        ENDIF   
C        ENDDO
C       ENDDO
C      ENDDO
      
      DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1
            E_ML_zavg(i,j) = E_ML_zavg(i,j) + 
     &               drF(k)*hfacC(i,j,k,bi,bj)*E_ML(i,j,k)      
        ENDDO
       ENDDO
      ENDDO
      
      DO j=1-Oly+1,sNy+Oly-1
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
       
       
      IF(Prograde) THEN
C Scalings for Kgm in prograde flows              
          

C     Compute bulk Richardson number
        DO j=1-Oly,sNy+Oly-1
         DO i=1-Olx,sNx+Olx-1
          IF ( dSigmaDrZint(i,j).NE. 0.0 _d 0 ) THEN  
          temp_2D(i,j)= -fCori(i,j,bi,bj)**2*r_Low(i,j,bi,bj)
     &              *abs(dSigmaDrZint(i,j)
     &              /(dSigmaDxZint(i,j)**2+dSigmaDyZint(i,j)**2
     &               *gravity*recip_rhoConst)) 
          ENDIF
         ENDDO
        ENDDO    
        
C      IF ( SmoothSiso ) THEN      
C      DO j=1-Oly+1,sNy+Oly-2
C       DO i=1-Olx,sNx+Olx-1      
C          Ri(i,j) = ( temp_2D(i,j)*1. _d 0
C     &                + temp_2D(i,j-1)*maskC(i,j-1,1,bi,bj)
C     &                + temp_2D(i,j+1)*maskC(i,j+1,1,bi,bj))
C     &               /( 1. _d 0
C     &                 + maskC(i,j-1,1,bi,bj)
C     &                 + maskC(i,j+1,1,bi,bj))      
C     &                 * maskC(i,j,1,bi,bj)
C       ENDDO
C      ENDDO     
          
C      ELSE
       DO j=1-Oly,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1   
            Ri(i,j) = temp_2D(i,j)
        ENDDO
       ENDDO     
C      ENDIF
      
      
C     Compute Kgm
      
       DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1   
         GM_ML_temp(i,j,k) 
     &        = 0.05/(8*SBurger(i,j)**2+1)
     &             *sqrt(Ri(i,j))/fCori(i,j,bi,bj)*E_ML_zavg(i,j);
C            Enforce lower and upper bounds on the diffusivity
             GM_ML_temp(i,j,k) 
     &                = MIN(GM_ML_temp(i,j,k),GM_ML_maxVal_K)
             GM_ML_temp(i,j,k) 
     &                = MAX(GM_ML_temp(i,j,k),GM_ML_minVal_K)        
           ENDDO
       ENDDO
       ENDDO       
           
          
      ELSE
C Scalings for Kgm in retrograde flows     
          
      DO j=1-Oly+1,sNy+Oly-1
       DO i=1-Olx,sNx+Olx-1
        IF (Beta_t(i,j).NE. 0.0 _d 0) THEN
           Lrhines(i,j) = sqrt(Ue(i,j)/Beta_t(i,j))
        ELSE
           Lrhines(i,j) = 0.0 _d 0
        ENDIF
       ENDDO
       ENDDO     
                   
       IF(GM_Burger) THEN  
C  Use empirical function of Burger number to constrain Kgm       
      DO j=1-Oly+1,sNy+Oly-1
       DO i=1-Olx,sNx+Olx-1
        IF (H(i,j) .LT. 2.25 _d 3) THEN
          Lmix(i,j) = Lrhines(i,j)/pi
        ElSE
          Lmix(i,j) = min(Lrhines(i,j)/pi,Ld(i,j))
        ENDIF
       ENDDO
       ENDDO
              
       
       DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1   
         GM_ML_temp(i,j,k) 
     &        = 0.24*(SBurger(i,j) + 1/(209.0*(SBurger(i,j)+0.0025)) )
     &             *Ue(i,j)*Lmix(i,j)
C            Enforce lower and upper bounds on the diffusivity
             GM_ML_temp(i,j,k) 
     &                = MIN(GM_ML_temp(i,j,k),GM_ML_maxVal_K)
             GM_ML_temp(i,j,k) 
     &                = MAX(GM_ML_temp(i,j,k),GM_ML_minVal_K)        
           ENDDO
       ENDDO
       ENDDO   
       
      ELSE
C  Use empirical function of slope parameter to constrain Kgm 

C     Compute slope parameter
        DO j=1-Oly,sNy+Oly-1
         DO i=1-Olx,sNx+Olx-1
          IF ( Siso(i,j) .NE. 0.0 _d 0 ) THEN  
          delta(i,j)= dHdy(i,j)/Siso(i,j)
          ENDIF
         ENDDO
        ENDDO    
        
C    predict kgm
      DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1   
            
C         GM_ML_temp(i,j,k) 
C     &        =     2.5 _d 0 *( abs(delta(i,j)) +
C     &              1/(0.05*(abs(delta(i,j))+0.05)) )
         
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

       ENDIF
     
C ENDIF for prograde or retrograde          
      ENDIF
       
#ifdef ALLOW_DIAGNOSTICS
C     Diagnostics
      IF ( useDiagnostics ) THEN
          
      IF(Prograde) THEN          
        CALL DIAGNOSTICS_FILL(E_ML_zavg,'EKEANNI1',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(Ri,    'EKEANNI3',0,1,2,bi,bj,myThid)
      ELSE  
        CALL DIAGNOSTICS_FILL(Ue,'EKEANNI1',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(Lmix,    'EKEANNI3',0,1,2,bi,bj,myThid)
      ENDIF  
      
      IF(GM_Burger) THEN
        CALL DIAGNOSTICS_FILL(SBurger, 'EKEANNI2',0,1,2,bi,bj,myThid)
      ELSE
        CALL DIAGNOSTICS_FILL(delta, 'EKEANNI2',0,1,2,bi,bj,myThid)  
      ENDIF
      
        CALL DIAGNOSTICS_FILL(E_ML,  'EKEANNI4',0,Nr,2,bi,bj,myThid)
      ENDIF
#endif
       
       
      ELSE
          
C     Input: depth-mean along-shore velocity
C     Calculate the barotropic velocity by vertically integrating
C     and then dividing by the depth of the water column
C     Note that Uzavg is at the Cell certer.
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx,sNx+Olx-1
            Uzavg(i,j) = Uzavg(i,j) + 
     &               drF(k)*hfacC(i,j,k,bi,bj)*uFld_C(i,j,k)      
        ENDDO
       ENDDO
      ENDDO
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx-1
        IF (r_Low(i,j,bi,bj).NE. 0.0 _d 0) THEN
C         The minus sign is because r_Low<0
          Uzavg(i,j) = -Uzavg(i,j)/r_Low(i,j,bi,bj)
          ELSE
          Uzavg(i,j) = 0.0 _d 0
        ENDIF
       ENDDO
       ENDDO


       
            
C    Calulate diffusivity
      DO k=1,Nr
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx,sNx+Olx-1
         IF (k.LE.kLow_C(i,j)) THEN
             Input(1,1) = abs(Uzavg(i,j))/0.1492
             Input(2,1) = SBurger(i,j)
             Input(3,1) = (4000-abs(H(i,j)))/4000
             Input(4,1) = Ld(i,j)/10415.94
             ML0 = matmul(GM_ML_w0,Input) + GM_ML_b0
             ML0 = ((1.+0.2)*ML0+(1.0-0.2)*abs(ML0))/2.0;
             ML1 = matmul(GM_ML_w1,ML0) + GM_ML_b1
             ML1 = ((1.+0.2)*ML1+(1.0-0.2)*abs(ML1))/2.0;
             ML2(1:1,1:1) = matmul(GM_ML_w2(1:1,1:32),ML1(1:32,1:1)) 
     &                           + GM_ML_b2(1,1)
             GM_ML_temp(i,j,k) = ML2(1,1)/abs(ML2(1,1))
     &                           *ML2(1,1)*ML2(1,1)*233.85
C            Enforce lower and upper bounds on the diffusivity
             GM_ML_temp(i,j,k) 
     &                = MIN(GM_ML_temp(i,j,k),GM_ML_maxVal_K)
             GM_ML_temp(i,j,k) 
     &                = MAX(GM_ML_temp(i,j,k),GM_ML_minVal_K)
           ENDIF
        ENDDO
       ENDDO
       ENDDO     
       
       

#ifdef ALLOW_DIAGNOSTICS
C     Diagnostics
      IF ( useDiagnostics ) THEN
        CALL DIAGNOSTICS_FILL(Uzavg,   'GMANN_I1',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(SBurger, 'GMANN_I2',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(H,       'GMANN_I3',0,1,2,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(Ld,    'GMANN_I4',0,1,2,bi,bj,myThid)
      ENDIF
#endif


C ENDIF for  IF ( GM_MLEner )    
      ENDIF
       
       IF ( SmoothKgm ) THEN 
          DO k=1,Nr
           DO j=1-Oly+2,sNy+Oly-2
            DO i=1-Olx,sNx+Olx-1     
             GM_ML_k(i,j,k,bi,bj) = ( GM_ML_temp(i,j,k)*1. _d 0
     &                + GM_ML_temp(i,j-1,k)*maskC(i,j-1,k,bi,bj)
     &                + GM_ML_temp(i,j+1,k)*maskC(i,j+1,k,bi,bj))
     &               /( 1. _d 0
     &                 + maskC(i,j-1,k,bi,bj)
     &                 + maskC(i,j+1,k,bi,bj))      
     &                 * maskC(i,j,k,bi,bj)
            ENDDO
           ENDDO
          ENDDO 
                
       ELSE
           
          DO k=1,Nr
           DO j=1-Oly+1,sNy+Oly-1
            DO i=1-Olx,sNx+Olx-1    
             GM_ML_k(i,j,k,bi,bj) =   GM_ML_temp(i,j,k)
            ENDDO
           ENDDO
          ENDDO 
       ENDIF

       
C ENDIF for IF ( .NOT. GM_useML )
         ENDIF   
C     ----------    Kgm part ends    ------------


#endif /* ALLOW_GM_ML */

      RETURN
      END
