#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

#ifdef ALLOW_GMREDI
# include "GMREDI_OPTIONS.h"
#endif

      SUBROUTINE CALC_EDDY_STRESS( bi, bj, myThid )

C     !DESCRIPTION: \bv
C     *==========================================================*
C     | S/R CALC_EDDY_STRESS
C     | o Calculates the eddy stress when running a residual
C     |   ocean model
C     *==========================================================*
C     | Calculates the eddy stress.  Later this will be added to
C     | gU the same as external sources (e.g. wind stress, bottom
C     | friction, etc.
C     *==========================================================*
C     | Added by WHY
C     | Calculates Du'v'/Dy using u'v' predicted from ANN
C     *==========================================================*
C     \ev

      IMPLICIT NONE

#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "FFIELDS.h"
#ifdef ALLOW_GMREDI
# include "GMREDI.h"
#endif

C     !INPUT/OUTPUT PARAMETERS:
C     == Routine arguments ==
      INTEGER bi,bj
      INTEGER myThid


C     !LOCAL VARIABLES:
C     == Local variables ==
C     Loop counters
      INTEGER i,j,k,kp1,km1,klow
         
      _RL UpVp_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL DupvpDy_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL DupvpDy_ML_temp2(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Depth_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL temp2

#ifdef ALLOW_EDDYPSI
C     Interpolated stream function and coriolis
      _RL psix, psiy, coriU, coriV

C     Calculate the eddy stress from the eddy induced streamfunction
#ifdef ALLOW_GMREDI
      IF ( GM_InMomAsStress ) THEN
#endif
        DO k=1,Nr

         DO j=1-OLy,sNy+OLy-1
          DO i=1-OLx+1,sNx+OLx
#ifdef ALLOW_GMREDI
           psiy = op25*(GM_PsiY(i,  j  ,k,bi,bj)
     &                 +GM_PsiY(i,  j+1,k,bi,bj)
     &                 +GM_PsiY(i-1,j  ,k,bi,bj)
     &                 +GM_PsiY(i-1,j+1,k,bi,bj))
#else
           psiy = op25*(eddyPsiY(i,  j  ,k,bi,bj)
     &                 +eddyPsiY(i,  j+1,k,bi,bj)
     &                 +eddyPsiY(i-1,j  ,k,bi,bj)
     &                 +eddyPsiY(i-1,j+1,k,bi,bj))
#endif
           coriU = op5*(fcori(i-1,j,bi,bj)
     &                 +fCori(i  ,j,bi,bj))
           tauxEddy(i,j,k,bi,bj) =  rhoConst*coriU*psiy
          ENDDO
         ENDDO

         DO j=1-OLy+1,sNy+OLy
          DO i=1-OLx,sNx+OLx-1
#ifdef ALLOW_GMREDI
           psix = op25*(GM_PsiX(i,  j  ,k,bi,bj)
     &                 +GM_PsiX(i+1,j  ,k,bi,bj)
     &                 +GM_PsiX(i  ,j-1,k,bi,bj)
     &                 +GM_PsiX(i+1,j-1,k,bi,bj))
#else
           psix = op25*(eddyPsiX(i,  j  ,k,bi,bj)
     &                 +eddyPsiX(i+1,j  ,k,bi,bj)
     &                 +eddyPsiX(i  ,j-1,k,bi,bj)
     &                 +eddyPsiX(i+1,j-1,k,bi,bj))
#endif
           coriV = op5*(fcori(i,j-1,bi,bj)
     &                 +fCori(i,j  ,bi,bj))
           tauyEddy(i,j,k,bi,bj) = -rhoConst*coriV*psix
          ENDDO
         ENDDO

C-    end k loop
        ENDDO

#ifdef ALLOW_DIAGNOSTICS
        IF ( useDiagnostics ) THEN
          CALL DIAGNOSTICS_FILL( tauxEddy, 'TAUXEDDY',
     &                           0, Nr, 1, bi, bj, myThid )
          CALL DIAGNOSTICS_FILL( tauyEddy, 'TAUYEDDY',
     &                           0, Nr, 1, bi, bj, myThid )
        ENDIF
#endif
#ifdef ALLOW_GMREDI
      ENDIF
#endif
#endif /* ALLOW_EDDYPSI */




C  Starts to calculate Du'v'/Dy

#ifdef ALLOW_GMREDI
      IF ( UpVp_useML ) THEN


      
C     Initialize        
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx
        DupvpDy_ML_temp2(i,j)  = 0. _d 0
        Depth_temp(i,j)  = 0. _d 0
       ENDDO
      ENDDO      
     
      
      
      
C    Make a local copy of u'v'
      DO k=1,Nr
      DO j=1-OLy,sNy+OLy
       DO i=1-OLx,sNx+OLx    
          UpVp_ML_temp(i,j,k) = UpVp_ML(i,j,k,bi,bj)
        ENDDO
      ENDDO   
      ENDDO

      
C     Calculate meridional gradients of u'v' at U points       
      DO k=1,Nr
      DO j=1-OLy+1,sNy+OLy-1
       DO i=1-OLx+1,sNx+OLx    
          temp(i,j,k)=0.25*(MaskS(i,j,k,bi,bj)
     &        *_recip_dyC(i,j,bi,bj)
     &        *(UpVp_ML_temp(i,j,k)-UpVp_ML_temp(i,j-1,k))
     &         +MaskS(i,j+1,k,bi,bj)
     &        *_recip_dyC(i,j+1,bi,bj)
     &        *(UpVp_ML_temp(i,j+1,k)-UpVp_ML_temp(i,j,k)))
     &        +0.25*(MaskS(i-1,j,k,bi,bj)
     &        *_recip_dyC(i-1,j,bi,bj)
     &        *(UpVp_ML_temp(i-1,j,k)-UpVp_ML_temp(i-1,j-1,k))
     &         +MaskS(i-1,j+1,k,bi,bj)
     &        *_recip_dyC(i-1,j+1,bi,bj)
     &        *(UpVp_ML_temp(i-1,j+1,k)-UpVp_ML_temp(i-1,j,k)))   
C            Enforce lower and upper bounds 
C             temp(i,j,k) = 
C     &                     MIN(temp(i,j,k),DUpVpDy_ML_maxVal)
C             temp(i,j,k) = 
C     &                     MAX(temp(i,j,k),-DUpVpDy_ML_maxVal)        
        ENDDO
      ENDDO   
      ENDDO
  
C     Modifiy near-bottom values              
      DO k=1,Nr
      DO j=1-OLy+1,sNy+OLy-1
      DO i=1-OLx+1,sNx+OLx     
      
         IF (k .GT. kLowC(i,j,bi,bj)-2 .AND. 
     &       k .LE. kLowC(i,j,bi,bj)) THEN 
C                klow = kLowC(i,j,bi,bj)-3
C                temp(i,j,k) = temp(i,j,klow)               
                temp(i,j,k) = 0. _d 0  
         ENDIF
              
      ENDDO
      ENDDO   
      ENDDO             
              
     

C     Du'v'/Dy = 0 at surface as u'v' has been enforced to 0     


C     Use values in k level to replace extreme values of Du'v'/Dy in k-1 level

      DO k=Nr,1,-1
      DO j=1-OLy+1,sNy+OLy-1
      DO i=1-OLx+1,sNx+OLx  
          IF (k .LE. kLowC(i,j,bi,bj)-1) THEN
             kp1 = min(k+1,Nr)
             IF(abs(temp(i,j,k)) .GT. DUpVpDy_ML_maxVal) THEN
                temp(i,j,k) = temp(i,j,kp1)   
             ENDIF
          ENDIF
      ENDDO
      ENDDO   
      ENDDO 

      
C     Use values in k-1 level to replace extreme values of Du'v'/Dy in k level

C      DO k=2,Nr
C      DO j=1-OLy+1,sNy+OLy-1
C      DO i=1-OLx+1,sNx+OLx  
C          IF (k .LE. kLowC(i,j,bi,bj)) THEN
C             km1 = max(k-1,1)
C             IF(abs(temp(i,j,k)) .GT. DUpVpDy_ML_maxVal) THEN
C                temp(i,j,k) = temp(i,j,km1)   
C             ENDIF
C          ENDIF
C      ENDDO
C      ENDDO   
C      ENDDO      
      
      
      
      
      
      
C      Use value in the left to replace extreme values in Du'v'/Dy
C      This could lead to problem because we only have information of 
C      a local tile rather than the whole domaine
C
C      DO k=1,Nr
C      DO j=1-OLy+2,sNy+OLy-1
C      DO i=1-OLx+1,sNx+OLx         
C             IF(abs(temp(i,j,k)) .GT. DUpVpDy_ML_maxVal) THEN
C                temp(i,j,k) = temp(i,j-1,k)   
C            ENDIF
C      ENDDO
C      ENDDO   
C      ENDDO  


C Use the mean value of surrounding points to replace extreme values in Du'v'/Dy
C      DO k=1,Nr
C       DO j=1-Oly+2,sNy+Oly-2
C        DO i=1-Olx+1,sNx+Olx
C        IF (k.LE.kLowC(i,j,bi,bj)) THEN
C              kp1 = min(k+1,Nr)
C              km1 = max(k-1,1)
C              temp2 = ( temp(i,j-1,k)*maskC(i,j-1,k,bi,bj)
C     &                + temp(i,j+1,k)*maskC(i,j+1,k,bi,bj)
C     &                + temp(i,j,kp1)*maskC(i,j,kp1,bi,bj)              
C     &                + temp(i,j,km1)*maskC(i,j,km1,bi,bj))
C     &               /(  maskC(i,j-1,k,bi,bj)
C     &                 + maskC(i,j+1,k,bi,bj)
C     &                 + maskC(i,j,kp1,bi,bj)
C     &                 + maskC(i,j,km1,bi,bj))      
C     &                 * maskC(i,j,k,bi,bj)
C              IF (abs(temp(i,j,k)) .GT. 2*abs(temp2)
C     &         .AND. maskC(i,j,k,bi,bj) .NE. 0. _d 0) THEN
C                     temp(i,j,k)=temp2
C              ENDIF
C        ENDIF   
C        ENDDO
C       ENDDO
C      ENDDO
      
      
              
C     Options to use depth-dependent or depth-mean version of predicted
C     Reynolds stress. The depth-dependet version is problematic if 
C     Uz is included in the ANN inputs for u'v'        
              
      IF ( UpVp_VertStruc ) THEN
C     Depth-dependent version   
          
C      DO k=1,Nr
C       DO j=1-Oly+2,sNy+Oly-2
C        DO i=1-Olx+1,sNx+Olx       
C      DupvpDy_ML_temp(i,j,k) = temp(i,j,k)
C         ENDDO
C       ENDDO
C      ENDDO
                
      
C     Smooth DupvpDy vertically       
       DO k=1,Nr
       kp1 = min(k+1,Nr)
       km1 = max(k-1,1)
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx+1,sNx+Olx     
       DupvpDy_ML_temp(i,j,k) = ( temp(i,j,k)*1. _d 0*drF(k)
     &            + temp(i,j,kp1)*maskC(i,j,kp1,bi,bj)*drF(kp1)
     &            + temp(i,j,km1)*maskC(i,j,km1,bi,bj)*drF(km1))
     &               /( 1. _d 0*drF(k)
     &                 + maskC(i,j,kp1,bi,bj)*drF(kp1)
     &                 + maskC(i,j,km1,bi,bj)*drF(km1))      
     &                 * maskC(i,j,k,bi,bj)
        ENDDO
       ENDDO
      ENDDO   
      

      ELSE
C     Depth-mean version             
      DO k=3,Nr      
       DO j=1-Oly+1,sNy+Oly-1
        DO i=1-Olx+1,sNx+Olx
        IF (k.LE. kLowC(i,j,bi,bj)-2) THEN            
            DupvpDy_ML_temp2(i,j) = DupvpDy_ML_temp2(i,j) + 
     &               drF(k)*hfacC(i,j,k,bi,bj)*temp(i,j,k) 
            Depth_temp(i,j) = Depth_temp(i,j)+drF(k)*hfacC(i,j,k,bi,bj)
        ENDIF
        ENDDO 
       ENDDO
      ENDDO
      
      DO k=1,Nr 
      DO j=1-Oly+1,sNy+Oly-1
       DO i=1-Olx+1,sNx+Olx
        IF (k.LE. kLowC(i,j,bi,bj)) THEN
      DupvpDy_ML_temp(i,j,k) = DupvpDy_ML_temp2(i,j)/Depth_temp(i,j)
        ENDIF
       ENDDO
       ENDDO    
      ENDDO  
          
      ENDIF   

      
      IF ( SmoothDUpVpDy ) THEN 
      DO k=1,Nr
      DO j=1-OLy+2,sNy+OLy-2
C this means OLy should be larger or equal to 3          
       DO i=1-OLx+1,sNx+OLx        
          DupvpDy_ML(i,j,k,bi,bj) = ( DupvpDy_ML_temp(i,j,k)*1. _d 0
     &               + DupvpDy_ML_temp(i,j-1,k)*maskC(i,j-1,k,bi,bj)
     &               + DupvpDy_ML_temp(i,j+1,k)*maskC(i,j+1,k,bi,bj))
     &               /( 1. _d 0
     &                 + maskC(i,j-1,k,bi,bj)
     &                 + maskC(i,j+1,k,bi,bj))      
     &                 * maskC(i,j,k,bi,bj)
C     &                             *(4000-abs(rC(k)))/2000
          ENDDO
       ENDDO 
      ENDDO  
      ELSE
          
      DO k=1,Nr
      DO j=1-OLy+2,sNy+OLy-2
       DO i=1-OLx+1,sNx+OLx    
          DupvpDy_ML(i,j,k,bi,bj) = DupvpDy_ML_temp(i,j,k)
C     &                             *(4000-abs(rC(k)))/2000
        ENDDO
      ENDDO   
      ENDDO
      ENDIF   
      
      
      
      
      
#ifdef ALLOW_DIAGNOSTICS
C     Diagnostics
      IF ( useDiagnostics ) THEN
      CALL DIAGNOSTICS_FILL(UpVp_ML_temp,'UpVp_ML ',0,Nr,2,bi,bj,myThid)
      CALL DIAGNOSTICS_FILL(temp,'DupvpDy2',0,Nr,2,bi,bj,myThid)
C      CALL DIAGNOSTICS_FILL(DupvpDy_ML,'DupvpDy ',0,Nr,1,bi,bj,myThid)
      ENDIF
#endif


      ENDIF
#endif

      RETURN
      END
