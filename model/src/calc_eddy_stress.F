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
#include "EESUPPORT.h"
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
      INTEGER :: stat(MPI_STATUS_SIZE),ierror
      INTEGER :: nproc,procNum  
      _RL UpVp_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL temp_new(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL temp_other(1:sNx,1:sNy,Nr)
      _RL temp_Allproc(1:sNx,1:sNy*nPy,Nr)
      _RL temp_Allproc_new(1:sNx,1:sNy*nPy,Nr)
      _RL DupvpDy_ML_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly,Nr)
      _RL DupvpDy_ML_temp2(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL Depth_temp(1-Olx:sNx+Olx,1-Oly:sNy+Oly)
      _RL temp2, tempDepth, tempStruc 

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




C  Starts to calculate Reynolds stress (e.g., EMF divergence) 
C  based on ANN-inferred EMF calculated by gmredi_calc_ML.F

#ifdef ALLOW_GMREDI
      IF ( UpVp_useML) THEN
          
C     Initialize        
      DO j=1-Oly,sNy+Oly
       DO i=1-Olx,sNx+Olx
        DupvpDy_ML_temp2(i,j)  = 0. _d 0
        Depth_temp(i,j)  = 0. _d 0
       ENDDO
      ENDDO      
     
     
C    Make a local copy of ANN-inferred EMF
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
        ENDDO
      ENDDO   
      ENDDO
  
C     Modifiy near-bottom values  
C     Values in deepest (2 cells) are set to zero
      DO k=1,Nr
      DO j=1-OLy+1,sNy+OLy-1
      DO i=1-OLx+1,sNx+OLx     
      
         IF (k .GT. kLowC(i,j,bi,bj)-2 .AND. 
     &       k .LE. kLowC(i,j,bi,bj)) THEN 
C     kLowC   :: index of the r-lowest "wet cell" (2D)          
                temp(i,j,k) = 0. _d 0  
         ENDIF
              
      ENDDO
      ENDDO   
      ENDDO             
              
 
   


      IF (LeftReplacement) THEN
C     Replace any extreme value of the EMF divergence with the
C     value on its nearest shoreward grid point.        
      
      
C   Send data of EMF divergence from all processors to processor zero       
      nproc=nPx*nPy
      call mpi_barrier(mpi_comm_world,ierror)
      DO procNum=1, nproc-1              
      IF (mpiMyId.eq.procNum.and.mpiMyId.ne.0.and.nproc.gt.1) THEN
      call mpi_send(temp(1:sNx,1:sNy,1:Nr),sNx*sNy*Nr,
     &              MPI_DOUBLE_PRECISION,0,
     &              procNum+1000,mpi_comm_world,ierror)              
      ENDIF

      IF (mpiMyId.eq.0.and.nproc.gt.1) THEN
      call mpi_recv(temp_other(1:sNx,1:sNy,1:Nr),sNx*sNy*Nr,
     &              MPI_DOUBLE_PRECISION,procNum,
     &              procNum+1000,mpi_comm_world,stat,ierror)
      temp_Allproc(1:sNx,1:sNy,1:Nr)=temp(1:sNx,1:sNy,1:Nr)
      temp_Allproc(:,1+sNy*procNum:sNy+sNy*procNum,:)=temp_other(:,:,:)
      ENDIF         
      call mpi_barrier(mpi_comm_world,ierror)
      ENDDO   
      call mpi_barrier(mpi_comm_world,ierror)

      
      IF (mpiMyId.eq.0) THEN     
C   Replcae extreme values of EMF divergence by the values at the left grid (j-1)
C   This replacement procedure is done by processor zero       
      DO k=1,Nr
      DO j=2,sNy*nPy
      DO i=1,sNx
             IF(abs(temp_Allproc(i,j,k)) .GT. DUpVpDy_ML_maxVal) THEN
                temp_Allproc(i,j,k) = temp_Allproc(i,j-1,k)
            ENDIF
      ENDDO
      ENDDO
      ENDDO
               
      DO k=1,Nr
      DO j=1,sNy+OLy
      DO i=1-OLx,sNx+OLx
      temp_new(i,j,k) = temp_Allproc(1,j,k)
      ENDDO
      ENDDO   
      ENDDO      
      ENDIF      
      call mpi_barrier(mpi_comm_world,ierror)
       

C   send the modified EMF divergence to other processors    
      call mpi_bcast(temp_Allproc,sNx*sNy*nPy*Nr,MPI_DOUBLE_PRECISION,
     &                0,mpi_comm_world,ierror)          
      call mpi_barrier(mpi_comm_world,ierror)
          
      IF (mpiMyId.ne.0.and.mpiMyId.ne.nproc-1) THEN
      DO k=1,Nr
      DO j=1-OLy,sNy+OLy
      DO i=1-OLx,sNx+OLx          
                temp_new(i,j,k) = temp_Allproc(1,mpiMyId*sNy+j,k)   
      ENDDO
      ENDDO   
      ENDDO  
      
      ELSEIF(mpiMyId.eq.nproc-1) THEN
          
      DO k=1,Nr
      DO j=1-OLy,sNy
      DO i=1-OLx,sNx+OLx          
                temp_new(i,j,k) = temp_Allproc(1,mpiMyId*sNy+j,k)   
      ENDDO
      ENDDO   
      ENDDO               
          
      ENDIF  
      call mpi_barrier(mpi_comm_world,ierror)
    
     
      
      
      ELSE
C  Simply bound the EMF divergence

      DO k=1,Nr
      DO j=1-OLy+1,sNy+OLy-1
      DO i=1-OLx+1,sNx+OLx         
       temp_new(i,j,k) = MIN(temp(i,j,k),DUpVpDy_ML_maxVal)
       temp_new(i,j,k) = MAX(temp_new(i,j,k),-DUpVpDy_ML_maxVal)    
      ENDDO
      ENDDO   
      ENDDO    
          
          
      ENDIF
      
    
              
C     Options to use depth-dependent or depth-mean version of predicted
C     Reynolds stress. The depth-dependet version is problematic if 
C     Uz is included in the ANN inputs for u'v'        
              
      IF ( UpVp_VertStruc ) THEN
C     Depth-dependent version   
          
      DO k=1,Nr
       DO j=1-Oly,sNy+Oly
        DO i=1-Olx+1,sNx+Olx       
      DupvpDy_ML_temp(i,j,k) = temp_new(i,j,k)
         ENDDO
       ENDDO
      ENDDO                     

      ELSE
C     Depth-mean version                
      DO k=1,Nr   
       DO j=1,sNy        
        DO i=1-Olx+1,sNx+Olx
        IF (k.LE. kLowC(i,j,bi,bj)-2) THEN            
            DupvpDy_ML_temp2(i,j) = DupvpDy_ML_temp2(i,j) + 
     &               drF(k)*hfacC(i,j,k,bi,bj)*temp_new(i,j,k) 
            Depth_temp(i,j) = Depth_temp(i,j)+drF(k)*hfacC(i,j,k,bi,bj)
        ENDIF
        ENDDO 
       ENDDO
      ENDDO
      
      DO k=1,Nr 
      DO j=1,sNy
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
      
       
      ELSEIF(UpVp_Efold) THEN 
C Add an artificial vertical structure for u'v' with e-folding depth 400 m                   
      DO j=1-OLy+2,sNy+OLy-2
       DO i=1-OLx+1,sNx+OLx                 
              tempDepth  = 0. _d 0
              tempStruc  = 0. _d 0    
           DO k=1,Nr
              tempDepth = tempDepth + drF(k)*hfacC(i,j,k,bi,bj)
              tempStruc = tempStruc + exp(-abs(rC(k))/400)
     &                               *drF(k)*hfacC(i,j,k,bi,bj)
           ENDDO       
           IF ( tempStruc.NE. 0. _d 0 ) THEN   
           DO k=1,Nr    
          DupvpDy_ML(i,j,k,bi,bj) = DupvpDy_ML_temp(i,j,k)
     &                             *exp(-abs(rC(k))/400)
     &                             *tempDepth/tempStruc
           ENDDO     
           ELSE         
           DO k=1,Nr    
           DupvpDy_ML(i,j,k,bi,bj) = 0. _d 0
           ENDDO
           ENDIF                 
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
      CALL DIAGNOSTICS_FILL(temp_new,'DupvpDy2',0,Nr,2,bi,bj,myThid)
      ENDIF
#endif


      ENDIF
#endif

      RETURN
      END
