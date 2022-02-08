#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"
#ifdef ALLOW_GMREDI
# include "GMREDI_OPTIONS.h"
#endif

C--  File UpVpML_tendency_apply.F: Routines to apply UpVpML tendencies
C--   Contents
C--   o TAUEDDY_TENDENCY_APPLY_U

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: UpVpML_TENDENCY_APPLY_U
C     !INTERFACE:
      SUBROUTINE UpVpML_TENDENCY_APPLY_U(
     U                     gU_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | S/R UpVpML_TENDENCY_APPLY_U
C     | o Contains problem specific forcing for zonal velocity.
C     *==========================================================*
C     | Adds terms to gU for Reynolds stress divergence
C     *==========================================================*
C     \ev

C     !USES:
      IMPLICIT NONE
C     == Global data ==
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "FFIELDS.h"
#ifdef ALLOW_GMREDI
# include "GMREDI.h"
#endif

C     !INPUT/OUTPUT PARAMETERS:
C     gU_arr    :: the tendency array
C     iMin,iMax :: Working range of x-index for applying forcing.
C     jMin,jMax :: Working range of y-index for applying forcing.
C     k         :: Current vertical level index
C     bi,bj     :: Current tile indices
C     myTime    :: Current time in simulation
C     myIter    :: Current iteration number
C     myThid    :: my Thread Id number
      _RL     gU_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid

CEOP

C In dynamics.F(line 192): PARAMETER( jMin = 0 , jMax = sNy+1 )

C     !LOCAL VARIABLES:
C     i, j      :: Loop counters
      INTEGER i, j

#ifdef ALLOW_GM_ML
      DO j=jMin,jMax
       DO i=iMin,iMax
C         Add to the tendency
             gU_arr(i,j) = gU_arr(i,j)
     &     - maskW(i,j,k,bi,bj) * DupvpDy_ML(i,j,k,bi,bj)
        ENDDO
      ENDDO
#endif


      RETURN
      END


