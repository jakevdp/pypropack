      SUBROUTINE SECOND( T )
*
      REAL       T
*
*  -- LAPACK auxiliary routine (preliminary version) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     July 26, 1991
*
*  Purpose
*  =======
*
*  SECOND returns the user time for a process in seconds.
*  This version gets the time from the system function ETIME.
*
*     .. Local Scalars ..
      REAL               T1
*     ..
*     .. Local Arrays ..
      REAL               TARRAY( 2 )
*     ..
*     .. External Functions ..

#ifdef _AIX
*      REAL               ETIME_
*      EXTERNAL           ETIME_
      EXTERNAL REAL ETIME_
#else
*      REAL               ETIME
*      EXTERNAL           ETIME
      EXTERNAL REAL ETIME
#endif 

*     ..
*     .. Executable Statements ..
*

#ifdef _AIX
      T1 = ETIME_( TARRAY )
      T = T1
#else
      T1 = ETIME( TARRAY )
      T  = TARRAY( 1 )
#endif

      RETURN
*
*     End of SECOND
*
      END
