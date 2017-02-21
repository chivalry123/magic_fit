MODULE bregman

implicit none 

!
! Public subroutines for compressive sensing using Bregman iterations.
! For large problems, use SplitBregman.
!
! REFERENCES:
!
! The Bregman algorithm is described in W. Yin, S. Osher, D. Goldfarb, and J. Darbon, 
! "Bregman Iterative Algorithms for L1-Minimization with Applications to Compressed Sensing,"
! SIAM J. Img. Sci. 1, 143 (2008).
!
! Fixed-point continuation is described in E. T. Hale, W. Yin, and Y. Zhang, 
! "A fixed-point continuation method for l1-regularized minimization with applications 
! to compressed sensing," CAAM TR07-07, Rice University (2007).
!
! The split Bregman algorithm is described in T. Goldstein and S. Osher,
! "The split Bregman method for L1 regularized problems",
! SIAM Journal on Imaging Sciences Volume 2 Issue 2, Pages 323-343 (2009).
!
! The conjugae gradient algorithm is described in S. Boyd and L. Vandenberghe,
! "Convex Optimization" (Cambridge University Press, 2004).
!
! Coded by Vidvuds Ozolins (UCLA Materials Science & Eng).
! Last modified: June 19, 2012
!
public BregmanFPC
public SplitBregman
public CGmin
public Shrink

!
! The same as above, but takes advantage of sparse sensing matrices.
!
public SparseBregmanFPC
public SparseSplitBregman
public SparseCGmin

!
! Private parameters for the fixed-point continuation (FPC) algorithm.
!
! MaxFPCit - maximum number of FPC iterations
! nPrint   - number of FPC its to print out gradients and residuals;
!            specify nPrint > MaxFPCit if output is not desired
! FPCgtol  - tolerance for gradient in FPC iterations; the algorithm will iterate until
!            FPCgtol > ||grad||_infty / mu - 1d0
! FPCxtol  - tolerance for the solution; the algorithm will iterate until
!            FPCxtol > ||u-uprev||_2 / ||uprev||_1
!
integer, private                  :: MaxFPCit = 60000
integer, private                  :: nPrint = 1000
double precision, private         :: FPCgtol = 1d-1
double precision, private         :: FPCxtol = 1d-4

!
! Tolerance factor for the CG algorithm: CG stops if ||grad|| <  CGtol*||deltaU||/||U||
!
double precision                  :: CGtol = 1d-1

CONTAINS

SUBROUTINE BregmanFPC(nBregIt, tol, tau, mu, N, M, A, b, u)
  ! 
  ! Performs Bregman iterations using fixed-point continuation (FPC)
  ! as the solver for the mixed L1/L2 problem:
  !
  !    u = arg min_u { mu ||u||_1 + 1/2 ||Au-b||_2^2 }
  !
  ! The Bregman algorithm is described in W. Yin, S. Osher, D. Goldfarb, and J. Darbon, 
  ! "Bregman Iterative Algorithms for L1-Minimization with Applications to Compressed Sensing,"
  ! SIAM J. Img. Sci. 1, 143 (2008).
  !
  ! Input parameters:
  !   nBregIt - Number of outer Bregman loops (usually 5)
  !   tol     - Required tolerance for the residual. The algorithm stops when
  !             tol > ||Au-b||_2 / ||b||_2
  !   tau     - step size for the FPC iterations;
  !             tau = MIN(1.999d0, -1.665d0 * M/N + 2.665d0)
  !   mu      - weight for the L1 norm of the solution
  !   N   - number of unknown expansion coefficients [ =length(u) ]
  !   M       - number of measurements [ =length(b) ]
  !   A       - sensing matrix of dimensions (M,N); the maximum eigenvalue of A.AT should be <= 1
  !             this can be enforced by dividing both A and b with lambda=Sqrt(Max(Eigenvalues(A.Transpose(A))))
  !   b       - array with measured signal values of length M 
  !   u       - solution array of length N
  !
  ! Output:
  !   u       - solution
  !

  integer, intent(in) :: nBregIt, N, M
  double precision, intent(in) :: tol, tau, mu
  double precision,intent(in) :: A(M,N), b(M)
  double precision,intent(inout) :: u(N)

  double precision, allocatable:: uprev(:), grad(:), residual(:), x(:), bp(:)

  integer j, k
  double precision crit1, crit2

  ALLOCATE( grad(N), uprev(N), bp(M), x(M), residual(M) )

  bp = b

  do k = 1, nBregIt

     do j = 1, MaxFPCit
        uprev = u

        x = MATMUL(A, u) - bp
        grad = MATMUL(TRANSPOSE(A), x)
        u = u - tau * grad

        !        print *, ' norm of residual = ', SQRT(dot_product(x,x))
        !        print *, ' norm of gradient = ', SQRT(dot_product(grad,grad))
        !        print *, ' norm of u = ', SQRT(dot_product(u,u))

        call Shrink(u, N, mu*tau)
        !        print *, ' norm of u after shrinkage = ', SQRT(dot_product(u,u))

        crit1 = MAXVAL(ABS(grad)) / mu - 1d0
        crit2 = SQRT(DOT_PRODUCT(u-uprev,u-uprev)) / MAX(SUM(ABS(uprev)),1d0)

        if( MOD(j,nPrint) == 0 ) print '(i6,2f14.6)', j, crit1, crit2
        if( crit1 < FPCgtol .and. crit2 < FPCxtol ) exit
     end do

     if ( j >= MaxFPCit ) then
        print '(2(a,f12.8))', " Unconverged FPC: ||deltaU||/||U|| =", crit2, " ||grad||/mu - 1 = ", crit1 
     end if

     residual = MATMUL(A,u) - b

     crit1 =  SQRT(DOT_PRODUCT(residual,residual)) / SQRT(DOT_PRODUCT(b,b))
     print '(a,i2,a,i6,a,f12.6)', ' Finished Bregman loop #',k,' after ', &
          j,' FPC steps, ||Au-b||/||b|| = ', crit1

     if( crit1 <= tol ) then
        exit
     else
        bp = bp - residual
     end if
  end do

  DEALLOCATE( grad, uprev, bp, x, residual )

  return
end SUBROUTINE BregmanFPC


SUBROUTINE SparseBregmanFPC(nBregIt, tol, tau, mu, N, M, nA, Aval, iAx, b, u)
  ! 
  ! Performs Bregman iterations using fixed-point continuation (FPC)
  ! as the solver for the mixed L1/L2 problem:
  !
  !    u = arg min_u { mu ||u||_1 + 1/2 ||Au-b||_2^2 }
  !
  ! This version uses a sparse sensing matrix A.
  !
  ! The Bregman algorithm is described in W. Yin, S. Osher, D. Goldfarb, and J. Darbon, 
  ! "Bregman Iterative Algorithms for L1-Minimization with Applications to Compressed Sensing,"
  ! SIAM J. Img. Sci. 1, 143 (2008).
  !
  ! Input parameters:
  !   nBregIt - Number of outer Bregman loops (usually 5)
  !   tol     - Required tolerance for the residual. The algorithm stops when
  !             tol > ||Au-b||_2 / ||b||_2
  !   tau     - step size for the FPC iterations;
  !             tau = MIN(1.999d0, -1.665d0 * dble(M) / dble(N) + 2.665d0)
  !   mu      - weight for the L1 norm of the solution
  !   N       - number of unknown expansion coefficients [ =length(u) ]
  !   M       - number of measurements [ =length(b) ]
  !   nA      - number of nonzero values of the sensing matrix A
  !   Aval    - nonzero values of the sensing matrix; the maximum eigenvalue of A.AT should be <= 1; 
  !             this can be enforced by dividing both A and b with lambda=Sqrt(Max(Eigenvalues(A.Transpose(A))))
  !   iAx     - indices of the nonzero elements of A; values of iAx are within the range (1:M,1:N)
  !   b       - array with measured signal values (of length M)
  !   u       - solution array (of length N)
  !
  ! Output:
  !   u       - solution
  !
  integer, intent(in) :: nBregIt, N, M, nA, iAx(nA,2)
  double precision, intent(in) :: tol, tau, mu
  double precision,intent(in) :: Aval(nA), b(M)
  double precision,intent(inout) :: u(N)

  double precision, allocatable:: uprev(:), grad(:), residual(:), x(:), bp(:)

  integer i, j, k
  double precision crit1, crit2

  ALLOCATE( grad(N), uprev(N), bp(M), x(M), residual(M) )

  bp = b

  do k = 1, nBregIt

     do j = 1, MaxFPCit
        uprev = u

#ifdef IntelMKL
        !
        ! Use BLAS calls if Intel MKL is available.
        !
        call mkl_dcoogemv('N', M, Aval, iAx(:,1), iAx(:,2), nA, u, x)
        x = x - bp
        call mkl_dcoogemv('N', N, Aval, iAx(:,2), iAx(:,1), nA, x, grad)
#else
        ! 
        ! Manual matrix-vector multiplies
        !
        x = 0d0
        do i = 1, nA
           x(iAx(i,1)) = x(iAx(i,1)) + Aval(i) * u(iAx(i,2))
        end do
        x = x - bp

        grad = 0d0
        do i = 1, nA
           grad(iAx(i,2)) = grad(iAx(i,2)) + Aval(i) * x(iAx(i,1))
        end do
#endif

        u = u - tau * grad

        !        print *, ' norm of residual = ', SQRT(dot_product(x,x))
        !        print *, ' norm of gradient = ', SQRT(dot_product(grad,grad))
        !        print *, ' norm of u = ', SQRT(dot_product(u,u))

        call Shrink(u, N, mu*tau)
        !        print *, ' norm of u after shrinkage = ', SQRT(dot_product(u,u))

        crit1 = MAXVAL(ABS(grad)) / mu - 1d0
        crit2 = SQRT(DOT_PRODUCT(u-uprev,u-uprev)) / MAX(SUM(ABS(uprev)),1d0)

        if( MOD(j,nPrint) == 0 ) print '(i6,2f14.6)', j, crit1, crit2
        if( crit1 < FPCgtol .and. crit2 < FPCxtol ) exit
     end do

     if ( j>= MaxFPCit ) then
        print '(2(a,f12.8))', " Unconverged FPC: ||deltaU||/||U|| =", crit2, " ||grad||/mu - 1 = ", crit1 
     end if

#ifdef IntelMKL
     !
     ! Use BLAS calls if Intel MKL is available.
     !
     call mkl_dcoogemv('N', M, Aval, iAx(:,1), iAx(:,2), nA, u, residual)
#else
     ! 
     ! Manual matrix-vector multiply
     !
     residual = 0d0
     do i = 1, nA
        residual(iAx(i,1)) = residual(iAx(i,1)) + Aval(i) * u(iAx(i,2))
     end do
#endif

     residual = residual - b

     crit1 =  SQRT(DOT_PRODUCT(residual,residual)) / SQRT(DOT_PRODUCT(b,b))
     print '(a,i2,a,i6,a,f12.6)', ' Finished Bregman loop #',k,' after ', &
          j,' FPC steps, ||Au-b||/||b|| = ', crit1

     if( crit1 <= tol ) then
        exit
     else
        bp = bp - residual
     end if
  end do

  DEALLOCATE( grad, uprev, bp, x, residual )

  return
end SUBROUTINE SparseBregmanFPC


SUBROUTINE SplitBregman(MaxIt, tol, mu, lambda, N, M, A, f, u)
  ! 
  ! Performs split Bregman iterations using conjugate gradients (CG) for the
  ! L2 minimizations and shrinkage for L1 minimizations.
  !
  !    u = arg min_u { ||u||_1 + mu/2 ||Au-f||_2^2 }
  !
  ! The algorithm is described in T. Goldstein and S. Osher,
  ! "The split Bregman method for L1 regularized problems",
  ! SIAM Journal on Imaging Sciences Volume 2 Issue 2, Pages 323-343 (2009).
  !
  ! Input parameters:
  !   MaxIt   - Number of outer split Bregman loops
  !   tol     - Required tolerance for the residual. The algorithm stops when
  !             tol > ||Au-f||_2 / ||f||_2
  !   mu      - weight for the L1 norm of the solution
  !   lambda  - weight for the split constraint (affects speed of convergence, not the result)
  !   N       - number of unknown expansion coefficients [ =length(u) ]
  !   M       - number of measurements [ =length(f) ]
  !   A       - sensing matrix A
  !   f       - array with measured signal values (of length M)
  !   u       - solution array (of length N)
  !
  ! Output:
  !   u       - solution
  !
  integer, intent(in)             :: MaxIt, N, M
  double precision, intent(in)    :: tol, mu, lambda
  double precision,intent(in)     :: A(M,N), f(M)
  double precision,intent(inout)  :: u(N)

  integer k, MaxCGit
  double precision crit1, crit2
  double precision, allocatable:: uprev(:), b(:), d(:), bp(:)

  MaxCGit = MAX(10,N/2)
  crit1 = 1d0
  crit2 = 1d0

  ALLOCATE( uprev(N), b(N), d(N), bp(N) )

  b = 0d0
  d = 0d0

  do k = 1, MaxIt

     uprev = u

     bp = d - b
     call CGmin(N, M, A, f, bp, mu, lambda, MaxCGit, CGtol*crit1, u)

     d = b + u
     call Shrink(d, N, 1d0/lambda)

     crit1 = SQRT(dot_product(u-uprev,u-uprev)/dot_product(u,u))
     crit2 = SQRT(dot_product(u-d,u-d)/dot_product(u,u))
     print '(a,i3,2(a,f10.6))', ' SplitBregman: it=', k, ', ||deltaU||/||U|| = ', crit1, ', ||d-U||/||U|| = ', crit2
     if ( crit1 <= tol ) exit

     b = b + u - d

  end do

  if ( crit1 > tol ) then
     print '(2(a,f10.6))', ' Did not reach prescribed accuracy in SplitBregman: ||deltaU||/||U|| = ', crit1, &
          ', ||d-U||/||U|| = ', crit2
  end if

  DEALLOCATE( uprev, b, d, bp )
  return

end SUBROUTINE SplitBregman


SUBROUTINE SparseSplitBregman(MaxIt, tol, mu, lambda, N, M, nA, Aval, iAx, f, u)
  ! 
  ! Performs split Bregman iterations using conjugate gradients (CG) for the
  ! L2 minimizations and shrinkage for L1 minimizations.
  !
  !    u = arg min_u { ||u||_1 + mu/2 ||Au-f||_2^2 }
  !
  ! This version uses sparse sensing matrices A.
  !
  ! The algorithm is described in T. Goldstein and S. Osher,
  ! "The split Bregman method for L1 regularized problems",
  ! SIAM Journal on Imaging Sciences Volume 2 Issue 2, Pages 323-343 (2009).
  !
  ! Input parameters:
  !   MaxIt   - Number of outer split Bregman loops
  !   tol     - Required tolerance for the residual. The algorithm stops when
  !             tol > ||Au-f||_2 / ||f||_2
  !   mu      - weight for the L1 norm of the solution
  !   lambda  - weight for the split constraint (affects speed of convergence, not the result)
  !   N       - number of unknown expansion coefficients [ =length(u) ]
  !   M       - number of measurements [ =length(f) ]
  !   nA      - number of nonzero values of the sensing matrix A
  !   Aval    - nonzero values of the sensing matrix; the maximum eigenvalue of A.AT should be <= 1; 
  !             this can be enforced by dividing both A and f with lambda=Sqrt(Max(Eigenvalues(A.Transpose(A))))
  !   iAx     - indices of the nonzero elements of A; values of iAx are within the range (1:M,1:N)
  !   f       - array with measured signal values (of length M)
  !   u       - solution array (of length N)
  !
  ! Output:
  !   u       - solution
  !
  integer, intent(in) :: MaxIt, N, M, nA, iAx(nA,2)
  double precision, intent(in) :: tol, mu, lambda
  double precision,intent(in) :: Aval(nA), f(M)
  double precision,intent(inout) :: u(N)

  integer k, MaxCGit, Nloop
  double precision crit1, crit2
  double precision, allocatable:: uprev(:), b(:), d(:), bp(:)

  MaxCGit = MAX(10,N/5)
  crit1 = 1d0
  crit2 = 1d0

  ALLOCATE( uprev(N), b(N), d(N), bp(N) )
  b = 0d0
  d = 0d0

  do k = 1, MaxIt

     uprev = u

     bp = d - b
     call SparseCGmin(N, M, nA, Aval, iAx, f, bp, mu, lambda, MaxCGit, CGtol*crit1, u)

     d = b + u
     call Shrink(d, N, 1d0/lambda)

     crit1 = SQRT(DOT_PRODUCT(u-uprev,u-uprev)/DOT_PRODUCT(u,u))
     crit2 = SQRT(DOT_PRODUCT(u-d,u-d)/DOT_PRODUCT(u,u))
     print '(a,i3,2(a,f10.6))', ' SplitBregman: it=', k, ', ||deltaU||/||U|| = ', crit1, &
          ', ||d-U||/||U|| = ', crit2
     if ( crit1 <= tol ) exit

     b = b + u - d

  end do

  if ( crit1 > tol ) then
     print '(2(a,f10.6))', ' Did not reach prescribed accuracy in SplitBregman: ||deltaU||/||U|| = ', crit1, &
          ', ||d-U||/||U|| = ', crit2
  end if

  DEALLOCATE( uprev, b, d, bp )
  return

end SUBROUTINE SparseSplitBregman


SUBROUTINE CGmin(N, M, A, f, b, mu, lambda, MaxIt, gtol, u)
  !
  ! Conjugate gradient routine to perform L2-based minimization of 
  !
  !     min_u { mu/2 ||Au-f||_2^2 + lambda/2 ||b-u||_2^2 }
  !
  ! Algorithm is described in S. Boyd and L. Vandenberghe,
  ! "Convex Optimization" (Cambridge University Press, 2004).
  !
  ! Inut parameters:
  !    A    - sensing matrix of dimensions (M,N)
  !    M    - number of measurements
  !    N    - number of expansion coefficients
  !    f    - values of measurements
  !    b    - vector enforcing the split-off L1 constraint
  !    mu   - weight of the L2 constraint on Au=f
  !    lambda - weight of the split-off constraint
  !    MaxIt  - max. number of CG iterations
  !    gtol - tolerance for the gradient; exit if gtol > ||grad||_2
  !    u    - starting guess for the solution
  !
  ! Output parameter:
  !    u    - converged solution
  !
  integer, intent(in) :: MaxIt, N, M
  double precision, intent(in) :: gtol, mu, lambda
  double precision,intent(in) :: A(M,N), f(M), b(N)
  double precision,intent(inout) :: u(N)

  integer k
  double precision, allocatable:: p(:), r(:), rp(:), x(:)
  double precision beta, alpha, delta, deltaprev

  ALLOCATE( p(N), r(N), rp(N), x(M) )

  x = MATMUL(A,u) - f
  r = -( mu * MATMUL(TRANSPOSE(A),x) - lambda * (b - u) )

  p = r
  delta = DOT_PRODUCT(r,r)
  !    print '(a,f16.4)', ' initial norm of gradient = ', SQRT(delta)

  do k = 1, MaxIt

     x = MATMUL(A,p)
     rp = mu * MATMUL(TRANSPOSE(A),x) + lambda * p

     alpha = delta / DOT_PRODUCT(p,rp)

     !     print '(a,f16.6)', ' gradient check = ', DOT_PRODUCT(r,r - alpha * rp)

     u = u + alpha * p
     r = r - alpha * rp

     deltaprev = delta
     delta = DOT_PRODUCT(r,r)

     !     print '(a,f16.4)', ' norm of gradient = ', SQRT(delta)

     if( SQRT(delta) < gtol ) then
        !     print '(a,i4,a)', ' Finished after ', k, ' CG iterations.'
        exit
     end if

     beta = delta / deltaprev
     p = beta * p + r

  end do

  if ( SQRT(delta) > gtol ) then
     print '(a,f12.6)', ' Unconverged gradient after CGmin = ', SQRT(delta)
  end if

  DEALLOCATE( p, r, rp, x )

  return
end SUBROUTINE CGmin



SUBROUTINE SparseCGmin(N, M, nA, Aval, iAx, f, b, mu, lambda, MaxIt, gtol, u)
  !
  ! Conjugate gradient routine to perform L2-based minimization of 
  !
  !     min_u { mu/2 ||Au-f||_2^2 + lambda/2 ||b-u||_2^2 }
  !
  ! This version uses a sparse sensing matrix A.
  !
  ! Algorithm is described in S. Boyd and L. Vandenberghe,
  ! "Convex Optimization" (Cambridge University Press, 2004).
  !
  ! Inut parameters:
  !    M    - number of measurements
  !    N    - number of expansion coefficients
  !    nA   - number of nonzero values of the sensing matrix A
  !    Aval - nonzero values of the sensing matrix
  !    iAx    - indices of the nonzero elements of A; values of iAx are within the range (1:M,1:N)
  !    f    - values of measurements
  !    b    - vector enforcing the split-off L1 constraint
  !    mu   - weight of the L2 constraint on Au=f
  !    lambda - weight of the split-off constraint
  !    MaxIt  - max. number of CG iterations
  !    gtol - tolerance for the gradient; exit if gtol > ||grad||_2
  !    u    - starting guess for the solution
  !
  ! Output parameter:
  !    u    - converged solution
  !
  integer, intent(in)            :: MaxIt, N, M, nA, iAx(nA,2)
  double precision, intent(in)   :: gtol, mu, lambda
  double precision,intent(in)    :: Aval(nA), f(M), b(N)
  double precision,intent(inout) :: u(N)

  integer i, k
  double precision, allocatable:: p(:), r(:), rp(:), x(:)
  double precision beta, alpha, delta, deltaprev

  ALLOCATE( p(N), r(N), rp(N), x(M) )

#ifdef IntelMKL
  !
  ! Use BLAS calls if Intel MKL is available.
  !
  call mkl_dcoogemv('N', M, Aval, iAx(:,1), iAx(:,2), nA, u, x)
  x = x - f
  call mkl_dcoogemv('N', N, Aval, iAx(:,2), iAx(:,1), nA, x, r)
  r = -( mu * r - lambda * (b - u) )
#else
  !
  ! Manual matrix-vector multiplies.
  !
  x = 0d0
  do i = 1, nA
     x(iAx(i,1)) = x(iAx(i,1)) + Aval(i) * u(iAx(i,2))
  end do
  x = x - f

  r = 0d0
  do i = 1, nA
     r(iAx(i,2)) = r(iAx(i,2)) + Aval(i) * x(iAx(i,1))
  end do
  r = -( mu * r - lambda * (b - u) )
#endif

  p = r
  delta = DOT_PRODUCT(r,r)
  !      print '(a,f16.4)', ' norm of gradient = ', SQRT(delta)

  do k = 1, MaxIt

#ifdef IntelMKL
     !
     ! Use BLAS calls if Intel MKL is available.
     !
     call mkl_dcoogemv('N', M, Aval, iAx(1:nA,1), iAx(1:nA,2), nA, p, x)
     call mkl_dcoogemv('N', N, Aval, iAx(1:nA,2), iAx(1:nA,1), nA, x, rp)
#else
     !
     ! Manual matrix-vector multiplies.
     !
     x = 0d0
     do i = 1, nA
        x(iAx(i,1)) = x(iAx(i,1)) + Aval(i) * p(iAx(i,2))
     end do

     rp = 0d0
     do i = 1, nA
        rp(iAx(i,2)) = rp(iAx(i,2)) + Aval(i) * x(iAx(i,1))
     end do
#endif

     rp = mu * rp + lambda * p

     alpha = delta / DOT_PRODUCT(p,rp)
     !     print '(a,f16.6)', ' gradient check = ', dot_product(r,r - alpha * rp)

     u = u + alpha * p
     r = r - alpha * rp

     deltaprev = delta
     delta = DOT_PRODUCT(r,r)
     !     print '(a,f16.4)', ' norm of gradient = ', SQRT(delta)

     if( SQRT(delta) < gtol ) then
        !     print '(a,i4,a)', ' Finished after ', k, ' CG iterations.'
        exit
     end if

     beta = delta / deltaprev
     p = beta * p + r

  end do

  if ( SQRT(delta) > gtol ) then
     print '(a,f12.6)', ' Unconverged gradient after CGmin = ', SQRT(delta)
  end if

  DEALLOCATE( p, r, rp, x )

  return
end SUBROUTINE SparseCGmin



subroutine Shrink( u, N, alpha )
  !
  ! Defines L1-based shrinkage.
  !
  integer, intent(in)  :: N
  double precision, intent(in) :: alpha
  double precision, intent(inout) :: u(N)

  where (u > 0d0) 
     u = MAX(ABS(u) - alpha, 0d0)
  elsewhere
     u = - MAX(ABS(u) - alpha, 0d0)
  end where

  return
end subroutine Shrink


end MODULE bregman
