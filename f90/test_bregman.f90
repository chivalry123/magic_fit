program test_bregman

  use bregman
  
  implicit none

  integer  :: i
  integer  :: Ncorr, Nstruc, nA
  double precision :: tol, tau, mu, sigma, lambda, err

  integer, allocatable, dimension(:,:)          :: iAx
  double precision, dimension(:,:), allocatable :: A
  double precision, dimension(:), allocatable   :: Aval, En, ecis, Epred, Apred, x
  double precision, external                    :: RMSfit
  double precision, parameter                   :: eci_cutoff=1d-8

  character(len=32) :: arg
  integer :: method, argc

  open(10,file='Amatrix.dat',status='old',form='formatted')
  read(10,*)
  read(10,*)
  read(10,*) Nstruc, Ncorr, nA
  print '(3i12)', Nstruc, Ncorr, nA

  allocate( Aval(nA), iAx(nA,2), En(Nstruc), ecis(Ncorr), A(Nstruc,Ncorr) )

  do i = 1, nA
     read(10,*) iAx(i,:), Aval(i)
  end do
  close(10)

  A = 0d0
  do i = 1, nA
     A(iAx(i,1),iAx(i,2)) = Aval(i)
  end do
  
  open(10,file='En.dat',status='old',form='formatted')
  do i = 1, Nstruc
     read(10,*) En(i)
  end do
  close(10)
!  En = 1000 * En

  print *, ' Finished reading correlation matrix and energies.'

  sigma = 1d0
  print '(a,f12.6)', ' Sigma = ', sigma

  A = A / sigma
  Aval = Aval / sigma
  En = En / sigma




  argc= iargc()
  print *
  if (argc < 3) then
    print *, "test_bregman method mu lambda(no effect on split methods)"
    print *, "method: 1 (Bregman, DEFAULT) 2 (Sparse) 3 (Split) 4 (Split-Sparse)"
    method = 1
    mu = 0.001
    lambda = 1.
  else
     call get_command_argument(1, arg)
     read(arg, *) method
     call get_command_argument(2, arg)
     read(arg, *) mu
     call get_command_argument(3, arg)
     read(arg, *) lambda
  end if
  print *, "method=", method
  print *, "mu=", mu
  print *, "lambda=", lambda
  tau = MIN(1.999d0, -1.665d0 * dble(Nstruc) / dble(Ncorr) + 2.665d0)
  print '(a,f12.6)', ' FPC step size tau = ', tau


  ecis = 0d0
  select case (method)

     case (1)
  print *
  print *, ' Calling Bregman FPC ... '
  call BregmanFPC(5, 1d-3, tau, mu, Ncorr, Nstruc, A, En, ecis)



case (2)
  print *, ' Calling Sparse Bregman FPC ... '
  call SparseBregmanFPC(5, 1d-3, tau, mu, Ncorr, Nstruc, nA, Aval, iAx, En, ecis)


  err = sigma * RMSfit(Ncorr, Nstruc, nA, Aval, iAx, ecis, En)
  print '(a,f12.8)', ' RMS error of the fit = ', err
  


case (3)
  print *
  print *, ' Calling Split Bregman ... '
  mu = 1/mu
  call SplitBregman(100, 1d-3, mu, lambda, Ncorr, Nstruc, A, En, ecis)



case (4)
  print *
  print *, ' Calling Sparse Split Bregman ... '
  mu = 1/mu
  call SparseSplitBregman(100, 1d-4, mu, lambda, Ncorr, Nstruc, nA, Aval, iAx, En, ecis)


  err = sigma * RMSfit(Ncorr, Nstruc, nA, Aval, iAx, ecis, En)
  print '(a,f12.8)', ' RMS error of the fit = ', err

  case DEFAULT
     print *, 'Unknown method ', method

end select

  print *, 'found the following nonzero ECIs: '
  do i = 1, Ncorr
     if( ABS(ecis(i)) > eci_cutoff ) then
        print '(i6,3x,f14.10)', i, ecis(i)
     end if
  end do


!!$    open(10,file='ecis.out',status='unknown',form='formatted')
!!$    do n = 1, Ncorr
!!$       if( ABS(ecis(n)) > 1d-2 ) then
!!$          write(10,'(i6,3x,f12.4)') n, ecis(n)
!!$       end if
!!$    end do
!!$    close(10)

!!$    open(10,file='4k-corr.mtx',status='old',form='formatted')
!!$    read(10,*)
!!$    read(10,*)
!!$    read(10,*) NSpred, j, nApred
!!$    print '(3i12)', NSpred, j, nApred
!!$    allocate( Apred(nApred), iApred(nApred,2), Epred(NSpred), x(NSpred) )
!!$    do n = 1, nApred
!!$       read(10,*) iApred(n,:), Apred(n)
!!$    end do
!!$    close(10)
!!$    print '(a)', ' Read correlations from 4k-corr.mtx.'
!!$
!!$    open(10,file='4k.dat',status='old',form='formatted')
!!$    do n = 1, NSpred
!!$       read(10,*) Epred(n)
!!$    end do
!!$    close(10)
!!$    print '(a)', ' Read energies from 4k-corr.mtx.'
!!$
!!$    call getEnergies(Ncorr, NSpred, nApred, Apred, nApred, iApred, ecis, x)
!!$
!!$    err = sqrt(dot_product(x-Epred,x-Epred)/dble(NSpred))
!!$    print '(a,f12.4)', ' RMS prediction error = ', err
!!$
!!$    err = 0d0
!!$    n = 0
!!$    do j = 1, NSpred
!!$       if( Epred(j) <= 0d0 ) then
!!$          n = n + 1
!!$          err = err + (Epred(j)-x(j))**2
!!$       end if
!!$    end do
!!$    err = sqrt(err/dble(n))
!!$    print '(a,f12.4)', ' RMS prediction error for E<0 = ', err
!!$    deallocate( Apred, Epred, iApred)

    deallocate( Aval, iAx, En, ecis, A )

end program test_bregman

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine getEnergies(Ncorr, Nstruc, nA, Aval, iAx, ecis, En)

  implicit none
  integer, intent(in) :: Ncorr, Nstruc, nA, iAx(nA,2)
  double precision,intent(in) :: Aval(nA), ecis(Ncorr)
  double precision,intent(out) :: En(Nstruc)
    
  integer n
    
  En = 0d0
  do n = 1, nA
     En(iAx(n,1)) = En(iAx(n,1)) + Aval(n) * ecis(iAx(n,2))
  end do
  !  call mkl_dcoogemv('N', Nstruc, Aval, iAx(:,1), iAx(:,2), nA, ecis, En)

  return

end subroutine getEnergies

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double precision function RMSfit(Ncorr, Nstruc, nA, Aval, iAx, ecis, En)

  implicit none
  integer, intent(in) :: Ncorr, Nstruc, nA, iAx(nA,2)
  double precision,intent(in) :: Aval(nA), ecis(Ncorr)
  double precision,intent(in) :: En(Nstruc)
  
  integer n
  double precision, allocatable :: x(:)
  
  allocate(x(Nstruc))
  
  call getEnergies(Ncorr, Nstruc, nA, Aval, iAx, ecis, x)
  x = x - En
  RMSfit = sqrt(dot_product(x,x)/dble(Nstruc))
  
  deallocate( x )
  return
  
end function RMSfit

