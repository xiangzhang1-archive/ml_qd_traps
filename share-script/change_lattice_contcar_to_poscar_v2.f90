   program change_lattice_contcar_to_poscar

   implicit none
   integer,parameter :: num_type_max=20
   real*8,parameter :: bohr=0.529177d0
   integer :: i,j,k,con,stat
   integer :: num_atom,num_type
   character(len=3),dimension(num_type_max) :: type_name
   integer,dimension(num_type_max) :: type_num_atom
   character(len=2),allocatable,dimension(:) :: atom_type
   integer,allocatable,dimension(:) :: atom_type_id
   real*8,allocatable,dimension(:,:) :: atom_direct_coord
   real*8,dimension(3,3) :: latt_vec_old,latt_vec_new
   real*8,dimension(3) :: latt_vec_change
   real*8 :: scale_fac
   character(len=300) :: filein,fileout,filein2
   character(len=200) :: line,sys_name
   character,allocatable,dimension(:,:) :: atom_dy
   character :: dynamics, wrap

   filein = 'CONTCAR'
   filein2 = 'change_lattice_contcar_to_poscar_v2.in'
   fileout = 'POSCAR'

   open(unit=35,file=filein2,status='old',action='read')
   read(35,*) latt_vec_change(:)
   read(35,*) wrap
   read(35,*) dynamics
   close(35)

   open(unit=34,file=filein,status='old',action='read')
   read(34,*) sys_name
   read(34,*) scale_fac
   do i=1,3
      read(34,*) latt_vec_old(i,:)
   enddo
   latt_vec_new = 0.0d0
   do i=1,3
      latt_vec_new(i,i) = latt_vec_old(i,i)+latt_vec_change(i)
   enddo
   read(34,'(A)') line
   num_type = 0
   type_name = ''
   con = 1
   do while(con .le. len(line))
      if (line(con:con) .ne. ' ') then 
         num_type = num_type+1
         type_name(num_type) = line(con:con+1)
         con = con+2
      else
         con = con+1
      endif
   enddo
   write(*,*) 'Number of atom type = ',num_type
   close(34)
   
   open(unit=35,file=filein,status='old',action='read')
   do i=1,6
      read(35,*) 
   enddo
   type_num_atom = 0
   read(35,*) type_num_atom(1:num_type)
   num_atom = sum(type_num_atom)
   write(*,*) 'Number of atoms = ',num_atom
   read(35,*)
   if (dynamics .eq. 'y' .or. dynamics .eq. 'Y') read(35,*)
   allocate(atom_direct_coord(num_atom,3))
   allocate(atom_dy(num_atom,3))
   atom_direct_coord = 0.0d0
   do i=1,num_atom
      if (dynamics .eq. 'y' .or. dynamics .eq. 'Y') then
         read(35,*) atom_direct_coord(i,:),atom_dy(i,:)
      else
         read(35,*) atom_direct_coord(i,:)
      endif
   enddo
   close(35)

   do i=1,num_atom
      do j=1,3
         if (trim(wrap) .eq. 'Y' .or. trim(wrap) .eq. 'y') then
            if (atom_direct_coord(i,j) > 0.5d0) then
               atom_direct_coord(i,j) = atom_direct_coord(i,j) - 1.0d0
            endif
         endif
         atom_direct_coord(i,j) = atom_direct_coord(i,j)*latt_vec_old(j,j)/latt_vec_new(j,j)
      enddo
   enddo

   open(unit=36,file=fileout,status='replace',action='write')
   write(36,*) trim(sys_name)
   write(36,*) scale_fac
   do i=1,3
      write(36,'(3f20.9)') latt_vec_new(i,:)
   enddo
   write(36,*) type_name(1:num_type) 
   write(36,*) type_num_atom(1:num_type)
   if (dynamics .eq. 'y' .or. dynamics .eq. 'Y') then
      write(36,'(a18)') 'Selective Dynamics'
   endif
   write(36,'(a6)') 'Direct'
   do i=1,num_atom
      if (dynamics .eq. 'y' .or. dynamics .eq. 'Y') then
         write(36,'(3f20.9,3a3)') atom_direct_coord(i,:),atom_dy(i,:)
      else
         write(36,'(3f20.9)') atom_direct_coord(i,:)
      endif
   enddo
   close(36)

   end program change_lattice_contcar_to_poscar
