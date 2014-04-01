NME
===

Nonlinear Maxwell's Equations solver

Solver for a 2D inhomogeneous nonlinear medium using PETSc.

Requires PETSc >= 3.4

Meshes are not provided and can be built with any mesh generation
software. The meshes must be in a list of vertices / triangles 
format.

Sample usage instructions after compilation:
$ mpiexec -n 100 ./solver -usermesh 1 -fV vertex.dat -fC cells.dat \

  -maxmode 10 -b 0.01 \
  
  -ksp_type gmres -ksp_gmres_restart 200 	-ksp_gmres_modifiedgramschmidt \
  
  -pc_type asm -pc_asm_overlap 10 -pc_asm_type basic \
  
  -sub_pc_type ilu 

Here we use 100 processes with an input mesh stored in {vertex.dat, cells.dat}
and solve for a maximum of 10 modes with the nonlinear strength 0.01.
The solver is a GMRES solver with a restart of 200. The preconditioner
is ASM along with it's options and the sub pc type for the asm solver
is the standard ILU solver done locally at each process.

These option provide an example case and one can change this depending
on the problem instance. They are not necessarily the best options for
all cases.
