include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

maincode: multiplecombined.o
	-${CLINKER} -o maincode multiplecombined.o ${PETSC_KSP_LIB} ${PETSC_DM_LIB} -lgsl -lgslcblas
	${RM} -f multiplecombined.o

runcode:
	${mpi} -n 2 ./maincode -usermesh 1 -fC ~/Dropbox/gvresearch/petsc/distmesh/circleMeshs/Circle.2.tri \
	-fV ~/Dropbox/gvresearch/petsc/distmesh/circleMeshs/Circle.2.points -holes 1 -maxmode 5 -b 0.01 \
	-pc_type lu -pc_factor_mat_solver_package superlu_dist

runvtk:
	${mpi} -n 2 ./testvtk

testvtk: testVTK.o
	-${CLINKER} -o testvtk testVTK.o ${PETSC_DM_LIB}
	${RM} -f testVTK.o
