//Time-stamp: <2013-09-29 00:30:22 garnet>
static char help[] = "Code for running both the plain and the holes problems";

// Parameters used with the exact solution:
// \epsilon = 9 on the whole domain.
// If holes is selected then inside the holes \epsilon=1
// \mu = 1
// a (forcing constant) = 150
// \omega = alpha*2*Pi (Since there is a difference between the old convention and the new one.)
// Solves for n values of alpha = {0.5*1, 0.5*2, ..., 0.5*n} where n=4

// To run with the Triangle mesh: mpiexec -n 2 ./multiplecombined -refinement_limit 0.001 -pc_type lu -pc_factor_mat_solver_package mumps
// To run with a user mesh: mpiexec -n 2 ./multiplecombined -usermesh 1 -fC Square.1.tri -fV Square.1.points -pc_type lu -pc_factor_mat_solver_package mumps

#include<petscksp.h>
#include<petscdmplex.h>
#include<math.h>
#include"gsl/gsl_math.h"
#include"gsl/gsl_integration.h"

#define DIM 2			/* Problem dimension */
#define RADIUS 1.0/40		/* Hole Radius */
#define HOLESX 10
#define HOLESY 8
#define NUMALPHA 1		/* Number of frequencies */
PetscReal alphavals[NUMALPHA] = {0.5};
/* #define NUMALPHA 4		/\* Number of frequencies *\/ */
/* PetscReal alphavals[NUMALPHA] = {0.2, 0.4, 0.6, 0.8}; */

const double XCenters[HOLESX] = {0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95};
const double YCenters[HOLESY] = {0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9};

typedef enum {INTERIOR,LEFT,TOP,RIGHT,BOTTOM} boundary;

typedef struct _p_appCtx *AppCtx;
typedef struct _p_parameters *Parameters;
typedef struct _p_model *Model;
typedef struct _p_meshOptions *MeshOptions;

typedef PetscErrorCode (*CreateMesh)(AppCtx); /* Mesh generation options. */
typedef PetscErrorCode (*Solver)(Mat,AppCtx,Vec*); /* Type of problem to solve. */

typedef struct {
  PetscScalar centroid[2];
  PetscScalar area;
} CellInfo;

typedef PetscScalar (*ComputeEps)(const CellInfo*,PetscScalar,PetscScalar); /* Choice of holes/plain problems */

struct _p_parameters {
  PetscScalar	 mu;
  PetscScalar    epsIn;
  PetscScalar    epsOut;
  PetscScalar    b;
  PetscScalar	 omega;
  PetscReal      alpha;
};

struct _p_model {
  Mat		 delta;
  Vec		 G,C,Vin;
  Vec		 cellGeom;
  IS		 cellIS;
  Solver         solverType;
  ComputeEps    problemType;
};

struct _p_meshOptions {
  PetscReal	 refinementLimit;
  PetscBool	 interpolate;
  PetscBool      writedistmesh;
  PetscBool	 usermesh;
  CreateMesh     meshtype;
  char fileCell[PETSC_MAX_PATH_LEN];
  char fileVertex[PETSC_MAX_PATH_LEN];
};

struct _p_appCtx{
  MPI_Comm	comm;
  DM		dm;
  PetscInt	numLocalCells;
  PetscInt	maxMode;
  PetscInt      maxIter;
  PetscBool	debug;
  MeshOptions	meshOptions;
  Model		model;
  PetscBool     saveDelta;
  Parameters	parameters;
  PetscBool     diverged;
};

PetscErrorCode ProcessOptions(AppCtx); /* User options */
PetscErrorCode CreateMeshTriangle(AppCtx); /* Automatic mesh generation */
PetscErrorCode CreateMeshFromFile(AppCtx); /* Load distmesh created triangulations */
PetscErrorCode readMesh(MPI_Comm comm, const char cname[], const char vname[], PetscInt *numC, PetscInt *numV, int **cell, double **vert);
PetscErrorCode ComputeParameters(AppCtx,Model,Parameters); /* Form L/G/C inc and Delta */
PetscErrorCode ComputeCellInfo(Parameters,Model,DM,Vec,Vec,IS,PetscInt,PetscInt);   /* Form C0 */
PetscErrorCode ComputeCellGeom(const PetscScalar*,CellInfo*); /* Computes area/centroid */
PetscErrorCode ComputeDelta(Parameters,Model,DM,Vec,Vec,Vec,Vec);
PetscScalar ComputeForcing(const PetscScalar*,PetscScalar);
PetscErrorCode LinearSolver(Mat,AppCtx,Vec*);
PetscErrorCode NonlinearSolver(Mat,AppCtx,Vec*);
PetscErrorCode FixedPointSolver(KSP*,AppCtx,Vec*); /* Called from NonlinearSolver */
PetscErrorCode WriteModes(Vec*,PetscInt,PetscReal);    /* Fourier coefficients for each mode written into seperate files */
PetscErrorCode WriteCentroid(Model,DM,Vec*);	      /* Plotting utility */
PetscErrorCode WriteCoords(DM);		      /* Triangle vertex points */
PetscErrorCode WriteCells(DM);		      /* Distmesh style triangle list */
PetscErrorCode WriteSolVTK(DM,Vec*,PetscInt,PetscReal);

////////////////////////////////////////////////////////////////
// Simple inline functions
////////////////////////////////////////////////////////////////
PetscScalar PlainPermittivity(const CellInfo *cv, PetscScalar epsIn, PetscScalar epsOut)
{
  return epsOut;
}

PetscScalar HolesPermittivity(const CellInfo *cv, PetscScalar epsIn, PetscScalar epsOut)
{
  PetscInt	i,j;
  double	xSq,ySq;
  double	distance;
  double	xCentroid = PetscRealPart(cv->centroid[0]);
  double	yCentroid = PetscRealPart(cv->centroid[1]);
  PetscScalar	eps	  = epsOut;

  for (i=0; i<HOLESX; ++i) {
    for (j=0; j<HOLESY; ++j) {
      xSq = gsl_pow_2(xCentroid-XCenters[i]);
      ySq = gsl_pow_2(yCentroid-YCenters[j]);
      distance = sqrt(xSq + ySq);
      if (distance < RADIUS) eps = epsIn;
    }
  }
  return eps;
}

PETSC_STATIC_INLINE PetscScalar InteriorDist(const CellInfo *cv1,const CellInfo *cv2)
{
  PetscScalar xSq = PetscPowComplex(cv1->centroid[0]-cv2->centroid[0],2);
  PetscScalar ySq = PetscPowComplex(cv1->centroid[1]-cv2->centroid[1],2);
  return PetscSqrtComplex(xSq+ySq);
}

PETSC_STATIC_INLINE PetscScalar BoundaryDist(const CellInfo *cv1,const PetscScalar *edgeCoords)
{
  PetscScalar xAvg = (edgeCoords[0]+edgeCoords[2])/2.0;
  PetscScalar yAvg = (edgeCoords[1]+edgeCoords[3])/2.0;
  PetscScalar xSq = PetscPowComplex(cv1->centroid[0]-xAvg,2);
  PetscScalar ySq = PetscPowComplex(cv1->centroid[1]-yAvg,2);
  return PetscSqrtComplex(xSq+ySq);
}

PETSC_STATIC_INLINE PetscScalar LengthOfEdge(const PetscScalar *edgeCoords)
{
  PetscScalar xSq = PetscPowComplex(edgeCoords[0]-edgeCoords[2],2);
  PetscScalar ySq = PetscPowComplex(edgeCoords[1]-edgeCoords[3],2);
  return PetscSqrtComplex(xSq+ySq);
}

// used by gsl
static inline double forcfunc(double x, void *params)
{
  double alpha = *(double *)params;
  double feval = exp(-alpha*gsl_pow_2(x-0.5));
  return feval;
}

int GetLineCount(const char *filename)
{
  FILE *fp = fopen(filename,"r");
  int ch;
  int count=0;
  do
    {
      ch = fgetc(fp);
      if( ch== '\n') count++;
    }while( ch != EOF );
  PetscPrintf(PETSC_COMM_WORLD,"Number of lines is %d\n",count);
  fclose(fp);
  return count;
}

////////////////////////////////////////////////////////////////
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx		 user;
  PetscInt		 i;
  Vec			*sol;
  PetscErrorCode	 ierr;

  PetscInitialize(&argc, &argv, (char*)0, help);
  ierr = PetscNew(struct _p_appCtx,&user);CHKERRQ(ierr);
  ierr = PetscNew(struct _p_parameters,&user->parameters);CHKERRQ(ierr);
  ierr = PetscNew(struct _p_model,&user->model);CHKERRQ(ierr);
  ierr = PetscNew(struct _p_meshOptions,&user->meshOptions);CHKERRQ(ierr);

  /* Set up command line options */
  ierr = ProcessOptions(user);CHKERRQ(ierr);

  /* Create/load mesh */
  ierr = (*user->meshOptions->meshtype)(user);CHKERRQ(ierr);

  /* Set up problem variables */
  ierr = ComputeParameters(user,user->model,user->parameters);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(user->model->Vin,user->maxMode,&sol);CHKERRQ(ierr);

  /* Solve all problem based on linear/nonlinear type for each value of alpha */
  for (i=0; i<NUMALPHA; ++i) {
    user->diverged = PETSC_FALSE;
    if (user->parameters->alpha < 1.e-10) {/* Solve for inbuilt alphavals */
      user->parameters->alpha = alphavals[i];
      user->parameters->omega = alphavals[i]*2.0*PETSC_PI;
  }
    user->parameters->omega = alphavals[i]*2.0*PETSC_PI;
    PetscPrintf(PETSC_COMM_WORLD,"Starting solver for %d\n",i);
    ierr = (user->model->solverType)(user->model->delta,user,sol);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Finished solving for %d\n",i);
    if (!user->diverged) {
      /* ierr = WriteSolVTK(user->dm,sol,user->maxMode,user->parameters->alpha);CHKERRQ(ierr); */
      /* Write solution (fourier coefficients) to binary file */
      ierr = WriteModes(sol,user->maxMode,alphavals[i]);CHKERRQ(ierr);
      if (!i) {
      	/* Write corresponding coordinates solution for plotting*/
      	ierr = WriteCentroid(user->model,user->dm,sol);CHKERRQ(ierr);
      }
    }
  }
  /* Free workspace */
  ierr = VecDestroyVecs(user->maxMode,&sol);CHKERRQ(ierr);
  ierr = MatDestroy(&user->model->delta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->model->C);CHKERRQ(ierr);
  ierr = VecDestroy(&user->model->G);CHKERRQ(ierr);
  ierr = VecDestroy(&user->model->Vin);CHKERRQ(ierr);
  ierr = ISDestroy(&user->model->cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
  ierr = PetscFree(sol);CHKERRQ(ierr);
  ierr = PetscFree(user->model);CHKERRQ(ierr);
  ierr = PetscFree(user->meshOptions);CHKERRQ(ierr);
  ierr = PetscFree(user->parameters);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(AppCtx user)
{
  MPI_Comm		comm		= PETSC_COMM_WORLD;
  PetscBool		interpolate	= PETSC_FALSE;
  PetscReal		refinementLimit	= 0.2;
  PetscBool		debug		= PETSC_FALSE;
  PetscBool		writedistmesh	= PETSC_FALSE;
  PetscBool		usermesh	= PETSC_FALSE;
  PetscBool		holes		= PETSC_FALSE;
  PetscBool		saveDelta	= PETSC_FALSE;
  PetscInt		maxMode		= 1;
  PetscInt		maxIter		= 100;
  PetscScalar		b		= 0.0;
  PetscScalar		mu		= 1.0;
  PetscScalar           epsIn		= 1.0;
  PetscScalar           epsOut		= 9.0;
  PetscReal             alpha           = 1.0;
  PetscErrorCode	ierr;

  PetscFunctionBegin;
  PetscStrcpy(user->meshOptions->fileCell,"\0");
  PetscStrcpy(user->meshOptions->fileVertex,"\0");
  ierr = PetscOptionsBegin(comm,"","Problem specification","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit","The maximum allowed area of a triangle","",refinementLimit,&refinementLimit,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate","Generate intermediate elements","",interpolate,&interpolate,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-debug","Allow debugging information to be printed","",debug,&debug,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-usermesh","Load a user defined mesh","",usermesh,&usermesh,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-writedistmesh","Write mesh in Distmesh format","",writedistmesh,&writedistmesh,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-holes","Solve the problem with holes","",holes,&holes,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-savedelta","Should the operators be saved?","",saveDelta,&saveDelta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-fV","File containing vertex list\n","",user->meshOptions->fileVertex,user->meshOptions->fileVertex,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-fC","File containing cell list\n","",user->meshOptions->fileCell,user->meshOptions->fileCell,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-maxmode","Number of modes to solve","",maxMode,&maxMode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-maxIter","Maximum number of fixed point iterations","",maxIter,&maxIter,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-mu","Value for mu","",mu,&mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-epsIn","Value for eps inside holes","",epsIn,&epsIn,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-epsOut","Value for eps outside holes","",epsOut,&epsOut,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Value for alpha","",alpha,&alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-b","Nonlinearity strength","",b,&b,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user->comm	  = comm;
  user->debug	  = debug;
  user->maxMode   = maxMode;
  user->maxIter   = maxIter;
  user->saveDelta = saveDelta;

  user->model->solverType  = (user->maxMode > 1) ? NonlinearSolver : LinearSolver;
  user->model->problemType = holes ? HolesPermittivity : PlainPermittivity;

  user->meshOptions->refinementLimit = refinementLimit;
  user->meshOptions->interpolate     = interpolate;
  user->meshOptions->usermesh	     = usermesh;
  user->meshOptions->writedistmesh   = writedistmesh;
  user->meshOptions->meshtype	     = (user->meshOptions->usermesh) ? CreateMeshFromFile : CreateMeshTriangle;

  user->parameters->mu	   = mu;
  user->parameters->epsIn  = epsIn;
  user->parameters->epsOut = epsOut;
  user->parameters->alpha   = alpha;
  user->parameters->omega  = alpha*2.0*PETSC_PI;
  user->parameters->b	   = b;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMeshTriangle"
PetscErrorCode CreateMeshTriangle(AppCtx user)
{
  MPI_Comm		comm		= user->comm;
  PetscBool             interpolate     = user->meshOptions->interpolate;
  PetscReal             refinementLimit = user->meshOptions->refinementLimit;
  DM                    dmEdges;
  IS			iscopy;
  PetscSection          coordSection;
  Vec                   coordinates;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateBoxMesh(comm,2,interpolate,&user->dm);CHKERRQ(ierr);
  if (user->dm) {
    DM		refinedMesh     = NULL;
    DM		distributedMesh = NULL;
    PetscInt	cStart,cEnd;
    ierr = DMPlexGetHeightStratum(user->dm,0,&cStart,&cEnd);CHKERRQ(ierr);

    ierr = DMPlexSetRefinementUniform(user->dm,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementLimit(user->dm,refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(user->dm,PETSC_COMM_WORLD,&refinedMesh);CHKERRQ(ierr);

    if (refinedMesh) {
      ierr     = DMDestroy(&user->dm);CHKERRQ(ierr);
      user->dm = refinedMesh;
      PetscPrintf(PETSC_COMM_WORLD,"Mesh refinement done\n");
    }
    if (user->meshOptions->writedistmesh) {
        ierr = WriteCoords(user->dm);CHKERRQ(ierr);
	ierr = WriteCells(user->dm);CHKERRQ(ierr);
    }
    ierr   = DMPlexDistribute(user->dm,"chaco",1,&distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
      user->dm  = distributedMesh;
      PetscPrintf(PETSC_COMM_WORLD,"Mesh distribution done\n");
    }
  }
  ierr = DMPlexGetCellNumbering(user->dm,&iscopy);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(user->dm,&coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(user->dm,&coordinates);CHKERRQ(ierr);
  ierr = DMPlexInterpolate(user->dm,&dmEdges);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(dmEdges,coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmEdges,coordinates);CHKERRQ(ierr);
  ierr = ISDuplicate(iscopy,&user->model->cellIS);CHKERRQ(ierr);
  ierr = ISCopy(iscopy,user->model->cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
  user->dm = dmEdges;
  ierr = DMSetFromOptions(user->dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeParameters"
PetscErrorCode ComputeParameters(AppCtx user, Model model, Parameters params)
{
  MPI_Comm		 comm		 = user->comm;
  DM			 dm		 = user->dm;
  IS			 cellIS		 = model->cellIS;
  PetscInt		 cStart,cEnd;
  const PetscInt	*isArr;
  PetscSection		 cellSection;
  PetscInt		 i,numLocalCells = 0;
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&cellSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(cellSection,cStart,cEnd);CHKERRQ(ierr);
  for (i=cStart; i<cEnd; ++i) {
    ierr = PetscSectionSetDof(cellSection,i,sizeof(CellInfo)/sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(cellSection);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,cellSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellSection);CHKERRQ(ierr);

  ierr = ISGetIndices(cellIS,&isArr);CHKERRQ(ierr);
  for (i=0; i<(cEnd-cStart); ++i) if (isArr[i] >= 0) numLocalCells++;
  ierr = ISRestoreIndices(cellIS,&isArr);CHKERRQ(ierr);

  /* Create C,G and L vectors. */
  ierr = VecCreate(comm,&model->C);CHKERRQ(ierr);
  ierr = VecSetSizes(model->C,numLocalCells,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(model->C);CHKERRQ(ierr);
  ierr = VecDuplicate(model->C,&model->G);CHKERRQ(ierr);
  ierr = VecDuplicate(model->C,&model->Vin);CHKERRQ(ierr);

  /* Compute the cell properties */
  ierr = DMGetLocalVector(dm,&model->cellGeom);CHKERRQ(ierr);
  ierr = ComputeCellInfo(params,model,dm,model->cellGeom,model->C,cellIS,cStart,cEnd);CHKERRQ(ierr);

  /* Compute delta/vin/G */
  ierr = MatCreate(comm,&model->delta);CHKERRQ(ierr);
  ierr = MatSetSizes(model->delta,numLocalCells,numLocalCells,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(model->delta);CHKERRQ(ierr);
  ierr = ComputeDelta(params,model,dm,model->G,model->cellGeom,model->Vin,model->C);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&model->cellGeom);CHKERRQ(ierr);
  user->numLocalCells = numLocalCells;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeCellGeom"
PetscErrorCode ComputeCellGeom(const PetscScalar *c,CellInfo *cv)
{
  PetscFunctionBegin;
  cv->centroid[0] = 1.0/3.0*(c[0]+c[2]+c[4]);
  cv->centroid[1] = 1.0/3.0*(c[1]+c[3]+c[5]);
  cv->area  = 0.5 * PetscAbsScalar(c[0]*(c[3]-c[5]) + c[2]*(c[5]-c[1]) + c[4]*(c[1]-c[3]));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeCellInfo"
PetscErrorCode ComputeCellInfo(Parameters params,Model model,DM dm,Vec cell,Vec C,IS cellIS,PetscInt cStart,PetscInt cEnd)
{
  PetscScalar		*cellArr,*capArr;
  PetscScalar		 eps;
  const PetscInt	*isArr;
  PetscInt		 i,numvals = 0;
  Vec			 coordinates;
  PetscSection		 coordSection;
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm,&coordSection);CHKERRQ(ierr);
  ierr = VecGetArray(cell,&cellArr);CHKERRQ(ierr);
  ierr = VecGetArray(C,&capArr);CHKERRQ(ierr);
  ierr = ISGetIndices(cellIS,&isArr);CHKERRQ(ierr);
  for(i=cStart; i<cEnd; ++i){
    PetscScalar	*coords = NULL;
    CellInfo		*cv;
    ierr = DMPlexVecGetClosure(dm,coordSection,coordinates,i,NULL,&coords);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRef(dm,i,cellArr,&cv);CHKERRQ(ierr);
    ierr = ComputeCellGeom(coords,cv);CHKERRQ(ierr);
    if(isArr[i] >= 0) {
      eps = (*model->problemType)(cv,params->epsIn,params->epsOut);
      capArr[numvals++] = eps*(cv->area);
    }
    ierr = DMPlexVecRestoreClosure(dm,coordSection,coordinates,i,NULL,&coords);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(cellIS,&isArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(C,&capArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(cell,&cellArr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeDelta"
PetscErrorCode ComputeDelta(Parameters params,Model model,DM dm,Vec G,Vec cellGeom,Vec Vin,Vec C)
{
  Mat			 delta	  = model->delta;
  IS			 cellIS	  = model->cellIS;
  PetscInt		 cStart,cEnd,i,j,k,supSize;
  PetscInt		 locPoint,rowID;
  PetscInt               colID[4] = {0};
  const PetscInt	*isArr;
  PetscScalar           *cellArr,*gArr,*vinArr,*cArr;
  PetscSection		 coordSection;
  Vec			 coordinates;
  PetscScalar		 edgeLength,dist;
  PetscReal		 TOL	  = 1e-10; /* To check for vertical/horizontal boundary edges */
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = ISGetIndices(cellIS,&isArr);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm,&coordSection);CHKERRQ(ierr);
  ierr = VecGetArray(cellGeom,&cellArr);CHKERRQ(ierr);
  ierr = VecGetArray(G,&gArr);CHKERRQ(ierr);
  ierr = VecGetArray(C,&cArr);CHKERRQ(ierr);
  ierr = VecGetArray(Vin,&vinArr);CHKERRQ(ierr);
  locPoint = 0;
  for (i=cStart; i<cEnd; ++i) {
    if(isArr[i] < 0) continue;	/* Ghost Cell */
    PetscInt		 numVals    = 0;
    PetscInt		 numCone;
    PetscScalar values[4]	    = {0.0};
    PetscScalar		*coords	    = NULL;
    const PetscInt	*edgePoints = NULL;
    const CellInfo	*cv1;
    rowID			    = isArr[i];	/* Global ID's */
    colID[numVals++]		    = isArr[i];

    ierr = DMPlexVecGetClosure(dm,coordSection,coordinates,i,NULL,&coords);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRef(dm,i,cellArr,&cv1);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm,i,&edgePoints);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm,i,&numCone);CHKERRQ(ierr);
    for (j=0; j<numCone; ++j) {
      const PetscInt	*cellPoints = NULL;
      const CellInfo	*cv2;
      PetscScalar	*edgeCoords = NULL;
      boundary		 bound	    = INTERIOR;
      PetscScalar	 L;

      ierr = DMPlexVecGetClosure(dm,coordSection,coordinates,edgePoints[j],NULL,&edgeCoords);CHKERRQ(ierr);
      edgeLength = LengthOfEdge(edgeCoords);
      ierr = DMPlexGetSupportSize(dm,edgePoints[j],&supSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm,edgePoints[j],&cellPoints);CHKERRQ(ierr);
      if(supSize > 1) {
      	PetscInt neighCell = cellPoints[0] == i ? cellPoints[1] : cellPoints[0];
      	ierr = DMPlexPointLocalRead(dm,i,cellArr,&cv1);CHKERRQ(ierr);
      	ierr = DMPlexPointLocalRead(dm,neighCell,cellArr,&cv2);CHKERRQ(ierr);
      	dist = InteriorDist(cv1,cv2);
	L = params->mu*(dist/edgeLength);
      	values[numVals] = -1.0/L; // Negative because it is off diagonal
      	colID[numVals++] = (isArr[neighCell] < 0) ? -(isArr[neighCell]+1) : isArr[neighCell];
      } else {
      	// Do boundary computations.
      	ierr = DMPlexPointLocalRead(dm,i,cellArr,&cv1);CHKERRQ(ierr);
      	if(PetscAbsReal(PetscRealPart(edgeCoords[0]-edgeCoords[2])) < TOL){ /* Vertical edge */
	  dist = 2.0*BoundaryDist(cv1,edgeCoords);
      	  bound = (PetscRealPart(edgeCoords[0]) < PetscRealPart(cv1->centroid[0])) ? LEFT : RIGHT;
	  L = params->mu*(dist/edgeLength);
	  if(bound==LEFT) {
	    values[0] += 1.0/L; // Positive sign since it only comes into the diagonal part.
	    vinArr[locPoint] = ComputeForcing(edgeCoords,L);
	  } else {
	    gArr[locPoint] = PetscSqrtComplex(edgeLength*dist*cArr[locPoint]/(L*cv1->area));
	  }
      	} else { // End right/left
	  dist = 2.0*BoundaryDist(cv1,edgeCoords);
	  L = params->mu*(dist/edgeLength);
	  gArr[locPoint] = PetscSqrtComplex(edgeLength*dist*cArr[locPoint]/(L*cv1->area));
	}//End bottom/top
      }
      ierr = DMPlexVecRestoreClosure(dm,coordSection,coordinates,edgePoints[j],NULL,&edgeCoords);CHKERRQ(ierr);
    } // End j
    locPoint++;
    ierr = DMPlexVecRestoreClosure(dm,coordSection,coordinates,i,NULL,&coords);CHKERRQ(ierr);
    // compute diagonal L part.
    for(k=1; k<numVals; ++k) values[0] += -values[k];
    ierr = MatSetValues(delta,1,&rowID,numVals,colID,values,INSERT_VALUES);CHKERRQ(ierr);
  } // End i
  ierr = MatAssemblyBegin(delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(Vin,&vinArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&gArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(C,&cArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(cellGeom,&cellArr);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cellIS,&isArr);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LinearSolver"
PetscErrorCode LinearSolver(Mat delta,AppCtx user,Vec *sol)
{
  Model			model  = user->model;
  KSP			ksp;
  Mat			delta0;
  Vec			work;
  PetscScalar		omega  = user->parameters->omega;
  PetscScalar		iOmega = PETSC_i*omega;	// Note: Omega already includes the 2*pi here.
  PetscScalar		sOmega = -1.0*omega*omega;
  PetscViewer viewer;
  PetscErrorCode	ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate(delta,MAT_COPY_VALUES,&delta0);CHKERRQ(ierr);
  ierr = MatGetVecs(delta,sol,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(sol[0],&work);CHKERRQ(ierr);
  /* G part */
  ierr = VecCopy(model->G,work);CHKERRQ(ierr);
  ierr = VecScale(work,iOmega);CHKERRQ(ierr);
  ierr = MatDiagonalSet(delta0,work,ADD_VALUES);CHKERRQ(ierr);
  /* C part */
  ierr = VecCopy(model->C,work);CHKERRQ(ierr);
  ierr = VecScale(work,sOmega);CHKERRQ(ierr);
  ierr = MatDiagonalSet(delta0,work,ADD_VALUES);CHKERRQ(ierr);
  /* Solve system */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,delta0,delta0,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ///
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"delta.bin",FILE_MODE_WRITE,&viewer);
  ierr = MatView(delta0,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"rhs.bin",FILE_MODE_WRITE,&viewer);
  ierr = VecView(model->Vin,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ///
  ierr = KSPSolve(ksp,model->Vin,sol[0]);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&delta0);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeForcing"
PetscScalar ComputeForcing(const PetscScalar *edge,PetscScalar L)
{
  double	start, end;
  double	result, error;
  gsl_function	F;
  double	alpha = 150.0;
  size_t	limit = 1000;

  PetscFunctionBegin;
  start = PetscRealPart(edge[1]-edge[3])< 0.0 ? PetscRealPart(edge[1]) : PetscRealPart(edge[3]);
  end = PetscRealPart(edge[1]-edge[3])> 0.0 ? PetscRealPart(edge[1]) : PetscRealPart(edge[3]);
  F.function = &forcfunc;
  F.params = &alpha;
  gsl_integration_qng(&F, start, end, 1e-8, 1e-8, &result, &error, &limit);
  PetscScalar finalResult = (PetscScalar) result/(end-start);
  finalResult /= L;
  PetscFunctionReturn(finalResult);
}


#undef __FUNCT__
#define __FUNCT__ "NonlinearSolver"
PetscErrorCode NonlinearSolver(Mat delta,AppCtx user,Vec *sol)
{
  PetscInt		i;
  PetscInt		maxMode = user->maxMode;
  PetscScalar		omega	= user->parameters->omega;
  PetscScalar		iOmega	= PETSC_i*omega;
  PetscScalar		sOmega	= -1.0*omega*omega;
  PetscScalar           kiOmega,ksqsOmega;
  Vec			work;
  KSP			ksp[user->maxMode];
  Mat			del[user->maxMode];
  char			filename[PETSC_MAX_PATH_LEN];
  PetscViewer		viewer;
  PetscErrorCode	ierr;

  PetscFunctionBegin;
  ierr = MatGetVecs(delta,PETSC_NULL,&work);CHKERRQ(ierr);
  for (i=0; i<maxMode; ++i) { /* 0 corresponds to mode 1 */
    kiOmega = ((PetscScalar)(i+1))*iOmega;
    ksqsOmega = ((PetscScalar)((i+1)*(i+1)))*sOmega;
    ierr = MatDuplicate(delta,MAT_COPY_VALUES,&del[i]);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(work,kiOmega,ksqsOmega,0.0,user->model->G,user->model->C);CHKERRQ(ierr);
    ierr = MatDiagonalSet(del[i],work,ADD_VALUES);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp[i]);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp[i],del[i],del[i],SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp[i],PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp[i]);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp[i]);CHKERRQ(ierr);
  }

  /* Save delta matrices for slepc */
  if (user->saveDelta) {
    for (i=0; i<maxMode; ++i) {
      ierr = PetscSNPrintf(filename,sizeof(filename),"Delta_%d.bin",i+1);CHKERRQ(ierr);
      PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);
      ierr = MatView(del[i],viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }

  /* Call nonlinear solver */
  ierr = FixedPointSolver(ksp,user,sol);CHKERRQ(ierr);

  for (i=0; i<maxMode; ++i) {
    KSPDestroy(&ksp[i]);
    MatDestroy(&del[i]);
  }
  VecDestroy(&work);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FixedPointSolver"
PetscErrorCode FixedPointSolver(KSP ksp[],AppCtx user,Vec *osol)
{
  PetscInt		 i,k,l;
  PetscInt		 maxIter    = user->maxIter;
  PetscInt		 maxMode    = user->maxMode;
  Vec                    Vin = user->model->Vin;
  Vec			 convolve,rhs,work1,work2;
  Vec			*harmonics,*harmonicsOld;
  PetscReal		 TOL	    = 1.0e-10; /* Fixed point iteration tolerance */
  PetscReal		 modeNorm,diff;
  PetscScalar		 scaleFac;
  Parameters             params = user->parameters;
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(Vin,&convolve);CHKERRQ(ierr);
  ierr = VecDuplicate(Vin,&rhs);CHKERRQ(ierr);
  ierr = VecDuplicate(Vin,&work1);CHKERRQ(ierr);
  ierr = VecDuplicate(Vin,&work2);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Vin,maxMode,&harmonics);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Vin,maxMode,&harmonicsOld);CHKERRQ(ierr);

  for (i=0; i<maxMode; ++i){
    ierr = VecSet(harmonics[i],0.0);CHKERRQ(ierr);
    ierr = VecSet(harmonicsOld[i],0.0);CHKERRQ(ierr);
  }
  for (i=0; i<maxIter; ++i) {
    for (k=1; k<maxMode+1; ++k) {
      scaleFac = params->b*params->omega*params->omega;
      ierr = VecSet(rhs,0.0);CHKERRQ(ierr);
      if (k==1) VecCopy(Vin,rhs);
      ierr = VecSet(convolve,0.0);CHKERRQ(ierr);
      for (l=-maxMode; l<maxMode+1; ++l) {
  	if (l>0) VecCopy(harmonics[l-1],work1);
  	else if (l<0) {
  	  ierr = VecCopy(harmonics[-l-1],work1);CHKERRQ(ierr);
	  ierr = VecConjugate(work1);CHKERRQ(ierr);
  	} else VecSet(work1,0.0);
  	if (((k-l) > 0) && ((k-l) <= maxMode)) VecCopy(harmonics[k-l-1],work2);
  	else if (((k-l) < 0) && ((k-l)>= -maxMode)) {
  	  ierr = VecCopy(harmonics[l-k-1],work2);CHKERRQ(ierr);
	  ierr = VecConjugate(work2);CHKERRQ(ierr);
  	} else VecSet(work2,0.0);
  	ierr = VecPointwiseMult(work1,work1,work2);CHKERRQ(ierr);
  	ierr = VecAXPY(convolve,1.0,work1);CHKERRQ(ierr);
      }// End l
      ierr = VecPointwiseMult(convolve,convolve,user->model->C);CHKERRQ(ierr);
      scaleFac *= (PetscScalar) (k*k);
      ierr = VecScale(convolve,scaleFac);CHKERRQ(ierr);
      ierr = VecAXPY(rhs,-1.0,convolve);CHKERRQ(ierr);

      ierr = KSPSolve(ksp[k-1],rhs,harmonicsOld[k-1]);CHKERRQ(ierr);
    }// End k
    // Check for convergence.
    diff = 0.0;
    for (l=0; l<maxMode; ++l) {
      ierr = VecWAXPY(work1,-1.0,harmonicsOld[l],harmonics[l]);CHKERRQ(ierr);
      ierr = VecNorm(work1,NORM_INFINITY,&modeNorm);CHKERRQ(ierr);
      diff += modeNorm;
      ierr = VecCopy(harmonicsOld[l],harmonics[l]);CHKERRQ(ierr);
    }
    if (diff < TOL) {
      ierr = VecNorm(harmonics[0],NORM_INFINITY,&modeNorm);CHKERRQ(ierr);
      if (modeNorm > 1.e+2) {
	PetscPrintf(PETSC_COMM_WORLD,"Fixed-point iteration diverged\n");
	user->diverged = PETSC_TRUE;
      }
      PetscPrintf(PETSC_COMM_WORLD,"Number of iterations is %d\n",i);
      break;
    }
  }// End i
  for (l=0; l<maxMode; ++l) {
    VecCopy(harmonics[l],osol[l]);
  }
  VecDestroy(&convolve);
  VecDestroy(&rhs);
  VecDestroy(&work1);
  VecDestroy(&work2);
  VecDestroyVecs(user->maxMode,&harmonics);
  VecDestroyVecs(user->maxMode,&harmonicsOld);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WriteModes"
PetscErrorCode WriteModes(Vec *sol,PetscInt maxMode,PetscReal alphaval)
{
  PetscViewer viewer;
  PetscInt i;
  char filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<maxMode; ++i) {
    ierr = PetscSNPrintf(filename,sizeof(filename),"Mode_%d_%g.bin",i+1,alphaval);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(sol[i],viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WriteCentroid"
PetscErrorCode WriteCentroid(Model model,DM dm,Vec *osol)
{
  IS			 cellIS = model->cellIS;
  Vec			 sol	= osol[0];
  PetscScalar		*cellArr,*xArr,*yArr,*aArr;
  Vec			 cellGeom;
  Vec			 x,y,area;
  PetscInt		 i,ind=0;
  PetscInt		 lSize;
  const PetscInt	*isArr;
  PetscViewer		 viewer;
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(sol,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(sol,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(sol,&area);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&cellGeom);CHKERRQ(ierr);
  ierr = VecGetLocalSize(cellGeom,&lSize);CHKERRQ(ierr);
  ierr = VecGetArray(cellGeom,&cellArr);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xArr);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yArr);CHKERRQ(ierr);
  ierr = VecGetArray(area,&aArr);CHKERRQ(ierr);
  ierr = ISGetIndices(cellIS,&isArr);CHKERRQ(ierr);
  for(i=0; i<lSize/3; ++i){
    if (isArr[i] >= 0){
      xArr[ind] = cellArr[3*i];
      yArr[ind] = cellArr[3*i+1];
      aArr[ind++] = cellArr[3*i+2];
    }
  }
  ierr = ISRestoreIndices(cellIS,&isArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(cellGeom,&cellArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yArr);CHKERRQ(ierr);
  ierr = VecRestoreArray(area,&aArr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&cellGeom);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"x.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"y.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(y,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"area.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(area,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&area);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMeshFromFile"
PetscErrorCode CreateMeshFromFile(AppCtx user)
{
  PetscInt		 numVertices,numCells;
  PetscMPIInt		 rank;
  double		*coordinates;
  int			*cellList;
  DM			 dmEdges,dmDist;
  IS			 iscopy;
  PetscSection		 coordSection;
  Vec			 coordinateVec;
  PetscInt		 dim=2,numCorners=3;
  const char		*vname	     = user->meshOptions->fileVertex;
  const char		*cname	     = user->meshOptions->fileCell;
  char partitioner[2048];
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  numVertices = GetLineCount(vname);
  numCells    = GetLineCount(cname);
  ierr = PetscStrcpy(partitioner,"chaco");CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = readMesh(PETSC_COMM_WORLD,cname,vname,&numCells,&numVertices,&cellList,&coordinates);CHKERRQ(ierr);
  if(rank) {numCells=0;numVertices=0;}
  ierr = DMPlexCreateFromCellList(PETSC_COMM_WORLD,dim,numCells,numVertices,numCorners,PETSC_FALSE,cellList,dim,coordinates,&user->dm);CHKERRQ(ierr);
  if (user->meshOptions->writedistmesh) {
    ierr = WriteCoords(user->dm);CHKERRQ(ierr);
    ierr = WriteCells(user->dm);CHKERRQ(ierr);
  }
  ierr = DMPlexDistribute(user->dm,partitioner,1,&dmDist);CHKERRQ(ierr);
  if (dmDist) {
      ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
      user->dm = dmDist;
      ierr = DMSetFromOptions(user->dm);CHKERRQ(ierr);
  }
  ierr = DMPlexGetCellNumbering(user->dm,&iscopy);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(user->dm,&coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(user->dm,&coordinateVec);CHKERRQ(ierr);
  ierr = DMPlexInterpolate(user->dm,&dmEdges);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(dmEdges,coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmEdges,coordinateVec);CHKERRQ(ierr);
  ierr = ISDuplicate(iscopy,&user->model->cellIS);CHKERRQ(ierr);
  ierr = ISCopy(iscopy,user->model->cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
  user->dm = dmEdges;
  ierr = DMSetFromOptions(user->dm);CHKERRQ(ierr);
  ierr = PetscFree(coordinates);CHKERRQ(ierr);
  ierr = PetscFree(cellList);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "readMesh"
PetscErrorCode readMesh(MPI_Comm comm, const char cname[],const char vname[],PetscInt *numC,PetscInt *numV,int **cell,double **vert)
{
  FILE			*fdc, *fdv;
  int			*cellList    = NULL;
  double		*coordinates = NULL;
  PetscMPIInt		 rank;
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    /* Allocate space for the cells and coordinates */
    ierr = PetscMalloc(sizeof(int)*(*numC)*3, &cellList);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(double)*(*numV)*2, &coordinates);CHKERRQ(ierr);

    fdc = fopen(cname,"r");
    /* Read cell list file */
    if (!fdc) {
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Error opening cell list file %s\n", cname);
    } else {
      PetscInt i = 0;
      int tempc;

      while (fscanf(fdc,"%d",&tempc) != EOF) {
        cellList[i++] = tempc;	/* NOTE: Starts from 0!! */
      }
      fclose(fdc);
      ierr = PetscPrintf(comm, "reading cell list successful\n");CHKERRQ(ierr);
    }
    /* Read vertex list file */
    fdv = fopen(vname,"r");
    if (!fdv) {
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Error opening vertex list file %s\n", vname);
    } else {
      PetscInt i = 0;
      double   tempv;
      while (fscanf(fdv,"%lf",&tempv) != EOF)
        coordinates[i++] = tempv;
      fclose(fdv);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"reading vertex list successful\n");CHKERRQ(ierr);
    }
  } else {
    *numC = 0;
    *numV = 0;
  }
  *cell = cellList;
  *vert = coordinates;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "WriteCoords"
PetscErrorCode WriteCoords(DM dm)
{
  PetscSection		 coordSection;
  Vec			 coordinates;
  PetscInt		 cStart,cEnd,pStart,pEnd;
  PetscInt		 i;
  FILE			*fp;
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm,&coordSection);CHKERRQ(ierr);
  fp   = fopen("pointlist.txt","w");
  for(i=pStart; i<pEnd; ++i) {
    PetscScalar *coords = NULL;

    ierr = DMPlexVecGetClosure(dm,coordSection,coordinates,i,NULL,&coords);CHKERRQ(ierr);
    fprintf(fp,"%.10e\t%.10e\n",PetscRealPart(coords[0]),PetscRealPart(coords[1]));
    ierr = DMPlexVecRestoreClosure(dm,coordSection,coordinates,i,NULL,&coords);CHKERRQ(ierr);
  }
  fclose(fp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WriteCells"
PetscErrorCode WriteCells(DM dm)
{
  PetscSection		 coordSection;
  Vec			 coordinates;
  PetscInt		 cStart,cEnd,pStart,pEnd;
  PetscInt		 i;
  FILE			*fp;
  PetscErrorCode	 ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm,&coordSection);CHKERRQ(ierr);
  fp   = fopen("celllist.txt","w");
  for(i=cStart; i<cEnd; ++i) {
    const PetscInt *points = NULL;

    ierr = DMPlexGetCone(dm,i,&points);CHKERRQ(ierr);
    fprintf(fp,"%d\t%d\t%d\n",points[0],points[1],points[2]);
  }
  fclose(fp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WriteSolVTK"
PetscErrorCode WriteSolVTK(DM dm,Vec *sol,PetscInt maxmode,PetscReal alpha)
{
  DM                    cdm;
  char			modeName[PETSC_MAX_PATH_LEN];
  PetscSection		section;
  Vec			coordinates,local;
  PetscInt		i,cStart,cEnd;
  PetscViewer		viewer;
  PetscErrorCode	ierr;

  PetscFunctionBegin;
  ierr = DMClone(dm,&cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(cdm,coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm,&section);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(cdm,section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(cdm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&section);CHKERRQ(ierr);
  for(i=cStart;i<cEnd;++i){
    ierr = PetscSectionSetDof(section,i,1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(cdm,section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = DMSetFromOptions(cdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(cdm,&local);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(cdm,&local);CHKERRQ(ierr);

  ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,"mysol.vtk",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  for(int i=0;i<maxmode;++i) {
    PetscSNPrintf(modeName,sizeof(modeName),"Mode_%d",i+1);
    PetscObjectSetName((PetscObject)sol[i],modeName);
    PetscObjectReference((PetscObject)sol[i]);
    VecView(sol[i],viewer);
  }
  PetscViewerDestroy(&viewer);
  DMDestroy(&cdm);

  PetscFunctionReturn(0);
}
