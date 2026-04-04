#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "HYPRE.h"
#include "HYPRE_struct_ls.h"

#define HYPRE_CALL(call) do { \
  HYPRE_Int ierr__ = (call); \
  if (ierr__) { \
    int rank__; MPI_Comm_rank(MPI_COMM_WORLD, &rank__); \
    fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n", rank__, __FILE__, __LINE__, (int)ierr__); \
    MPI_Abort(MPI_COMM_WORLD, ierr__); \
  } \
} while (0)

static void ParseArgs(int argc, char **argv,
                      int *nx, int *ny, int *nz,
                      char *solver,
                      double *tol, int *maxit,
                      int *print_level,
                      int *relax_type,
                      int *rap_type,
                      int *pc_maxit,
                      int *two_norm,
                      int *px, int *py, int *pz,
                      int *print_decomp)
{
  *nx = 64;
  *ny = 64;
  *nz = 64;
  strcpy(solver, "pfmg");
  *tol = 1e-10;
  *maxit = 100;
  *print_level = 1;
  *relax_type = 1;
  *rap_type = 1;
  *pc_maxit = 1;
  *two_norm = 1;
  *px = 0;
  *py = 0;
  *pz = 0;
  *print_decomp = 0;

  for (int a = 1; a < argc; ++a) {
    if (!strcmp(argv[a], "-nx") && a + 1 < argc) *nx = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-ny") && a + 1 < argc) *ny = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-nz") && a + 1 < argc) *nz = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-solver") && a + 1 < argc) {
      strncpy(solver, argv[++a], 31);
      solver[31] = '\0';
    }
    else if (!strcmp(argv[a], "-tol") && a + 1 < argc) *tol = atof(argv[++a]);
    else if (!strcmp(argv[a], "-maxit") && a + 1 < argc) *maxit = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-print") && a + 1 < argc) *print_level = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-relax-type") && a + 1 < argc) *relax_type = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-rap-type") && a + 1 < argc) *rap_type = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-pc-maxit") && a + 1 < argc) *pc_maxit = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-two-norm") && a + 1 < argc) *two_norm = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-px") && a + 1 < argc) *px = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-py") && a + 1 < argc) *py = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-pz") && a + 1 < argc) *pz = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-print-decomp") && a + 1 < argc) *print_decomp = atoi(argv[++a]);
  }
}

static void BlockPartition1D(int n, int p, int coord, int *ilo, int *ihi)
{
  int base = n / p;
  int rem  = n % p;
  int nloc = base + (coord < rem ? 1 : 0);
  int start0 = coord * base + (coord < rem ? coord : rem); /* 0-based */
  *ilo = start0 + 1;        /* global interior indexing starts at 1 */
  *ihi = start0 + nloc;
}

static void FillMatrixHostLocal(int gnx, int gny, int gnz,
                                int ilower[3], int iupper[3],
                                double hx, double hy, double hz,
                                double *vals7)
{
  const double cx = 1.0 / (hx * hx);
  const double cy = 1.0 / (hy * hy);
  const double cz = 1.0 / (hz * hz);

  const int nx = iupper[0] - ilower[0] + 1;
  const int ny = iupper[1] - ilower[1] + 1;
  const int nz = iupper[2] - ilower[2] + 1;

  size_t idx = 0;
  for (int k = 0; k < nz; ++k) {
    int K = ilower[2] + k;
    for (int j = 0; j < ny; ++j) {
      int J = ilower[1] + j;
      for (int i = 0; i < nx; ++i, ++idx) {
        int I = ilower[0] + i;
        vals7[7 * idx + 0] = 2.0 * cx + 2.0 * cy + 2.0 * cz;
        vals7[7 * idx + 1] = (I > 1)   ? -cx : 0.0;
        vals7[7 * idx + 2] = (I < gnx) ? -cx : 0.0;
        vals7[7 * idx + 3] = (J > 1)   ? -cy : 0.0;
        vals7[7 * idx + 4] = (J < gny) ? -cy : 0.0;
        vals7[7 * idx + 5] = (K > 1)   ? -cz : 0.0;
        vals7[7 * idx + 6] = (K < gnz) ? -cz : 0.0;
      }
    }
  }
}

static void FillRhsAndX0HostLocal(int ilower[3], int iupper[3],
                                  double hx, double hy, double hz,
                                  double *b, double *x0)
{
  const double pi = 3.141592653589793238462643383279502884;
  const int nx = iupper[0] - ilower[0] + 1;
  const int ny = iupper[1] - ilower[1] + 1;
  const int nz = iupper[2] - ilower[2] + 1;

  size_t idx = 0;
  for (int k = 0; k < nz; ++k) {
    int K = ilower[2] + k;
    for (int j = 0; j < ny; ++j) {
      int J = ilower[1] + j;
      for (int i = 0; i < nx; ++i, ++idx) {
        int I = ilower[0] + i;
        double x = (double)I * hx;
        double y = (double)J * hy;
        double z = (double)K * hz;
        double uex = sin(pi * x) * sin(pi * y) * sin(pi * z);
        double f = 3.0 * pi * pi * uex;
        b[idx]  = f;
        x0[idx] = 0.0;
      }
    }
  }
}

static double ReduceMax(double v)
{
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

static void ReduceError(double l2sum_local, double linf_local,
                        double *l2err, double *linf)
{
  double l2sum_global = 0.0;
  double linf_global = 0.0;
  MPI_Reduce(&l2sum_local, &l2sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&linf_local,  &linf_global,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  *l2err = sqrt(l2sum_global);
  *linf = linf_global;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int nx, ny, nz, maxit, print_level, relax_type, rap_type, pc_maxit, two_norm;
  int px, py, pz, print_decomp;
  char solver[32];
  double tol;
  ParseArgs(argc, argv, &nx, &ny, &nz, solver, &tol, &maxit,
            &print_level, &relax_type, &rap_type, &pc_maxit, &two_norm,
            &px, &py, &pz, &print_decomp);

  if (nx < 1 || ny < 1 || nz < 1) {
    if (rank == 0) fprintf(stderr, "nx, ny, nz must all be >= 1\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int dims[3];
  if (px > 0 || py > 0 || pz > 0) {
    dims[0] = (px > 0) ? px : 1;
    dims[1] = (py > 0) ? py : 1;
    dims[2] = (pz > 0) ? pz : 1;
    if (dims[0] * dims[1] * dims[2] != size) {
      if (rank == 0) {
        fprintf(stderr, "Requested decomposition px*py*pz = %d*%d*%d = %d, but MPI size = %d\n",
                dims[0], dims[1], dims[2], dims[0] * dims[1] * dims[2], size);
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  } else {
    dims[0] = 0;
    dims[1] = 0;
    dims[2] = 0;
    MPI_Dims_create(size, 3, dims);
  }

  if (dims[0] > nx || dims[1] > ny || dims[2] > nz) {
    if (rank == 0) {
      fprintf(stderr, "Decomposition %d x %d x %d is incompatible with grid %d x %d x %d\n",
              dims[0], dims[1], dims[2], nx, ny, nz);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int periods[3] = {0, 0, 0};
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
  if (cart_comm == MPI_COMM_NULL) {
    fprintf(stderr, "[%d] MPI_Cart_create failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int cart_rank = -1;
  int coords[3] = {0, 0, 0};
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

  int ilower[3], iupper[3];
  BlockPartition1D(nx, dims[0], coords[0], &ilower[0], &iupper[0]);
  BlockPartition1D(ny, dims[1], coords[1], &ilower[1], &iupper[1]);
  BlockPartition1D(nz, dims[2], coords[2], &ilower[2], &iupper[2]);

  int local_nx = iupper[0] - ilower[0] + 1;
  int local_ny = iupper[1] - ilower[1] + 1;
  int local_nz = iupper[2] - ilower[2] + 1;
  size_t local_N = (size_t)local_nx * (size_t)local_ny * (size_t)local_nz;
  size_t global_N = (size_t)nx * (size_t)ny * (size_t)nz;
  unsigned long long local_N_ull = (unsigned long long)local_N;
  unsigned long long max_local_N_ull = 0ULL;
  MPI_Allreduce(&local_N_ull, &max_local_N_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

  const double hx = 1.0 / (double)(nx + 1);
  const double hy = 1.0 / (double)(ny + 1);
  const double hz = 1.0 / (double)(nz + 1);

  if (rank == 0) {
    printf("hypre Struct 3D Poisson MMS (distributed CPU / host path)\n");
    printf("  sizeof(HYPRE_Real): %zu bytes\n", sizeof(HYPRE_Real));
    printf("  requested solver  : %s\n", solver);
    printf("  MPI decomposition : %d x %d x %d\n", dims[0], dims[1], dims[2]);
    fflush(stdout);
  }

  if (print_decomp) {
    printf("[rank %d] coords=(%d,%d,%d) ilower=(%d,%d,%d) iupper=(%d,%d,%d) local_N=%zu\n",
           rank, coords[0], coords[1], coords[2],
           ilower[0], ilower[1], ilower[2],
           iupper[0], iupper[1], iupper[2], local_N);
    fflush(stdout);
  }

  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST));

  double t_total0 = MPI_Wtime();
  double t0, t1;
  double t_grid_stencil = 0.0;
  double t_fill_mat = 0.0, t_fill_rhs = 0.0;
  double t_mat_set = 0.0, t_mat_assemble = 0.0;
  double t_rhs_set = 0.0, t_rhs_assemble = 0.0;
  double t_x0_set = 0.0, t_x0_assemble = 0.0;
  double t_solver_object_setup = 0.0;
  double t_solver_setup = 0.0, t_solver_solve = 0.0;
  double t_solution_fetch = 0.0, t_error_eval = 0.0;

  HYPRE_StructGrid grid;
  HYPRE_StructStencil stencil;
  HYPRE_StructMatrix A;
  HYPRE_StructVector b, x;

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructGridCreate(cart_comm, 3, &grid));
  HYPRE_CALL(HYPRE_StructGridSetExtents(grid, ilower, iupper));
  HYPRE_CALL(HYPRE_StructGridAssemble(grid));

  HYPRE_CALL(HYPRE_StructStencilCreate(3, 7, &stencil));
  int offsets[7][3] = {
    { 0, 0, 0},
    {-1, 0, 0}, { 1, 0, 0},
    { 0,-1, 0}, { 0, 1, 0},
    { 0, 0,-1}, { 0, 0, 1}
  };
  for (int s = 0; s < 7; ++s) {
    HYPRE_CALL(HYPRE_StructStencilSetElement(stencil, s, offsets[s]));
  }
  t1 = MPI_Wtime();
  t_grid_stencil = t1 - t0;

  double *Avals = (double*) malloc((size_t)7 * local_N * sizeof(double));
  double *bvals = (double*) malloc(local_N * sizeof(double));
  double *xvals = (double*) malloc(local_N * sizeof(double));
  if (!Avals || !bvals || !xvals) {
    fprintf(stderr, "[%d] Host allocation failed for local_N=%zu\n", rank, local_N);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  t0 = MPI_Wtime();
  FillMatrixHostLocal(nx, ny, nz, ilower, iupper, hx, hy, hz, Avals);
  t1 = MPI_Wtime();
  t_fill_mat = t1 - t0;

  t0 = MPI_Wtime();
  FillRhsAndX0HostLocal(ilower, iupper, hx, hy, hz, bvals, xvals);
  t1 = MPI_Wtime();
  t_fill_rhs = t1 - t0;

  HYPRE_CALL(HYPRE_StructMatrixCreate(cart_comm, grid, stencil, &A));
  HYPRE_CALL(HYPRE_StructMatrixInitialize(A));
  HYPRE_Int entries[7] = {0, 1, 2, 3, 4, 5, 6};
  HYPRE_CALL(HYPRE_StructMatrixSetSymmetric(A, 0));

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 7, entries, Avals));
  t1 = MPI_Wtime();
  t_mat_set = t1 - t0;

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixAssemble(A));
  t1 = MPI_Wtime();
  t_mat_assemble = t1 - t0;

  HYPRE_CALL(HYPRE_StructVectorCreate(cart_comm, grid, &b));
  HYPRE_CALL(HYPRE_StructVectorInitialize(b));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b, ilower, iupper, bvals));
  t1 = MPI_Wtime();
  t_rhs_set = t1 - t0;
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorAssemble(b));
  t1 = MPI_Wtime();
  t_rhs_assemble = t1 - t0;

  HYPRE_CALL(HYPRE_StructVectorCreate(cart_comm, grid, &x));
  HYPRE_CALL(HYPRE_StructVectorInitialize(x));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x, ilower, iupper, xvals));
  t1 = MPI_Wtime();
  t_x0_set = t1 - t0;
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorAssemble(x));
  t1 = MPI_Wtime();
  t_x0_assemble = t1 - t0;

  int iters = -1;
  double final_rel = -1.0;

  if (!strcmp(solver, "smg")) {
    HYPRE_StructSolver smg;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructSMGCreate(cart_comm, &smg));
    HYPRE_CALL(HYPRE_StructSMGSetMemoryUse(smg, 0));
    HYPRE_CALL(HYPRE_StructSMGSetMaxIter(smg, maxit));
    HYPRE_CALL(HYPRE_StructSMGSetTol(smg, tol));
    HYPRE_CALL(HYPRE_StructSMGSetPrintLevel(smg, print_level));
    HYPRE_CALL(HYPRE_StructSMGSetLogging(smg, 1));
    t1 = MPI_Wtime();
    t_solver_object_setup = t1 - t0;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructSMGSetup(smg, A, b, x));
    t1 = MPI_Wtime();
    t_solver_setup = t1 - t0;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructSMGSolve(smg, A, b, x));
    t1 = MPI_Wtime();
    t_solver_solve = t1 - t0;

    HYPRE_CALL(HYPRE_StructSMGGetNumIterations(smg, &iters));
    HYPRE_CALL(HYPRE_StructSMGGetFinalRelativeResidualNorm(smg, &final_rel));
    HYPRE_CALL(HYPRE_StructSMGDestroy(smg));
  }
  else if (!strcmp(solver, "pcg_pfmg")) {
    HYPRE_StructSolver pcg, pfmg;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPCGCreate(cart_comm, &pcg));
    HYPRE_CALL(HYPRE_StructPCGSetMaxIter(pcg, maxit));
    HYPRE_CALL(HYPRE_StructPCGSetTol(pcg, tol));
    HYPRE_CALL(HYPRE_StructPCGSetTwoNorm(pcg, two_norm));
    HYPRE_CALL(HYPRE_StructPCGSetPrintLevel(pcg, print_level));
    HYPRE_CALL(HYPRE_StructPCGSetLogging(pcg, 1));

    HYPRE_CALL(HYPRE_StructPFMGCreate(cart_comm, &pfmg));
    HYPRE_CALL(HYPRE_StructPFMGSetMaxIter(pfmg, pc_maxit));
    HYPRE_CALL(HYPRE_StructPFMGSetTol(pfmg, 0.0));
    HYPRE_CALL(HYPRE_StructPFMGSetPrintLevel(pfmg, 0));
    HYPRE_CALL(HYPRE_StructPFMGSetLogging(pfmg, 0));
    HYPRE_CALL(HYPRE_StructPFMGSetRelaxType(pfmg, relax_type));
    HYPRE_CALL(HYPRE_StructPFMGSetRAPType(pfmg, rap_type));
    HYPRE_CALL(HYPRE_StructPFMGSetZeroGuess(pfmg));

    HYPRE_CALL(HYPRE_StructPCGSetPrecond(
      pcg,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSolve,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSetup,
      pfmg));
    t1 = MPI_Wtime();
    t_solver_object_setup = t1 - t0;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPCGSetup(pcg, A, b, x));
    t1 = MPI_Wtime();
    t_solver_setup = t1 - t0;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPCGSolve(pcg, A, b, x));
    t1 = MPI_Wtime();
    t_solver_solve = t1 - t0;

    HYPRE_CALL(HYPRE_StructPCGGetNumIterations(pcg, &iters));
    HYPRE_CALL(HYPRE_StructPCGGetFinalRelativeResidualNorm(pcg, &final_rel));
    HYPRE_CALL(HYPRE_StructPFMGDestroy(pfmg));
    HYPRE_CALL(HYPRE_StructPCGDestroy(pcg));
  }
  else {
    HYPRE_StructSolver pfmg;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPFMGCreate(cart_comm, &pfmg));
    HYPRE_CALL(HYPRE_StructPFMGSetMaxIter(pfmg, maxit));
    HYPRE_CALL(HYPRE_StructPFMGSetTol(pfmg, tol));
    HYPRE_CALL(HYPRE_StructPFMGSetPrintLevel(pfmg, print_level));
    HYPRE_CALL(HYPRE_StructPFMGSetLogging(pfmg, 1));
    HYPRE_CALL(HYPRE_StructPFMGSetRelaxType(pfmg, relax_type));
    HYPRE_CALL(HYPRE_StructPFMGSetRAPType(pfmg, rap_type));
    t1 = MPI_Wtime();
    t_solver_object_setup = t1 - t0;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPFMGSetup(pfmg, A, b, x));
    t1 = MPI_Wtime();
    t_solver_setup = t1 - t0;

    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPFMGSolve(pfmg, A, b, x));
    t1 = MPI_Wtime();
    t_solver_solve = t1 - t0;

    HYPRE_CALL(HYPRE_StructPFMGGetNumIterations(pfmg, &iters));
    HYPRE_CALL(HYPRE_StructPFMGGetFinalRelativeResidualNorm(pfmg, &final_rel));
    HYPRE_CALL(HYPRE_StructPFMGDestroy(pfmg));
  }

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorGetBoxValues(x, ilower, iupper, xvals));
  t1 = MPI_Wtime();
  t_solution_fetch = t1 - t0;

  double l2sum_local = 0.0, linf_local = 0.0;
  const double pi = 3.141592653589793238462643383279502884;
  t0 = MPI_Wtime();
  size_t idx = 0;
  for (int k = 0; k < local_nz; ++k) {
    int K = ilower[2] + k;
    for (int j = 0; j < local_ny; ++j) {
      int J = ilower[1] + j;
      for (int i = 0; i < local_nx; ++i, ++idx) {
        int I = ilower[0] + i;
        double xx = (double)I * hx;
        double yy = (double)J * hy;
        double zz = (double)K * hz;
        double uex = sin(pi * xx) * sin(pi * yy) * sin(pi * zz);
        double err = fabs(xvals[idx] - uex);
        l2sum_local += err * err * hx * hy * hz;
        if (err > linf_local) linf_local = err;
      }
    }
  }
  t1 = MPI_Wtime();
  t_error_eval = t1 - t0;

  double l2err = 0.0, linf = 0.0;
  ReduceError(l2sum_local, linf_local, &l2err, &linf);

  double total_wall = MPI_Wtime() - t_total0;

  t_grid_stencil       = ReduceMax(t_grid_stencil);
  t_fill_mat           = ReduceMax(t_fill_mat);
  t_fill_rhs           = ReduceMax(t_fill_rhs);
  t_mat_set            = ReduceMax(t_mat_set);
  t_mat_assemble       = ReduceMax(t_mat_assemble);
  t_rhs_set            = ReduceMax(t_rhs_set);
  t_rhs_assemble       = ReduceMax(t_rhs_assemble);
  t_x0_set             = ReduceMax(t_x0_set);
  t_x0_assemble        = ReduceMax(t_x0_assemble);
  t_solver_object_setup= ReduceMax(t_solver_object_setup);
  t_solver_setup       = ReduceMax(t_solver_setup);
  t_solver_solve       = ReduceMax(t_solver_solve);
  t_solution_fetch     = ReduceMax(t_solution_fetch);
  t_error_eval         = ReduceMax(t_error_eval);
  total_wall           = ReduceMax(total_wall);

  if (rank == 0) {
    printf("hypre Struct 3D Poisson MMS (distributed CPU / host path)\n");
    printf("  grid              : %d x %d x %d interior points\n", nx, ny, nz);
    printf("  total unknowns    : %zu\n", global_N);
    printf("  local unknowns max: %llu\n", max_local_N_ull);
    printf("  MPI ranks         : %d\n", size);
    printf("  MPI decomposition : %d x %d x %d\n", dims[0], dims[1], dims[2]);
    printf("  solver            : %s\n", solver);
    printf("  relax_type        : %d\n", relax_type);
    printf("  rap_type          : %d\n", rap_type);
    if (!strcmp(solver, "pcg_pfmg")) {
      printf("  pc_maxit          : %d\n", pc_maxit);
      printf("  two_norm          : %d\n", two_norm);
    }
    printf("  hx,hy,hz          : %.12e %.12e %.12e\n", hx, hy, hz);
    printf("  iterations        : %d\n", iters);
    printf("  final relres      : %.12e\n", final_rel);
    printf("  L2 error          : %.12e\n", l2err);
    printf("  Linf error        : %.12e\n", linf);
    printf("timings (seconds, max over ranks)\n");
    printf("  grid_stencil       : %.6f\n", t_grid_stencil);
    printf("  fill_mat           : %.6f\n", t_fill_mat);
    printf("  fill_rhs           : %.6f\n", t_fill_rhs);
    printf("  matrix_set         : %.6f\n", t_mat_set);
    printf("  matrix_assemble    : %.6f\n", t_mat_assemble);
    printf("  rhs_set            : %.6f\n", t_rhs_set);
    printf("  rhs_assemble       : %.6f\n", t_rhs_assemble);
    printf("  x0_set             : %.6f\n", t_x0_set);
    printf("  x0_assemble        : %.6f\n", t_x0_assemble);
    printf("  solver_obj_setup   : %.6f\n", t_solver_object_setup);
    printf("  solver_setup       : %.6f\n", t_solver_setup);
    printf("  solver_solve       : %.6f\n", t_solver_solve);
    printf("  solution_fetch     : %.6f\n", t_solution_fetch);
    printf("  error_eval         : %.6f\n", t_error_eval);
    printf("  total_wall         : %.6f\n", total_wall);
    fflush(stdout);
  }

  HYPRE_CALL(HYPRE_StructMatrixDestroy(A));
  HYPRE_CALL(HYPRE_StructVectorDestroy(b));
  HYPRE_CALL(HYPRE_StructVectorDestroy(x));
  HYPRE_CALL(HYPRE_StructStencilDestroy(stencil));
  HYPRE_CALL(HYPRE_StructGridDestroy(grid));
  HYPRE_CALL(HYPRE_Finalize());

  free(Avals);
  free(bvals);
  free(xvals);

  MPI_Comm_free(&cart_comm);
  MPI_Finalize();
  return 0;
}
