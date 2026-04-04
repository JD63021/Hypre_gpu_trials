#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <cstddef>

#include <mpi.h>
#include <cuda_runtime.h>

extern "C" {
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
}

#define HYPRE_CALL(call) do { \
  HYPRE_Int ierr__ = (call); \
  if (ierr__) { \
    int rank__; MPI_Comm_rank(MPI_COMM_WORLD, &rank__); \
    std::fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n", rank__, __FILE__, __LINE__, (int)ierr__); \
    MPI_Abort(MPI_COMM_WORLD, ierr__); \
  } \
} while (0)

#define CUDA_CALL(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    int rank__; MPI_Comm_rank(MPI_COMM_WORLD, &rank__); \
    std::fprintf(stderr, "[%d] CUDA ERROR %s:%d: %s\n", rank__, __FILE__, __LINE__, cudaGetErrorString(err__)); \
    MPI_Abort(MPI_COMM_WORLD, (int)err__); \
  } \
} while (0)

static __global__ void FillMatrixKernel(
    int nx, int ny, int nz,
    double hx, double hy, double hz,
    double *vals7)
{
  std::size_t idx = (std::size_t)blockIdx.x * (std::size_t)blockDim.x + (std::size_t)threadIdx.x;
  std::size_t N = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
  if (idx >= N) return;

  int i = (int)(idx % (std::size_t)nx);
  std::size_t tmp = idx / (std::size_t)nx;
  int j = (int)(tmp % (std::size_t)ny);
  int k = (int)(tmp / (std::size_t)ny);

  const double cx = 1.0 / (hx * hx);
  const double cy = 1.0 / (hy * hy);
  const double cz = 1.0 / (hz * hz);

  vals7[7 * idx + 0] = 2.0 * cx + 2.0 * cy + 2.0 * cz;
  vals7[7 * idx + 1] = (i > 0)      ? -cx : 0.0;
  vals7[7 * idx + 2] = (i < nx - 1) ? -cx : 0.0;
  vals7[7 * idx + 3] = (j > 0)      ? -cy : 0.0;
  vals7[7 * idx + 4] = (j < ny - 1) ? -cy : 0.0;
  vals7[7 * idx + 5] = (k > 0)      ? -cz : 0.0;
  vals7[7 * idx + 6] = (k < nz - 1) ? -cz : 0.0;
}

static __global__ void FillRhsAndX0Kernel(
    int nx, int ny, int nz,
    double hx, double hy, double hz,
    double *b, double *x0)
{
  std::size_t idx = (std::size_t)blockIdx.x * (std::size_t)blockDim.x + (std::size_t)threadIdx.x;
  std::size_t N = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
  if (idx >= N) return;

  int i = (int)(idx % (std::size_t)nx);
  std::size_t tmp = idx / (std::size_t)nx;
  int j = (int)(tmp % (std::size_t)ny);
  int k = (int)(tmp / (std::size_t)ny);

  double x = (double)(i + 1) * hx;
  double y = (double)(j + 1) * hy;
  double z = (double)(k + 1) * hz;
  const double pi = 3.141592653589793238462643383279502884;
  double uex = sin(pi * x) * sin(pi * y) * sin(pi * z);
  double f = 3.0 * pi * pi * uex;

  b[idx]  = f;
  x0[idx] = 0.0;
}

static void ParseArgs(int argc, char **argv,
                      int &nx, int &ny, int &nz,
                      double &tol, int &maxit,
                      int &print, int &device,
                      int &pc_maxit, int &pc_relax_type, int &pc_rap_type,
                      int &two_norm)
{
  nx = 64;
  ny = 64;
  nz = 64;
  tol = 1e-10;
  maxit = 100;
  print = 1;
  device = 0;
  pc_maxit = 1;       // one PFMG V-cycle as preconditioner
  pc_relax_type = 1;  // weighted Jacobi
  pc_rap_type = 1;    // non-Galerkin 7-pt coarse operator
  two_norm = 1;

  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "-nx") && a + 1 < argc) nx = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-ny") && a + 1 < argc) ny = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-nz") && a + 1 < argc) nz = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-tol") && a + 1 < argc) tol = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-maxit") && a + 1 < argc) maxit = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-print") && a + 1 < argc) print = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-device") && a + 1 < argc) device = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-pc-maxit") && a + 1 < argc) pc_maxit = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-pc-relax-type") && a + 1 < argc) pc_relax_type = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-pc-rap-type") && a + 1 < argc) pc_rap_type = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-two-norm") && a + 1 < argc) two_norm = std::atoi(argv[++a]);
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    if (rank == 0) std::fprintf(stderr, "This demo currently supports exactly 1 MPI rank.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int nx, ny, nz, maxit, print, device, pc_maxit, pc_relax_type, pc_rap_type, two_norm;
  double tol;
  ParseArgs(argc, argv, nx, ny, nz, tol, maxit, print, device,
            pc_maxit, pc_relax_type, pc_rap_type, two_norm);

  CUDA_CALL(cudaSetDevice(device));
  CUDA_CALL(cudaFree(0));

  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, device));
  if (rank == 0) {
    std::printf("Running on \"%s\", major %d, minor %d, total memory %.2f GiB\n",
                prop.name, prop.major, prop.minor,
                (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("MaxSharedMemoryPerBlock %zu, MaxSharedMemoryPerBlockOptin %zu\n",
                prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);
    std::fflush(stdout);
  }

  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
  HYPRE_CALL(HYPRE_SetSpGemmUseVendor(0));
  HYPRE_CALL(HYPRE_SetUseGpuRand(1));

  const double hx = 1.0 / (double)(nx + 1);
  const double hy = 1.0 / (double)(ny + 1);
  const double hz = 1.0 / (double)(nz + 1);
  const std::size_t N = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;

  int ilower[3] = {1, 1, 1};
  int iupper[3] = {nx, ny, nz};

  double t_total0 = MPI_Wtime();
  double t0, t1;
  float t_kernel_mat_ms = 0.0f, t_kernel_rhs_ms = 0.0f;
  double t_grid_stencil = 0.0, t_mat_set = 0.0, t_mat_assemble = 0.0;
  double t_rhs_set = 0.0, t_rhs_assemble = 0.0, t_x0_set = 0.0, t_x0_assemble = 0.0;
  double t_pc_setup = 0.0, t_solver_setup = 0.0, t_solver_solve = 0.0, t_solution_fetch = 0.0, t_error_eval = 0.0;

  HYPRE_StructGrid grid;
  HYPRE_StructStencil stencil;
  HYPRE_StructMatrix A;
  HYPRE_StructVector b, x;

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid));
  HYPRE_CALL(HYPRE_StructGridSetExtents(grid, ilower, iupper));
  HYPRE_CALL(HYPRE_StructGridAssemble(grid));

  HYPRE_CALL(HYPRE_StructStencilCreate(3, 7, &stencil));
  int offsets[7][3] = {
    { 0, 0, 0},
    {-1, 0, 0},
    { 1, 0, 0},
    { 0,-1, 0},
    { 0, 1, 0},
    { 0, 0,-1},
    { 0, 0, 1}
  };
  for (int s = 0; s < 7; ++s) {
    HYPRE_CALL(HYPRE_StructStencilSetElement(stencil, s, offsets[s]));
  }
  t1 = MPI_Wtime();
  t_grid_stencil = t1 - t0;

  double *Avals = nullptr, *bvals = nullptr, *xvals = nullptr;
  CUDA_CALL(cudaMallocManaged((void**)&Avals, (std::size_t)7 * N * sizeof(double), cudaMemAttachGlobal));
  CUDA_CALL(cudaMallocManaged((void**)&bvals, N * sizeof(double), cudaMemAttachGlobal));
  CUDA_CALL(cudaMallocManaged((void**)&xvals, N * sizeof(double), cudaMemAttachGlobal));

  cudaEvent_t e0, e1;
  CUDA_CALL(cudaEventCreate(&e0));
  CUDA_CALL(cudaEventCreate(&e1));

  int block = 256;
  int gridk = (int)((N + (std::size_t)block - 1) / (std::size_t)block);

  CUDA_CALL(cudaEventRecord(e0));
  FillMatrixKernel<<<gridk, block>>>(nx, ny, nz, hx, hy, hz, Avals);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaEventRecord(e1));
  CUDA_CALL(cudaEventSynchronize(e1));
  CUDA_CALL(cudaEventElapsedTime(&t_kernel_mat_ms, e0, e1));

  CUDA_CALL(cudaEventRecord(e0));
  FillRhsAndX0Kernel<<<gridk, block>>>(nx, ny, nz, hx, hy, hz, bvals, xvals);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaEventRecord(e1));
  CUDA_CALL(cudaEventSynchronize(e1));
  CUDA_CALL(cudaEventElapsedTime(&t_kernel_rhs_ms, e0, e1));

  HYPRE_CALL(HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A));
  HYPRE_CALL(HYPRE_StructMatrixInitialize(A));
  int entries[7] = {0,1,2,3,4,5,6};
  HYPRE_CALL(HYPRE_StructMatrixSetSymmetric(A, 0));

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 7, entries, Avals));
  t1 = MPI_Wtime();
  t_mat_set = t1 - t0;

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixAssemble(A));
  t1 = MPI_Wtime();
  t_mat_assemble = t1 - t0;

  HYPRE_CALL(HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b));
  HYPRE_CALL(HYPRE_StructVectorInitialize(b));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b, ilower, iupper, bvals));
  t1 = MPI_Wtime();
  t_rhs_set = t1 - t0;
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorAssemble(b));
  t1 = MPI_Wtime();
  t_rhs_assemble = t1 - t0;

  HYPRE_CALL(HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x));
  HYPRE_CALL(HYPRE_StructVectorInitialize(x));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x, ilower, iupper, xvals));
  t1 = MPI_Wtime();
  t_x0_set = t1 - t0;
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorAssemble(x));
  t1 = MPI_Wtime();
  t_x0_assemble = t1 - t0;

  HYPRE_StructSolver pcg_solver;
  HYPRE_StructSolver pfmg_precond;
  int iters = -1;
  double final_rel = -1.0;

  HYPRE_CALL(HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &pfmg_precond));
  HYPRE_CALL(HYPRE_StructPFMGSetMaxIter(pfmg_precond, pc_maxit));
  HYPRE_CALL(HYPRE_StructPFMGSetTol(pfmg_precond, 0.0));
  HYPRE_CALL(HYPRE_StructPFMGSetPrintLevel(pfmg_precond, 0));
  HYPRE_CALL(HYPRE_StructPFMGSetLogging(pfmg_precond, 0));
  HYPRE_CALL(HYPRE_StructPFMGSetZeroGuess(pfmg_precond));
  HYPRE_CALL(HYPRE_StructPFMGSetRelaxType(pfmg_precond, pc_relax_type));
  HYPRE_CALL(HYPRE_StructPFMGSetRAPType(pfmg_precond, pc_rap_type));

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructPCGCreate(MPI_COMM_WORLD, &pcg_solver));
  HYPRE_CALL(HYPRE_StructPCGSetTol(pcg_solver, tol));
  HYPRE_CALL(HYPRE_StructPCGSetMaxIter(pcg_solver, maxit));
  HYPRE_CALL(HYPRE_StructPCGSetTwoNorm(pcg_solver, two_norm));
  HYPRE_CALL(HYPRE_StructPCGSetPrintLevel(pcg_solver, print));
  HYPRE_CALL(HYPRE_StructPCGSetLogging(pcg_solver, 1));
  HYPRE_CALL(HYPRE_StructPCGSetPrecond(
      pcg_solver,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSolve,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSetup,
      pfmg_precond));
  t1 = MPI_Wtime();
  t_pc_setup = t1 - t0;

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructPCGSetup(pcg_solver, A, b, x));
  t1 = MPI_Wtime();
  t_solver_setup = t1 - t0;

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructPCGSolve(pcg_solver, A, b, x));
  t1 = MPI_Wtime();
  t_solver_solve = t1 - t0;

  HYPRE_CALL(HYPRE_StructPCGGetNumIterations(pcg_solver, &iters));
  HYPRE_CALL(HYPRE_StructPCGGetFinalRelativeResidualNorm(pcg_solver, &final_rel));

  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructVectorGetBoxValues(x, ilower, iupper, xvals));
  CUDA_CALL(cudaDeviceSynchronize());
  t1 = MPI_Wtime();
  t_solution_fetch = t1 - t0;

  double l2sum = 0.0, linf = 0.0;
  t0 = MPI_Wtime();
  const double pi = 3.141592653589793238462643383279502884;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        std::size_t idx = ((std::size_t)k * (std::size_t)ny + (std::size_t)j) * (std::size_t)nx + (std::size_t)i;
        double xx = (double)(i + 1) * hx;
        double yy = (double)(j + 1) * hy;
        double zz = (double)(k + 1) * hz;
        double uex = std::sin(pi * xx) * std::sin(pi * yy) * std::sin(pi * zz);
        double err = std::fabs(xvals[idx] - uex);
        l2sum += err * err * hx * hy * hz;
        if (err > linf) linf = err;
      }
    }
  }
  double l2err = std::sqrt(l2sum);
  t1 = MPI_Wtime();
  t_error_eval = t1 - t0;

  double total_wall = MPI_Wtime() - t_total0;

  if (rank == 0) {
    std::printf("hypre Struct 3D Poisson MMS (GPU kernel fill + Struct PCG solve + PFMG preconditioner)\n");
    std::printf("  grid            : %d x %d x %d interior points\n", nx, ny, nz);
    std::printf("  total unknowns  : %zu\n", N);
    std::printf("  MPI ranks       : %d\n", size);
    std::printf("  solver          : pcg\n");
    std::printf("  preconditioner  : pfmg\n");
    std::printf("  pc_maxit        : %d\n", pc_maxit);
    std::printf("  pc_relax_type   : %d\n", pc_relax_type);
    std::printf("  pc_rap_type     : %d\n", pc_rap_type);
    std::printf("  two_norm        : %d\n", two_norm);
    std::printf("  device          : %d\n", device);
    std::printf("  hx,hy,hz        : %.12e %.12e %.12e\n", hx, hy, hz);
    std::printf("  iterations      : %d\n", iters);
    std::printf("  final relres    : %.12e\n", final_rel);
    std::printf("  L2 error        : %.12e\n", l2err);
    std::printf("  Linf error      : %.12e\n", linf);
    std::printf("timings (seconds unless noted)\n");
    std::printf("  grid_stencil     : %.6f\n", t_grid_stencil);
    std::printf("  kernel_mat_ms    : %.6f\n", (double)t_kernel_mat_ms);
    std::printf("  kernel_rhs_ms    : %.6f\n", (double)t_kernel_rhs_ms);
    std::printf("  matrix_set       : %.6f\n", t_mat_set);
    std::printf("  matrix_assemble  : %.6f\n", t_mat_assemble);
    std::printf("  rhs_set          : %.6f\n", t_rhs_set);
    std::printf("  rhs_assemble     : %.6f\n", t_rhs_assemble);
    std::printf("  x0_set           : %.6f\n", t_x0_set);
    std::printf("  x0_assemble      : %.6f\n", t_x0_assemble);
    std::printf("  pcg_object_setup : %.6f\n", t_pc_setup);
    std::printf("  solver_setup     : %.6f\n", t_solver_setup);
    std::printf("  solver_solve     : %.6f\n", t_solver_solve);
    std::printf("  solution_fetch   : %.6f\n", t_solution_fetch);
    std::printf("  error_eval       : %.6f\n", t_error_eval);
    std::printf("  total_wall       : %.6f\n", total_wall);
    std::fflush(stdout);
  }

  HYPRE_CALL(HYPRE_StructPCGDestroy(pcg_solver));
  HYPRE_CALL(HYPRE_StructPFMGDestroy(pfmg_precond));
  HYPRE_CALL(HYPRE_StructMatrixDestroy(A));
  HYPRE_CALL(HYPRE_StructVectorDestroy(b));
  HYPRE_CALL(HYPRE_StructVectorDestroy(x));
  HYPRE_CALL(HYPRE_StructStencilDestroy(stencil));
  HYPRE_CALL(HYPRE_StructGridDestroy(grid));
  HYPRE_CALL(HYPRE_Finalize());

  CUDA_CALL(cudaEventDestroy(e0));
  CUDA_CALL(cudaEventDestroy(e1));
  CUDA_CALL(cudaFree(Avals));
  CUDA_CALL(cudaFree(bvals));
  CUDA_CALL(cudaFree(xvals));

  MPI_Finalize();
  return 0;
}
