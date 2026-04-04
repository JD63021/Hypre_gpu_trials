#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <algorithm>

#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"

#include <cuda_runtime.h>

#define CUDA_CALL(stmt) do { \
  cudaError_t _err = (stmt); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
    MPI_Abort(MPI_COMM_WORLD, -1); \
  } \
} while (0)

#define HYPRE_CALL(stmt) do { \
  int _ierr = (stmt); \
  if (_ierr) { \
    int _rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
    std::fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n", _rank, __FILE__, __LINE__, _ierr); \
    MPI_Abort(MPI_COMM_WORLD, _ierr); \
  } \
} while (0)

static inline long long idx3_h(int i, int j, int k, int nx, int ny)
{
  return (long long)i + (long long)nx * ((long long)j + (long long)ny * (long long)k);
}

__device__ __forceinline__ int idx3_d(int i, int j, int k, int nx, int ny)
{
  return i + nx * (j + ny * k);
}

__device__ __forceinline__ int xface_idx(int ii, int jj, int kk, int nx, int ny)
{
  return ii + (nx + 1) * (jj + ny * kk);
}

__device__ __forceinline__ int yface_idx(int ii, int jj, int kk, int nx, int ny)
{
  return ii + nx * (jj + (ny + 1) * kk);
}

__device__ __forceinline__ int zface_idx(int ii, int jj, int kk, int nx, int ny)
{
  return ii + nx * (jj + ny * kk);
}

struct Params
{
  int nx = 32, ny = 32, nz = 32;
  double Lx = 1.0, Ly = 1.0, Lz = 1.0;
  double rho = 1.0, U_lid = 1.0, Re = 100.0;
  double dt = 2.0e-3;
  int nsteps = 200;
  int print_every = 20;
  int device = 0;
  int write_vtk = 0;
  std::string vtk_file = "cavity3d_struct_rhiechow_gpu.vtk";
  int vel_maxit = 200;
  double vel_tol = 1.0e-8;
  int p_maxit = 100;
  double p_tol = 1.0e-10;
  int p_relax_type = 1;
  int p_rap_type = 1;
  int monitor = 1;
};

static void ParseArgs(int argc, char **argv, Params &par)
{
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "-nx") && a + 1 < argc) par.nx = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-ny") && a + 1 < argc) par.ny = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-nz") && a + 1 < argc) par.nz = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-Lx") && a + 1 < argc) par.Lx = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-Ly") && a + 1 < argc) par.Ly = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-Lz") && a + 1 < argc) par.Lz = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-rho") && a + 1 < argc) par.rho = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-u-lid") && a + 1 < argc) par.U_lid = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-re") && a + 1 < argc) par.Re = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-dt") && a + 1 < argc) par.dt = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-nsteps") && a + 1 < argc) par.nsteps = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-print-every") && a + 1 < argc) par.print_every = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-device") && a + 1 < argc) par.device = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-write-vtk") && a + 1 < argc) par.write_vtk = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-vtk-file") && a + 1 < argc) par.vtk_file = argv[++a];
    else if (!std::strcmp(argv[a], "-vel-maxit") && a + 1 < argc) par.vel_maxit = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-vel-tol") && a + 1 < argc) par.vel_tol = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-maxit") && a + 1 < argc) par.p_maxit = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-tol") && a + 1 < argc) par.p_tol = std::atof(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-relax-type") && a + 1 < argc) par.p_relax_type = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-p-rap-type") && a + 1 < argc) par.p_rap_type = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "-monitor") && a + 1 < argc) par.monitor = std::atoi(argv[++a]);
  }
}

static void print_device_info(int device)
{
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, device));
  std::printf("Running on \"%s\", major %d, minor %d, total memory %.2f GiB\n",
              prop.name, prop.major, prop.minor,
              (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  std::printf("MaxSharedMemoryPerBlock %zu, MaxSharedMemoryPerBlockOptin %zu\n",
              (size_t)prop.sharedMemPerBlock, (size_t)prop.sharedMemPerBlockOptin);
}

__global__ static void ZeroKernel(int N, double *x)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) x[idx] = 0.0;
}

__global__ static void BuildMomentumAndBCKernel(
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double rho, double nu, double dt,
    double U_lid,
    double *Avals7,
    double *bc_u,
    double *bc_v,
    double *bc_w,
    double *aP_inv)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = nx * ny * nz;
  if (idx >= N) return;

  int i = idx % nx;
  int j = (idx / nx) % ny;
  int k = idx / (nx * ny);

  double cx = nu / (dx * dx);
  double cy = nu / (dy * dy);
  double cz = nu / (dz * dz);
  double at = rho / dt;

  double diagP = at;
  double bu = 0.0, bv = 0.0, bw = 0.0;
  double west = 0.0, east = 0.0, south = 0.0, north = 0.0, bottom = 0.0, top = 0.0;

  if (i > 0) {
    west = -cx;
    diagP += cx;
  } else {
    diagP += 2.0 * cx;
  }
  if (i < nx - 1) {
    east = -cx;
    diagP += cx;
  } else {
    diagP += 2.0 * cx;
  }

  if (j > 0) {
    south = -cy;
    diagP += cy;
  } else {
    diagP += 2.0 * cy;
  }
  if (j < ny - 1) {
    north = -cy;
    diagP += cy;
  } else {
    diagP += 2.0 * cy;
    bu += 2.0 * cy * U_lid; // lid only affects u on north boundary
  }

  if (k > 0) {
    bottom = -cz;
    diagP += cz;
  } else {
    diagP += 2.0 * cz;
  }
  if (k < nz - 1) {
    top = -cz;
    diagP += cz;
  } else {
    diagP += 2.0 * cz;
  }

  Avals7[7 * idx + 0] = diagP;
  Avals7[7 * idx + 1] = west;
  Avals7[7 * idx + 2] = east;
  Avals7[7 * idx + 3] = south;
  Avals7[7 * idx + 4] = north;
  Avals7[7 * idx + 5] = bottom;
  Avals7[7 * idx + 6] = top;

  bc_u[idx] = bu;
  bc_v[idx] = bv;
  bc_w[idx] = bw;
  aP_inv[idx] = 1.0 / diagP;
}

__global__ static void BuildPressureKernel(
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    const double *aP_inv,
    int ref_idx,
    double *Avals7)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = nx * ny * nz;
  if (idx >= N) return;

  if (idx == ref_idx) {
    Avals7[7 * idx + 0] = 1.0;
    Avals7[7 * idx + 1] = 0.0;
    Avals7[7 * idx + 2] = 0.0;
    Avals7[7 * idx + 3] = 0.0;
    Avals7[7 * idx + 4] = 0.0;
    Avals7[7 * idx + 5] = 0.0;
    Avals7[7 * idx + 6] = 0.0;
    return;
  }

  int i = idx % nx;
  int j = (idx / nx) % ny;
  int k = idx / (nx * ny);

  double diagP = 0.0;
  double west = 0.0, east = 0.0, south = 0.0, north = 0.0, bottom = 0.0, top = 0.0;

  if (i > 0) {
    int W = idx - 1;
    double dW = 0.5 * (aP_inv[idx] + aP_inv[W]) / (dx * dx);
    diagP += dW;
    if (W != ref_idx) west = -dW;
  }
  if (i < nx - 1) {
    int E = idx + 1;
    double dE = 0.5 * (aP_inv[idx] + aP_inv[E]) / (dx * dx);
    diagP += dE;
    if (E != ref_idx) east = -dE;
  }
  if (j > 0) {
    int S = idx - nx;
    double dS = 0.5 * (aP_inv[idx] + aP_inv[S]) / (dy * dy);
    diagP += dS;
    if (S != ref_idx) south = -dS;
  }
  if (j < ny - 1) {
    int Nn = idx + nx;
    double dN = 0.5 * (aP_inv[idx] + aP_inv[Nn]) / (dy * dy);
    diagP += dN;
    if (Nn != ref_idx) north = -dN;
  }
  if (k > 0) {
    int B = idx - nx * ny;
    double dB = 0.5 * (aP_inv[idx] + aP_inv[B]) / (dz * dz);
    diagP += dB;
    if (B != ref_idx) bottom = -dB;
  }
  if (k < nz - 1) {
    int T = idx + nx * ny;
    double dT = 0.5 * (aP_inv[idx] + aP_inv[T]) / (dz * dz);
    diagP += dT;
    if (T != ref_idx) top = -dT;
  }

  Avals7[7 * idx + 0] = diagP;
  Avals7[7 * idx + 1] = west;
  Avals7[7 * idx + 2] = east;
  Avals7[7 * idx + 3] = south;
  Avals7[7 * idx + 4] = north;
  Avals7[7 * idx + 5] = bottom;
  Avals7[7 * idx + 6] = top;
}

__global__ static void CellGradientsKernel(
    int nx, int ny, int nz,
    const double *phi,
    double dx, double dy, double dz,
    double *gx, double *gy, double *gz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = nx * ny * nz;
  if (idx >= N) return;

  int i = idx % nx;
  int j = (idx / nx) % ny;
  int k = idx / (nx * ny);

  int im = (i > 0)      ? idx - 1       : idx;
  int ip = (i < nx - 1) ? idx + 1       : idx;
  int jm = (j > 0)      ? idx - nx      : idx;
  int jp = (j < ny - 1) ? idx + nx      : idx;
  int km = (k > 0)      ? idx - nx * ny : idx;
  int kp = (k < nz - 1) ? idx + nx * ny : idx;

  if (i == 0) gx[idx] = (phi[ip] - phi[idx]) / dx;
  else if (i == nx - 1) gx[idx] = (phi[idx] - phi[im]) / dx;
  else gx[idx] = (phi[ip] - phi[im]) / (2.0 * dx);

  if (j == 0) gy[idx] = (phi[jp] - phi[idx]) / dy;
  else if (j == ny - 1) gy[idx] = (phi[idx] - phi[jm]) / dy;
  else gy[idx] = (phi[jp] - phi[jm]) / (2.0 * dy);

  if (k == 0) gz[idx] = (phi[kp] - phi[idx]) / dz;
  else if (k == nz - 1) gz[idx] = (phi[idx] - phi[km]) / dz;
  else gz[idx] = (phi[kp] - phi[km]) / (2.0 * dz);
}

__global__ static void RhieChowFacesXKernel(
    int nx, int ny, int nz,
    const double *u,
    const double *p,
    const double *dpdx,
    const double *aP_inv,
    double dx,
    double *Uf)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int NF = (nx + 1) * ny * nz;
  if (idx >= NF) return;

  int i = idx % (nx + 1);
  int j = (idx / (nx + 1)) % ny;
  int k = idx / ((nx + 1) * ny);

  if (i == 0 || i == nx) {
    Uf[idx] = 0.0;
    return;
  }

  int P = idx3_d(i - 1, j, k, nx, ny);
  int E = idx3_d(i,     j, k, nx, ny);

  double ubar = 0.5 * (u[P] + u[E]);
  double dbar = 0.5 * (aP_inv[P] + aP_inv[E]);
  double gradp_bar = 0.5 * (dpdx[P] + dpdx[E]);
  double dp_face = (p[E] - p[P]) / dx;
  Uf[idx] = ubar - dbar * (dp_face - gradp_bar);
}

__global__ static void RhieChowFacesYKernel(
    int nx, int ny, int nz,
    const double *v,
    const double *p,
    const double *dpdy,
    const double *aP_inv,
    double dy,
    double *Vf)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int NF = nx * (ny + 1) * nz;
  if (idx >= NF) return;

  int i = idx % nx;
  int j = (idx / nx) % (ny + 1);
  int k = idx / (nx * (ny + 1));

  if (j == 0 || j == ny) {
    Vf[idx] = 0.0;
    return;
  }

  int S = idx3_d(i, j - 1, k, nx, ny);
  int Nn = idx3_d(i, j, k, nx, ny);

  double vbar = 0.5 * (v[S] + v[Nn]);
  double dbar = 0.5 * (aP_inv[S] + aP_inv[Nn]);
  double gradp_bar = 0.5 * (dpdy[S] + dpdy[Nn]);
  double dp_face = (p[Nn] - p[S]) / dy;
  Vf[idx] = vbar - dbar * (dp_face - gradp_bar);
}

__global__ static void RhieChowFacesZKernel(
    int nx, int ny, int nz,
    const double *w,
    const double *p,
    const double *dpdz,
    const double *aP_inv,
    double dz,
    double *Wf)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int NF = nx * ny * (nz + 1);
  if (idx >= NF) return;

  int i = idx % nx;
  int j = (idx / nx) % ny;
  int k = idx / (nx * ny);

  if (k == 0 || k == nz) {
    Wf[idx] = 0.0;
    return;
  }

  int B = idx3_d(i, j, k - 1, nx, ny);
  int T = idx3_d(i, j, k, nx, ny);

  double wbar = 0.5 * (w[B] + w[T]);
  double dbar = 0.5 * (aP_inv[B] + aP_inv[T]);
  double gradp_bar = 0.5 * (dpdz[B] + dpdz[T]);
  double dp_face = (p[T] - p[B]) / dz;
  Wf[idx] = wbar - dbar * (dp_face - gradp_bar);
}

__device__ __forceinline__ double upwind_x(
    const double *phi, int nx, int ny, int nz,
    int iL, int j, int k, double F, double bc_face)
{
  if (F >= 0.0) {
    if (iL < 0) return bc_face;
    return phi[idx3_d(iL, j, k, nx, ny)];
  } else {
    int iR = iL + 1;
    if (iR >= nx) return bc_face;
    return phi[idx3_d(iR, j, k, nx, ny)];
  }
}

__device__ __forceinline__ double upwind_y(
    const double *phi, int nx, int ny, int nz,
    int i, int jS, int k, double F, double bc_face)
{
  if (F >= 0.0) {
    if (jS < 0) return bc_face;
    return phi[idx3_d(i, jS, k, nx, ny)];
  } else {
    int jN = jS + 1;
    if (jN >= ny) return bc_face;
    return phi[idx3_d(i, jN, k, nx, ny)];
  }
}

__device__ __forceinline__ double upwind_z(
    const double *phi, int nx, int ny, int nz,
    int i, int j, int kB, double F, double bc_face)
{
  if (F >= 0.0) {
    if (kB < 0) return bc_face;
    return phi[idx3_d(i, j, kB, nx, ny)];
  } else {
    int kT = kB + 1;
    if (kT >= nz) return bc_face;
    return phi[idx3_d(i, j, kT, nx, ny)];
  }
}

__global__ static void ConvectionKernel(
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    const double *u,
    const double *v,
    const double *w,
    const double *Uf,
    const double *Vf,
    const double *Wf,
    double U_lid,
    double *conv_u,
    double *conv_v,
    double *conv_w)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = nx * ny * nz;
  if (idx >= N) return;

  int i = idx % nx;
  int j = (idx / nx) % ny;
  int k = idx / (nx * ny);

  double Fe = Uf[xface_idx(i + 1, j, k, nx, ny)];
  double Fw = Uf[xface_idx(i,     j, k, nx, ny)];
  double Fn = Vf[yface_idx(i, j + 1, k, nx, ny)];
  double Fs = Vf[yface_idx(i, j,     k, nx, ny)];
  double Ft = Wf[zface_idx(i, j, k + 1, nx, ny)];
  double Fb = Wf[zface_idx(i, j, k, nx, ny)];

  double ue = upwind_x(u, nx, ny, nz, i,     j, k, Fe, 0.0);
  double uw = upwind_x(u, nx, ny, nz, i - 1, j, k, Fw, 0.0);
  double un = upwind_y(u, nx, ny, nz, i, j,     k, Fn, U_lid);
  double us = upwind_y(u, nx, ny, nz, i, j - 1, k, Fs, 0.0);
  double ut = upwind_z(u, nx, ny, nz, i, j, k,     Ft, 0.0);
  double ub = upwind_z(u, nx, ny, nz, i, j, k - 1, Fb, 0.0);

  double ve = upwind_x(v, nx, ny, nz, i,     j, k, Fe, 0.0);
  double vw = upwind_x(v, nx, ny, nz, i - 1, j, k, Fw, 0.0);
  double vn = upwind_y(v, nx, ny, nz, i, j,     k, Fn, 0.0);
  double vs = upwind_y(v, nx, ny, nz, i, j - 1, k, Fs, 0.0);
  double vt = upwind_z(v, nx, ny, nz, i, j, k,     Ft, 0.0);
  double vb = upwind_z(v, nx, ny, nz, i, j, k - 1, Fb, 0.0);

  double we = upwind_x(w, nx, ny, nz, i,     j, k, Fe, 0.0);
  double ww = upwind_x(w, nx, ny, nz, i - 1, j, k, Fw, 0.0);
  double wn = upwind_y(w, nx, ny, nz, i, j,     k, Fn, 0.0);
  double ws = upwind_y(w, nx, ny, nz, i, j - 1, k, Fs, 0.0);
  double wt = upwind_z(w, nx, ny, nz, i, j, k,     Ft, 0.0);
  double wb = upwind_z(w, nx, ny, nz, i, j, k - 1, Fb, 0.0);

  conv_u[idx] = (Fe * ue - Fw * uw) / dx + (Fn * un - Fs * us) / dy + (Ft * ut - Fb * ub) / dz;
  conv_v[idx] = (Fe * ve - Fw * vw) / dx + (Fn * vn - Fs * vs) / dy + (Ft * vt - Fb * vb) / dz;
  conv_w[idx] = (Fe * we - Fw * ww) / dx + (Fn * wn - Fs * ws) / dy + (Ft * wt - Fb * wb) / dz;
}

__global__ static void BuildVelocityRhsKernel(
    int N,
    double at,
    const double *uold,
    const double *conv,
    const double *gradp,
    const double *bc,
    double *rhs)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) rhs[idx] = at * uold[idx] - conv[idx] - gradp[idx] + bc[idx];
}

__global__ static void DivergenceKernel(
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    const double *Uf,
    const double *Vf,
    const double *Wf,
    double *divU)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = nx * ny * nz;
  if (idx >= N) return;

  int i = idx % nx;
  int j = (idx / nx) % ny;
  int k = idx / (nx * ny);

  divU[idx] = (Uf[xface_idx(i + 1, j, k, nx, ny)] - Uf[xface_idx(i, j, k, nx, ny)]) / dx
            + (Vf[yface_idx(i, j + 1, k, nx, ny)] - Vf[yface_idx(i, j, k, nx, ny)]) / dy
            + (Wf[zface_idx(i, j, k + 1, nx, ny)] - Wf[zface_idx(i, j, k, nx, ny)]) / dz;
}

__global__ static void BuildPressureRhsKernel(int N, int ref_idx, const double *divU, double *rhs)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) rhs[idx] = (idx == ref_idx) ? 0.0 : -divU[idx];
}

__global__ static void VelocityCorrectionKernel(
    int N,
    const double *ustar,
    const double *aP_inv,
    const double *gradphi,
    double *u)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) u[idx] = ustar[idx] - aP_inv[idx] * gradphi[idx];
}

__global__ static void PressureUpdateKernel(int N, const double *phi, double *p)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) p[idx] += phi[idx];
}

__global__ static void CorrectFacesXWithPhiKernel(
    int nx, int ny, int nz,
    const double *Uf_star,
    const double *phi,
    const double *aP_inv,
    double dx,
    double *Uf_corr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int NF = (nx + 1) * ny * nz;
  if (idx >= NF) return;
  int i = idx % (nx + 1);
  int j = (idx / (nx + 1)) % ny;
  int k = idx / ((nx + 1) * ny);
  if (i == 0 || i == nx) { Uf_corr[idx] = 0.0; return; }
  int P = idx3_d(i - 1, j, k, nx, ny);
  int E = idx3_d(i,     j, k, nx, ny);
  double dbar = 0.5 * (aP_inv[P] + aP_inv[E]);
  Uf_corr[idx] = Uf_star[idx] - dbar * (phi[E] - phi[P]) / dx;
}

__global__ static void CorrectFacesYWithPhiKernel(
    int nx, int ny, int nz,
    const double *Vf_star,
    const double *phi,
    const double *aP_inv,
    double dy,
    double *Vf_corr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int NF = nx * (ny + 1) * nz;
  if (idx >= NF) return;
  int i = idx % nx;
  int j = (idx / nx) % (ny + 1);
  int k = idx / (nx * (ny + 1));
  if (j == 0 || j == ny) { Vf_corr[idx] = 0.0; return; }
  int S = idx3_d(i, j - 1, k, nx, ny);
  int Nn = idx3_d(i, j, k, nx, ny);
  double dbar = 0.5 * (aP_inv[S] + aP_inv[Nn]);
  Vf_corr[idx] = Vf_star[idx] - dbar * (phi[Nn] - phi[S]) / dy;
}

__global__ static void CorrectFacesZWithPhiKernel(
    int nx, int ny, int nz,
    const double *Wf_star,
    const double *phi,
    const double *aP_inv,
    double dz,
    double *Wf_corr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int NF = nx * ny * (nz + 1);
  if (idx >= NF) return;
  int i = idx % nx;
  int j = (idx / nx) % ny;
  int k = idx / (nx * ny);
  if (k == 0 || k == nz) { Wf_corr[idx] = 0.0; return; }
  int B = idx3_d(i, j, k - 1, nx, ny);
  int T = idx3_d(i, j, k, nx, ny);
  double dbar = 0.5 * (aP_inv[B] + aP_inv[T]);
  Wf_corr[idx] = Wf_star[idx] - dbar * (phi[T] - phi[B]) / dz;
}

static void write_vtk_cell_centered(
    const std::string &filename,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    const double *u, const double *v, const double *w, const double *p)
{
  std::ofstream out(filename.c_str());
  out << "# vtk DataFile Version 3.0\n";
  out << "3D cavity collocated cell-centered structured grid\n";
  out << "ASCII\n";
  out << "DATASET STRUCTURED_POINTS\n";
  out << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " " << (nz + 1) << "\n";
  out << "ORIGIN 0 0 0\n";
  out << "SPACING " << dx << " " << dy << " " << dz << "\n";
  out << "CELL_DATA " << ((long long)nx * ny * nz) << "\n";
  out << "VECTORS velocity double\n";
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        long long id = idx3_h(i, j, k, nx, ny);
        out << u[id] << " " << v[id] << " " << w[id] << "\n";
      }
  out << "SCALARS pressure double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        long long id = idx3_h(i, j, k, nx, ny);
        out << p[id] << "\n";
      }
}

static void host_monitor(
    int step, int nsteps,
    int N,
    const double *u, const double *v, const double *w,
    const double *divU)
{
  double umax = 0.0, vmax = 0.0, wmax = 0.0;
  double div_l2 = 0.0, div_inf = 0.0;
  for (int i = 0; i < N; ++i) {
    umax = std::max(umax, std::abs(u[i]));
    vmax = std::max(vmax, std::abs(v[i]));
    wmax = std::max(wmax, std::abs(w[i]));
    div_l2 += divU[i] * divU[i];
    div_inf = std::max(div_inf, std::abs(divU[i]));
  }
  div_l2 = std::sqrt(div_l2 / (double)N);
  std::printf("step %5d / %5d : max|u,v,w| = [%9.3e %9.3e %9.3e], div L2 = %.3e, div Linf = %.3e\n",
              step, nsteps, umax, vmax, wmax, div_l2, div_inf);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    if (rank == 0) std::fprintf(stderr, "This GPU demo currently supports exactly 1 MPI rank.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Params par;
  ParseArgs(argc, argv, par);

  CUDA_CALL(cudaSetDevice(par.device));
  CUDA_CALL(cudaFree(0));

  if (rank == 0) print_device_info(par.device);

  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
  HYPRE_CALL(HYPRE_SetSpGemmUseVendor(0));
  HYPRE_CALL(HYPRE_SetUseGpuRand(1));

  const int nx = par.nx, ny = par.ny, nz = par.nz;
  const int N = nx * ny * nz;
  const int NFx = (nx + 1) * ny * nz;
  const int NFy = nx * (ny + 1) * nz;
  const int NFz = nx * ny * (nz + 1);
  const double dx = par.Lx / (double)nx;
  const double dy = par.Ly / (double)ny;
  const double dz = par.Lz / (double)nz;
  const double nu = par.U_lid * par.Lx / par.Re;
  const double at = par.rho / par.dt;
  const int ref_idx = 0;

  if (rank == 0) {
    std::printf("=====================================================================\n");
    std::printf("3D cavity, collocated FVM, Struct + Rhie-Chow, GPU assembly + solve\n");
    std::printf("Reference logic follows the attached MATLAB IPCS/Rhie-Chow structure.\n");
    std::printf("Grid          : %d x %d x %d = %d cells\n", nx, ny, nz, N);
    std::printf("Re            : %.6g\n", par.Re);
    std::printf("nu            : %.6e\n", nu);
    std::printf("dt            : %.6e\n", par.dt);
    std::printf("nsteps        : %d\n", par.nsteps);
    std::printf("Velocity solve: Struct PCG + Struct Jacobi\n");
    std::printf("Pressure solve: Struct PCG + Struct PFMG(preconditioner)\n");
    std::printf("write_vtk     : %d\n", par.write_vtk);
    std::printf("=====================================================================\n");
  }

  int ilower[3] = {1, 1, 1};
  int iupper[3] = {nx, ny, nz};
  int offsets[7][3] = {{0,0,0}, {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
  int entries7[7] = {0,1,2,3,4,5,6};

  double t0 = MPI_Wtime();
  HYPRE_StructGrid grid;
  HYPRE_StructStencil stencil;
  HYPRE_CALL(HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid));
  HYPRE_CALL(HYPRE_StructGridSetExtents(grid, ilower, iupper));
  HYPRE_CALL(HYPRE_StructGridAssemble(grid));
  HYPRE_CALL(HYPRE_StructStencilCreate(3, 7, &stencil));
  for (int s = 0; s < 7; ++s) HYPRE_CALL(HYPRE_StructStencilSetElement(stencil, s, offsets[s]));
  double t_grid = MPI_Wtime() - t0;

  double *Avals_mom = nullptr, *Avals_phi = nullptr;
  double *bc_u = nullptr, *bc_v = nullptr, *bc_w = nullptr, *aP_inv = nullptr;
  double *u = nullptr, *v = nullptr, *w = nullptr, *p = nullptr;
  double *ustar = nullptr, *vstar = nullptr, *wstar = nullptr, *phi = nullptr;
  double *gx = nullptr, *gy = nullptr, *gz = nullptr;
  double *Uf = nullptr, *Vf = nullptr, *Wf = nullptr;
  double *Uf_corr = nullptr, *Vf_corr = nullptr, *Wf_corr = nullptr;
  double *conv_u = nullptr, *conv_v = nullptr, *conv_w = nullptr;
  double *rhs = nullptr, *divU = nullptr;

  CUDA_CALL(cudaMallocManaged((void**)&Avals_mom, (size_t)7 * N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&Avals_phi, (size_t)7 * N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&bc_u, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&bc_v, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&bc_w, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&aP_inv, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&u, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&v, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&w, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&p, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&ustar, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&vstar, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&wstar, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&phi, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&gx, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&gy, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&gz, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&Uf, (size_t)NFx * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&Vf, (size_t)NFy * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&Wf, (size_t)NFz * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&Uf_corr, (size_t)NFx * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&Vf_corr, (size_t)NFy * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&Wf_corr, (size_t)NFz * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&conv_u, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&conv_v, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&conv_w, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&rhs, (size_t)N * sizeof(double)));
  CUDA_CALL(cudaMallocManaged((void**)&divU, (size_t)N * sizeof(double)));

  int block = 256;
  int gridN = (N + block - 1) / block;
  int gridFx = (NFx + block - 1) / block;
  int gridFy = (NFy + block - 1) / block;
  int gridFz = (NFz + block - 1) / block;

  ZeroKernel<<<gridN, block>>>(N, u);
  ZeroKernel<<<gridN, block>>>(N, v);
  ZeroKernel<<<gridN, block>>>(N, w);
  ZeroKernel<<<gridN, block>>>(N, p);
  CUDA_CALL(cudaGetLastError());

  cudaEvent_t e0, e1;
  CUDA_CALL(cudaEventCreate(&e0));
  CUDA_CALL(cudaEventCreate(&e1));
  float t_kernel_mom_ms = 0.0f, t_kernel_phi_ms = 0.0f;

  CUDA_CALL(cudaEventRecord(e0));
  BuildMomentumAndBCKernel<<<gridN, block>>>(nx, ny, nz, dx, dy, dz, par.rho, nu, par.dt, par.U_lid,
                                             Avals_mom, bc_u, bc_v, bc_w, aP_inv);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaEventRecord(e1));
  CUDA_CALL(cudaEventSynchronize(e1));
  CUDA_CALL(cudaEventElapsedTime(&t_kernel_mom_ms, e0, e1));

  CUDA_CALL(cudaEventRecord(e0));
  BuildPressureKernel<<<gridN, block>>>(nx, ny, nz, dx, dy, dz, aP_inv, ref_idx, Avals_phi);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaEventRecord(e1));
  CUDA_CALL(cudaEventSynchronize(e1));
  CUDA_CALL(cudaEventElapsedTime(&t_kernel_phi_ms, e0, e1));

  HYPRE_StructMatrix A_mom, A_phi;
  double t_mom_set = 0.0, t_mom_assemble = 0.0, t_phi_set = 0.0, t_phi_assemble = 0.0;

  HYPRE_CALL(HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A_mom));
  HYPRE_CALL(HYPRE_StructMatrixInitialize(A_mom));
  HYPRE_CALL(HYPRE_StructMatrixSetSymmetric(A_mom, 1));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixSetBoxValues(A_mom, ilower, iupper, 7, entries7, Avals_mom));
  t_mom_set = MPI_Wtime() - t0;
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixAssemble(A_mom));
  t_mom_assemble = MPI_Wtime() - t0;

  HYPRE_CALL(HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A_phi));
  HYPRE_CALL(HYPRE_StructMatrixInitialize(A_phi));
  HYPRE_CALL(HYPRE_StructMatrixSetSymmetric(A_phi, 1));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixSetBoxValues(A_phi, ilower, iupper, 7, entries7, Avals_phi));
  t_phi_set = MPI_Wtime() - t0;
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructMatrixAssemble(A_phi));
  t_phi_assemble = MPI_Wtime() - t0;

  CUDA_CALL(cudaFree(Avals_mom)); Avals_mom = nullptr;
  CUDA_CALL(cudaFree(Avals_phi)); Avals_phi = nullptr;

  HYPRE_StructVector b_mom, x_mom, b_phi, x_phi;
  HYPRE_CALL(HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b_mom));
  HYPRE_CALL(HYPRE_StructVectorInitialize(b_mom));
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b_mom, ilower, iupper, rhs));
  HYPRE_CALL(HYPRE_StructVectorAssemble(b_mom));

  HYPRE_CALL(HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x_mom));
  HYPRE_CALL(HYPRE_StructVectorInitialize(x_mom));
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x_mom, ilower, iupper, u));
  HYPRE_CALL(HYPRE_StructVectorAssemble(x_mom));

  HYPRE_CALL(HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b_phi));
  HYPRE_CALL(HYPRE_StructVectorInitialize(b_phi));
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b_phi, ilower, iupper, rhs));
  HYPRE_CALL(HYPRE_StructVectorAssemble(b_phi));

  HYPRE_CALL(HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x_phi));
  HYPRE_CALL(HYPRE_StructVectorInitialize(x_phi));
  HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x_phi, ilower, iupper, phi));
  HYPRE_CALL(HYPRE_StructVectorAssemble(x_phi));

  HYPRE_StructSolver vel_solver, vel_prec, p_solver, p_prec;
  double t_vel_setup = 0.0, t_p_setup = 0.0;

  HYPRE_CALL(HYPRE_StructPCGCreate(MPI_COMM_WORLD, &vel_solver));
  HYPRE_CALL(HYPRE_StructPCGSetTol(vel_solver, par.vel_tol));
  HYPRE_CALL(HYPRE_StructPCGSetMaxIter(vel_solver, par.vel_maxit));
  HYPRE_CALL(HYPRE_StructPCGSetTwoNorm(vel_solver, 1));
  HYPRE_CALL(HYPRE_StructPCGSetPrintLevel(vel_solver, 0));
  HYPRE_CALL(HYPRE_StructPCGSetLogging(vel_solver, 1));
  HYPRE_CALL(HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &vel_prec));
  HYPRE_CALL(HYPRE_StructJacobiSetMaxIter(vel_prec, 1));
  HYPRE_CALL(HYPRE_StructJacobiSetTol(vel_prec, 0.0));
  HYPRE_CALL(HYPRE_StructPCGSetPrecond(vel_solver,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructJacobiSolve,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructJacobiSetup,
      vel_prec));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructPCGSetup(vel_solver, A_mom, b_mom, x_mom));
  t_vel_setup = MPI_Wtime() - t0;

  HYPRE_CALL(HYPRE_StructPCGCreate(MPI_COMM_WORLD, &p_solver));
  HYPRE_CALL(HYPRE_StructPCGSetTol(p_solver, par.p_tol));
  HYPRE_CALL(HYPRE_StructPCGSetMaxIter(p_solver, par.p_maxit));
  HYPRE_CALL(HYPRE_StructPCGSetTwoNorm(p_solver, 1));
  HYPRE_CALL(HYPRE_StructPCGSetPrintLevel(p_solver, 0));
  HYPRE_CALL(HYPRE_StructPCGSetLogging(p_solver, 1));
  HYPRE_CALL(HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &p_prec));
  HYPRE_CALL(HYPRE_StructPFMGSetMaxIter(p_prec, 1));
  HYPRE_CALL(HYPRE_StructPFMGSetTol(p_prec, 0.0));
  HYPRE_CALL(HYPRE_StructPFMGSetPrintLevel(p_prec, 0));
  HYPRE_CALL(HYPRE_StructPFMGSetLogging(p_prec, 0));
  HYPRE_CALL(HYPRE_StructPFMGSetRelaxType(p_prec, par.p_relax_type));
  HYPRE_CALL(HYPRE_StructPFMGSetRAPType(p_prec, par.p_rap_type));
  HYPRE_CALL(HYPRE_StructPCGSetPrecond(p_solver,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSolve,
      (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSetup,
      p_prec));
  t0 = MPI_Wtime();
  HYPRE_CALL(HYPRE_StructPCGSetup(p_solver, A_phi, b_phi, x_phi));
  t_p_setup = MPI_Wtime() - t0;

  double t_loop0 = MPI_Wtime();
  double t_u_solve = 0.0, t_v_solve = 0.0, t_w_solve = 0.0, t_phi_solve = 0.0;

  for (int step = 1; step <= par.nsteps; ++step) {
    CellGradientsKernel<<<gridN, block>>>(nx, ny, nz, p, dx, dy, dz, gx, gy, gz);
    RhieChowFacesXKernel<<<gridFx, block>>>(nx, ny, nz, u, p, gx, aP_inv, dx, Uf);
    RhieChowFacesYKernel<<<gridFy, block>>>(nx, ny, nz, v, p, gy, aP_inv, dy, Vf);
    RhieChowFacesZKernel<<<gridFz, block>>>(nx, ny, nz, w, p, gz, aP_inv, dz, Wf);
    ConvectionKernel<<<gridN, block>>>(nx, ny, nz, dx, dy, dz, u, v, w, Uf, Vf, Wf,
                                       par.U_lid, conv_u, conv_v, conv_w);
    CUDA_CALL(cudaGetLastError());

    BuildVelocityRhsKernel<<<gridN, block>>>(N, at, u, conv_u, gx, bc_u, rhs);
    CUDA_CALL(cudaGetLastError());
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b_mom, ilower, iupper, rhs));
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x_mom, ilower, iupper, u));
    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPCGSolve(vel_solver, A_mom, b_mom, x_mom));
    t_u_solve += MPI_Wtime() - t0;
    HYPRE_CALL(HYPRE_StructVectorGetBoxValues(x_mom, ilower, iupper, ustar));

    BuildVelocityRhsKernel<<<gridN, block>>>(N, at, v, conv_v, gy, bc_v, rhs);
    CUDA_CALL(cudaGetLastError());
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b_mom, ilower, iupper, rhs));
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x_mom, ilower, iupper, v));
    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPCGSolve(vel_solver, A_mom, b_mom, x_mom));
    t_v_solve += MPI_Wtime() - t0;
    HYPRE_CALL(HYPRE_StructVectorGetBoxValues(x_mom, ilower, iupper, vstar));

    BuildVelocityRhsKernel<<<gridN, block>>>(N, at, w, conv_w, gz, bc_w, rhs);
    CUDA_CALL(cudaGetLastError());
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b_mom, ilower, iupper, rhs));
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x_mom, ilower, iupper, w));
    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPCGSolve(vel_solver, A_mom, b_mom, x_mom));
    t_w_solve += MPI_Wtime() - t0;
    HYPRE_CALL(HYPRE_StructVectorGetBoxValues(x_mom, ilower, iupper, wstar));

    RhieChowFacesXKernel<<<gridFx, block>>>(nx, ny, nz, ustar, p, gx, aP_inv, dx, Uf);
    RhieChowFacesYKernel<<<gridFy, block>>>(nx, ny, nz, vstar, p, gy, aP_inv, dy, Vf);
    RhieChowFacesZKernel<<<gridFz, block>>>(nx, ny, nz, wstar, p, gz, aP_inv, dz, Wf);
    DivergenceKernel<<<gridN, block>>>(nx, ny, nz, dx, dy, dz, Uf, Vf, Wf, divU);
    BuildPressureRhsKernel<<<gridN, block>>>(N, ref_idx, divU, rhs);
    CUDA_CALL(cudaGetLastError());
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(b_phi, ilower, iupper, rhs));
    ZeroKernel<<<gridN, block>>>(N, phi);
    HYPRE_CALL(HYPRE_StructVectorSetBoxValues(x_phi, ilower, iupper, phi));
    CUDA_CALL(cudaGetLastError());
    t0 = MPI_Wtime();
    HYPRE_CALL(HYPRE_StructPCGSolve(p_solver, A_phi, b_phi, x_phi));
    t_phi_solve += MPI_Wtime() - t0;
    HYPRE_CALL(HYPRE_StructVectorGetBoxValues(x_phi, ilower, iupper, phi));

    CellGradientsKernel<<<gridN, block>>>(nx, ny, nz, phi, dx, dy, dz, gx, gy, gz);
    VelocityCorrectionKernel<<<gridN, block>>>(N, ustar, aP_inv, gx, u);
    VelocityCorrectionKernel<<<gridN, block>>>(N, vstar, aP_inv, gy, v);
    VelocityCorrectionKernel<<<gridN, block>>>(N, wstar, aP_inv, gz, w);
    PressureUpdateKernel<<<gridN, block>>>(N, phi, p);
    CUDA_CALL(cudaGetLastError());

    if (par.monitor && (step == 1 || step == par.nsteps || (par.print_every > 0 && step % par.print_every == 0))) {
      CorrectFacesXWithPhiKernel<<<gridFx, block>>>(nx, ny, nz, Uf, phi, aP_inv, dx, Uf_corr);
      CorrectFacesYWithPhiKernel<<<gridFy, block>>>(nx, ny, nz, Vf, phi, aP_inv, dy, Vf_corr);
      CorrectFacesZWithPhiKernel<<<gridFz, block>>>(nx, ny, nz, Wf, phi, aP_inv, dz, Wf_corr);
      DivergenceKernel<<<gridN, block>>>(nx, ny, nz, dx, dy, dz, Uf_corr, Vf_corr, Wf_corr, divU);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());
      if (rank == 0) host_monitor(step, par.nsteps, N, u, v, w, divU);
    }
  }

  CUDA_CALL(cudaDeviceSynchronize());
  double t_loop = MPI_Wtime() - t_loop0;

  int vel_iters = -1, p_iters = -1;
  double vel_rel = -1.0, p_rel = -1.0;
  HYPRE_CALL(HYPRE_StructPCGGetNumIterations(vel_solver, &vel_iters));
  HYPRE_CALL(HYPRE_StructPCGGetFinalRelativeResidualNorm(vel_solver, &vel_rel));
  HYPRE_CALL(HYPRE_StructPCGGetNumIterations(p_solver, &p_iters));
  HYPRE_CALL(HYPRE_StructPCGGetFinalRelativeResidualNorm(p_solver, &p_rel));

  if (rank == 0) {
    std::printf("\nTimings (seconds unless noted)\n");
    std::printf("  grid+stencil        : %.6f\n", t_grid);
    std::printf("  kernel_mom_ms       : %.6f\n", t_kernel_mom_ms);
    std::printf("  kernel_phi_ms       : %.6f\n", t_kernel_phi_ms);
    std::printf("  mom_set             : %.6f\n", t_mom_set);
    std::printf("  mom_assemble        : %.6f\n", t_mom_assemble);
    std::printf("  phi_set             : %.6f\n", t_phi_set);
    std::printf("  phi_assemble        : %.6f\n", t_phi_assemble);
    std::printf("  vel_setup           : %.6f\n", t_vel_setup);
    std::printf("  pressure_setup      : %.6f\n", t_p_setup);
    std::printf("  time_loop_total     : %.6f\n", t_loop);
    std::printf("  u_solve_total       : %.6f\n", t_u_solve);
    std::printf("  v_solve_total       : %.6f\n", t_v_solve);
    std::printf("  w_solve_total       : %.6f\n", t_w_solve);
    std::printf("  phi_solve_total     : %.6f\n", t_phi_solve);
    std::printf("  avg_step            : %.6f\n", t_loop / (double)par.nsteps);
    std::printf("  last vel PCG iters  : %d, relres %.3e\n", vel_iters, vel_rel);
    std::printf("  last p   PCG iters  : %d, relres %.3e\n", p_iters, p_rel);
  }

  if (par.write_vtk) {
    CUDA_CALL(cudaDeviceSynchronize());
    if (rank == 0) {
      write_vtk_cell_centered(par.vtk_file, nx, ny, nz, dx, dy, dz, u, v, w, p);
      std::printf("Wrote VTK file: %s\n", par.vtk_file.c_str());
    }
  }

  HYPRE_CALL(HYPRE_StructPCGDestroy(vel_solver));
  HYPRE_CALL(HYPRE_StructJacobiDestroy(vel_prec));
  HYPRE_CALL(HYPRE_StructPCGDestroy(p_solver));
  HYPRE_CALL(HYPRE_StructPFMGDestroy(p_prec));
  HYPRE_CALL(HYPRE_StructVectorDestroy(b_mom));
  HYPRE_CALL(HYPRE_StructVectorDestroy(x_mom));
  HYPRE_CALL(HYPRE_StructVectorDestroy(b_phi));
  HYPRE_CALL(HYPRE_StructVectorDestroy(x_phi));
  HYPRE_CALL(HYPRE_StructMatrixDestroy(A_mom));
  HYPRE_CALL(HYPRE_StructMatrixDestroy(A_phi));
  HYPRE_CALL(HYPRE_StructStencilDestroy(stencil));
  HYPRE_CALL(HYPRE_StructGridDestroy(grid));
  HYPRE_CALL(HYPRE_Finalize());

  if (bc_u) CUDA_CALL(cudaFree(bc_u));
  if (bc_v) CUDA_CALL(cudaFree(bc_v));
  if (bc_w) CUDA_CALL(cudaFree(bc_w));
  if (aP_inv) CUDA_CALL(cudaFree(aP_inv));
  if (u) CUDA_CALL(cudaFree(u));
  if (v) CUDA_CALL(cudaFree(v));
  if (w) CUDA_CALL(cudaFree(w));
  if (p) CUDA_CALL(cudaFree(p));
  if (ustar) CUDA_CALL(cudaFree(ustar));
  if (vstar) CUDA_CALL(cudaFree(vstar));
  if (wstar) CUDA_CALL(cudaFree(wstar));
  if (phi) CUDA_CALL(cudaFree(phi));
  if (gx) CUDA_CALL(cudaFree(gx));
  if (gy) CUDA_CALL(cudaFree(gy));
  if (gz) CUDA_CALL(cudaFree(gz));
  if (Uf) CUDA_CALL(cudaFree(Uf));
  if (Vf) CUDA_CALL(cudaFree(Vf));
  if (Wf) CUDA_CALL(cudaFree(Wf));
  if (Uf_corr) CUDA_CALL(cudaFree(Uf_corr));
  if (Vf_corr) CUDA_CALL(cudaFree(Vf_corr));
  if (Wf_corr) CUDA_CALL(cudaFree(Wf_corr));
  if (conv_u) CUDA_CALL(cudaFree(conv_u));
  if (conv_v) CUDA_CALL(cudaFree(conv_v));
  if (conv_w) CUDA_CALL(cudaFree(conv_w));
  if (rhs) CUDA_CALL(cudaFree(rhs));
  if (divU) CUDA_CALL(cudaFree(divU));

  MPI_Finalize();
  return 0;
}
