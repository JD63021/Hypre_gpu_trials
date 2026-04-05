# README

File:
`cavity3d_struct_rhiechow_gpu_fixproj_matlabmatch_pvd_atend.cu`

This program solves the 3D lid-driven cavity problem on a uniform Cartesian grid using a collocated finite-volume formulation with Rhie-Chow interpolation, a projection / IPCS-style pressure correction, and hypre Struct solvers running on the GPU.

The code is designed as a structured-grid CFD demo that keeps the discretization and data layout light, while still using a pressure-velocity coupling strategy suitable for collocated variables.

## 1. What physical problem it solves

The code solves incompressible flow in a unit box-like cavity:

- domain: `[0,Lx] x [0,Ly] x [0,Lz]`
- top lid moves in the `+x` direction with speed `U_lid`
- all other walls are no-slip stationary walls
- fluid density is constant
- viscosity is set from the Reynolds number as

  `nu = U_lid * Lx / Re`

The unknowns are:

- cell-centered velocity components `u, v, w`
- cell-centered pressure `p`

The mesh is uniform and structured:

- `nx x ny x nz` control volumes
- cell sizes: `dx = Lx/nx`, `dy = Ly/ny`, `dz = Lz/nz`

## 2. Numerical method in plain words

The program uses a collocated finite-volume approach, meaning both velocity and pressure live at the same cell centers.

That creates the classic checkerboarding risk for pressure, so the code uses Rhie-Chow face interpolation.

The time advancement is projection-style:

1. Start from the old pressure and old velocity.
2. Build Rhie-Chow face velocities from the old fields.
3. Use those face fluxes to form explicit convection terms.
4. Solve three implicit momentum equations to get intermediate velocities `u*`, `v*`, `w*`.
5. Build a pressure-correction equation from the divergence of the intermediate face fluxes.
6. Solve for pressure correction `phi`.
7. Correct velocity and update pressure.
8. Repeat.

So this is not a fully coupled monolithic solve. It is a segregated velocity-pressure algorithm.

## 3. Discretization details

### 3.1 Momentum equations

The momentum operator uses:

- implicit transient term
- implicit diffusion term
- explicit convection term
- explicit old-pressure gradient term
- Dirichlet wall contributions folded into the right-hand side

The intermediate velocity equations are of the form

`(rho/dt) u* - nu Laplacian(u*) = (rho/dt) u^n - convection(u^n) - dp/dx + BC_terms`

and similarly for `v*` and `w*`.

In the code, the momentum matrix is the same for all three velocity components because:

- the grid is uniform
- viscosity is constant
- the implicit operator is just transient + diffusion

That matrix is assembled only once.

### 3.2 Diffusion stencil

The momentum matrix uses a 7-point structured stencil:

- center
- west/east
- south/north
- bottom/top

This corresponds to a standard second-order finite-volume / finite-difference style Laplacian on a Cartesian mesh.

Boundary conditions are handled by modifying the diagonal and adding boundary contributions to the RHS.

For the moving lid:

- only the north boundary contribution in the `u` equation gets the nonzero lid value
- `v` and `w` stay zero on the lid

### 3.3 Pressure correction equation

The pressure-correction operator is built from the inverse momentum diagonal `aP_inv`.

The code forms a variable-coefficient Poisson-like equation using face-averaged `aP_inv` values. In practical terms, this is the structured-grid analogue of the standard projection correction:

`div( aP_inv grad(phi) ) = - div(U*_face)`

A pressure reference cell is imposed by pinning one cell index `ref_idx = 0`.

That removes the nullspace and makes the linear system nonsingular.

### 3.4 Rhie-Chow interpolation

The face fluxes are not computed by simple interpolation of cell-centered velocities alone.

Instead, for each face the code uses the Rhie-Chow correction pattern:

`face velocity = interpolated cell velocity - d_bar * (face pressure difference - interpolated cell pressure gradient)`

implemented separately for x, y, and z faces.

This is what prevents pressure checkerboarding on a collocated grid.

### 3.5 Convection term

The convection operator is built explicitly from face fluxes.

The code uses first-order upwinding at faces:

- pick the upwind cell value based on the sign of the face flux
- use wall values when the upwind direction points to a boundary

This is simple and robust, though it is only first-order accurate in space for convection.

## 4. What is assembled, and when

This part is important.

The expensive linear operators are not rebuilt every time step.

### Built once at startup

1. hypre Struct grid
2. hypre 7-point stencil
3. momentum matrix `A_mom`
4. pressure-correction matrix `A_phi`
5. velocity solver objects
6. pressure solver objects
7. solver setups / multigrid hierarchy

So yes:

- the momentum matrix is assembled once
- the pressure matrix is assembled once
- the pressure PFMG hierarchy is built once
- the PCG setups are done once

### Updated every time step

Only the right-hand sides and solution vectors are updated each step:

- momentum RHS for `u*`
- momentum RHS for `v*`
- momentum RHS for `w*`
- pressure-correction RHS
- solution guesses

This is why the code stays fast after setup.

## 5. GPU work split: what runs on the GPU

The code is explicitly meant for GPU assembly + GPU solve.

### GPU kernels used for assembly / field operations

CUDA kernels are used for:

- zeroing arrays
- building the momentum stencil values
- building the pressure stencil values
- computing cell-centered pressure gradients
- computing Rhie-Chow face velocities
- computing convection terms
- building velocity RHS vectors
- computing divergence of face fluxes
- building pressure RHS
- correcting velocities
- updating pressure
- correcting face fluxes for monitoring

### hypre device execution

The program calls:

- `HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE)`
- `HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE)`

and stores field arrays in CUDA managed memory.

The structured matrices and vectors are then set and solved through hypre in device mode.

So the intended path is:

- stencil/value generation on GPU
- linear solves on GPU

### What still happens on CPU

Some things are still host-side:

- command-line parsing
- printing
- final ASCII VTK writing
- final PVD writing
- monitoring reductions done by pulling data to host-visible managed memory after synchronization

So this is not a "never touch the CPU" code. But the main numerical work is GPU-side.

## 6. Linear solvers used

### Velocity solves

For each component, the code solves the same momentum matrix with:

- outer solver: hypre Struct PCG
- preconditioner: hypre Struct Jacobi

This is used three times per time step:

- once for `u*`
- once for `v*`
- once for `w*`

### Pressure solve

For the pressure correction, the code uses:

- outer solver: hypre Struct PCG
- preconditioner: hypre Struct PFMG

PFMG is the structured multigrid preconditioner and is the main reason the pressure solve is efficient on large structured grids.

## 7. Data layout in the code

The code stores:

- cell-centered fields of size `N = nx*ny*nz`
- x-face fluxes of size `(nx+1)*ny*nz`
- y-face fluxes of size `nx*(ny+1)*nz`
- z-face fluxes of size `nx*ny*(nz+1)`

Major arrays include:

- `u, v, w` : corrected cell-centered velocity
- `ustar, vstar, wstar` : intermediate velocity after momentum solve
- `p` : current pressure
- `phi` : pressure correction
- `gx, gy, gz` : cell-centered gradients
- `Uf, Vf, Wf` : Rhie-Chow face velocities
- `Uf_corr, Vf_corr, Wf_corr` : face velocities after pressure correction, used mainly for monitoring
- `conv_u, conv_v, conv_w` : explicit convection terms
- `rhs` : reusable RHS work array
- `divU` : cell-centered divergence of face fluxes
- `aP_inv` : inverse diagonal of the momentum operator
- `bc_u, bc_v, bc_w` : boundary contributions to the momentum RHS

## 8. Time-step loop, step by step

Every time step does the following:

### Step A: pressure gradient from old pressure

`CellGradientsKernel` computes `gx, gy, gz` from the current pressure field `p`.

### Step B: Rhie-Chow face velocities from old fields

The code computes old face velocities:

- `Uf`
- `Vf`
- `Wf`

using the old cell-centered velocity, pressure, and pressure gradients.

### Step C: explicit convection

`ConvectionKernel` uses those face fluxes and upwinded cell values to form:

- `conv_u`
- `conv_v`
- `conv_w`

### Step D: three momentum solves

For each component:

1. build RHS
2. put RHS into the hypre Struct vector
3. solve with PCG + Jacobi
4. fetch result into `ustar`, `vstar`, or `wstar`

### Step E: intermediate face fluxes

Rhie-Chow is applied again, now using the intermediate velocities `u*`, `v*`, `w*` together with the old pressure.

This gives intermediate face velocities whose divergence is used to form the pressure-correction RHS.

### Step F: pressure-correction solve

1. compute divergence of intermediate face fluxes
2. build RHS as `-div(U*_face)`
3. solve the pressure-correction equation for `phi`

### Step G: correction step

1. compute gradients of `phi`
2. correct cell-centered velocities
3. update pressure by `p = p + phi`

### Step H: optional monitoring

If monitoring is enabled, the code also corrects the face fluxes with `phi`, recomputes divergence, and prints:

- max absolute `u, v, w`
- L2 norm of divergence
- Linf norm of divergence

### Step I: optional snapshot writing

If `write_every > 0`, the code writes `.vtk` snapshots at the requested interval and also at the final step.

## 9. Output files

### Final VTK

If `-write-vtk 1` is used, the code writes a final ASCII VTK file given by `-vtk-file`.

This contains:

- cell-centered velocity vector
- cell-centered pressure scalar

### Time-series snapshots

If `-write-every N` is used, the code writes files like:

- `prefix_step0000000.vtk`
- `prefix_step0000020.vtk`
- etc.

If `-write-initial 1` is given, step 0 is also written.

### PVD collection

The `.pvd` file is not updated during the run.

Instead, all snapshot names and times are stored in memory and the `.pvd` collection file is written only once at the very end.

This was done to avoid ParaView issues with partially updated or path-mismatched `.pvd` files.

The `.pvd` references VTK files by basename, so it expects:

- the `.pvd` file
- all snapshot `.vtk` files

to be in the same folder.

## 10. Important implementation choices and limitations

### Single MPI rank only

The current code aborts unless run with exactly one MPI rank.

So this is a single-GPU, single-rank demo.

### Uniform structured mesh only

The solver assumes:

- Cartesian grid
- uniform spacing in each direction
- one structured box

It is not for unstructured meshes.

### First-order upwind convection

The convection operator is robust but low order.

### Constant-coefficient momentum matrix

Because only transient + diffusion are implicit, the velocity matrix does not change in time.

That is why the code can reuse the same setup every step.

### Pressure equation fixed after startup

The pressure matrix depends on `aP_inv`, and `aP_inv` is taken from the fixed implicit momentum diagonal. Since that diagonal is constant in this code, the pressure matrix also stays constant and is assembled only once.

### ASCII VTK output

The VTK writer is simple and portable, but ASCII files can become large for many snapshots.

## 11. Why this code is fast for large grids

This code is much lighter than a generic unstructured assembled FEM path because:

- the stencil is fixed and compact
- the grid is structured
- the pressure solve uses hypre Struct multigrid
- the same matrices are reused every time step
- only RHS vectors change during the loop
- major field operations are CUDA kernels

That is why it can handle very large Cartesian CFD grids much more easily than a heavy generic sparse route.

## 12. Relation to SIMPLE / projection ideas

Conceptually, this code is closest to a projection / IPCS-style collocated finite-volume solver.

It already contains the critical collocated ingredient needed for practical pressure-velocity coupling:

- Rhie-Chow face interpolation

If extended toward a SIMPLE-family solver, the main ingredients are already present in spirit:

- collocated storage
- face-flux construction
- diagonal-based pressure correction coefficient `aP_inv`
- pressure Poisson / correction solve
- velocity correction step

But this exact program is a transient projection-style implementation, not a classical steady SIMPLE loop.

## 13. Main command-line options

Important runtime arguments:

- `-nx -ny -nz` : grid resolution
- `-Lx -Ly -Lz` : domain size
- `-rho` : density
- `-u-lid` : lid speed
- `-re` : Reynolds number
- `-dt` : time step
- `-nsteps` : number of time steps
- `-vel-maxit` : max PCG iterations for each velocity solve
- `-vel-tol` : velocity solve tolerance
- `-p-maxit` : max PCG iterations for pressure correction
- `-p-tol` : pressure solve tolerance
- `-p-relax-type` : PFMG smoother type
- `-p-rap-type` : PFMG RAP type
- `-monitor` : print step diagnostics
- `-print-every` : print frequency
- `-write-vtk` : write final VTK
- `-vtk-file` : final VTK filename
- `-write-every` : snapshot interval
- `-write-initial` : also write step 0 snapshot
- `-output-prefix` : prefix for snapshot files and final PVD
- `-device` : CUDA device index

## 14. Build and run

The build used for this file is:

```text
cd ~/hypretrials
cp /mnt/data/cavity3d_struct_rhiechow_gpu_fixproj_matlabmatch_pvd_atend.cu .

export PETSC_DIR=/home/jd/src/petsc
export PETSC_ARCH=arch-linux-cuda-amgx-opt
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
hash -r

"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -ccbin mpicxx \
  -I"$PETSC_DIR/include" \
  -I"$PETSC_DIR/$PETSC_ARCH/include" \
  -c cavity3d_struct_rhiechow_gpu_fixproj_matlabmatch_pvd_atend.cu \
  -o cavity3d_struct_rhiechow_gpu_fixproj_matlabmatch_pvd_atend.o

mpicxx cavity3d_struct_rhiechow_gpu_fixproj_matlabmatch_pvd_atend.o \
  -o cavity3d_struct_rhiechow_gpu_fixproj_matlabmatch_pvd_atend \
  -L"$PETSC_DIR/$PETSC_ARCH/lib" \
  -Wl,-rpath,"$PETSC_DIR/$PETSC_ARCH/lib" \
  -L"$CUDA_HOME/lib64" \
  -lHYPRE -lcudart -lm
```

Example run:

```text
mpirun -n 1 ./cavity3d_struct_rhiechow_gpu_fixproj_matlabmatch_pvd_atend \
  -nx 100 -ny 100 -nz 100 \
  -re 2000 -dt 1e-3 -nsteps 500 \
  -vel-maxit 300 -vel-tol 1e-8 \
  -p-maxit 150 -p-tol 1e-10 \
  -p-relax-type 1 -p-rap-type 1 \
  -print-every 20 -monitor 1 \
  -write-every 20 -write-initial 1 \
  -output-prefix cavity100_re2000 \
  -write-vtk 1 -vtk-file cavity100_re2000_final.vtk \
  -device 0
```

## 15. Short summary

This code is a structured-grid 3D incompressible cavity solver that:

- uses cell-centered collocated finite volumes
- uses Rhie-Chow interpolation for pressure-velocity coupling
- uses a projection / IPCS-style correction step
- assembles fixed structured momentum and pressure matrices once
- reuses the same solver setup every time step
- performs the main numerical work on the GPU through CUDA kernels and hypre Struct device solves
- writes VTK snapshots and builds the PVD collection at the end

That is why it is both light enough to scale to very large Cartesian grids and still close in spirit to practical collocated CFD methods.
