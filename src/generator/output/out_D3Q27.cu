/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul  6 22:45:07 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, int ave_type>
__global__
void Cu_Average_D3Q27(int n_ids_idev_L,long int n_maxcells,int n_maxcblocks,ufloat_t tau_L,ufloat_t tau_ratio,int *id_set_idev_L,int *cells_ID_mask,ufloat_t *cells_f_F,int *cblock_ID_nbr,int *cblock_ID_nbr_child,int *cblock_ID_mask,int *cblock_ID_onb)
{
    //
    // p = 1
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 2
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 3
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 4
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 4*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 5
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp+tmp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 5*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 6
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 6*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 7
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 7*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 8
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx+ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 8*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 9
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp+tmp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 9*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 9*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 10
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 10*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 10*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 11
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp+tmp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 11*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 11*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 12
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(ny+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 12*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 12*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 13
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx)-(ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 13*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 13*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 14
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny)-(nx));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 14*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 14*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 15
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx)-(nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 15*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 15*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 16
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz)-(nx));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp);
            tmpz = (vzp+tmp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 16*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 16*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 17
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny)-(nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 17*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 17*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 18
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz)-(ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp+tmp);
            tmpx = (vxp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 18*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 18*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 19
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+ny+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp+tmp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 19*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 19*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 20
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / (-(nx+ny+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 20*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 20*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 21
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+ny)-(nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 21*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 21*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 22
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nz)-(nx+ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp+tmp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 22*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 22*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 23
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx+nz)-(ny));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp+tmp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 23*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 23*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 24
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny)-(nx+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 24*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 24*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 25
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((ny+nz)-(nx));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp+tmp);
            tmpz = (vzp+tmp);
            tmpx = (vxp)-(tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 25*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 25*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    //
    // p = 26
    //
    
    b_id_p = -8;
    dist_p = -1;
    for (int p = 0; p < n_f; p += 1)
    {
        f_p = binned_face_ids[N_f+p];
        vx1 = geom_f_face_X[f_p + 0*n_faces_a]; vy1 = geom_f_face_X[f_p + 1*n_faces_a]; vz1 = geom_f_face_X[f_p + 2*n_faces_a];
        vx2 = geom_f_face_X[f_p + 3*n_faces_a]; vy2 = geom_f_face_X[f_p + 4*n_faces_a]; vz2 = geom_f_face_X[f_p + 5*n_faces_a];
        vx3 = geom_f_face_X[f_p + 6*n_faces_a]; vy3 = geom_f_face_X[f_p + 7*n_faces_a]; vz3 = geom_f_face_X[f_p + 8*n_faces_a];
        ex1 = vx2-vx1; ey1 = vy2-vy1; ez1 = vz2-vz1;
        ex2 = vx3-vx1; ey2 = vy3-vy1; ez2 = vz3-vz1;
        Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
        tmp = Tsqrt(nx*nx + ny*ny + nz*nz); nx /= tmp; ny /= tmp; nz /= tmp;
        tmp = ((vx1-vxp)*nx + (vy1-vyp)*ny + (vz1-vzp)*nz) / ((nx)-(ny+nz));
        if (tmp > (ufloat_g_t)0.0 && tmp < dx_L)
        {
            tmpy = (vyp)-(tmp);
            tmpz = (vzp)-(tmp);
            tmpx = (vxp+tmp);
            if (CheckPointInTriangle(tmpx, tmpy, tmpz, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2))
            {
                if (tmp < dist_p || dist_p < 0)
                    dist_p = tmp;
            }
        }
    }
    cells_ID_mask_b[valid_block*M_CBLOCK + threadIdx.x + 26*n_maxcells_b] = b_id_p; // This will be updated to read actual boundary Ids later.
    cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 26*n_maxcells_b] = dist_p;
    if (dist_p > 0)
        check_cell_mask = true;
    
    gIMP(1.000000000000000
    1.000000000000000
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Average_D3Q27(int i_dev, int L, int var, ufloat_t tau_L, ufloat_t tau_ratio_L)
{
	if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_INTERFACE)
	{
		Cu_Average_D3Q27<ufloat_t,ufloat_g_t,AP,0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_BLOCK)
	{
		Cu_Average_D3Q27<ufloat_t,ufloat_g_t,AP,1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_GRID)
	{
		Cu_Average_D3Q27<ufloat_t,ufloat_g_t,AP,2><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, tau_L, tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

