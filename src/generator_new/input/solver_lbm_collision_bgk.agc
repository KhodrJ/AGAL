# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_collision
FILE_DIR ../solver_lbm/

ROUTINE_NAME Collide
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L

KERNEL_REQUIRE int n_ids_idev_L           | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t tau_L
KERNEL_REQUIRE int *id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]


# Declarations.
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG __shared__ int s_ID_cblock[M_TBLOCK];
REG int I = threadIdx.x % 4;
REG int J = (threadIdx.x / 4) % 4;
INIF Ldim==3
    int K = (threadIdx.x / 4) / 4;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int nbr_kap_c = -1;
REG int block_on_boundary = -1;
INFOR p 1   1 Lsize 1
	REG int nbr_<p> __attribute__((unused)) = -1;
END_INFOR
INFOR p 1   0 Lsize 1
	REG f_<p> = (ufloat_t)0.0;
END_INFOR
REG ufloat_t cdotu = (ufloat_t)0.0;
REG ufloat_t udotu = (ufloat_t)0.0;


TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(block_on_boundary==1)
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG block_on_boundary=cblock_ID_mask[i_kap_b];

// Retrieve DDFs in alternating index. Compute macroscopic properties.
INFOR p 1   0 Lsize 1
    REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
END_INFOR
REG-vs rho = gSUM(i,0 Lsize 1,f_<i>);
REG-vsvz u = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc0(<i>))*f_<i>))) / rho_kap;
REG-vsvz v = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc1(<i>))*f_<i>))) / rho_kap;
INIF Ldim==2
    REG-vmz udotu = u*u + v*v;
INELSE
    REG-vsvz w = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc2(<i>))*f_<i>))) / rho_kap;
    REG udotu = u*u + v*v + w*w;
END_INIF
<

// Apply the turbulence model.
<

// Perform collision.
INFOR p 1   0 Lsize 1
	REG-vz cdotu = gNz( Lc0(<p>)*u + Lc1(<p>)*v + Lc2(<p>)*w );
	REG-vm f_<p> = omegap*f_<p> + omega*rho*(ufloat_t)gD(Lw(<p>))*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
END_INFOR
<

// Apply no-slip or anti-bounce back boundary conditions. Free slip is implemented elsewhere.
<
REG block_on_boundary = cblock_ID_onb[i_kap_b];
OUTIF (block_on_boundary)
	# Get the neighbor block indices.
	INFOR q 1   1 Lsize 1
		REG nbr_<q> = cblock_ID_nbr[i_kap + <q>*n_maxcblocks];
	END_INFOR
	
	# Get the neighbor cell indices.
	#INFOR p 1   1 Lsize 1
		#//
		#// DDF <p>
		#//
		
		#INFOR q 1   1 Lsize 1
			#INIF ((Lc0(<p>)==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<q>)==0))
				#// Consider nbr <q>.
				
			#END_INIF
		#END_INFOR
		#<
	#END_INFOR
	
	INFOR q 1   1 Lsize 1
		//
		// nbr <q>
		//
		
		INFOR p 1   1 Lsize 1
			INIF ( (Lc0(<q>)==0 or (Lc0(<q>)!=0 and Lc0(<q>)==Lc0(<p>))) and (Lc1(<q>)==0 or (Lc1(<q>)!=0 and Lc1(<q>)==Lc1(<p>))) and (Lc2(<q>)==0 or (Lc2(<q>)!=0 and Lc2(<q>)==Lc2(<p>))) )
				// Consider DDF <p>
			END_INIF
		END_INFOR
	END_INFOR
END_OUTIF
<

// Write DDFs in proper index.
INFOR p 1   0 Lsize 1
	REG-v cells_f_F[i_kap*M_CBLOCK + threadIdx.x + <p>*n_maxcells] = f_<p>;
END_INFOR

TEMPLATE NEW_BLOCK
END_TEMPLATE
