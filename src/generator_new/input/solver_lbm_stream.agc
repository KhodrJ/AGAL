# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
#FILE_NAME testname
#FILE_DIR ./output/

ROUTINE_NAME Stream
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
REG int Ip = I;
REG int J = (threadIdx.x / 4) % 4;
REG int Jp = J;
INIF Ldim==3
    int K = (threadIdx.x / 4) / 4;
    int Kp = K;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int nbr_kap_c = -1;
REG int nbr_kap_b = -1;
INFOR p 1   1 Lsize 1
	REG int nbr_<p> __attribute__((unused)) = -1;
END_INFOR
REG ufloat_t F_p = (ufloat_t)0.0;
REG ufloat_t F_pb = (ufloat_t)0.0;


TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(block_on_boundary==1)
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b]
TEMPLATE ARG REG block_on_boundary=cblock_ID_mask[i_kap_b]

// Load neighbor block indices.
INFOR q 1     1 Lsize 1
	REG nbr_<q> = cblock_ID_nbr[i_kap + <q>*n_maxcblocks];
END_INFOR
<

# Loop over all DDFs (except resting ones).
INFOR p 1   1 Lsize 1
	INIF ( (Ldim==2 and (<p>==1 or (<p>+1)%3==0)) or (Ldim==3 and ((((<p>-1)%2==0)and(<p><25))or(<p>==26))) )
		
		//
		// DDF <p>
		//
		
		// Load DDFs in current cells.
		REG F_p = cells_f_F[i_kap_p*M_CBLOCK + threadIdx.x + <p>*n_maxcells];
		<
		
		// Compute neighbor cell indices.
		REG-v Ip = I + Lc0(<p>);
		REG-v Jp = J + Lc1(<p>);
		REG-v Kp = K + Lc2(<p>);
		<
		
		// Assign the correct neighbor cell-block ID.
		INFOR q 1   1 Lsize 1
			INIF ((Lc0(<p>)==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<q>)==0))
				// Consider nbr <q>.
				OUTIFL ( gNa( gCOND(Lc0(<q>): 1,(Ip==4),-1,(Ip==-1),def.,(Ip>=0)and(Ip<4)) and \
						gCOND(Lc1(<q>): 1,(Jp==4),-1,(Jp==-1),def.,(Jp>=0)and(Jp<4)) and \
						gCOND(Ldim: 3,gCOND(Lc2(<q>): 1,(Kp==4),-1,(Kp==-1),def.,(Kp>=0)and(Ip<4)), def.,true) )\
				)
					REG nbr_kap_b = nbr_<q>;
					#cblock_ID_nbr[i_kap_b + <q>*n_maxcblocks];
				END_OUTIFL
			END_INIF
		END_INFOR
		<
		
		// Correct the neighbor cell indices.
		REG Ip = (4 + (Ip % 4)) % 4;
		REG Jp = (4 + (Jp % 4)) % 4;
		REG Kp = (4 + (Kp % 4)) % 4;
		REG nbr_kap_c = Ip + 4*Jp + 16*Kp;
		<
		
		// Retrieve neighboring DDFs, if applicable.
		REG F_pb = -1;
		OUTIFL ( nbr_kap_b>=0 )
			REG-v F_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + Lpb(<p>)*n_maxcells];
		END_OUTIFL
		<
		
		// Exchange, if applicable.
		OUTIF ( F_pb>=0 )
			REG-v cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + Lpb(<p>)*n_maxcells] = F_p;
			REG cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + <p>*n_maxcells] = F_pb;
		END_OUTIF
		<
		
	END_INIF
END_INFOR

TEMPLATE NEW_BLOCK
END_TEMPLATE
