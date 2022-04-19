/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement non-blocking 1D communication scheme
//       along X axis.
//
// SUMMARY:
//     - 1D splitting along X
// NEW:
//     - >>> Non-blocking communications <<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex3(lbm_comm_t * comm, int total_width, int total_height)
{
	// We use the same implementation as ex1.s
	lbm_comm_init_ex1(comm, total_width, total_height);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex3(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
    MPI_Request request_s, request_r;
	MPI_Status status;
	// There are 9 directions per cell.
	int h = DIRECTIONS*(comm->height);
	// The first process in the 1-row grid.
	if (comm->rank_x == 0) {
		MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width-1, 0), h, MPI_DOUBLE, 1, 99, MPI_COMM_WORLD, &request_r);
		MPI_Wait(&request_r, &status);
		MPI_Isend(lbm_mesh_get_cell(mesh, comm->width-2, 0), h, MPI_DOUBLE, 1, 99, MPI_COMM_WORLD, &request_s);
	// Intermediate or last process in the 1-row grid.
	} else {
		// EVEN : Receive from L, receive from R, send to R, send to L.
		if(comm->rank_x%2 == 0) {
			MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), h, MPI_DOUBLE, comm->rank_x-1, 99, MPI_COMM_WORLD, &request_r);
			MPI_Wait(&request_r, &status);
			// Intermediate process in the 1-row grid.
			if (comm->rank_x+1 != comm->nb_x){
			MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width-1, 0), h, MPI_DOUBLE, comm->rank_x+1, 99, MPI_COMM_WORLD, &request_r);
			MPI_Wait(&request_r, &status);
			MPI_Isend(lbm_mesh_get_cell(mesh, comm->width-2, 0), h, MPI_DOUBLE, comm->rank_x+1, 99, MPI_COMM_WORLD, &request_s);
			}
			MPI_Isend(lbm_mesh_get_cell(mesh, 1, 0), h, MPI_DOUBLE, comm->rank_x-1, 99, MPI_COMM_WORLD, &request_s);
		// ODD : Send to L, send to R, receive from R, receive from L.
		} else {
			MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), h,MPI_DOUBLE,comm->rank_x-1,99,MPI_COMM_WORLD);
			// Intermediate process in the 1-row grid.
			if (comm->rank_x+1 != comm->nb_x) {
			MPI_Isend(lbm_mesh_get_cell(mesh, comm->width-2, 0), h, MPI_DOUBLE, comm->rank_x+1, 99, MPI_COMM_WORLD, &request_s);
			MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width-1, 0), h, MPI_DOUBLE, comm->rank_x+1, 99, MPI_COMM_WORLD, &request_r);
			MPI_Wait(&request_r, &status);
			}
			MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), h, MPI_DOUBLE, comm->rank_x-1, 99, MPI_COMM_WORLD, &request_r);
			MPI_Wait(&request_r, &status);
		}
	}
}