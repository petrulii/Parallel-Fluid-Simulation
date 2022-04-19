/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
//
// GOAL: Implement a 1D communication scheme along
//       X axis with blocking communications.
//
// SUMMARY:
//     - 1D splitting along X
//     - Blocking communications
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex1(lbm_comm_t * comm, int total_width, int total_height)
{
	// The splitting parameters for the current task.
	int rank;
	int comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	// The number of tasks along X axis and Y axis.
	comm->nb_x = comm_size;
	comm->nb_y = 1;

	// The current task position in the splitting.
	comm->rank_x = rank;
	comm->rank_y = 0;

	// The local sub-domain size.
	comm->width = (total_width/comm_size) + 2;
	comm->height = total_height + 2;

	// The absolute position in the global mesh.
	comm->x = rank*(total_width/comm_size);
	comm->y = 0;

	#ifndef NDEBUG
	lbm_comm_print( comm );
	#endif
}

/****************************************************/
void lbm_comm_ghost_exchange_ex1(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	// Receive from L, send to R, receive from R, send to L.
	MPI_Status status;
	// There are 9 directions per cell.
	int h = DIRECTIONS*(comm->height);
	// The first process in the 1-row grid.
	if(comm->rank_x == 0){
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0),h,MPI_DOUBLE,1,99,MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0),h,MPI_DOUBLE,1,99,MPI_COMM_WORLD,&status);
	// Intermediate or last process in the 1-row grid.
	}else{
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), h,MPI_DOUBLE,comm->rank_x-1,99,MPI_COMM_WORLD,&status);
		// Intermediate process in the 1-row grid.
		if(comm->rank_x+1 != comm->nb_x){
			MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0),h,MPI_DOUBLE,comm->rank_x+1,99,MPI_COMM_WORLD);
			MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0),h,MPI_DOUBLE,comm->rank_x+1,99,MPI_COMM_WORLD,&status);
		}
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), h,MPI_DOUBLE,comm->rank_x-1,99,MPI_COMM_WORLD);
	}
}