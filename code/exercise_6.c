/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication with non-blocking
//       messages.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - MPI type for non contiguous cells
// NEW:
//     - Non-blocking communications
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

int rank_Calc(lbm_comm_t * comm,int cord0,int cord1);

/****************************************************/
void lbm_comm_init_ex6(lbm_comm_t * comm, int total_width, int total_height)
{
	//we use the same implementation than ex5
	lbm_comm_init_ex5(comm, total_width, total_height);
}

/****************************************************/
void lbm_comm_release_ex6(lbm_comm_t * comm)
{
	//we use the same implementation than ext 5
	lbm_comm_release_ex5(comm);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex6(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	//
	// TODO: Implement the 2D communication with :
	//         - non-blocking MPI functions
	//         - use MPI type for non contiguous side 
	//
	// To be used:
	//    - DIRECTIONS: the number of doubles composing a cell
	//    - double[9] lbm_mesh_get_cell(mesh, x, y): function to get the address of a particular cell.
	//    - comm->width : The with of the local sub-domain (containing the ghost cells)
	//    - comm->height : The height of the local sub-domain (containing the ghost cells)
	//
	// TIP: create a function to get the target rank from x,y task coordinate.
	// TIP: You can use MPI_PROC_NULL on borders.
	// TIP: send the corner values 2 times, with the up/down/left/write communication
	//      and with the diagonal communication in a second time, this avoid
	//      special cases for border tasks.
	// TIP: The previous trick require to make two batch of non-blocking communications.

	//example to access cell
	//double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
	//double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);

	//TODO:
	//   - implement left/write communications
	//   - implement top/bottom communication (non contiguous)
	//   - implement diagonal communications
	MPI_Status status;
	MPI_Request request_s, request_r;
	
	int h = DIRECTIONS*(comm->height);
	int rank_r;
	int rank_l;
	int rank_u;
	int rank_d;
	int rank_dr;
	int rank_dl;
	int rank_ur;
	int rank_ul;

	// RIGHT communication.
	rank_r =rank_Calc(comm,1,0); 
	// LEFT communication.
	rank_l =rank_Calc(comm,-1,0);
	// UP communication.
	rank_u =rank_Calc(comm,0,-1);
	// DOWN communication.
	rank_d =rank_Calc(comm,0,1);
	// DOWN-RIGHT communication.
	rank_dr =rank_Calc(comm,1,1);
	// DOWN-LEFT communication.
	rank_dl =rank_Calc(comm,-1,1);
	// UP-RIGHT communication.
	rank_ur =rank_Calc(comm,1,-1);
	// UP-LEFT communication.
	rank_ul =rank_Calc(comm,-1,-1);
	
	// Receive from L, send to R, receive from R, send to L.
	// Receive from U, send to D, receive from D, send to U.
	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), h, MPI_DOUBLE, rank_l, 89, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Isend(lbm_mesh_get_cell(mesh, comm->width-2, 0), h, MPI_DOUBLE, rank_r, 89, MPI_COMM_WORLD, &request_s);
	MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width-1, 0), h, MPI_DOUBLE, rank_r, 89, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Isend(lbm_mesh_get_cell(mesh, 1, 0), h, MPI_DOUBLE, rank_l, 89, MPI_COMM_WORLD, &request_s);

	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), 1,comm->type, rank_u, 99, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Isend(lbm_mesh_get_cell(mesh,  0, comm->height-2), 1,comm->type, rank_d, 99, MPI_COMM_WORLD, &request_s);
	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, comm->height-1), 1,comm->type, rank_d, 99, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Isend(lbm_mesh_get_cell(mesh, 0, 1), 1,comm->type, rank_u, 99, MPI_COMM_WORLD, &request_s);

	MPI_Isend(lbm_mesh_get_cell(mesh, comm->width-2, comm->height-2), 9, MPI_DOUBLE, rank_dr, 79, MPI_COMM_WORLD, &request_s);
	MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width-1, comm->height-1), 9, MPI_DOUBLE, rank_dr, 79, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Isend(lbm_mesh_get_cell(mesh, 1, comm->height-2), 9, MPI_DOUBLE, rank_dl, 79, MPI_COMM_WORLD, &request_s);
	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, comm->height-1), 9, MPI_DOUBLE, rank_dl, 79, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width-1, 0), 9, MPI_DOUBLE, rank_ur, 79, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Isend(lbm_mesh_get_cell(mesh, comm->width-2, 1), 9, MPI_DOUBLE, rank_ur, 79, MPI_COMM_WORLD, &request_s);
	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), 9, MPI_DOUBLE, rank_ul, 79, MPI_COMM_WORLD, &request_r);
	MPI_Wait(&request_r, &status);
	MPI_Isend(lbm_mesh_get_cell(mesh, 1, 1), 9, MPI_DOUBLE, rank_ul, 79, MPI_COMM_WORLD, &request_s);
}
