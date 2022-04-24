/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication scheme with
//      8 neighbors using MPI types for non contiguous
//      side.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
// NEW:
//     - >>> MPI type for non contiguous cells <<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

int rank_Calc(lbm_comm_t * comm,int cord0,int cord1);
/****************************************************/
void lbm_comm_init_ex5(lbm_comm_t * comm, int total_width, int total_height)
{
	//we use the same implementation than ex5 execpt for type creation
	lbm_comm_init_ex4(comm, total_width, total_height);

	//TODO: create MPI type for non contiguous side in comm->type.
	MPI_Type_vector( comm->width , DIRECTIONS,comm->height*DIRECTIONS  ,MPI_DOUBLE,&comm->type );
	MPI_Type_commit( &comm->type );
}

/****************************************************/
void lbm_comm_release_ex5(lbm_comm_t * comm)
{
	//we use the same implementation than ex5 except for type destroy
	lbm_comm_release_ex4(comm);

	//TODO: release MPI type created in init.
	MPI_Type_free(&comm->type);
}


/****************************************************/
void lbm_comm_ghost_exchange_ex5(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	//
	// TODO: Implement the 2D communication with :
	//         - blocking MPI functions
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

	//example to access cell
	//double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
	//double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);

	//TODO:
	//   - implement left/write communications
	//   - implement top/bottom communication (non contiguous)
	//   - implement diagonal communications
	MPI_Status status;
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
	MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), h, MPI_DOUBLE, rank_l, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), h, MPI_DOUBLE, rank_r, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), h, MPI_DOUBLE, rank_r, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), h, MPI_DOUBLE, rank_l, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), 1,comm->type, rank_u, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh,  0, comm->height-2), 1,comm->type, rank_d, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, 0, comm->height-1), 1,comm->type, rank_d, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, 0, 1), 1,comm->type, rank_u, 99, MPI_COMM_WORLD);	
	MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, comm->height-2), 9, MPI_DOUBLE, rank_dr, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, comm->height-1), 9, MPI_DOUBLE, rank_dr, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, 1, comm->height-2), 9, MPI_DOUBLE, rank_dl, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, 0, comm->height-1), 9, MPI_DOUBLE, rank_dl, 99, MPI_COMM_WORLD, &status);
	MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), 9, MPI_DOUBLE, rank_ur, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 1), 9, MPI_DOUBLE, rank_ur, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), 9, MPI_DOUBLE, rank_ul, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, 1, 1), 9, MPI_DOUBLE, rank_ul, 99, MPI_COMM_WORLD);
}
