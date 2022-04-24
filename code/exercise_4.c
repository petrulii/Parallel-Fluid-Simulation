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
//       8 neighbors using manual copy for non
//       contiguous side and blocking communications
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
//     - Manual copy for non continguous cells
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"
#include <string.h>

/****************************************************/
void lbm_comm_init_ex4(lbm_comm_t * comm, int total_width, int total_height)
{
	// The splitting parameters for the current task.
	int rank;
	int comm_size;
	int dims[] = {0, 0};
	int period[2] = {0, 0};
	int coords[2];
	int reorder = 0;
	MPI_Comm communicator;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	// The number of tasks along X axis and Y axis.
	// Letting MPI choose the dimension decomposition.
	MPI_Dims_create(comm_size, 2, dims);
	comm->nb_x = dims[0];
	comm->nb_y = dims[1];
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, reorder, &communicator);
	MPI_Cart_coords(communicator, rank, 2, coords);
	comm->communicator = communicator;
	//printf("Coordinates [%d, %d], I'm Process %d/%d\n", coords[0], coords[1], rank, comm_size);

	// The current task position in the splitting
	comm->rank_x = coords[0];
	comm->rank_y = coords[1];

	// The local sub-domain size.
	comm->width = (total_width / comm->nb_x) + 2;
	comm->height = (total_height / comm->nb_y) + 2;

	// The absolute position in the global mesh.
	comm->x = comm->rank_x*(total_width / comm->nb_x);
	comm->y = comm->rank_y*(total_height / comm->nb_y);

	// Temporary copy buffer for every step.
	comm->buffer_recv_down = malloc(sizeof(double) * DIRECTIONS * comm->width);
	comm->buffer_recv_up = malloc(sizeof(double) * DIRECTIONS * comm->width);
	comm->buffer_send_down = malloc(sizeof(double) * DIRECTIONS * comm->width);
	comm->buffer_send_up = malloc(sizeof(double) * DIRECTIONS * comm->width);

	//if debug print comm
	//lbm_comm_print(comm);
}

/****************************************************/
void lbm_comm_release_ex4(lbm_comm_t * comm)
{
	free(comm->buffer_recv_down);
	free(comm->buffer_recv_up);
	free(comm->buffer_send_down);
	free(comm->buffer_send_up);
}

int get_rank(int x, int y, int width)
{
	return y*width+x;
}

void fill_buffer(double* dest, double* src, int width, int height, int to_cell)
{
	width = width/DIRECTIONS;
	if (to_cell == 1) {
		for (int i=0; i<width; i++){
			for (int k=0; k<DIRECTIONS; k++){
				dest[i*height+k] = src[i*DIRECTIONS+k];
			}
		}
	} else {
		for (int i=0; i<width; i++){
			for (int k=0; k<DIRECTIONS; k++){
				dest[i*DIRECTIONS+k] = src[i*height+k];
			}
		}
	}
}

void print_buffer(double* buffer, int n)
{
	printf("BUFFER : ");
	for (int i=0; i<n; i++)
		printf("%.1f ", buffer[i]);
	printf("\n");
}

int rank_Calc(lbm_comm_t * comm,int cord0,int cord1){
	int coords[2];
	int rank;
	coords[0] = comm->rank_x+cord0;
	coords[1] = comm->rank_y+cord1;
	if (coords[0] < 0 || coords[1] < 0 || coords[0] >= comm->nb_x || coords[1] >= comm->nb_y) {
		rank = MPI_PROC_NULL;
	} else {
		MPI_Cart_rank(comm->communicator, coords, &rank);
	}
	return rank;
}
/****************************************************/
void lbm_comm_ghost_exchange_ex4(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	MPI_Status status;
	int w = DIRECTIONS*(comm->width);
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

	if (rank_u != MPI_PROC_NULL){
		MPI_Recv(comm->buffer_recv_up, w, MPI_DOUBLE, rank_u, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, 0), comm->buffer_recv_up, w, h, 1);
	}
	if (rank_d != MPI_PROC_NULL){
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, 0, comm->height-2), w, h, 0);
		MPI_Send(comm->buffer_send_down, w, MPI_DOUBLE, rank_d, 99, MPI_COMM_WORLD);
	}
	if (rank_d != MPI_PROC_NULL){
		MPI_Recv(comm->buffer_recv_down, w, MPI_DOUBLE, rank_d, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, comm->height-1), comm->buffer_recv_down, w, h, 1);
	}
	if (rank_u != MPI_PROC_NULL){
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 1), w, h, 0);
		MPI_Send(comm->buffer_send_up, w, MPI_DOUBLE, rank_u, 99, MPI_COMM_WORLD);
	}

	MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, comm->height-2), 9, MPI_DOUBLE, rank_dr, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, comm->height-1), 9, MPI_DOUBLE, rank_dr, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, 1, comm->height-2), 9, MPI_DOUBLE, rank_dl, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, 0, comm->height-1), 9, MPI_DOUBLE, rank_dl, 99, MPI_COMM_WORLD, &status);
	MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), 9, MPI_DOUBLE, rank_ur, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 1), 9, MPI_DOUBLE, rank_ur, 99, MPI_COMM_WORLD);
	MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), 9, MPI_DOUBLE, rank_ul, 99, MPI_COMM_WORLD, &status);
	MPI_Send(lbm_mesh_get_cell(mesh, 1, 1), 9, MPI_DOUBLE, rank_ul, 99, MPI_COMM_WORLD);

}