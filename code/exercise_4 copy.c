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

void corner_cell(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	// Receive from L, send to R, receive from R, send to L.
	// Receive from U, send to D, receive from D, send to U.
	MPI_Status status;
	// s is the size of the corner row for UP/DOWN communication, TODO: CORRECT when n != m.
	int w = DIRECTIONS*(comm->width);
	int h = DIRECTIONS*(comm->height);
	int rank;
	int coords[2];
	// Corner cell (0,0).
	if (comm->rank_x == 0 && comm->rank_y == 0) {
		// LEFT/RIGHT communication.
		coords[0] = comm->rank_x+1;
		coords[1] = comm->rank_y;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		//printf("I'm [%d, %d], my L/R coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		// UP/DOWN communication.
		coords[0] = comm->rank_x;
		coords[1] = comm->rank_y+1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, 0, comm->height-2), w, h, 0);
		MPI_Send(comm->buffer_send_down, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, comm->height-1), comm->buffer_recv_down, w, h, 1);
		// DOWN-RIGHT communication.
		coords[0] = comm->rank_x+1;
		coords[1] = comm->rank_y+1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, comm->height-2), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, comm->height-1), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
	}
	// Corner cell (w-1,0).
	if (comm->rank_x == comm->nb_x-1 && comm->rank_y == 0) {
		coords[0] = comm->rank_x-1;
		coords[1] = comm->rank_y;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		coords[0] = comm->rank_x;
		coords[1] = comm->rank_y+1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, 0, comm->height-2), w, h, 0);
		MPI_Send(comm->buffer_send_down, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, comm->height-1), comm->buffer_recv_down, w, h, 1);
		// DOWN-LEFT communication.
		coords[0] = comm->rank_x-1;
		coords[1] = comm->rank_y+1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, comm->height-2), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, comm->height-1), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
	}
	// Corner cell (0,h-1).
	if (comm->rank_x == 0 && comm->rank_y == comm->nb_y-1) {
		coords[0] = comm->rank_x+1;
		coords[1] = comm->rank_y;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		coords[0] = comm->rank_x;
		coords[1] = comm->rank_y-1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		MPI_Recv(comm->buffer_recv_up, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, 0), comm->buffer_recv_up, w, h, 1);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 1), w, h, 0);
		MPI_Send(comm->buffer_send_up, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		// UP-RIGHT communication.
		coords[0] = comm->rank_x+1;
		coords[1] = comm->rank_y-1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		//printf("I'm [%d, %d], my UP-RIGHT coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 1), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
	}
	// Corner cell (w-1,h-1).
	if (comm->rank_x == comm->nb_x-1 && comm->rank_y == comm->nb_y-1) {
		coords[0] = comm->rank_x-1;
		coords[1] = comm->rank_y;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		//printf("I'm [%d, %d], my L/R coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), h, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		coords[0] = comm->rank_x;
		coords[1] = comm->rank_y-1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		MPI_Recv(comm->buffer_recv_up, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, 0), comm->buffer_recv_up, w, h, 1);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 1), w, h, 0);
		MPI_Send(comm->buffer_send_up, w, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		// UP-LEFT communication.
		coords[0] = comm->rank_x-1;
		coords[1] = comm->rank_y-1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 1), 9, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
	}
}

// TODO: FIX THESE ACOORDING TO THE PREVIOUS FUNCTION
void edge_cell(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
}

/****************************************************/
void lbm_comm_ghost_exchange_ex4(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	// Receive from L, send to R, receive from R, send to L.
	//MPI_Status status;
	// There are 9 directions per cell.
	//int s = DIRECTIONS*(comm->height);
	// Corner cells.
	if ((comm->rank_x == 0 && comm->rank_y == 0) || (comm->rank_x == comm->nb_x-1 && comm->rank_y == 0) ||
		(comm->rank_x == 0 && comm->rank_y == comm->nb_y-1) || (comm->rank_x == comm->nb_x-1 && comm->rank_y == comm->nb_y-1)) {
		corner_cell(comm, mesh);
	// Edge cells
	} else if ((comm->rank_x == 0) || (comm->rank_x == comm->nb_x-1) ||
				(comm->rank_y == 0) || (comm->rank_y == comm->nb_y-1)) {
		edge_cell(comm, mesh);
	// Intermediate cells.
	} else {
		// TODO
	}
	
	//
	// TODO: Implement the 2D communication with :
	//         - blocking MPI functions
	//         - manual copy in temp buffer for non contiguous side 
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
}
