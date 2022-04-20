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

void fill_buffer(double* dest, double* src, int n)
{
	int pos = 0;
	for (int i=0; i<n; i++){
		pos += i * sizeof(double) * DIRECTIONS;
		dest[pos] = src[pos];
	}
}

void corner_cell(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	// Receive from L, send to R, receive from R, send to L.
	// Receive from U, send to D, receive from D, send to U.
	MPI_Status status;
	int s = DIRECTIONS*(comm->width);
	int rank;
	int coords[2];
	// Corner cell (0,0).
	if (comm->rank_x == 0 && comm->rank_y == 0) {
		coords[0] = 1;
		coords[1] = 0;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my L/R coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		// Temporary buffer for UP/DOWN communication.
		coords[0] = 0;
		coords[1] = 1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my U/D coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, 0, comm->height-2), comm->width);
		MPI_Send(comm->buffer_send_down, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, comm->height-1), comm->buffer_recv_down, comm->width);
	}
	// Corner cell (w-1,0).
	if (comm->rank_x == comm->nb_x-1 && comm->rank_y == 0) {
		coords[0] = comm->nb_x-2;
		coords[1] = 0;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my L/R coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		coords[0] = comm->nb_x-1;
		coords[1] = 1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my U/D coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, 0, comm->height-2), comm->width);
		MPI_Send(comm->buffer_send_down, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, comm->height-1), comm->buffer_recv_down, comm->width);
	}
	// Corner cell (0,h-1).
	if (comm->rank_x == 0 && comm->rank_y == comm->nb_y-1) {
		coords[0] = 1;
		coords[1] = comm->rank_y;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my L/R coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		coords[0] = 0;
		coords[1] = comm->rank_y-1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my U/D coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Recv(comm->buffer_recv_up, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, 0), comm->buffer_recv_up, comm->width);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 1), comm->width);
		MPI_Send(comm->buffer_send_up, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
	}
	// Corner cell (w-1,h-1).
	if (comm->rank_x == comm->nb_x-1 && comm->rank_y == comm->nb_y-1) {
		coords[0] = comm->rank_x-1;
		coords[1] = comm->rank_y;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my L/R coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
		coords[0] = comm->rank_x;
		coords[1] = comm->rank_y-1;
		MPI_Cart_rank(comm->communicator, coords, &rank);
    	printf("I'm [%d, %d], my U/D coordinates [%d, %d], process %d\n", comm->rank_x, comm->rank_y, coords[0], coords[1], rank);
		MPI_Recv(comm->buffer_recv_up, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 0, 0), comm->buffer_recv_up, comm->width);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 1), comm->width);
		MPI_Send(comm->buffer_send_up, s, MPI_DOUBLE, rank, 99, MPI_COMM_WORLD);
	}
}

void edge_cell(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	// Receive from L, send to R, receive from R, send to L.
	// Receive from U, send to D, receive from D, send to U.
	MPI_Status status;
	int s = DIRECTIONS*(comm->height);
	int w = comm->nb_x;
	// Edge cell (x,0).
	if (comm->rank_y == 0) {
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,0,w), 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), s, MPI_DOUBLE, get_rank(comm->rank_x+1,0,w), 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), s, MPI_DOUBLE, get_rank(comm->rank_x+1,0,w), 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,0,w), 99, MPI_COMM_WORLD);
		// Temporary buffer for UP/DOWN communication.
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, comm->height-2, comm->rank_y), comm->width);
		MPI_Send(comm->buffer_send_down, s, MPI_DOUBLE, get_rank(comm->rank_x,1,w), 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, s, MPI_DOUBLE, get_rank(comm->rank_x,1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, comm->height-1, comm->rank_y), comm->buffer_recv_down, comm->width);
	}
	// Edge cell (x,h-1).
	if (comm->rank_y == comm->nb_y-1) {
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,comm->rank_y,w), 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), s, MPI_DOUBLE, get_rank(comm->rank_x+1,comm->rank_y,w), 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), s, MPI_DOUBLE, get_rank(comm->rank_x+1,comm->rank_y,w), 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,comm->rank_y,w), 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_up, s, MPI_DOUBLE, get_rank(comm->rank_x, comm->rank_y-1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 1, 0), comm->buffer_recv_up, comm->width);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 0), comm->width);
		MPI_Send(comm->buffer_send_up, s, MPI_DOUBLE, get_rank(comm->rank_x, comm->rank_y-1,w), 99, MPI_COMM_WORLD);
	}
	// Edge cell (0,y).
	if (comm->rank_x == 0) {
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), s, MPI_DOUBLE, get_rank(1,comm->rank_y,w), 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), s, MPI_DOUBLE, get_rank(1,comm->rank_y,w), 99, MPI_COMM_WORLD, &status);
		MPI_Recv(comm->buffer_recv_up, s, MPI_DOUBLE, get_rank(0, comm->rank_y-1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 1, 0), comm->buffer_recv_up, comm->width);
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, comm->height-2, 0), comm->width);
		MPI_Send(comm->buffer_send_down, s, MPI_DOUBLE, get_rank(0, comm->rank_y+1,w), 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, s, MPI_DOUBLE, get_rank(0, comm->rank_y+1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, comm->height-1, 0), comm->buffer_recv_down, comm->width);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 0), comm->width);
		MPI_Send(comm->buffer_send_up, s, MPI_DOUBLE, get_rank(0, comm->rank_y-1,w), 99, MPI_COMM_WORLD);
	}
	// Edge cell (x,w-1).
	if (comm->rank_x == comm->nb_x-1) {
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,comm->rank_y,w), 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,comm->rank_y,w), 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_up, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y-1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 1, 0), comm->buffer_recv_up, comm->width);
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, comm->height-2, 0), comm->width);
		MPI_Send(comm->buffer_send_down, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y+1,w), 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y+1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, comm->height-1, 0), comm->buffer_recv_down, comm->width);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 0), comm->width);
		MPI_Send(comm->buffer_send_up, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y-1,w), 99, MPI_COMM_WORLD);
	}
}

/****************************************************/
void lbm_comm_ghost_exchange_ex4(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	// Receive from L, send to R, receive from R, send to L.
	MPI_Status status;
	// There are 9 directions per cell.
	int s = DIRECTIONS*(comm->height);
	int w = comm->nb_x;
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
		MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,comm->rank_y,w), 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, comm->width-2, 0), s, MPI_DOUBLE, get_rank(comm->rank_x+1,comm->rank_y,w), 99, MPI_COMM_WORLD);
		MPI_Recv(lbm_mesh_get_cell(mesh, comm->width-1, 0), s, MPI_DOUBLE, get_rank(comm->rank_x+1,comm->rank_y,w), 99, MPI_COMM_WORLD, &status);
		MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), s, MPI_DOUBLE, get_rank(comm->rank_x-1,comm->rank_y,w), 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_up, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y-1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, 1, 0), comm->buffer_recv_up, comm->width);
		fill_buffer(comm->buffer_send_down, lbm_mesh_get_cell(mesh, comm->height-2, 0), comm->width);
		MPI_Send(comm->buffer_send_down, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y+1,w), 99, MPI_COMM_WORLD);
		MPI_Recv(comm->buffer_recv_down, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y+1,w), 99, MPI_COMM_WORLD, &status);
		fill_buffer(lbm_mesh_get_cell(mesh, comm->height-1, 0), comm->buffer_recv_down, comm->width);
		fill_buffer(comm->buffer_send_up, lbm_mesh_get_cell(mesh, 0, 0), comm->width);
		MPI_Send(comm->buffer_send_up, s, MPI_DOUBLE, get_rank(comm->rank_x,comm->rank_y-1,w), 99, MPI_COMM_WORLD);
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
