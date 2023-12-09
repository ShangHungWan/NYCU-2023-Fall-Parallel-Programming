#include <mpi.h>
#include <cstdio>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        scanf("%d%d%d", n_ptr, m_ptr, l_ptr);

        *a_mat_ptr = new int[*n_ptr * *m_ptr];
        *b_mat_ptr = new int[*m_ptr * *l_ptr];

        for (int i = 0; i < *n_ptr; i++)
            for (int j = 0; j < *m_ptr; j++)
                scanf("%d", (*a_mat_ptr) + i * (*m_ptr) + j);

        for (int i = 0; i < *m_ptr; i++)
            for (int j = 0; j < *l_ptr; j++)
                scanf("%d", (*b_mat_ptr) + i * (*l_ptr) + j);
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int private_n, private_m, private_l;

    if (rank == 0)
    {
        private_n = n;
        private_m = m;
        private_l = l;
    }

    MPI_Bcast(&private_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&private_m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&private_l, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int from = rank * private_n / size;
    int to = (rank + 1) * private_n / size;

    int *private_a_mat = new int[private_n * private_m];
    int *private_b_mat = new int[private_m * private_l];
    int *c_mat = new int[private_n * private_l];
    int *lengths = new int[size];

    if (rank == 0)
    {
        for (int i = 0; i < private_n; i++)
            for (int j = 0; j < private_m; j++)
                private_a_mat[i * private_m + j] = a_mat[i * private_m + j];
        for (int i = 0; i < private_m; i++)
            for (int j = 0; j < private_l; j++)
                private_b_mat[i * private_l + j] = b_mat[i * private_l + j];
        for (int i = 0; i < private_n * private_l; i++)
            c_mat[i] = 0;
    }

    MPI_Bcast(private_a_mat, private_m * private_l, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(private_b_mat, private_n * private_m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(c_mat, private_n * private_l, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = from; i < to; i++)
        for (int j = 0; j < private_l; j++)
        {
            c_mat[i * private_l + j] = 0;
            for (int k = 0; k < private_m; k++)
            {
                c_mat[i * private_l + j] += private_a_mat[i * private_m + k] * private_b_mat[k * private_l + j];
            }
        }

    if (rank != 0)
    {
        int length = (to - from) * private_l;
        MPI_Send(&length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(c_mat + from * private_l, length, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for (int i = 1, start_line = to; i < size; i++, start_line += lengths[i - 1] / private_l)
        {
            MPI_Recv(lengths + i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(c_mat + start_line * private_l, lengths[i], MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = 0; i < private_n; i++)
        {
            for (int j = 0; j < private_l; j++)
                printf("%d ", c_mat[i * private_l + j]);
            printf("\n");
        }
    }

    delete[] private_a_mat;
    delete[] private_b_mat;
    delete[] c_mat;
    delete[] lengths;
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        delete[] a_mat;
        delete[] b_mat;
    }
}