#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TAG_ROOT   10
#define TAG_COUNT  20
#define TAG_TOUR   30

#define MAXN 10000

int n;             
int root;     
int *adj_sizes;
int **adj;           
int *local_tour;       
int local_count;


void euler_rec(int u, int parent) {
    local_tour[local_count++] = u;
    for (int i = 0; i < adj_sizes[u]; i++) {
        int v = adj[u][i];
        if (v == parent) continue;
        euler_rec(v, u);
        local_tour[local_count++] = u;
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr,"Uso: %s <arquivo_tree.txt> [root]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        root = (argc >= 3) ? atoi(argv[2]) : 0;

        FILE *f = fopen(argv[1],"r");
        if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD,2); }

        fscanf(f,"%d",&n);
        adj_sizes = calloc(n, sizeof(int));
        adj = malloc(n * sizeof(int*));

        int u,v;
        int *deg = calloc(n,sizeof(int));
        for (int i = 0; i < n-1; i++) {
            fscanf(f,"%d%d",&u,&v);
            deg[u]++; deg[v]++;
        }
        for (int i = 0; i < n; i++) {
            adj_sizes[i] = deg[i];
            adj[i] = malloc(deg[i] * sizeof(int));
            deg[i] = 0;
        }
        rewind(f);
        fscanf(f,"%d",&n);
        for (int i = 0; i < n-1; i++) {
            fscanf(f,"%d%d",&u,&v);
            adj[u][ deg[u]++ ] = v;
            adj[v][ deg[v]++ ] = u;
        }
        fclose(f);
        free(deg);
    } else {
        adj_sizes = calloc(MAXN, sizeof(int));
        adj = malloc(MAXN * sizeof(int*));
    }

    MPI_Bcast(&n,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&root,  1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(adj_sizes, n, MPI_INT, 0, MPI_COMM_WORLD);

    int total = 0;
    if (rank == 0) {
        for (int i = 0; i < n; i++) total += adj_sizes[i];
    }
    MPI_Bcast(&total, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *flat = malloc(total * sizeof(int));
    if (rank == 0) {
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < adj_sizes[i]; j++)
                flat[idx++] = adj[i][j];
        }
    }
    MPI_Bcast(flat, total, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        int idx = 0;
        for (int i = 0; i < n; i++) {
            adj[i] = malloc(adj_sizes[i] * sizeof(int));
            for (int j = 0; j < adj_sizes[i]; j++)
                adj[i][j] = flat[idx++];
        }
    }
    free(flat);

    local_tour  = malloc((2*n-1) * sizeof(int));
    local_count = 0;

    if (rank == 0) {
        int nchild = adj_sizes[root];
        for (int i = 0; i < nchild; i++) {
            int child = adj[root][i];
            MPI_Send(&child, 1, MPI_INT, child, TAG_ROOT, MPI_COMM_WORLD);
        }

        int *global_tour = malloc((2*n-1)*sizeof(int));
        int pos = 0;
        global_tour[pos++] = root;

        for (int i = 0; i < nchild; i++) {
            int child = adj[root][i];
            int cnt;
            MPI_Recv(&cnt,      1, MPI_INT, child, TAG_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(global_tour+pos, cnt, MPI_INT, child, TAG_TOUR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pos += cnt;
            global_tour[pos++] = root;
        }

        printf("Circuito de Euler (len=%d):\n", pos);
        for (int i = 0; i < pos; i++)
            printf("%d ", global_tour[i]);
        printf("\n");
        free(global_tour);
    } else {
        MPI_Status st;
        int subtree_root;
        MPI_Recv(&subtree_root, 1, MPI_INT, 0, TAG_ROOT, MPI_COMM_WORLD, &st);
        euler_rec(subtree_root, root);
        MPI_Send(&local_count, 1, MPI_INT, 0, TAG_COUNT, MPI_COMM_WORLD);
        MPI_Send(local_tour,    local_count, MPI_INT, 0, TAG_TOUR, MPI_COMM_WORLD);
    }

    for (int i = 0; i < n; i++) free(adj[i]);
    free(adj); free(adj_sizes);
    free(local_tour);

    MPI_Finalize();
    return 0;
}