/*
 ============================================================================
 Name        : waveparallel.c
 Author      : Engin Aybey
 Version     : v1.0
 Copyright   : All rights reserved.
 Description : Solving 2D wave equation using stationary method with MPI
               Utt=c^2*(Uxx^2+Uyy^2)

 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define TRUE 1
#define FALSE 0

void para_range(int, int, int, int, int*, int*);
int min(int, int);


int main(int argc, char *argv[])
{
    int nmax=16; // size of matrix with boundary conditions
    int n=nmax-2;
    int  irank, nprocs, myrank ;
    int jsta, jend, jsta2, jend1, jnext, jprev, jlen;
    int is,ie,js,je;
    int ista, iend, ista2, iend1, inext, iprev, ilen;
    MPI_Request isend1,isend2,irecv1,irecv2,jsend1,jsend2,jrecv1,jrecv2;
    MPI_Status istatus;
    MPI_Status istatus1;
    int rank, size;
    int newrank, newsize;
    double start_time;
    double end_time;
    double tot,time;
    MPI_Comm new_comm,commrow,commcol;
    int dim[2], period[2], reorder;
    int coord[2],coord1[2], id, ndims;
    int belongs[2];
    int colrank,rowrank;
    int tag=1;
    int i,j,ii,jj;
    int T=1; //time
    int L=20; // T and L determine "dt" .They can be choosen 
               // arbitrarily how much amount of time step is required.
               // Time step(dt) affects 
               // the oscillation velocity of wave and iteration number to converge. 
    float dx,dy,dt,x,y,c;
    float lx,ly;
    lx=M_PI; //0<x<pi
    ly=M_PI; //0<y<pi
    dx=lx/(float)(n+1);
    dy=ly/(float)(n+1);
    dt=(float)T/(float)L; //time step
    c=sqrt(0.5);//wave equation constant

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    MPI_Request request,sendreq1,recvreq1;
    int s_root_procs=(int)sqrt((double)size); //square root of processes
    int num=(int)((double)(n+2)/(double)s_root_procs);

    // to find third layer we need previous two layer.
    // Thus we allocate three matrices, Uk-1=U_pre, Uk=U, Uk+1=U_up
    double** u_pre = (double **)malloc((num+2)*sizeof(double*));
    double** u = (double **)malloc((num+2)*sizeof(double*));
    double** u_up = (double **)malloc((num+2)*sizeof(double*));
    for (i=0; i<num+2; i++){
        u_pre[i] = (double *)malloc(sizeof(double)*(num+2));
        u[i] = (double *)malloc(sizeof(double)*(num+2));
        u_up[i] = (double *)malloc(sizeof(double)*(num+2));
     }
    // allocate vectors for ghost regions buffers.
    double* works1 = (double *)malloc(sizeof(double)*num);
    double* workr1 = (double *)malloc(sizeof(double)*num);
    double* works2 = (double *)malloc(sizeof(double)*num);
    double* workr2 = (double *)malloc(sizeof(double)*num);


    if ( size!=4&&size!=16&&size!=64 ){
        fprintf(stdout,"Please run the program with 4 or 16 or 64 CPUs!!\n");
        MPI_Finalize();
        exit(1);
    }
    // Create a 2D cartesian topology
    ndims=2;            /*  2D matrix grid */
    dim[0]=s_root_procs;           /* rows */
    dim[1]=s_root_procs;           /* columns */
    period[0]=TRUE;     /* row periodic (each column forms a ring) */
    period[1]=TRUE;    /* columns periodic */
    reorder=TRUE;       /* allows processes reordered for efficiency */
    MPI_Cart_create(MPI_COMM_WORLD,ndims,dim,period,reorder,&new_comm);

    MPI_Cart_coords(new_comm, rank, ndims, coord);
    MPI_Cart_rank(new_comm,coord, &newrank);

/* Create 1D row subgrid */
    belongs[0]=0;
    belongs[1]=1; /* this dimension belongs to subgrid */

    MPI_Cart_sub(new_comm, belongs, &commrow);
    MPI_Comm_rank(commrow,&rowrank);

    /* Create 1D column subgrids */
    belongs[0]=1; /* this dimension belongs to subgrid */
    belongs[1]=0;

    MPI_Cart_sub(new_comm, belongs, &commcol);
    MPI_Comm_rank(commcol,&colrank);

    // set global starting and ending indices to fill 
    // the first layer of wave equation in continuous way. 
    para_range(0, nmax-1, s_root_procs, coord[1], &jsta, &jend);

    jsta2 = jsta; 
    jend1 = jend;

    if(coord[1]==0) jsta2=1;
    if(coord[1]==s_root_procs-1) jend1=n;

    para_range(0, nmax-1, s_root_procs, coord[0], &ista, &iend);
    ista2 = ista; 
    iend1 = iend;

    if(coord[0]==0) ista2=1;
    if(coord[0]==s_root_procs-1) iend1=n;
    
    // set local starting and ending indices of stencils 
    if(colrank==0){
    is=2;
    }else{
    is=1;
    }
    if(colrank==s_root_procs-1){
    ie=num-1;
    }else{
    ie=num;
    }
    if(rowrank==0){
    js=2;
    }else{
    js=1;
    }
    if(rowrank==s_root_procs-1){
    je=num-1;
    }else{
    je=num;
    }
    
    // neighbour processes
    coord1[0]=coord[0];
    coord1[1]=coord[1]+1;
    MPI_Cart_rank(new_comm, coord1, &jnext);
    coord1[0]=coord[0];
    coord1[1]=coord[1]-1;
    MPI_Cart_rank(new_comm, coord1, &jprev);
    coord1[0]=coord[0]+1;
    coord1[1]=coord[1];
    MPI_Cart_rank(new_comm, coord1, &inext);
    coord1[0]=coord[0]-1;
    coord1[1]=coord[1];
    MPI_Cart_rank(new_comm, coord1, &iprev);
//    printf("----%d %d %d %d %d ",newrank,jnext,jprev,inext,iprev);
    for(i=0; i<=num+1; i++){
        for(j=0; j<=num+1; j++){
            u_pre[i][j] = 0.0;
            u[i][j] = 0.0;
            u_up[i][j] = 0.0;
        }
    }
    // fill the first layer
    ii=is;
    for (i=ista2; i<=iend1; i++) {
        y = i*dy;
        jj=js;
        for (j=jsta2; j<=jend1; j++) {
               x = j*dx;
                u_pre[ii][jj] = x*(M_PI-x)*y*(M_PI-y);
               jj++;
        }
        ii++;
  }
    //apply boundary conditions
    if(colrank==0){
    for(j=1; j<=num; j++) u_pre[1][j] = 0.0; // first row
    }
    if(rowrank==0){
    for(i=1; i<=num; i++) u_pre[i][1] = 0.0; // first column
    }
    if(colrank==s_root_procs-1){
    for(j=1; j<=num; j++) u_pre[num][j] = 0.0; // last row
    }
    if(rowrank==s_root_procs-1){
    for(i=1; i<=num; i++) u_pre[i][num] = 0.0; // last column
    }
    
    if(coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) works1[i-1]=u_pre[i][num];

    if(coord[1] != 0)
        for(i=1; i<=num; i++) works2[i-1]=u_pre[i][1];
    // communication of ghost regions
    MPI_Isend(&works1[0], num, MPI_DOUBLE, jnext, tag, new_comm, &isend1);
    MPI_Isend(&works2[0], num, MPI_DOUBLE, jprev, tag, new_comm, &isend2);
    MPI_Isend(&u_pre[num][1], num, MPI_DOUBLE, inext, tag, new_comm, &jsend1);
    MPI_Isend(&u_pre[1][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jsend2);
    MPI_Irecv(&workr1[0], num, MPI_DOUBLE, jprev, tag, new_comm, &irecv1);
    MPI_Irecv(&workr2[0], num, MPI_DOUBLE, jnext, tag, new_comm, &irecv2);
    MPI_Irecv(&u_pre[0][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jrecv1);
    MPI_Irecv(&u_pre[num+1][1], num, MPI_DOUBLE, inext, tag, new_comm, &jrecv2);

    MPI_Wait(&isend1, &istatus);
    MPI_Wait(&isend2, &istatus);
    MPI_Wait(&jsend1, &istatus);
    MPI_Wait(&jsend2, &istatus);
    MPI_Wait(&irecv1, &istatus);
    MPI_Wait(&irecv2, &istatus);
    MPI_Wait(&jrecv1, &istatus);
    MPI_Wait(&jrecv2, &istatus);
    
    if (coord[1] != 0)
        for(i=1; i<=num; i++) u_pre[i][0] = workr1[i-1];

    if (coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) u_pre[i][num+1] = workr2[i-1];
  //MPI_Barrier(new_comm);  
  //MPI_Barrier(MPI_COMM_WORLD);  

    int pr; //a procs id whichever you want to see
    pr=0; //atoi(argv[1]);

    // see the part of first layer
    if(newrank==pr){
    printf("\n The first row and the last row & the first column  and the last column are the  ghost regions \n");
    printf("The part of First layer with ghost regions in process %d\n",pr);
    for (i=0; i<=num+1; i++){
        for(j=0; j<=num+1; j++){
           printf("%7.3f",u_pre[i][j]);
        }
        printf("\n");
    }
        printf("\n");
    }

  // Applying initial condition 2 for second layer
  // From Ut=0,we get a reduced explicit difference formula like in below loop.
  for (i=1; i<=num; i++)
  	for (j=1; j<=num; j++)
        // interior points 
        u[i][j] = u_pre[i][j]+(c*c*(dt*dt)/(2*dx*dx))*(u_pre[i][j+1]-2*u_pre[i][j]+u_pre[i][j-1])
                  +(c*c*(dt*dt)/(2*dy*dy))*(u_pre[i+1][j]-2*u_pre[i][j]+u_pre[i-1][j]);

    //apply boundary conditions
    if(colrank==0){
    for(j=1; j<=num; j++) u[1][j] = 0.0; // first row
    }
    if(rowrank==0){
    for(i=1; i<=num; i++) u[i][1] = 0.0; // first column
    }
    if(colrank==s_root_procs-1){
    for(j=1; j<=num; j++) u[num][j] = 0.0; // last row
    }
    if(rowrank==s_root_procs-1){
    for(i=1; i<=num; i++) u[i][num] = 0.0; // last column
    }
    
    if(coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) works1[i-1]=u[i][num];

    if(coord[1] != 0)
        for(i=1; i<=num; i++) works2[i-1]=u[i][1];
    
    // communication of ghost regions
    tag++;
    MPI_Isend(&works1[0], num, MPI_DOUBLE, jnext, tag, new_comm, &isend1);
    MPI_Isend(&works2[0], num, MPI_DOUBLE, jprev, tag, new_comm, &isend2);
    MPI_Isend(&u[num][1], num, MPI_DOUBLE, inext, tag, new_comm, &jsend1);
    MPI_Isend(&u[1][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jsend2);
    MPI_Irecv(&workr1[0], num, MPI_DOUBLE, jprev, tag, new_comm, &irecv1);
    MPI_Irecv(&workr2[0], num, MPI_DOUBLE, jnext, tag, new_comm, &irecv2);
    MPI_Irecv(&u[0][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jrecv1);
    MPI_Irecv(&u[num+1][1], num, MPI_DOUBLE, inext, tag, new_comm, &jrecv2);

    MPI_Wait(&isend1, &istatus);
    MPI_Wait(&isend2, &istatus);
    MPI_Wait(&jsend1, &istatus);
    MPI_Wait(&jsend2, &istatus);
    MPI_Wait(&irecv1, &istatus);
    MPI_Wait(&irecv2, &istatus);
    MPI_Wait(&jrecv1, &istatus);
    MPI_Wait(&jrecv2, &istatus);
    
    if (coord[1] != 0)
        for(i=1; i<=num; i++) u[i][0] = workr1[i-1];

    if (coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) u[i][num+1] = workr2[i-1];
    

   // see the part of second layer
   if(newrank==pr){
    printf("The part of Second layer with ghost regions in process %d \n",pr);
    for (i=0; i<=num+1; i++){
        for(j=0; j<=num+1; j++){
           printf("%7.3f",u[i][j]);
        }
        printf("\n");
    }
    }
   
  int i_up,j_up;
  float C1=cos(M_PI/(float)(nmax-1))+cos(M_PI/(float)(nmax-1));
  float omega=4.0/(2.0+sqrt(4.0-C1*C1));// 0<w<1 ==> Under-relaxation
                                        // w=1 ==> Gauss-Seidel
                                        // 1<w<2 ==> Over-relaxation
  float tol=0.000001;
  int maxiter=1000;
  float temp=0.0;
  double err1=10.0;
  double err2=0.0;
  double err3=10.0;
  double err4;
  int r=0;
  int rsum=0;
  double timesum=0.0;
  double time1;
  // Red Black SOR algorithm for third layer
  // choosing proper c,dt,dx,dy and omega constants is important 
  // for obtaining correct result and reducing iteration number.
  // For the other time steps, "while loop" can be placed in a "for loop"
  // with end of loop, U_pre=U and U=U_up.
  while (err1>tol&r<maxiter) {
        err1=0.0;
        // red points
        start_time = MPI_Wtime();
        i_up=is;
  	for (i=is; i<=ie; i++){
                j_up=(i_up+1)%2+1;
		for (j=(i+1)%2+1; j<=je; j+=2){
		temp = (2*u[i][j]-u_pre[i][j]+(c*c*(dt*dt)/(2.0*dx*dx))*(u_up[i][j+1]-2*u_up[i][j]+u_up[i][j-1]+u_pre[i][j+1]-2*u_pre[i][j]+u_pre[i][j-1]) + (c*c*(dt*dt)/(2.0*dy*dy))*(u_up[i+1][j]-2*u_up[i][j]+u_up[i-1][j]+u_pre[i+1][j]-2*u_pre[i][j]+u_pre[i-1][j]))-u_up[i_up][j_up];
                u_up[i_up][j_up]=u_up[i_up][j_up]+omega*temp;
                j_up+=2;
                if(fabs(temp)>err1) err1=fabs(temp);
		}
            i_up++;
	}
        
        end_time = MPI_Wtime();
        time=(double)(end_time - start_time);
        timesum+=time;

        
         //apply boundary conditions
    if(colrank==0){
    for(j=1; j<=num; j++) u_up[1][j] = 0.0; // first row
    }
    if(rowrank==0){
    for(i=1; i<=num; i++) u_up[i][1] = 0.0; // first column
    }
    if(colrank==s_root_procs-1){
    for(j=1; j<=num; j++) u_up[num][j] = 0.0; // last row
    }
    if(rowrank==s_root_procs-1){
    for(i=1; i<=num; i++) u_up[i][num] = 0.0; // last column
    }
    
    if(coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) works1[i-1]=u_up[i][num];

    if(coord[1] != 0)
        for(i=1; i<=num; i++) works2[i-1]=u_up[i][1];
    
    // communication of ghost regions
    tag++;
    MPI_Isend(&works1[0], num, MPI_DOUBLE, jnext, tag, new_comm, &isend1);
    MPI_Isend(&works2[0], num, MPI_DOUBLE, jprev, tag, new_comm, &isend2);
    MPI_Isend(&u_up[num][1], num, MPI_DOUBLE, inext, tag, new_comm, &jsend1);
    MPI_Isend(&u_up[1][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jsend2);
    MPI_Irecv(&workr1[0], num, MPI_DOUBLE, jprev, tag, new_comm, &irecv1);
    MPI_Irecv(&workr2[0], num, MPI_DOUBLE, jnext, tag, new_comm, &irecv2);
    MPI_Irecv(&u_up[0][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jrecv1);
    MPI_Irecv(&u_up[num+1][1], num, MPI_DOUBLE, inext, tag, new_comm, &jrecv2);

    MPI_Wait(&isend1, &istatus);
    MPI_Wait(&isend2, &istatus);
    MPI_Wait(&jsend1, &istatus);
    MPI_Wait(&jsend2, &istatus);
    MPI_Wait(&irecv1, &istatus);
    MPI_Wait(&irecv2, &istatus);
    MPI_Wait(&jrecv1, &istatus);
    MPI_Wait(&jrecv2, &istatus);
    
    if (coord[1] != 0)
        for(i=1; i<=num; i++) u_up[i][0] = workr1[i-1];

    if (coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) u_up[i][num+1] = workr2[i-1];

        // black points
        start_time = MPI_Wtime();
  	i_up=is;
  	for (i=is; i<=ie; i++){
                j_up=i_up%2+1;
		for (j=i%2+1; j<=je; j+=2){
                temp = (2*u[i][j]-u_pre[i][j]+(c*c*(dt*dt)/(2.0*dx*dx))*(u_up[i][j+1]-2*u_up[i][j]+u_up[i][j-1]+u_pre[i][j+1]-2*u_pre[i][j]+u_pre[i][j-1]) + (c*c*(dt*dt)/(2.0*dy*dy))*(u_up[i+1][j]-2*u_up[i][j]+u_up[i-1][j]+u_pre[i+1][j]-2*u_pre[i][j]+u_pre[i-1][j]))-u_up[i_up][j_up];
                u_up[i_up][j_up]=u_up[i_up][j_up]+omega*temp;
                j_up+=2;
                if(fabs(temp)>err1) err1=fabs(temp);
		}
            i_up++;
	}
        end_time = MPI_Wtime();
        time=(double)(end_time - start_time);
        timesum+=time;
     //apply boundary conditions
    if(colrank==0){
    for(j=1; j<=num; j++) u_up[1][j] = 0.0; // first row
    }
    if(rowrank==0){
    for(i=1; i<=num; i++) u_up[i][1] = 0.0; // first column
    }
    if(colrank==s_root_procs-1){
    for(j=1; j<=num; j++) u_up[num][j] = 0.0; // last row
    }
    if(rowrank==s_root_procs-1){
    for(i=1; i<=num; i++) u_up[i][num] = 0.0; // last column
    }
    
    if(coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) works1[i-1]=u_up[i][num];

    if(coord[1] != 0)
        for(i=1; i<=num; i++) works2[i-1]=u_up[i][1];
    
    // communication of ghost regions
    tag++;
    MPI_Isend(&works1[0], num, MPI_DOUBLE, jnext, tag, new_comm, &isend1);
    MPI_Isend(&works2[0], num, MPI_DOUBLE, jprev, tag, new_comm, &isend2);
    MPI_Isend(&u_up[num][1], num, MPI_DOUBLE, inext, tag, new_comm, &jsend1);
    MPI_Isend(&u_up[1][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jsend2);
    MPI_Irecv(&workr1[0], num, MPI_DOUBLE, jprev, tag, new_comm, &irecv1);
    MPI_Irecv(&workr2[0], num, MPI_DOUBLE, jnext, tag, new_comm, &irecv2);
    MPI_Irecv(&u_up[0][1], num, MPI_DOUBLE, iprev, tag, new_comm, &jrecv1);
    MPI_Irecv(&u_up[num+1][1], num, MPI_DOUBLE, inext, tag, new_comm, &jrecv2);

    MPI_Wait(&isend1, &istatus);
    MPI_Wait(&isend2, &istatus);
    MPI_Wait(&jsend1, &istatus);
    MPI_Wait(&jsend2, &istatus);
    MPI_Wait(&irecv1, &istatus);
    MPI_Wait(&irecv2, &istatus);
    MPI_Wait(&jrecv1, &istatus);
    MPI_Wait(&jrecv2, &istatus);
    
    if (coord[1] != 0)
        for(i=1; i<=num; i++) u_up[i][0] = workr1[i-1];

    if (coord[1] != s_root_procs-1)
        for(i=1; i<=num; i++) u_up[i][num+1] = workr2[i-1];
    r++;
    MPI_Allreduce(&err1,&err2,1,MPI_DOUBLE,MPI_MIN,new_comm);
    err1 = err2;
  }

    MPI_Reduce(&r,&rsum,1,MPI_INT,MPI_SUM,0,new_comm);
    MPI_Reduce(&timesum,&tot,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Barrier(new_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){ 
               printf("\nRunning Time for %d X %d = %f\n\n",nmax,nmax,tot/(double)size);
               printf("\nIteration number:%d \n",rsum/size);
               printf("\nw = %f \n",omega);
               printf("\n");
    }
    MPI_Barrier(new_comm);
    
    // see the part of third layer
    if(newrank==pr){
    printf("The part of Third layer with ghost regions in process %d \n",pr);
    for (i=0; i<num+2; i++){
        for(j=0; j<num+2; j++){
           printf("%7.3f",u_up[i][j]);
        }
        printf("\n");
    }
    }
    

    MPI_Finalize();

    return 0;

}

void para_range(int n1, int n2, int nprocs, int myrank, int *ista, int *iend){
int iwork1, iwork2;

iwork1 = (n2-n1+1)/nprocs;
iwork2 = (n2-n1+1) % nprocs;
*ista= myrank*iwork1 + n1 + min(myrank, iwork2);
*iend = *ista + iwork1 -1;
if(iwork2>myrank) *iend = *iend +1;
}

int min(int x, int y){
    int v;
    if (x>=y) v = y;
    else v = x;
    return v;
}

