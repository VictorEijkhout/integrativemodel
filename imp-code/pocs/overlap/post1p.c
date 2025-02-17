#ifdef TRACE
if (procno==0)
  printf("post 1p for step=%d\n",step+1);
#endif

#ifdef TIME
post1start_t[step+1] = tsc();
#endif

#ifdef FAKE
MPI_Isend( outputs1[step+1]+0*N*THICK,1,topsend, proc_top,0,comm, requests1+0 );
MPI_Isend( outputs1[step+1]+1*N*THICK,1,botsend, proc_bot,0,comm, requests1+1 );
MPI_Isend( outputs1[step+1]+2*N*THICK,1,leftsend, proc_left,0,comm, requests1+2 );
MPI_Isend( outputs1[step+1]+3*N*THICK,1,rightsend, proc_right,0,comm, requests1+3 );

MPI_Irecv( inputs1[step+1]+0*N*THICK,1,toprecv, proc_top,0,comm, requests1+4 );
MPI_Irecv( inputs1[step+1]+1*N*THICK,1,botrecv, proc_bot,0,comm, requests1+5 );
MPI_Irecv( inputs1[step+1]+2*N*THICK,1,leftrecv, proc_left,0,comm, requests1+6 );
MPI_Irecv( inputs1[step+1]+3*N*THICK,1,rightrecv, proc_right,0,comm, requests1+7 );
#else
MPI_Isend( outputs1[step+1],1,topsend, proc_top,0,comm, requests1+0 );
MPI_Isend( outputs1[step+1],1,botsend, proc_bot,0,comm, requests1+1 );
MPI_Isend( outputs1[step+1],1,leftsend, proc_left,0,comm, requests1+2 );
MPI_Isend( outputs1[step+1],1,rightsend, proc_right,0,comm, requests1+3 );

MPI_Irecv( inputs1[step+1],1,toprecv, proc_top,0,comm, requests1+4 );
MPI_Irecv( inputs1[step+1],1,botrecv, proc_bot,0,comm, requests1+5 );
MPI_Irecv( inputs1[step+1],1,leftrecv, proc_left,0,comm, requests1+6 );
MPI_Irecv( inputs1[step+1],1,rightrecv, proc_right,0,comm, requests1+7 );
#endif

#ifdef TIME
post1stop_t[step+1] = tsc();
#endif
