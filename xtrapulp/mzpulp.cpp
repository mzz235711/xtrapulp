/*
//@HEADER
// *****************************************************************************
//
//  XtraPuLP: Xtreme-Scale Graph Partitioning using Label Propagation
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <sstream>
#include <string.h>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include "xtrapulp.h"
#include "mzpulp.h"
#include "dist_graph.h"
#include "generate.h"
#include "comms.h"
#include "io_pp.h"
#include "pulp_util.h"
#include "util.h"


namespace pulp{

#ifdef VID_64BIT
#define EFFORMAT "%I64u%I64u%n\n"
#else
#define EFFORMAT "%u%u%n\n"
#endif



    MPI_Comm pulp_comm;

  void mzpulp(MPI_Comm comm_, int workernum, int workerID,  int partnum,std::vector<VertexID> input_edges,  int32_t* &final_parts, uint64_t &total_vnum, uint64_t file_size) 
  {

    srand(time(0));
    setbuf(stdout, 0);

    verbose = false;
    debug = false;
    verify = false;

    
    time_t start_time, end_time;

    char input_filename[1024]; input_filename[0] = '\0';
    char graphname[1024]; graphname[0] = '\0';
    char ouput_graph[]="output.path"; 
    char* graph_name = ouput_graph;
    nprocs = workernum;
    procid = workerID;
    pulp_comm = comm_;
    
    start_time = time(NULL);

    MPI_Barrier(pulp_comm);
    char  num_parts_str[10];
    sprintf(num_parts_str, "%d", partnum);
    char parts_out[1024]; parts_out[0] = '\0';
    char parts_in[1024]; parts_in[0] = '\0';
    strcat(input_filename, graph_name);
    int32_t num_parts = partnum;
    double vert_balance = 1.1;
    double edge_balance = 1.1;
    uint64_t num_runs = 1;
    bool output_time = true;
    bool output_quality = false;
    bool gen_rmat = false;
    bool gen_rand = false;
    bool gen_hd = false;
    uint64_t gen_n = 0;
    uint64_t gen_m_per_n = 16;
    bool offset_vids = false;
    int pulp_seed = rand();
    bool do_bfs_init = true;
    bool do_lp_init = false;
    bool do_repart = false;
    bool do_edge_balance = false;
    bool do_maxcut_balance = false;
    VertexID* origin_edges;
    uint64_t edge_num;

    char c;

    graph_gen_data_t ggi;
    dist_graph_t g;
    pulp_part_control_t ppc = {vert_balance, edge_balance, 
      do_lp_init, do_bfs_init, do_repart, do_edge_balance, do_maxcut_balance,
      false, pulp_seed};

     mpi_data_t comm;
      init_comm_data(&comm);
      
      double elt = omp_get_wtime();
      strcat(graphname, input_filename);
       origin_edges = pulp::load_graph_edges_32(input_edges , &ggi, offset_vids,  file_size,  &edge_num, total_vnum);
    end_time = time(NULL);
    if (nprocs > 1)
    {
      exchange_edges(&ggi, &comm);
      create_graph(&ggi, &g);
      relabel_edges(&g);
    }
    else
    {
      create_graph_serial(&ggi, &g);
    }
    queue_data_t q;
    init_queue_data(&g, &q);
    get_ghost_degrees(&g, &comm, &q);
    pulp_data_t pulp;
    init_pulp_data(&g, &pulp, num_parts);
    double total_elt = 0.0;
    for (uint32_t i = 0; i < num_runs; ++i)
    {
      double elt = 0.0;
      if (parts_in[0] != '\0')
      {
        elt = omp_get_wtime();
        read_parts(parts_in, &g, &pulp, offset_vids);
      }
      elt = omp_get_wtime();
      xtrapulp(&g, &ppc, &comm, &pulp, &q);
      total_elt += omp_get_wtime() - elt;
      if (output_quality)
      {
        part_eval(&g, &pulp);
      }

      char temp_out[1024]; temp_out[0] = '\0';
      strcat(temp_out, parts_out);
      if (strlen(temp_out) == 0)
      {
        strcat(temp_out, graph_name);
        strcat(temp_out, ".parts.");
        strcat(temp_out, num_parts_str);
      }
      if (num_runs > 1)
      {
        std::stringstream ss; 
        ss << "." << i;
        strcat(temp_out, ss.str().c_str());
      }
      elt = omp_get_wtime();
      output_parts(temp_out, &g, pulp.local_parts, offset_vids, final_parts);
    }
    
    clear_graph(&g);
    clear_comm_data(&comm);
    clear_queue_data(&q);
    MPI_Barrier(pulp_comm);
    end_time  = time(NULL);
    return ;
  }
}
