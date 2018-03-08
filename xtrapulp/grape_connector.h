#ifndef MZPULP
#define MZPULP
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
#include <vector>
//#include "grape/fragment/edge.h"
#include "config.h"
#include "xtrapulp.h"
#include "dist_graph.h"
#include "generate.h"
#include "comms.h"
#include "io_pp.h"
#include "pulp_util.h"
#include "util.h"

 
namespace pulp{

extern int procid, nprocs;
extern int seed;
extern bool verbose, debug, verify;
//uint32_t static getLine(char * filename);
//void parsetoxpulp(char *filename,char *outputfile ,uint64_t &filesize, std::vector<VertexID> &numstream, std::vector<grape::edata_t> & values);
//void print_usage(char** argv);
//void print_usage_full(char** argv);
void grape_connector(MPI_Comm comm_, int workernum, int workerID,  int partnum, std::vector<VertexID> input_edges, int32_t* &final_parts, uint64_t &total_vnum, uint64_t file_size);
}
#endif
