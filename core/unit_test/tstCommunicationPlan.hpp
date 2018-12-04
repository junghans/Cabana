/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_CommunicationPlan.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <vector>

namespace Test
{

//---------------------------------------------------------------------------//
void test1( const bool use_topology )
{
    // Make a communication plan.
    Cabana::CommunicationPlan<TEST_MEMSPACE> comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send all of its data.
    int num_data = 10;
    std::vector<int> export_ranks( num_data, my_rank );
    std::vector<int> neighbor_ranks( 1, my_rank );

    // Create the plan.
    if ( use_topology )
        comm_plan.createFromExportsAndTopology( export_ranks, neighbor_ranks );
    else
        comm_plan.createFromExportsOnly( export_ranks );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank(0), my_rank );
    EXPECT_EQ( comm_plan.numExport(0), num_data );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.numImport(0), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );

    // Create the export steering vector.
    comm_plan.createExportSteering( true, export_ranks );

    // Check the steering vector.
    auto steering = comm_plan.exportSteering();
    Kokkos::View<std::size_t*,Kokkos::HostSpace>
        host_steering( "host_steering", steering.extent(0) );
    Kokkos::deep_copy( host_steering, steering );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering(n) );
}

//---------------------------------------------------------------------------//
void test2( const bool use_topology )
{
    // Make a communication plan.
    Cabana::CommunicationPlan<TEST_MEMSPACE> comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send every other piece of data.
    int num_data = 10;
    std::vector<int> export_ranks( num_data );
    for ( int n = 0; n < num_data; ++n )
        export_ranks[n] = ( 0 == n%2 ) ? my_rank : -1;
    std::vector<int> neighbor_ranks( 1, my_rank );

    // Create the plan
    if ( use_topology )
        comm_plan.createFromExportsAndTopology( export_ranks, neighbor_ranks );
    else
        comm_plan.createFromExportsOnly( export_ranks );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank(0), my_rank );
    EXPECT_EQ( comm_plan.numExport(0), num_data / 2 );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data / 2 );
    EXPECT_EQ( comm_plan.numImport(0), num_data / 2);
    EXPECT_EQ( comm_plan.totalNumImport(), num_data / 2 );

    // Create the export steering vector.
    comm_plan.createExportSteering( true, export_ranks );

    // Check the steering vector.
    auto steering = comm_plan.exportSteering();
    Kokkos::View<std::size_t*,Kokkos::HostSpace>
        host_steering( "host_steering", steering.extent(0) );
    Kokkos::deep_copy( host_steering, steering );
    for ( int n = 0; n < num_data / 2; ++n )
        EXPECT_EQ( n * 2, host_steering(n) );
}

//---------------------------------------------------------------------------//
void test3( const bool use_topology )
{
    // Make a communication plan.
    Cabana::CommunicationPlan<TEST_MEMSPACE> comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Compute the inverse rank.
    int inverse_rank = my_size - my_rank - 1;

    // Every rank will communicate with the rank that is its inverse.
    int num_data = 10;
    std::vector<int> export_ranks( num_data, inverse_rank );
    std::vector<int> neighbor_ranks( 1, inverse_rank );

    // Create the plan with both export ranks and the topology.
    if ( use_topology )
        comm_plan.createFromExportsAndTopology( export_ranks, neighbor_ranks );
    else
        comm_plan.createFromExportsOnly( export_ranks );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank(0), inverse_rank );
    EXPECT_EQ( comm_plan.numExport(0), num_data );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.numImport(0), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );

    // Create the export steering vector.
    comm_plan.createExportSteering( true, export_ranks );

    // Check the steering vector.
    auto steering = comm_plan.exportSteering();
    Kokkos::View<std::size_t*,Kokkos::HostSpace>
        host_steering( "host_steering", steering.extent(0) );
    Kokkos::deep_copy( host_steering, steering );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering(n) );
}

//---------------------------------------------------------------------------//
void test4( const bool use_topology )
{
    // Make a communication plan.
    Cabana::CommunicationPlan<TEST_MEMSPACE> comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank will communicate with all other ranks. Interleave the sends.
    int num_data = 2 * my_size;
    std::vector<int> export_ranks( num_data );
    std::vector<int> neighbor_ranks( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        export_ranks[n] = n;
        export_ranks[n + my_size] = n;
        neighbor_ranks[n] = n;
    }

    // Create the plan
    if ( use_topology )
        comm_plan.createFromExportsAndTopology( export_ranks, neighbor_ranks );
    else
        comm_plan.createFromExportsOnly( export_ranks );

    // Check the plan. Note that if we are sending to ourselves (which we are)
    // that then that data is listed as the first neighbor.
    EXPECT_EQ( comm_plan.numNeighbor(), my_size );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );

    // self send
    EXPECT_EQ( comm_plan.neighborRank(0), my_rank );
    EXPECT_EQ( comm_plan.numExport(0), 2 );
    EXPECT_EQ( comm_plan.numImport(0), 2 );

    // others
    for ( int n = 1; n < my_size; ++n )
    {
        // the algorithm will swap this rank and the first one.
        if ( n == my_rank )
            EXPECT_EQ( comm_plan.neighborRank(n), 0 );
        else
            EXPECT_EQ( comm_plan.neighborRank(n), n );

        EXPECT_EQ( comm_plan.numExport(n), 2 );
        EXPECT_EQ( comm_plan.numImport(n), 2 );
    }

    // Create the export steering vector.
    comm_plan.createExportSteering( true, export_ranks );

    // Check the steering vector. The algorithm will pack the ids according to
    // send rank and self sends will appear first.
    auto steering = comm_plan.exportSteering();
    Kokkos::View<std::size_t*,Kokkos::HostSpace>
        host_steering( "host_steering", steering.extent(0) );
    Kokkos::deep_copy( host_steering, steering );

    // self sends
    EXPECT_EQ( host_steering(0), my_rank );
    EXPECT_EQ( host_steering(1), my_rank + my_size );

    // others
    for ( int n = 1; n < my_size; ++n )
    {
        if ( n == my_rank )
        {
            EXPECT_EQ( host_steering(2*n), 0 );
            EXPECT_EQ( host_steering(2*n+1), my_size );
        }
        else
        {
            EXPECT_EQ( host_steering(2*n), n );
            EXPECT_EQ( host_steering(2*n+1), n + my_size );
        }
    }
}

//---------------------------------------------------------------------------//
void test5( const bool use_topology )
{
    // Make a communication plan.
    Cabana::CommunicationPlan<TEST_MEMSPACE> comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank will communicate with all other ranks. Interleave the sends
    // and only send every other value.
    int num_data = 2 * my_size;
    std::vector<int> export_ranks( num_data );
    std::vector<int> neighbor_ranks( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        export_ranks[n] = -1;
        export_ranks[n + my_size] = n;
        neighbor_ranks[n] = n;
    }

    // Create the plan
    if ( use_topology )
        comm_plan.createFromExportsAndTopology( export_ranks, neighbor_ranks );
    else
        comm_plan.createFromExportsOnly( export_ranks );

    // Check the plan. Note that if we are sending to ourselves (which we are)
    // that then that data is listed as the first neighbor.
    EXPECT_EQ( comm_plan.numNeighbor(), my_size );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data / 2);
    EXPECT_EQ( comm_plan.totalNumImport(), num_data / 2);

    // self send
    EXPECT_EQ( comm_plan.neighborRank(0), my_rank );
    EXPECT_EQ( comm_plan.numExport(0), 1 );
    EXPECT_EQ( comm_plan.numImport(0), 1 );

    // others
    for ( int n = 1; n < my_size; ++n )
    {
        // the algorithm will swap this rank and the first one.
        if ( n == my_rank )
            EXPECT_EQ( comm_plan.neighborRank(n), 0 );
        else
            EXPECT_EQ( comm_plan.neighborRank(n), n );

        EXPECT_EQ( comm_plan.numExport(n), 1 );
        EXPECT_EQ( comm_plan.numImport(n), 1 );
    }

    // Create the export steering vector.
    comm_plan.createExportSteering( true, export_ranks );

    // Check the steering vector. The algorithm will pack the ids according to
    // send rank and self sends will appear first.
    auto steering = comm_plan.exportSteering();
    Kokkos::View<std::size_t*,Kokkos::HostSpace>
        host_steering( "host_steering", steering.extent(0) );
    Kokkos::deep_copy( host_steering, steering );

    // self sends
    EXPECT_EQ( host_steering(0), my_rank + my_size );

    // others
    for ( int n = 1; n < my_size; ++n )
    {
        if ( n == my_rank )
            EXPECT_EQ( host_steering(n), my_size );
        else
            EXPECT_EQ( host_steering(n), n + my_size );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, comm_plan_test_1 )
{ test1(true); }

TEST_F( TEST_CATEGORY, comm_plan_test_2 )
{ test2(true); }

TEST_F( TEST_CATEGORY, comm_plan_test_3 )
{ test3(true); }

TEST_F( TEST_CATEGORY, comm_plan_test_4 )
{ test4(true); }

TEST_F( TEST_CATEGORY, comm_plan_test_5 )
{ test5(true); }

TEST_F( TEST_CATEGORY, comm_plan_test_1_no_topo )
{ test1(false); }

TEST_F( TEST_CATEGORY, comm_plan_test_2_no_topo )
{ test2(false); }

TEST_F( TEST_CATEGORY, comm_plan_test_3_no_topo )
{ test3(false); }

TEST_F( TEST_CATEGORY, comm_plan_test_4_no_topo )
{ test4(false); }

TEST_F( TEST_CATEGORY, comm_plan_test_5_no_topo )
{ test5(false); }

//---------------------------------------------------------------------------//

} // end namespace Test
