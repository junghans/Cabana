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

#include <Cabana_Distributor.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_DeepCopy.hpp>

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
    Cabana::Distributor<TEST_MEMSPACE> distributor( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send all of its data.
    int num_data = 10;
    std::vector<int> export_ranks( num_data, my_rank );
    std::vector<int> neighbor_ranks( 1, my_rank );

    // Create the plan.
    if ( use_topology )
        distributor.createFromExportsAndNeighbors( export_ranks, neighbor_ranks );
    else
        distributor.createFromExports( export_ranks );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int,double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t data_src( num_data );
    auto slice_int_src = data_src.slice<0>();
    auto slice_dbl_src = data_src.slice<1>();

    // Fill the data.
    auto fill_func =
        KOKKOS_LAMBDA( const int i )
        {
            slice_int_src(i) = my_rank + i;
            slice_dbl_src(i,0) = my_rank + i;
            slice_dbl_src(i,1) = my_rank + i + 0.5;
        };
    Cabana::Experimental::RangePolicy<AoSoA_t::vector_length,TEST_EXECSPACE>
        range_policy( 0, num_data );
    Cabana::Experimental::parallel_for( range_policy, fill_func );

    // Create a second set of data to which we will migrate.
    AoSoA_t data_dst( num_data );
    auto slice_int_dst = data_dst.slice<0>();
    auto slice_dbl_dst = data_dst.slice<1>();

    // Do the migration
    Cabana::migrate( distributor, data_src, data_dst );

    // Check the migration.
    Cabana::AoSoA<DataTypes,Cabana::HostSpace> data_dst_host( num_data );
    auto slice_int_dst_host = data_dst_host.slice<0>();
    auto slice_dbl_dst_host = data_dst_host.slice<1>();
    Cabana::deep_copy( data_dst_host, data_dst );
    for ( int i = 0; i < num_data; ++i )
    {
        EXPECT_EQ( slice_int_dst_host(i), my_rank + i );
        EXPECT_EQ( slice_dbl_dst_host(i,0), my_rank + i );
        EXPECT_EQ( slice_dbl_dst_host(i,1), my_rank + i + 0.5 );
    }
}

//---------------------------------------------------------------------------//
void test2( const bool use_topology )
{
    // Make a communication plan.
    Cabana::Distributor<TEST_MEMSPACE> distributor( MPI_COMM_WORLD );

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
        distributor.createFromExportsAndNeighbors( export_ranks, neighbor_ranks );
    else
        distributor.createFromExports( export_ranks );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int,double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t data( num_data );
    auto slice_int = data.slice<0>();
    auto slice_dbl = data.slice<1>();

    // Fill the data.
    auto fill_func =
        KOKKOS_LAMBDA( const int i )
        {
            slice_int(i) = my_rank + i;
            slice_dbl(i,0) = my_rank + i;
            slice_dbl(i,1) = my_rank + i + 0.5;
        };
    Cabana::Experimental::RangePolicy<AoSoA_t::vector_length,TEST_EXECSPACE>
        range_policy( 0, num_data );
    Cabana::Experimental::parallel_for( range_policy, fill_func );

    // Do the migration in-place
    Cabana::migrate( distributor, data );

    // Check the migration.
    Cabana::AoSoA<DataTypes,Cabana::HostSpace> data_host( num_data / 2 );
    auto slice_int_host = data_host.slice<0>();
    auto slice_dbl_host = data_host.slice<1>();
    Cabana::deep_copy( data_host, data );

    // Check the migration. We received less than we sent so this should have
    // resized the aososa.
    EXPECT_EQ( data.size(), num_data / 2 );
    for ( int i = 0; i < num_data / 2; ++i )
    {
        EXPECT_EQ( slice_int_host(i), my_rank + 2 * i );
        EXPECT_EQ( slice_dbl_host(i,0), my_rank + 2.0 * i );
        EXPECT_EQ( slice_dbl_host(i,1), my_rank + 2.0 * i + 0.5 );
    }
}

//---------------------------------------------------------------------------//
void test3( const bool use_topology )
{
    // Make a communication plan.
    Cabana::Distributor<TEST_MEMSPACE> distributor( MPI_COMM_WORLD );

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
        distributor.createFromExportsAndNeighbors( export_ranks, neighbor_ranks );
    else
        distributor.createFromExports( export_ranks );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int,double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t data_src( num_data );
    auto slice_int_src = data_src.slice<0>();
    auto slice_dbl_src = data_src.slice<1>();

    // Fill the data.
    auto fill_func =
        KOKKOS_LAMBDA( const int i )
        {
            slice_int_src(i) = my_rank + i;
            slice_dbl_src(i,0) = my_rank + i;
            slice_dbl_src(i,1) = my_rank + i + 0.5;
        };
    Cabana::Experimental::RangePolicy<AoSoA_t::vector_length,TEST_EXECSPACE>
        range_policy( 0, num_data );
    Cabana::Experimental::parallel_for( range_policy, fill_func );

    // Create a second set of data to which we will migrate.
    AoSoA_t data_dst( num_data );
    auto slice_int_dst = data_dst.slice<0>();
    auto slice_dbl_dst = data_dst.slice<1>();

    // Do the migration with slices
    Cabana::migrate( distributor, slice_int_src, slice_int_dst );
    Cabana::migrate( distributor, slice_dbl_src, slice_dbl_dst );

    // Check the migration.
    Cabana::AoSoA<DataTypes,Cabana::HostSpace> data_dst_host( num_data );
    Cabana::deep_copy( data_dst_host, data_dst );
    auto slice_int_dst_host = data_dst_host.slice<0>();
    auto slice_dbl_dst_host = data_dst_host.slice<1>();
    for ( int i = 0; i < num_data; ++i )
    {
        EXPECT_EQ( slice_int_dst_host(i), inverse_rank + i );
        EXPECT_EQ( slice_dbl_dst_host(i,0), inverse_rank + i );
        EXPECT_EQ( slice_dbl_dst_host(i,1), inverse_rank + i + 0.5 );
    }
}

//---------------------------------------------------------------------------//
void test4( const bool use_topology )
{
    // Make a communication plan.
    Cabana::Distributor<TEST_MEMSPACE> distributor( MPI_COMM_WORLD );

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
        distributor.createFromExportsAndNeighbors( export_ranks, neighbor_ranks );
    else
        distributor.createFromExports( export_ranks );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int,double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t data_src( num_data );
    auto slice_int_src = data_src.slice<0>();
    auto slice_dbl_src = data_src.slice<1>();

    // Fill the data.
    auto fill_func =
        KOKKOS_LAMBDA( const int i )
        {
            slice_int_src(i) = my_rank;
            slice_dbl_src(i,0) = my_rank;
            slice_dbl_src(i,1) = my_rank + 0.5;
        };
    Cabana::Experimental::RangePolicy<AoSoA_t::vector_length,TEST_EXECSPACE>
        range_policy( 0, num_data );
    Cabana::Experimental::parallel_for( range_policy, fill_func );

    // Create a second set of data to which we will migrate.
    AoSoA_t data_dst( num_data );
    auto slice_int_dst = data_dst.slice<0>();
    auto slice_dbl_dst = data_dst.slice<1>();

    // Do the migration
    Cabana::migrate( distributor, data_src, data_dst );

    // Check the migration.
    Cabana::AoSoA<DataTypes,Cabana::HostSpace> data_dst_host( num_data );
    auto slice_int_dst_host = data_dst_host.slice<0>();
    auto slice_dbl_dst_host = data_dst_host.slice<1>();
    Cabana::deep_copy( data_dst_host, data_dst );

    // self sends
    EXPECT_EQ( slice_int_dst_host(0), my_rank );
    EXPECT_EQ( slice_dbl_dst_host(0,0), my_rank );
    EXPECT_EQ( slice_dbl_dst_host(0,1), my_rank + 0.5 );

    EXPECT_EQ( slice_int_dst_host(1), my_rank );
    EXPECT_EQ( slice_dbl_dst_host(1,0), my_rank );
    EXPECT_EQ( slice_dbl_dst_host(1,1), my_rank + 0.5 );

    // others
    for ( int i = 1; i < my_size; ++i )
    {
        if ( i == my_rank )
        {
            EXPECT_EQ( slice_int_dst_host(2*i), 0 );
            EXPECT_EQ( slice_dbl_dst_host(2*i,0), 0 );
            EXPECT_EQ( slice_dbl_dst_host(2*i,1), 0.5 );

            EXPECT_EQ( slice_int_dst_host(2*i + 1), 0 );
            EXPECT_EQ( slice_dbl_dst_host(2*i + 1,0), 0 );
            EXPECT_EQ( slice_dbl_dst_host(2*i + 1,1), 0.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_dst_host(2*i), i );
            EXPECT_EQ( slice_dbl_dst_host(2*i,0), i );
            EXPECT_EQ( slice_dbl_dst_host(2*i,1), i + 0.5 );

            EXPECT_EQ( slice_int_dst_host(2*i + 1), i );
            EXPECT_EQ( slice_dbl_dst_host(2*i + 1,0), i );
            EXPECT_EQ( slice_dbl_dst_host(2*i + 1,1), i + 0.5 );
        }
    }
}

//---------------------------------------------------------------------------//
void test5( const bool use_topology )
{
    // Make a communication plan.
    Cabana::Distributor<TEST_MEMSPACE> distributor( MPI_COMM_WORLD );

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
        distributor.createFromExportsAndNeighbors( export_ranks, neighbor_ranks );
    else
        distributor.createFromExports( export_ranks );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int,double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t data_src( num_data );
    auto slice_int_src = data_src.slice<0>();
    auto slice_dbl_src = data_src.slice<1>();

    // Fill the data.
    auto fill_func =
        KOKKOS_LAMBDA( const int i )
        {
            slice_int_src(i) = my_rank;
            slice_dbl_src(i,0) = my_rank;
            slice_dbl_src(i,1) = my_rank + 0.5;
        };
    Cabana::Experimental::RangePolicy<AoSoA_t::vector_length,TEST_EXECSPACE>
        range_policy( 0, num_data );
    Cabana::Experimental::parallel_for( range_policy, fill_func );

    // Create a second set of data to which we will migrate.
    AoSoA_t data_dst( my_size );
    auto slice_int_dst = data_dst.slice<0>();
    auto slice_dbl_dst = data_dst.slice<1>();

    // Do the migration with slices
    Cabana::migrate( distributor, slice_int_src, slice_int_dst );
    Cabana::migrate( distributor, slice_dbl_src, slice_dbl_dst );

    // Check the migration.
    Cabana::AoSoA<DataTypes,Cabana::HostSpace> data_host( my_size );
    auto slice_int_host = data_host.slice<0>();
    auto slice_dbl_host = data_host.slice<1>();
    Cabana::deep_copy( data_host, data_dst );

    // self sends
    EXPECT_EQ( slice_int_host(0), my_rank );
    EXPECT_EQ( slice_dbl_host(0,0), my_rank );
    EXPECT_EQ( slice_dbl_host(0,1), my_rank + 0.5 );

    // others
    for ( int i = 1; i < my_size; ++i )
    {
        if ( i == my_rank )
        {
            EXPECT_EQ( slice_int_host(i), 0 );
            EXPECT_EQ( slice_dbl_host(i,0), 0 );
            EXPECT_EQ( slice_dbl_host(i,1), 0.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host(i), i );
            EXPECT_EQ( slice_dbl_host(i,0), i );
            EXPECT_EQ( slice_dbl_host(i,1), i + 0.5 );
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, distributor_test_1 )
{ test1(true); }

TEST_F( TEST_CATEGORY, distributor_test_2 )
{ test2(true); }

TEST_F( TEST_CATEGORY, distributor_test_3 )
{ test3(true); }

TEST_F( TEST_CATEGORY, distributor_test_4 )
{ test4(true); }

TEST_F( TEST_CATEGORY, distributor_test_5 )
{ test5(true); }

TEST_F( TEST_CATEGORY, distributor_test_1_no_topo )
{ test1(false); }

TEST_F( TEST_CATEGORY, distributor_test_2_no_topo )
{ test2(false); }

TEST_F( TEST_CATEGORY, distributor_test_3_no_topo )
{ test3(false); }

TEST_F( TEST_CATEGORY, distributor_test_4_no_topo )
{ test4(false); }

TEST_F( TEST_CATEGORY, distributor_test_5_no_topo )
{ test5(false); }

//---------------------------------------------------------------------------//

} // end namespace Test
