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

#ifndef CABANA_DISTRIBUTOR_HPP
#define CABANA_DISTRIBUTOR_HPP

#include <Cabana_CommunicationPlan.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <vector>
#include <exception>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class Distributor

  \brief Distributor is a communication plan for migrating data from one
  uniquely-owned decomposition to another uniquely owned decomposition.

  The Distributor allows data to be migrated to an entirely new
  decomposition. Only uniquely-owned decompositions are handled (i.e. each
  local element in the source rank has a single unique destination rank).

  Export - things we are sending. Exports are uniquely owned in this
  communication plan.

  Import - things we are receiving. Imports are uniquely owned in this
  communication plan.
*/
template<class MemorySpace>
class Distributor : public CommunicationPlan<MemorySpace>
{
  public:

    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the distributor is defined.
    */
    Distributor( MPI_Comm comm )
        : CommunicationPlan<MemorySpace>( comm )
    {}

    /*!
      \brief Neighbor and export rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may be any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication plan.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    void createFromExportsAndNeighbors(
        const std::vector<int>& element_export_ranks,
        const std::vector<int>& neighbor_ranks )
    {
        this->createFromExportsAndTopology(
            element_export_ranks, neighbor_ranks );
        this->createExportSteering( true, element_export_ranks );
    }

    /*!
      \brief Export rank creator. Use this when you don't know who you will
      receiving from - only who you are sending to. This is less efficient
      than if we already knew who our neighbors were because we have to
      determine the topology of the point-to-point communication first.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    void createFromExports( const std::vector<int>& element_export_ranks )
    {
        this->createFromExportsOnly( element_export_ranks );
        this->createExportSteering( true, element_export_ranks );
    }
};

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_distributor : public std::false_type {};

template<typename MemorySpace>
struct is_distributor<Distributor<MemorySpace> >
    : public std::true_type {};

template<typename MemorySpace>
struct is_distributor<const Distributor<MemorySpace> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
namespace Impl
{

//---------------------------------------------------------------------------//
// Synchronously move data between a source and destination AoSoA by executing
// the forward communication plan.
template<class Distributor_t, class AoSoA_t>
void distributeData(
    const Distributor_t& distributor,
    const AoSoA_t& src,
    AoSoA_t& dst,
    int mpi_tag,
    typename std::enable_if<(is_distributor<Distributor_t>::value &&
                             is_aosoa<AoSoA_t>::value),
    int>::type * = 0 )
{
    // Get the MPI rank we are currently on.
    int my_rank = -1;
    MPI_Comm_rank( distributor.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = distributor.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay = ( distributor.neighborRank(0) == my_rank )
                           ? distributor.numExport(0) : 0;

    // Allocate a send buffer.
    std::size_t num_send = distributor.totalNumExport() - num_stay;
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Distributor_t::kokkos_memory_space>
        send_buffer( "distributor_send_buffer", num_send );

    // Allocate a receive buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Distributor_t::kokkos_memory_space>
        recv_buffer( "distributor_recv_buffer", distributor.totalNumImport() );

    // Get the steering vector for the sends.
    auto steering = distributor.exportSteering();

    // Gather the exports from the source AoSoA into the tuple-contiguous send
    // buffer or the receive buffer if the data is staying. We know that the
    // steering vector is ordered such that the data staying on this rank
    // comes first.
    auto build_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto tpl = src.getTuple( steering(i) );
            if ( i < num_stay )
                recv_buffer( i ) = tpl;
            else
                send_buffer( i - num_stay ) = tpl;
        };
    Kokkos::RangePolicy<typename Distributor_t::kokkos_execution_space>
        build_send_buffer_policy( 0, distributor.totalNumExport() );
    Kokkos::parallel_for( "Cabana::Impl::distributeData::build_send_buffer",
                          build_send_buffer_policy,
                          build_send_buffer_func );
    Kokkos::fence();

    // Post non-blocking receives.
    std::vector<MPI_Request> requests(0);
    requests.reserve( num_n );
    std::pair<std::size_t,std::size_t> recv_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_bounds.second =
            recv_bounds.first + distributor.numImport(n);

        if ( (distributor.numImport(n) > 0) &&
             (distributor.neighborRank(n) != my_rank) )
        {
            auto recv_subview = Kokkos::subview(
                recv_buffer, recv_bounds );

            requests.resize( requests.size() + 1 );

            MPI_Irecv( recv_subview.data(),
                       recv_subview.size() * sizeof(typename AoSoA_t::tuple_type),
                       MPI_CHAR,
                       distributor.neighborRank(n),
                       mpi_tag,
                       distributor.comm(),
                       &(requests.back()) );
        }

        recv_bounds.first = recv_bounds.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        if ( (distributor.numExport(n) > 0) &&
             (distributor.neighborRank(n) != my_rank) )
        {
            send_bounds.second =
                send_bounds.first + distributor.numExport(n);

            auto send_subview = Kokkos::subview(
                send_buffer, send_bounds );

            MPI_Send( send_subview.data(),
                      send_subview.size() * sizeof(typename AoSoA_t::tuple_type),
                      MPI_CHAR,
                      distributor.neighborRank(n),
                      mpi_tag,
                      distributor.comm() );

            send_bounds.first = send_bounds.second;
        }
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( requests.size() );
    MPI_Waitall( requests.size(), requests.data(), status.data() );

    // Extract the receive buffer into the destination AoSoA.
    auto extract_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        { dst.setTuple( i, recv_buffer(i) ); };
    Kokkos::RangePolicy<typename Distributor_t::kokkos_execution_space>
        extract_recv_buffer_policy( 0, distributor.totalNumImport() );
    Kokkos::parallel_for( "Cabana::Impl::distributeData::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
// Synchronously migrate data between two different decompositions using the
// distributor forward communication plan. Multiple AoSoA version.
template<class Distributor_t, class AoSoA_t>
void migrate( const Distributor_t& distributor,
              const AoSoA_t& src,
              AoSoA_t& dst,
              int mpi_tag = 1001,
              typename std::enable_if<(is_distributor<Distributor_t>::value &&
                                       is_aosoa<AoSoA_t>::value),
              int>::type * = 0 )
{
    // Check that src and dst are the right size.
    if ( src.size() != distributor.numExportElement() )
        throw std::runtime_error("Source is the wrong size for migration!");
    if ( dst.size() != distributor.totalNumImport() )
        throw std::runtime_error("Destination is the wrong size for migration!");

    // Move the data.
    Impl::distributeData( distributor, src, dst, mpi_tag );
}

//---------------------------------------------------------------------------//
// Synchronously migrate data between two different decompositions using the
// distributor forward communication plan. Single AoSoA version that will
// resize in-place. Note that resizing does not necessarily allocate more
// memory. The AoSoA memory will only increase if not enough has already been
// reserved/allocated for the needed number of elements.
template<class Distributor_t, class AoSoA_t>
void migrate( const Distributor_t& distributor,
              AoSoA_t& aosoa,
              int mpi_tag = 1001,
              typename std::enable_if<(is_distributor<Distributor_t>::value &&
                                       is_aosoa<AoSoA_t>::value),
              int>::type * = 0 )
{
    // Check that the AoSoA is the right size.
    if ( aosoa.size() != distributor.numExportElement() )
        throw std::runtime_error("AoSoA is the wrong size for migration!");

    // If the destination decomposition is bigger than the source
    // decomposition resize now so we have enough space to do the operation.
    bool dst_is_bigger =
        ( distributor.totalNumImport() > distributor.numExportElement() );
    if ( dst_is_bigger )
        aosoa.resize( distributor.totalNumImport() );

    // Move the data.
    Impl::distributeData( distributor, aosoa, aosoa, mpi_tag );

    // If the destination decomposition is smaller than the source
    // decomposition resize after we have moved the data.
    if ( !dst_is_bigger )
        aosoa.resize( distributor.totalNumImport() );
}

//---------------------------------------------------------------------------//
// Synchronously migrate data between two different decompositions using the
// distributor forward communication plan. Slice version. The user can do this
// in-place with the same slice but they will need to manage the resizing
// themselves as we can't resize slices.
template<class Distributor_t, class Slice_t>
void migrate( const Distributor_t& distributor,
              const Slice_t& src,
              Slice_t& dst,
              int mpi_tag = 1001,
              typename std::enable_if<(is_distributor<Distributor_t>::value &&
                                       is_slice<Slice_t>::value),
              int>::type * = 0 )
{
    // Check that src and dst are the right size.
    if ( src.size() != distributor.numExportElement() )
        throw std::runtime_error("Source is the wrong size for migration!");
    if ( dst.size() != distributor.totalNumImport() )
        throw std::runtime_error("Destination is the wrong size for migration!");

    // Get the number of components in the slice. The source and destination
    // should be the same.
    int num_comp = 1;
    for ( int d = 2; d < src.rank(); ++d )
        num_comp *= src.extent(d);

    // Get the raw slice data.
    auto src_data = src.data();
    auto dst_data = dst.data();

    // Get the MPI rank we are currently on.
    int my_rank = -1;
    MPI_Comm_rank( distributor.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = distributor.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay = ( distributor.neighborRank(0) == my_rank )
                           ? distributor.numExport(0) : 0;

    // Allocate a send buffer. Note this one is layout right so the components
    // are consecutive.
    std::size_t num_send = distributor.totalNumExport() - num_stay;
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Distributor_t::kokkos_memory_space>
        send_buffer( "distributor_send_buffer",
                     num_send, num_comp );

    // Allocate a receive buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Distributor_t::kokkos_memory_space>
        recv_buffer( "distributor_recv_buffer",
                     distributor.totalNumImport(), num_comp );

    // Get the steering vector for the sends.
    auto steering = distributor.exportSteering();

    // Gather from the source Slice into the contiguous send buffer or,
    // if it is part of the local copy, put it directly in the destination
    // Slice.
    auto build_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s_src = Slice_t::index_type::s( steering(i) );
            auto a_src = Slice_t::index_type::a( steering(i) );
            std::size_t src_offset = s_src*src.stride(0) + a_src;
            if ( i < num_stay )
                for ( int n = 0; n < num_comp; ++n )
                    recv_buffer( i, n ) =
                        src_data[ src_offset + n * Slice_t::vector_length ];
            else
                for ( int n = 0; n < num_comp; ++n )
                    send_buffer( i - num_stay, n ) =
                        src_data[ src_offset + n * Slice_t::vector_length ];
        };
    Kokkos::RangePolicy<typename Distributor_t::kokkos_execution_space>
        build_send_buffer_policy( 0, distributor.totalNumExport() );
    Kokkos::parallel_for( "Cabana::migrate::build_send_buffer",
                          build_send_buffer_policy,
                          build_send_buffer_func );
    Kokkos::fence();

    // Post non-blocking receives.
    std::vector<MPI_Request> requests(0);
    requests.reserve( num_n );
    std::pair<std::size_t,std::size_t> recv_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_bounds.second = recv_bounds.first + distributor.numImport(n);

        if ( (distributor.numImport(n) > 0) &&
             (distributor.neighborRank(n) != my_rank) )
        {
            auto recv_subview = Kokkos::subview(
                recv_buffer, recv_bounds, Kokkos::ALL );

            requests.resize( requests.size() + 1 );

            MPI_Irecv( recv_subview.data(),
                       recv_subview.size() * sizeof(typename Slice_t::value_type),
                       MPI_CHAR,
                       distributor.neighborRank(n),
                       mpi_tag,
                       distributor.comm(),
                       &(requests.back()) );
        }

        recv_bounds.first = recv_bounds.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        if ( (distributor.numExport(n) > 0) &&
             (distributor.neighborRank(n) != my_rank) )
        {
            send_bounds.second = send_bounds.first + distributor.numExport(n);

            auto send_subview = Kokkos::subview(
                send_buffer, send_bounds, Kokkos::ALL );

            MPI_Send( send_subview.data(),
                      send_subview.size() * sizeof(typename Slice_t::value_type),
                      MPI_CHAR,
                      distributor.neighborRank(n),
                      mpi_tag,
                      distributor.comm() );

            send_bounds.first = send_bounds.second;
        }
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( requests.size() );
    MPI_Waitall( requests.size(), requests.data(), status.data() );

    // Extract the data from the receive buffer into the destination Slice.
    auto extract_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s = Slice_t::index_type::s( i );
            auto a = Slice_t::index_type::a( i );
            std::size_t dst_offset = s*dst.stride(0) + a;
            for ( int n = 0; n < num_comp; ++n )
                dst_data[ dst_offset + n * Slice_t::vector_length ] =
                    recv_buffer( i, n );
        };
    Kokkos::RangePolicy<typename Distributor_t::kokkos_execution_space>
        extract_recv_buffer_policy( 0, distributor.totalNumImport() );
    Kokkos::parallel_for( "Cabana::migrate::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_DISTRIBUTOR_HPP
