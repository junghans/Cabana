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

#ifndef CABANA_HALO_HPP
#define CABANA_HALO_HPP

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
  \class Halo

  \brief Halo communication plan for scattering and gathering of ghosted
  data.

  The halo allows for scatter and gather operations between locally-owned and
  ghosted data. All data in the Halo (e.g. export and import data) is from the
  point of view of the forward *GATHER* operation such that, for example, the
  number of exports is the number of exports in the gather and the number of
  imports is the number of imports in the gather. The reverse *SCATTER*
  operation sends the ghosted data back the the uniquely-owned decomposition
  and resolves collisions with atomic addition. Based on input for the forward
  communication plan (where local data will be sent) the local number of
  ghosts is computed.

  Export - the local data we uniquely own that we will send to other ranks for
  those ranks to be used as ghosts.

  Import - the ghost data that we get from other ranks. The rank we get a
  ghost from is the unique owner of that data.
*/
template<class MemorySpace>
class Halo : public CommunicationPlan<MemorySpace>
{
  public:

    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the halo is defined.
    */
    Halo( MPI_Comm comm )
        : CommunicationPlan<MemorySpace>( comm )
    {}

    /*!
      \brief Neighbor and export rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be exporting to and receiving from) as it will be more
      efficient. In this case you already know the topology of the
      point-to-point communication but not how much data to send and receive
      from the neighbors.

      \param num_local The number of locally-owned elements on this rank.

      \param element_export_ranks The local ids of the elements that will be
      exported to other ranks to be used as ghosts. Element ids may be
      repeated in this list if they are sent to multiple destinations. Must be
      the same length as element_export_ranks

      \param element_export_ranks The ranks to which we will export each
      element in element_export_ranks. In this case each rank must be one of the
      neighbor ranks. Must be the same length as element_export_ranks. A rank is
      allowed to send to itself.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication plan.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.
    */
    void createFromExportsAndNeighbors(
        const std::size_t num_local,
        const std::vector<std::size_t>& element_export_ids,
        const std::vector<int>& element_export_ranks,
        const std::vector<int>& neighbor_ranks )
    {
        if ( element_export_ids.size() != element_export_ranks.size() )
            throw std::runtime_error("Export ids and ranks different sizes!");

        _num_local = num_local;
        this->createFromExportsAndTopology(
            element_export_ranks, neighbor_ranks );
        this->createExportSteering(
            false, element_export_ranks, element_export_ids );
    }

    /*!
      \brief Export rank creator. Use this when you don't know who you will
      receiving from - only who you are exporting to. This is less efficient
      than if we already knew who our neighbors were because we have to
      determine the topology of the point-to-point communication first.

      \param num_local The number of locally-owned elements on this rank.

      \param element_export_ranks The local ids of the elements that will be
      sent to other ranks to be used as ghosts. Element ids may be repeated in
      this list if they are sent to multiple destinations. Must be the same
      length as element_export_ranks.

      \param element_export_ranks The ranks to which we will export each
      element in element_export_ranks. Must be the same length as
      element_export_ranks. The neighbor ranks will be determined from this
      list. A rank is allowed to send to itself.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.
    */
    void createFromExports( const std::size_t num_local,
                            const std::vector<std::size_t>& element_export_ids,
                            const std::vector<int>& element_export_ranks )
    {
        if ( element_export_ids.size() != element_export_ranks.size() )
            throw std::runtime_error("Export ids and ranks different sizes!");

        _num_local = num_local;
        this->createFromExportsOnly( element_export_ranks );
        this->createExportSteering(
            false, element_export_ranks, element_export_ids );
    }

    /*!
      \brief Get the number of elements locally owned by this rank.
    */
    std::size_t numLocal() const
    { return _num_local; }

    /*!
      \brief Get the number of ghost elements this rank.
    */
    std::size_t numGhost() const
    { return this->totalNumImport(); }

  private:

    std::size_t _num_local;
};

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_halo : public std::false_type {};

template<typename MemorySpace>
struct is_halo<Halo<MemorySpace> >
    : public std::true_type {};

template<typename MemorySpace>
struct is_halo<const Halo<MemorySpace> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
// Synchronously gather data from the local decomposition to the ghosts using
// the halo forward communication plan. AoSoA version. This is a
// uniquely-owned to multiply-owned communication.
template<class Halo_t, class AoSoA_t>
void gather( const Halo_t& halo,
             AoSoA_t& aosoa,
             int mpi_tag = 1002,
             typename std::enable_if<(is_halo<Halo_t>::value &&
                                      is_aosoa<AoSoA_t>::value),
             int>::type * = 0 )
{
    // Check that the AoSoA is the right size.
    if ( aosoa.size() != halo.numLocal() + halo.numGhost() )
        throw std::runtime_error("AoSoA is the wrong size for scatter!");

    // Allocate a send buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Halo_t::kokkos_memory_space>
        send_buffer( "halo_send_buffer", halo.totalNumExport() );

    // Get the steering vector for the sends.
    auto steering = halo.exportSteering();

    // Gather from the local data into a tuple-contiguous send buffer.
    auto gather_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            send_buffer( i ) = aosoa.getTuple( steering(i) );
        };
    Kokkos::RangePolicy<typename Halo_t::kokkos_execution_space>
        gather_send_buffer_policy( 0, halo.totalNumExport() );
    Kokkos::parallel_for( "Cabana::gather::gather_send_buffer",
                          gather_send_buffer_policy,
                          gather_send_buffer_func );
    Kokkos::fence();

    // Allocate a receive buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Halo_t::kokkos_memory_space>
        recv_buffer( "halo_recv_buffer", halo.totalNumImport() );

    // Post non-blocking receives.
    int num_n = halo.numNeighbor();
    std::vector<MPI_Request> requests( num_n );
    std::pair<std::size_t,std::size_t> recv_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_bounds.second =
            recv_bounds.first + halo.numImport(n);

        auto recv_subview = Kokkos::subview(
            recv_buffer, recv_bounds );

        MPI_Irecv( recv_subview.data(),
                   recv_subview.size() * sizeof(typename AoSoA_t::tuple_type),
                   MPI_CHAR,
                   halo.neighborRank(n),
                   mpi_tag,
                   halo.comm(),
                   &(requests[n]) );

        recv_bounds.first = recv_bounds.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        send_bounds.second =
            send_bounds.first + halo.numExport(n);

        auto send_subview = Kokkos::subview(
            send_buffer, send_bounds );

        MPI_Send( send_subview.data(),
                  send_subview.size() * sizeof(typename AoSoA_t::tuple_type),
                  MPI_CHAR,
                  halo.neighborRank(n),
                  mpi_tag,
                  halo.comm() );

        send_bounds.first = send_bounds.second;
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( num_n );
    MPI_Waitall( num_n, requests.data(), status.data() );

    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = halo.numLocal();
    auto extract_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            std::size_t ghost_idx = i + num_local;
            aosoa.setTuple( ghost_idx, recv_buffer(i) );
        };
    Kokkos::RangePolicy<typename Halo_t::kokkos_execution_space>
        extract_recv_buffer_policy( 0, halo.totalNumImport() );
    Kokkos::parallel_for( "Cabana::gather::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Synchronously gather data from the local decomposition to the ghosts using
// the halo forward communication plan. Slice version. This is a
// uniquely-owned to multiply owned-communication.
template<class Halo_t, class Slice_t>
void gather( const Halo_t& halo,
             Slice_t& slice,
             int mpi_tag = 1002,
             typename std::enable_if<(is_halo<Halo_t>::value &&
                                      is_slice<Slice_t>::value),
             int>::type * = 0 )
{
    // Check that the Slice is the right size.
    if ( slice.size() != halo.numLocal() + halo.numGhost() )
        throw std::runtime_error("Slice is the wrong size for scatter!");

    // Get the number of components in the slice.
    int num_comp = 1;
    for ( int d = 2; d < slice.rank(); ++d )
        num_comp *= slice.extent(d);

    // Get the raw slice data.
    auto slice_data = slice.data();

    // Allocate a send buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::kokkos_memory_space>
        send_buffer( "halo_send_buffer", halo.totalNumExport(), num_comp );

    // Get the steering vector for the sends.
    auto steering = halo.exportSteering();

    // Gather from the local data into a tuple-contiguous send buffer.
    auto gather_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s = Slice_t::index_type::s( steering(i) );
            auto a = Slice_t::index_type::a( steering(i) );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( int n = 0; n < num_comp; ++n )
                send_buffer( i, n ) =
                    slice_data[ slice_offset + n * Slice_t::vector_length ];
        };
    Kokkos::RangePolicy<typename Halo_t::kokkos_execution_space>
        gather_send_buffer_policy( 0, halo.totalNumExport() );
    Kokkos::parallel_for( "Cabana::gather::gather_send_buffer",
                          gather_send_buffer_policy,
                          gather_send_buffer_func );
    Kokkos::fence();

    // Allocate a receive buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::kokkos_memory_space>
        recv_buffer( "halo_recv_buffer", halo.totalNumImport(), num_comp );

    // Post non-blocking receives.
    int num_n = halo.numNeighbor();
    std::vector<MPI_Request> requests( num_n );
    std::pair<std::size_t,std::size_t> recv_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_bounds.second =
            recv_bounds.first + halo.numImport(n);

        auto recv_subview = Kokkos::subview(
            recv_buffer, recv_bounds, Kokkos::ALL );

        MPI_Irecv( recv_subview.data(),
                   recv_subview.size() * sizeof(typename Slice_t::value_type),
                   MPI_CHAR,
                   halo.neighborRank(n),
                   mpi_tag,
                   halo.comm(),
                   &(requests[n]) );

        recv_bounds.first = recv_bounds.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        send_bounds.second =
            send_bounds.first + halo.numExport(n);

        auto send_subview = Kokkos::subview(
            send_buffer, send_bounds, Kokkos::ALL );

        MPI_Send( send_subview.data(),
                  send_subview.size() * sizeof(typename Slice_t::value_type),
                  MPI_CHAR,
                  halo.neighborRank(n),
                  mpi_tag,
                  halo.comm() );

        send_bounds.first = send_bounds.second;
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( num_n );
    MPI_Waitall( num_n, requests.data(), status.data() );

    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = halo.numLocal();
    auto extract_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            std::size_t ghost_idx = i + num_local;
            auto s = Slice_t::index_type::s( ghost_idx );
            auto a = Slice_t::index_type::a( ghost_idx );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( int n = 0; n < num_comp; ++n )
                slice_data[ slice_offset + Slice_t::vector_length * n ] =
                    recv_buffer( i, n );
        };
    Kokkos::RangePolicy<typename Halo_t::kokkos_execution_space>
        extract_recv_buffer_policy( 0, halo.totalNumImport() );
    Kokkos::parallel_for( "Cabana::gather::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Synchronously scatter data from the ghosts to the local decomposition of a
// slice using the halo reverse communication plan. This is a multiply-owned
// to uniquely owned communication.
template<class Halo_t, class Slice_t>
void scatter( const Halo_t& halo,
              Slice_t& slice,
              int mpi_tag = 1003,
              typename std::enable_if<(is_halo<Halo_t>::value &&
                                       is_slice<Slice_t>::value),
              int>::type * = 0 )
{
    // Check that the Slice is the right size.
    if ( slice.size() != halo.numLocal() + halo.numGhost() )
        throw std::runtime_error("Slice is the wrong size for scatter!");

    // Get the number of components in the slice.
    int num_comp = 1;
    for ( int d = 2; d < slice.rank(); ++d )
        num_comp *= slice.extent(d);

    // Get the raw slice data.
    auto slice_data = slice.data();

    // Allocate a send buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::kokkos_memory_space>
        send_buffer( "halo_send_buffer", halo.totalNumImport(), num_comp );

    // Extract the send buffer from the ghosted elements.
    std::size_t num_local = halo.numLocal();
    auto extract_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            std::size_t ghost_idx = i + num_local;
            auto s = Slice_t::index_type::s( ghost_idx );
            auto a = Slice_t::index_type::a( ghost_idx );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( int n = 0; n < num_comp; ++n )
                send_buffer( i, n ) =
                    slice_data[ slice_offset + Slice_t::vector_length * n ];
        };
    Kokkos::RangePolicy<typename Halo_t::kokkos_execution_space>
        extract_send_buffer_policy( 0, halo.totalNumImport() );
    Kokkos::parallel_for( "Cabana::scatter::extract_send_buffer",
                          extract_send_buffer_policy,
                          extract_send_buffer_func );
    Kokkos::fence();

    // Allocate a receive buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::kokkos_memory_space>
        recv_buffer( "halo_recv_buffer", halo.totalNumExport(), num_comp );

    // Post non-blocking receives.
    int num_n = halo.numNeighbor();
    std::vector<MPI_Request> requests( num_n );
    std::pair<std::size_t,std::size_t> recv_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_bounds.second = recv_bounds.first + halo.numExport(n);

        auto recv_subview = Kokkos::subview(
            recv_buffer, recv_bounds, Kokkos::ALL );

        MPI_Irecv( recv_subview.data(),
                   recv_subview.size() * sizeof(typename Slice_t::value_type),
                   MPI_CHAR,
                   halo.neighborRank(n),
                   mpi_tag,
                   halo.comm(),
                   &(requests[n]) );

        recv_bounds.first = recv_bounds.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_bounds = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        send_bounds.second = send_bounds.first + halo.numImport(n);

        auto send_subview = Kokkos::subview(
            send_buffer, send_bounds, Kokkos::ALL );

        MPI_Send( send_subview.data(),
                  send_subview.size() * sizeof(typename Slice_t::value_type),
                  MPI_CHAR,
                  halo.neighborRank(n),
                  mpi_tag,
                  halo.comm() );

        send_bounds.first = send_bounds.second;
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( num_n );
    MPI_Waitall( num_n, requests.data(), status.data() );

    // Get the steering vector for the sends.
    auto steering = halo.exportSteering();

    // Scatter the ghosts in the receive buffer into the local values.
    auto scatter_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s = Slice_t::index_type::s( steering(i) );
            auto a = Slice_t::index_type::a( steering(i) );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( int n = 0; n < num_comp; ++n )
                Kokkos::atomic_add(
                    slice_data + slice_offset + Slice_t::vector_length * n,
                    recv_buffer(i,n) );
        };
    Kokkos::RangePolicy<typename Halo_t::kokkos_execution_space>
        scatter_recv_buffer_policy( 0, halo.totalNumExport() );
    Kokkos::parallel_for( "Cabana::scatter::scatter_recv_buffer",
                          scatter_recv_buffer_policy,
                          scatter_recv_buffer_func );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_HALO_HPP
