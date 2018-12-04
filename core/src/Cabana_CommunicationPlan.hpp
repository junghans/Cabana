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

#ifndef CABANA_COMMUNICATIONPLAN_HPP
#define CABANA_COMMUNICATIONPLAN_HPP

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <vector>
#include <exception>
#include <algorithm>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class CommunicationPlan

  \brief Communication plan base clase.

  Export - elements we are sending.

  Import - elements we are receiving.
*/
template<class MemorySpace>
class CommunicationPlan
{
  public:

    // Cabana memory space.
    using memory_space = MemorySpace;

    // Kokkos memory space.
    using kokkos_memory_space = typename MemorySpace::kokkos_memory_space;

    // Kokkos execution space.
    using kokkos_execution_space = typename kokkos_memory_space::execution_space;

    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the distributor is defined.
    */
    CommunicationPlan( MPI_Comm comm )
        : _comm( comm )
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
    void createFromExportsAndTopology(
        const std::vector<int>& element_export_ranks,
        const std::vector<int>& neighbor_ranks )
    {
        // Store the number of export elements.
        _num_export_element = element_export_ranks.size();

        // Store the neighbors.
        _neighbors = neighbor_ranks;
        int num_n = _neighbors.size();

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( _comm, &my_rank );

        // If we are sending to ourself put that one first in the neighbor
        // list.
        for ( auto& n : _neighbors )
            if ( n == my_rank )
            {
                std::swap( n, _neighbors[0] );
                break;
            }

        // Initialize import/export sizes.
        _num_export.assign( num_n, 0 );
        _num_import.assign( num_n, 0 );

        // Count the number of sends to each neighbor.
        for ( int n = 0; n < num_n; ++n )
            _num_export[n] = std::count( element_export_ranks.begin(),
                                         element_export_ranks.end(),
                                         _neighbors[n] );

        // Post receives for the number of imports we will get.
        std::vector<MPI_Request> requests(0);
        requests.reserve( num_n );
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
            {
                requests.resize( requests.size() + 1 );
                MPI_Irecv( &_num_import[n],
                           1,
                           MPI_UNSIGNED_LONG,
                           _neighbors[n],
                           1001,
                           _comm,
                           &(requests.back()) );
            }
            else
                _num_import[n] = _num_export[n];

        // Send the number of exports to each of our neighbors.
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
                MPI_Send( &_num_export[n],
                          1,
                          MPI_UNSIGNED_LONG,
                          _neighbors[n],
                          1001,
                          _comm );

        // Wait on receives.
        std::vector<MPI_Status> status( requests.size() );
        MPI_Waitall( requests.size(), requests.data(), status.data() );

        // Get the total number of imports/exports.
        _total_num_export = 0;
        for ( const auto e : _num_export ) _total_num_export += e;
        _total_num_import = 0;
        for ( const auto i : _num_import ) _total_num_import += i;
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
    void createFromExportsOnly( const std::vector<int>& element_export_ranks )
    {
        // Store the number of export elements.
        _num_export_element = element_export_ranks.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( _comm, &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( _comm, &my_rank );

        // Count the number of sends this rank will do to other ranks.
        std::vector<int> num_neighbor( comm_size, 0 );
        for ( const auto export_rank : element_export_ranks )
            if ( export_rank >= 0 )
                ++num_neighbor[ export_rank ];

        // Extract the export ranks and number of exports and then flag the
        // send ranks.
        _neighbors.resize( 0 );
        _num_export.resize( 0 );
        _total_num_export = 0;
        for ( int r = 0; r < comm_size; ++r )
            if ( num_neighbor[r] > 0 )
            {
                _neighbors.push_back( r );
                _num_export.push_back( num_neighbor[r] );
                _total_num_export += num_neighbor[r];
                num_neighbor[r] = 1;
            }

        // Get the number of export ranks and initially allocate the import sizes.
        int num_export_rank = _neighbors.size();
        _num_import.assign( num_export_rank, 0 );

        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of imports to be the number of exports.
        bool self_send = false;
        for ( int n = 0; n < num_export_rank; ++n )
            if ( _neighbors[n] == my_rank )
            {
                std::swap( _neighbors[n], _neighbors[0] );
                std::swap( _num_export[n], _num_export[0] );
                _num_import[0] = _num_export[0];
                self_send = true;
                break;
            }

        // Determine how many total import ranks each neighbor has.
        MPI_Allreduce( MPI_IN_PLACE,
                       num_neighbor.data(),
                       comm_size,
                       MPI_INT,
                       MPI_SUM,
                       _comm );

        // Get the number of import ranks.
        int num_import_rank = num_neighbor[my_rank];
        if ( self_send ) --num_import_rank;

        // Post the expected number of receives and indicate we might get them
        // from any rank.
        std::vector<std::size_t> import_sizes( num_import_rank );
        std::vector<MPI_Request> requests( num_import_rank );
        for ( int n = 0; n < num_import_rank; ++n )
            MPI_Irecv( &import_sizes[n],
                       1,
                       MPI_UNSIGNED_LONG,
                       MPI_ANY_SOURCE,
                       1001,
                       _comm,
                       &requests[n] );

        // Do blocking sends. Dont do any self sends.
        int self_offset = (self_send) ? 1 : 0;
        for ( int n = self_offset; n < num_export_rank; ++n )
            MPI_Send( &_num_export[n],
                      1,
                      MPI_UNSIGNED_LONG,
                      _neighbors[n],
                      1001,
                      _comm );

        // Wait on non-blocking receives.
        std::vector<MPI_Status> status( requests.size() );
        MPI_Waitall( requests.size(), requests.data(), status.data() );

        // Extract the imports. If we did self sends we already know what
        // imports we got from that.
        _total_num_import = (self_send) ? _num_import[0] : 0;
        for ( int i = 0; i < num_import_rank; ++i )
        {
            // Increment the import count.
            _total_num_import += import_sizes[i];

            // See if the neighbor we received stuff from was someone we also
            // sent stuff to. If it was, just record what they sent us.
            bool found_neighbor = false;
            for ( int n = self_offset; n < num_export_rank; ++n )
                if ( _neighbors[n] == status[i].MPI_SOURCE )
                {
                    found_neighbor = true;
                    _num_import[i+self_offset] = import_sizes[i];
                    break;
                }

            // If this is a new neighbor (i.e. someone we didn't send anything
            // to) record this.
            if ( !found_neighbor )
            {
                _neighbors.push_back( status[i].MPI_SOURCE );
                _num_import.push_back( import_sizes[i] );
            }
        }
    }

    /*!
      \brief Create the export steering vector.

      Creates an array describing which export element ids are moved to which
      location in the send buffer of the communcation plan. Ordered such that
      if a rank sends to itself then those values come first.

      \param use_iota True if we use an ordered array from 0 to size-1 for the
      element export ids. This lets us avoid making the ids in cases where
      std::iota would have just been used.

      \param element_export_ranks The ranks to which we are exporting each
      element. We use this to build the steering vector.

      \param element_export_ids The local ids of the elements to be
      exported. This corresponds with the export ranks vector and must be the
      same length if defined. If use_iota is false this vector must be
      defined. If use_iota is true, this vector is ignored.
    */
    void createExportSteering(
        const bool use_iota,
        const std::vector<int>& element_export_ranks,
        const std::vector<std::size_t>& element_export_ids =
        std::vector<std::size_t>() )
    {
        if ( !use_iota &&
             (element_export_ids.size() != element_export_ranks.size()) )
            throw std::runtime_error("Export ids and ranks different sizes!");

        // Calculate the steering offsets for the exports.
        int num_n = _neighbors.size();
        std::vector<std::size_t> offsets( num_n, 0 );
        for ( int n = 1; n < num_n; ++n )
            offsets[n] = offsets[n-1] + _num_export[n-1];

        // Create the export steering vector for writing local elements into
        // the send buffer.
        std::vector<std::size_t> counts( num_n, 0 );
        Kokkos::View<std::size_t*,Kokkos::HostSpace>
            host_steering( "host_steering", _total_num_export );
        for ( std::size_t i = 0; i < element_export_ranks.size(); ++i )
            for ( int n = 0; n < num_n; ++n )
                if ( element_export_ranks[i] == _neighbors[n] )
                {
                    host_steering( offsets[n] + counts[n] ) =
                        (use_iota) ? i : element_export_ids[i];
                    ++counts[n];
                    break;
                }
        _export_steering = Kokkos::View<std::size_t*,kokkos_memory_space>(
            "export_steering", _total_num_export );
        Kokkos::deep_copy( _export_steering, host_steering );
    }

    /*!
      \brief Get the MPI communicator.
    */
    MPI_Comm comm() const
    { return _comm; }

    /*!
      \brief Get the number of neighbor ranks that this rank will communicate
      with.

      \return The number of MPI ranks that will exchange data with this rank.
    */
    int numNeighbor() const
    { return _neighbors.size(); }

    /*!
      \brief Given a local neighbor id get its rank in the MPI communicator.

      \param neighbor The local id of the neighbor to get the rank for.

      \return The MPI rank of the neighbor with the given local id.
    */
    int neighborRank( const int neighbor ) const
    { return _neighbors[neighbor]; }

    /*!
      \brief Get the number of elements this rank will export to a given neighbor.

      \param neighbor The local id of the neighbor to get the number of
      exports for.

      \return The number of elements this rank will export to the neighbor with the
      given local id.
     */
    std::size_t numExport( const int neighbor ) const
    { return _num_export[neighbor]; }

    /*!
      \brief Get the total number of exports this rank will do.

      \return The total number of elements this rank will export to its
      neighbors.
    */
    std::size_t totalNumExport() const
    { return _total_num_export; }

    /*!
      \brief Get the number of elements this rank will import from a given neighbor.

      \param neighbor The local id of the neighbor to get the number of
      imports for.

      \return The number of elements this rank will import from the neighbor
      with the given local id.
     */
    std::size_t numImport( const int neighbor ) const
    { return _num_import[neighbor]; }

    /*!
      \brief Get the total number of imports this rank will do.

      \return The total number of elements this rank will import from its
      neighhbors.
    */
    std::size_t totalNumImport() const
    { return _total_num_import; }

    /*!
      \brief Get the number of export elements.
    */
    std::size_t numExportElement() const
    { return _num_export_element; }

    /*!
      \brief Get the steering vector for the exports.

      \return The steering vector for the exports.

      The steering vector places exports in contiguous chunks by destination
      rank. The chunks are in consecutive order based on the local neighbor id
      (i.e. all elements going to neighbor with local id 0 first, then all
      elements going to neighbor with local id 1, etc.). Only indices of ghost
      elements are in the list of exports.
    */
    Kokkos::View<std::size_t*,kokkos_memory_space> exportSteering() const
    { return _export_steering; }

  private:

    MPI_Comm _comm;
    std::vector<int> _neighbors;
    std::size_t _total_num_export;
    std::size_t _total_num_import;
    std::vector<std::size_t> _num_export;
    std::vector<std::size_t> _num_import;
    std::size_t _num_export_element;
    Kokkos::View<std::size_t*,kokkos_memory_space> _export_steering;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_COMMUNICATIONPLAN_HPP
