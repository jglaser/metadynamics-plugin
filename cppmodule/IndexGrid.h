#ifndef __INDEX_GRID_H__
#define __INDEX_GRID_H__

/*! \file IndexGrid.h
    \brief Defines the IndexGrid class
 */

#include <vector>

//! Helper Class to cacluate a one-dimensional index for a d-dimensional grid
class IndexGrid
    {
    public:
        //! Constructs an index for one-dimensional grid of length 1
        IndexGrid();

        //! Constructs an index for a d-dimensional grid of given lengths
        /*! \param lengths List of grid points in every direction
         */
        IndexGrid(const std::vector<unsigned int>& lengths);
        virtual ~IndexGrid() {}

        //! Set the dimensions and lengths of the grid
        /*! \param lengths List of grid points in every direction
         */
        void setLengths(const std::vector<unsigned int>& lengths);

        //! Get a grid index for given coordinates
        /*! \param coords Coordinates of the grid point in d dimensions
         *  \returns The grid index
         */
        unsigned int getIndex(const std::vector<unsigned int>& coords);

        //! Get the coordinates for a given grid index
        /*! \param idx The grid index
         *  \param coords The grid coordinates (output variable)
         */
        void getCoordinates(const unsigned int idx, std::vector<unsigned int>& coords);

        //! Returns the total number of grid elements
        unsigned int getNumElements();

        //! Returns the length of the grid in a given direction
        /*! \param i Index of the direction
         */
        unsigned int getLength(const unsigned int i);

        //! Returns the dimensionsality of the grid
        unsigned int getDimension();

    private:
        std::vector<unsigned int> m_lengths;  //!< Stores the lengths in every direction
        std::vector<unsigned int> m_factors;  //!< Pre-calculated factors for converting between index and coordinates
    };

#endif // __INDEX_GRID_H__
