#ifndef __INDEX_GRID_H__
#define __INDEX_GRID_H__

#include <vector>

class IndexGrid
    {
    public:
        IndexGrid();
        IndexGrid(const std::vector<unsigned int>& lengths);
        virtual ~IndexGrid() {}

        void setLengths(const std::vector<unsigned int>& lengths);

        unsigned int getIndex(const std::vector<unsigned int>& coords);

        void getCoordinates(const unsigned int idx, std::vector<unsigned int>& coords);

        unsigned int getNumElements();

        unsigned int getLength(const unsigned int i);

        unsigned int getDimension();

    private:
        std::vector<unsigned int> m_lengths; 
        std::vector<unsigned int> m_factors;
    };

#endif // __INDEX_GRID_H__
