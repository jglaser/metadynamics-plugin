/*! \file IndexGrid.cc
 *  \brief Implements the IndexGrid class
 */
#include "IndexGrid.h"
#include <assert.h>

IndexGrid::IndexGrid()
    {
    m_lengths.resize(1);
    m_factors.resize(1);
    m_lengths[0] = 0;
    m_factors[0] = 1;
    }

IndexGrid::IndexGrid(const std::vector<unsigned int>& lengths)
    {
    setLengths(lengths);
    }

void IndexGrid::setLengths(const std::vector<unsigned int>& lengths)
    {
    m_lengths.resize(lengths.size());
    m_factors.resize(lengths.size());

    m_lengths = lengths;

    for (unsigned int i = 0; i < m_lengths.size(); i++)
        {
        m_factors[i] = (i == 0) ? 1 : ( m_lengths[i-1] * m_factors[i-1] );
        }
    }

unsigned int IndexGrid::getIndex(const std::vector<unsigned int>& coords)
    {
    assert(coords.size() == m_lengths.size());

    unsigned int idx = 0;
    for (unsigned int i = 0; i < m_lengths.size(); i++)
        {
        idx += coords[i] * m_factors[i];
        }

    return idx;
    }

void IndexGrid::getCoordinates(const unsigned int idx, std::vector<unsigned int>& coords)
    {
    assert(coords.size() == m_lengths.size());

    unsigned int rest = idx;
    for (int i = m_lengths.size()-1; i >= 0; i--)
        {
        coords[i] = rest/m_factors[i];
        rest -= coords[i]*m_factors[i];
        }

    assert(rest == 0);
    }

unsigned int IndexGrid::getNumElements()
    {
    unsigned int res = 1;
    for (unsigned int i = 0; i < m_lengths.size(); i++)
        res *= m_lengths[i];

    return res;
    }

unsigned int IndexGrid::getLength(const unsigned int i)
    {
    assert(m_lengths.size() > i);

    return m_lengths[i];
    }

unsigned int IndexGrid::getDimension()
    {
    return m_lengths.size();
    }
