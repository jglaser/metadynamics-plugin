#ifndef __COLLECTIVE_VARIABLE_H__
#define __COLLECTIVE_VARIABLE_H__

#include <hoomd/hoomd.h>

class CollectiveVariable : public ForceCompute
    {
    public:
        CollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef);
        virtual ~CollectiveVariable() {}

        virtual void computeForces(unsigned int timestep) = 0;

        virtual Scalar getCurrentValue(unsigned int timestep) = 0;

        virtual void setBiasFactor(Scalar bias)
            {
            m_bias = bias;
            }

    protected:
        Scalar m_bias;
    };

void export_CollectiveVariable();

#endif // __COLLECTIVE_VARIABLE_H__
