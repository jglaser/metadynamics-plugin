#ifndef __COLLECTIVE_VARIABLE_H__
#define __COLLECTIVE_VARIABLE_H__

#include <hoomd/hoomd.h>

#include <string.h>

class CollectiveVariable : public ForceCompute
    {
    public:
        CollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef, const std::string& name);
        virtual ~CollectiveVariable() {}

        virtual void computeForces(unsigned int timestep) = 0;

        virtual Scalar getCurrentValue(unsigned int timestep) = 0;

        virtual void setBiasFactor(Scalar bias)
            {
            m_bias = bias;
            }

        std::string getName()
            {
            return m_cv_name;
            }

    protected:
        Scalar m_bias;

        std::string m_cv_name;
    };

void export_CollectiveVariable();

#endif // __COLLECTIVE_VARIABLE_H__
