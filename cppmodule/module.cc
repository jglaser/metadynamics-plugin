// Include the defined classes that are to be exported to python
#include "IntegratorMetaDynamics.h"
#include "CollectiveVariable.h"
#include "LamellarOrderParameter.h"

#ifdef ENABLE_CUDA
#include "LamellarOrderParameterGPU.h"
#endif


// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_metadynamics)
    {
    export_CollectiveVariable();
    export_IntegratorMetaDynamics();
    export_LamellarOrderParameter();

#ifdef ENABLE_CUDA
    export_LamellarOrderParameterGPU();
#endif
    }
