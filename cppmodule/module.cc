// Include the defined classes that are to be exported to python
#include "IntegratorMetaDynamics.h"
#include "CollectiveVariable.h"
#include "LamellarOrderParameter.h"
#include "AspectRatio.h"
#include "OrderParameterMesh.h"

#ifdef ENABLE_CUDA
#include "LamellarOrderParameterGPU.h"
#include "OrderParameterMeshGPU.h"
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
    export_AspectRatio();
    export_OrderParameterMesh();

#ifdef ENABLE_CUDA
    export_LamellarOrderParameterGPU();
    export_OrderParameterMeshGPU();
#endif
    }
