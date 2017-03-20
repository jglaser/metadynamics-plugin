// Include the defined classes that are to be exported to python
#include "IntegratorMetaDynamics.h"
#include "CollectiveVariable.h"
#include "LamellarOrderParameter.h"
#include "AspectRatio.h"
#include "OrderParameterMesh.h"
#include "WellTemperedEnsemble.h"
#include "CollectiveWrapper.h"

#ifdef ENABLE_CUDA
#include "LamellarOrderParameterGPU.h"
#include "OrderParameterMeshGPU.h"
#endif


#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl_bind.h>

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_PLUGIN(_metadynamics)
    {
    pybind11::module m("_metadynamics");

    pybind11::bind_vector<int3>(m, "std_vector_int3");

    export_CollectiveVariable(m);
    export_IntegratorMetaDynamics(m);
    export_LamellarOrderParameter(m);
    export_AspectRatio(m);
    export_OrderParameterMesh(m);
    export_WellTemperedEnsemble(m);
    export_CollectiveWrapper(m);

#ifdef ENABLE_CUDA
    export_LamellarOrderParameterGPU(m);
    export_OrderParameterMeshGPU(m);
#endif

    return m.ptr();
    }
