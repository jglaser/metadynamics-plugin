// Include the defined classes that are to be exported to python
#include "IntegratorMetaDynamics.h"
#include "CollectiveVariable.h"
#include "LamellarOrderParameter.h"
#include "AspectRatio.h"
#include "OrderParameterMesh.h"
#include "WellTemperedEnsemble.h"
#include "CollectiveWrapper.h"
#include "CollectiveCallback.h"
#include "SteinhardtQl.h"
#include "Density.h"

#ifdef ENABLE_HIP
#include "LamellarOrderParameterGPU.h"
#include "OrderParameterMeshGPU.h"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_MODULE(_metadynamics, m)
    {
    pybind11::bind_vector<std::vector<int3 >>(m, "std_vector_int3");

    export_CollectiveVariable(m);
    export_IntegratorMetaDynamics(m);
    export_LamellarOrderParameter(m);
    export_AspectRatio(m);
    export_OrderParameterMesh(m);
    export_WellTemperedEnsemble(m);
    export_CollectiveWrapper(m);
    export_CollectiveCallback(m);
    export_SteinhardtQl(m);
    export_Density(m);

#ifdef ENABLE_HIP
    export_LamellarOrderParameterGPU(m);
    export_OrderParameterMeshGPU(m);
#endif
    }
