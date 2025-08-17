template <typename ufloat_g_t, int N_DIM>
__host__ __device__ __forceinline__
bool IncludeInBin
(
    const vec3<ufloat_g_t> &vm,
    const vec3<ufloat_g_t> &vM,
    const vec3<ufloat_g_t> &vBm,
    const vec3<ufloat_g_t> &vBM,
    const vec3<ufloat_g_t> &v1,
    const vec3<ufloat_g_t> &v2,
    const vec3<ufloat_g_t> &v3
)
{
    // vm/M are the bin dimensions.
    // vBm/M is the face bounding box.
    // v1/2/3 are the face vertices.
    
    ufloat_g_t tmp = (ufloat_g_t)(0.0);
    ufloat_g_t ex1 = (ufloat_g_t)(0.0);
    ufloat_g_t ey1 = (ufloat_g_t)(0.0);
    
if (N_DIM==2)
{
    // Only consider this calculation if the bounding box intersects the bin.
    if ( !( (vBm.x < vm.x && vBM.x < vm.x) || (vBm.x > vM.x && vBM.x > vM.x) || (vBm.y < vm.y && vBM.y < vm.y) || (vBm.y > vM.y && vBM.y > vM.y) ) )
    {
        // Check if bounding box is entirely inside current bin.
        if (vBm.x > vm.x && vBM.x < vM.x && vBm.y > vm.y && vBM.y < vM.y) { return true; }
        
        // Check if at least one of the vertices is inside the bin.
        if (v1.x > vm.x && v1.x < vM.x) { return true; }
        if (v2.x > vm.x && v2.x < vM.x) { return true; }
        if (v1.y > vm.y && v1.y < vM.y) { return true; }
        if (v2.y > vm.y && v2.y < vM.y) { return true; }
        
        // Check the bottom edge of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vm.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ex1,vm.x,vM.x)) { return true; }
        }
        
        // Check the top edge of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vM.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ex1,vm.x,vM.x)) { return true; }
        }
        
        // Check the left edge of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vm.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ey1,vm.y,vM.y)) { return true; }
        }
        
        // Check the right edge of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vM.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ey1,vm.y,vM.y)) { return true; }
        }
    }
}
else
{
    ufloat_g_t ez1 = (ufloat_g_t)(0.0);
    
    if ( !( (vBm.x < vm.x && vBM.x < vm.x) || (vBm.x > vM.x && vBM.x > vM.x) || (vBm.y < vm.y && vBM.y < vm.y) || (vBm.y > vM.y && vBM.y > vM.y) || (vBm.z < vm.z && vBM.z < vm.z) || (vBm.z > vM.z && vBM.z > vM.z) ) )
    {
        // Check if bounding box is entirely inside current bin.
        if (vBm.x > vm.x && vBM.x < vM.x && vBm.y > vm.y && vBM.y < vM.y && vBm.z > vm.z && vBM.z < vM.z) { return true; }
        
        // Check if bounding box completely surrounds the bin.
        if (vBm.x < vm.x && vBM.x > vM.x && vBm.y < vm.y && vBM.y > vM.y) { return true; }
        if (vBm.y < vm.y && vBM.y > vM.y && vBm.z < vm.z && vBM.z > vM.z) { return true; }
        if (vBm.z < vm.z && vBM.z > vM.z && vBm.x < vm.x && vBM.x > vM.x) { return true; }
        
        // Check if at least one of the vertices is inside the bin.
        if (v1.x > vm.x && v1.x < vM.x && v1.y > vm.y && v1.y < vM.y && v1.z > vm.z && v1.z < vM.z) { return true; }
        if (v2.x > vm.x && v2.x < vM.x && v2.y > vm.y && v2.y < vM.y && v2.z > vm.z && v2.z < vM.z) { return true; }
        if (v3.x > vm.x && v3.x < vM.x && v3.y > vm.y && v3.y < vM.y && v3.z > vm.z && v3.z < vM.z) { return true; }
        
        // Check the bottom face of the bin.
        {
            ez1 = v2.z-v1.z;
            tmp = (vm.z-v1.z)/(ez1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v3.z-v2.z;
            tmp = (vm.z-v2.z)/(ez1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v1.z-v3.z;
            tmp = (vm.z-v3.z)/(ez1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        
        // Check the top face of the bin.
        {
            ez1 = v2.z-v1.z;
            tmp = (vM.z-v1.z)/(ez1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v3.z-v2.z;
            tmp = (vM.z-v2.z)/(ez1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v1.z-v3.z;
            tmp = (vM.z-v3.z)/(ez1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        
        // Check the back face of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vm.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v3.y-v2.y;
            tmp = (vm.y-v2.y)/(ey1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v1.y-v3.y;
            tmp = (vm.y-v3.y)/(ey1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        
        // Check the front face of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vM.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v3.y-v2.y;
            tmp = (vM.y-v2.y)/(ey1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v1.y-v3.y;
            tmp = (vM.y-v3.y)/(ey1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        
        // Check the left face of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vm.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v3.x-v2.x;
            tmp = (vm.x-v2.x)/(ex1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v1.x-v3.x;
            tmp = (vm.x-v3.x)/(ex1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        
        // Check the right face of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vM.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v3.x-v2.x;
            tmp = (vM.x-v2.x)/(ex1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v1.x-v3.x;
            tmp = (vM.x-v3.x)/(ex1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
    }
}
    
    return false;
}
