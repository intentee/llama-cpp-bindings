use enumflags2::bitflags;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[bitflags]
#[repr(u16)]
pub enum BuildFeature {
    Cuda = 1 << 0,
    CudaNoVmm = 1 << 1,
    DynamicBackends = 1 << 2,
    DynamicLink = 1 << 3,
    FaultInjection = 1 << 4,
    Metal = 1 << 5,
    Openmp = 1 << 6,
    Rocm = 1 << 7,
    SharedStdcxx = 1 << 8,
    StaticStdcxx = 1 << 9,
    SystemGgml = 1 << 10,
    SystemGgmlStatic = 1 << 11,
    Vulkan = 1 << 12,
}
