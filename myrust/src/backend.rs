//! Backend selection, chosen at compile time by Cargo features.
//!
//! `--features wgpu` -> Metal (Apple silicon) / Vulkan / DX12 GPU.
//! default (`ndarray`) -> portable CPU backend, fast to build.
//!
//! Training needs autodiff, so the training backend is the compute backend
//! wrapped in `Autodiff`. Inference uses the bare compute backend.
use burn::backend::Autodiff;

#[cfg(feature = "wgpu")]
mod inner {
    pub use burn::backend::wgpu::{Wgpu, WgpuDevice};
    // The `fusion` feature (enabled by the `wgpu`/`metal` Cargo features) makes
    // this backend merge small element-wise kernels into fewer GPU dispatches
    // transparently — a big win for this tiny, dispatch-bound model.
    pub type Compute = Wgpu;
    pub fn device() -> WgpuDevice {
        WgpuDevice::default()
    }
    pub fn name() -> &'static str {
        if cfg!(feature = "metal") {
            "wgpu+fusion (Metal runtime)"
        } else {
            "wgpu+fusion (Metal/Vulkan/DX12)"
        }
    }
}

// ndarray is the fallback whenever wgpu is not enabled.
#[cfg(not(feature = "wgpu"))]
mod inner {
    pub use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type Compute = NdArray;
    pub fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }
    pub fn name() -> &'static str {
        "ndarray (CPU)"
    }
}

/// The forward-only backend (used by `generate`).
pub type Infer = inner::Compute;
/// The training backend, with reverse-mode autodiff.
pub type Train = Autodiff<inner::Compute>;

pub use inner::{device, name};
