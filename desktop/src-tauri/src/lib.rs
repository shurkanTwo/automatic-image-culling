#[cfg(feature = "desktop")]
mod app;
pub mod core;
mod process;
pub mod worker;
#[cfg(feature = "desktop")]
pub use app::run;
