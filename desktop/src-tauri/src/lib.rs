#[cfg(feature = "desktop")]
mod app;
pub mod core;
pub mod worker;
#[cfg(feature = "desktop")]
pub use app::run;
