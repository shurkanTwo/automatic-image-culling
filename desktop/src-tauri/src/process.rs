//! Every engine runs in an owned process group/job, including probes and descendants.
use std::{
    ops::{Deref, DerefMut},
    process::{Child, Command},
};

pub struct ManagedChild {
    child: Child,
    #[cfg(windows)]
    job: usize,
}
impl ManagedChild {
    pub fn spawn(command: &mut Command) -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = command.spawn()?;
        #[cfg(windows)]
        {
            let mut child = child;
            use std::os::windows::io::AsRawHandle;
            let job = unsafe { windows::CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job.is_null() {
                let error = std::io::Error::last_os_error();
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
            let limits = windows::ExtendedLimits {
                basic: windows::BasicLimits {
                    limit_flags: 0x2000,
                    ..Default::default()
                },
                ..Default::default()
            };
            let configured = unsafe {
                windows::SetInformationJobObject(
                    job,
                    9,
                    (&limits as *const windows::ExtendedLimits).cast(),
                    std::mem::size_of::<windows::ExtendedLimits>() as u32,
                ) != 0
                    && windows::AssignProcessToJobObject(job, child.as_raw_handle()) != 0
            };
            if !configured {
                let error = std::io::Error::last_os_error();
                let _ = child.kill();
                let _ = child.wait();
                unsafe {
                    windows::CloseHandle(job);
                };
                return Err(error);
            }
            return Ok(Self {
                child,
                job: job as usize,
            });
        }
        #[cfg(not(windows))]
        Ok(Self { child })
    }
    pub fn kill_tree(&mut self) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            extern "C" {
                fn kill(pid: i32, signal: i32) -> i32;
            }
            // spawn created a fresh process group whose ID equals the child's PID.
            unsafe {
                kill(-(self.child.id() as i32), 9);
            }
        }
        #[cfg(windows)]
        {
            if unsafe { windows::TerminateJobObject(self.job as *mut std::ffi::c_void, 1) } != 0 {
                return Ok(());
            }
        }
        if self.child.try_wait()?.is_some() {
            return Ok(());
        }
        self.child.kill()
    }
}
impl Deref for ManagedChild {
    type Target = Child;
    fn deref(&self) -> &Child {
        &self.child
    }
}
impl DerefMut for ManagedChild {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.child
    }
}
impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.kill_tree();
        let _ = self.child.wait();
        #[cfg(windows)]
        unsafe {
            windows::CloseHandle(self.job as *mut std::ffi::c_void);
        }
    }
}
#[cfg(windows)]
mod windows {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    pub struct BasicLimits {
        pub per_process_user_time: i64,
        pub per_job_user_time: i64,
        pub limit_flags: u32,
        pub minimum_working_set: usize,
        pub maximum_working_set: usize,
        pub active_process_limit: u32,
        pub affinity: usize,
        pub priority_class: u32,
        pub scheduling_class: u32,
    }
    #[repr(C)]
    #[derive(Default)]
    pub struct IoCounters {
        pub read_operations: u64,
        pub write_operations: u64,
        pub other_operations: u64,
        pub read_bytes: u64,
        pub write_bytes: u64,
        pub other_bytes: u64,
    }
    #[repr(C)]
    #[derive(Default)]
    pub struct ExtendedLimits {
        pub basic: BasicLimits,
        pub io: IoCounters,
        pub process_memory_limit: usize,
        pub job_memory_limit: usize,
        pub peak_process_memory: usize,
        pub peak_job_memory: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateJobObjectW(attributes: *const c_void, name: *const u16) -> *mut c_void;
        pub fn SetInformationJobObject(
            job: *mut c_void,
            class: i32,
            information: *const c_void,
            length: u32,
        ) -> i32;
        pub fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
        pub fn TerminateJobObject(job: *mut c_void, exit_code: u32) -> i32;
        pub fn CloseHandle(handle: *mut c_void) -> i32;
    }
}
