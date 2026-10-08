use crate::core::{path_string, Group, Photo, Result, Store};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use uuid::Uuid;

#[derive(Clone)]
pub struct Engine {
    executable: PathBuf,
    python: bool,
    working_dir: Option<PathBuf>,
}
impl Engine {
    pub fn discover(resources: PathBuf) -> Self {
        if let Some(path) = std::env::var_os("PHOTO_SELECT_ENGINE") {
            return Self {
                executable: path.into(),
                python: false,
                working_dir: None,
            };
        }
        let executable = resources
            .join("resources")
            .join("engine")
            .join(if cfg!(windows) {
                "photo-select-engine.exe"
            } else {
                "photo-select-engine"
            });
        if executable.is_file() {
            return Self {
                executable,
                python: false,
                working_dir: None,
            };
        }
        let python = std::env::var_os("PHOTO_SELECT_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }));
        Self {
            executable: python,
            python: true,
            working_dir: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")),
        }
    }
    pub fn executable(path: PathBuf) -> Self {
        Self {
            executable: path,
            python: false,
            working_dir: None,
        }
    }
    pub fn command(&self, operation: &str) -> Command {
        let mut command = Command::new(&self.executable);
        if self.python {
            command.args(["-m", "culling_engine"]);
            if let Some(dir) = &self.working_dir {
                command.current_dir(dir);
            }
        }
        command
            .arg(operation)
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        command
    }
    pub fn available(&self) -> bool {
        let Ok(mut child) = self.command("self-test").spawn() else {
            return false;
        };
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return status.success(),
                Err(_) => return false,
                Ok(None) => {}
            }
            if start.elapsed() > Duration::from_secs(10) {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
            thread::sleep(Duration::from_millis(30));
        }
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub project_id: String,
    pub job_id: String,
    pub phase: String,
    pub processed: u64,
    pub total: u64,
    pub current_file: Option<String>,
    pub failed: u64,
    pub message: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Detail {
    pub detail_path: String,
    pub width: u32,
    pub height: u32,
}
pub type Emitter = Arc<dyn Fn(&str, Value) + Send + Sync>;
struct Job {
    id: String,
    cancelled: AtomicBool,
    child: Mutex<Child>,
}
#[derive(Clone)]
pub struct Workers {
    pub store: Store,
    pub engine: Engine,
    jobs: Arc<Mutex<HashMap<String, Arc<Job>>>>,
    emit: Emitter,
}
impl Workers {
    pub fn new(store: Store, engine: Engine, emit: Emitter) -> Self {
        Self {
            store,
            engine,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            emit,
        }
    }
    pub fn start(&self, project_id: &str) -> Result<String> {
        let mut jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        if jobs.contains_key(project_id) {
            return Err("An import is already running for this project".into());
        }
        let project = self.store.project(project_id)?;
        let cache = self.store.prepare_cache(project_id)?;
        let mut command = self.engine.command("scan");
        command
            .arg("--source")
            .arg(&project.source_dir)
            .arg("--cache")
            .arg(path_string(&cache));
        let mut child = command
            .spawn()
            .map_err(|e| format!("Cannot start analysis engine: {e}"))?;
        let stdout = child.stdout.take().ok_or("Engine stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Engine stderr unavailable")?;
        if let Err(e) = self.store.set_status(project_id, "running", None) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
        let job = Arc::new(Job {
            id: Uuid::new_v4().to_string(),
            cancelled: AtomicBool::new(false),
            child: Mutex::new(child),
        });
        jobs.insert(project_id.into(), job.clone());
        let id = job.id.clone();
        let workers = self.clone();
        let project_id = project_id.to_string();
        thread::spawn(move || {
            let diagnostics = thread::spawn(move || read_diagnostics(stderr));
            let result = workers.consume(&project_id, &job, BufReader::new(stdout));
            // A malformed record must not leave a subprocess writing into a closed pipe indefinitely.
            if result.is_err() {
                if let Ok(mut child) = job.child.lock() {
                    let _ = child.kill();
                }
            }
            let status = wait_for_child(&job);
            let diagnostic = diagnostics.join().unwrap_or_default();
            let counts = result.as_ref().ok().copied().unwrap_or((0, 0, 0));
            let cancelled = job.cancelled.load(Ordering::SeqCst);
            let (mut phase, reason) = if cancelled {
                ("cancelled", None)
            } else {
                match (result, status) {
                    (Ok(_), Ok(s)) if s.success() => ("complete", None),
                    (Err(e), _) => ("error", Some(attach_diagnostic(e, &diagnostic))),
                    (_, Ok(s)) => (
                        "error",
                        Some(attach_diagnostic(
                            format!("Analysis engine exited with {s}"),
                            &diagnostic,
                        )),
                    ),
                    (_, Err(e)) => ("error", Some(e)),
                }
            };
            let status = match phase {
                "complete" => "completed",
                "cancelled" => "cancelled",
                _ => "failed",
            };
            let persistence = workers
                .store
                .set_status(&project_id, status, reason.clone());
            let final_message = match persistence {
                Ok(()) => reason,
                Err(e) => {
                    phase = "error";
                    Some(format!("Unable to save import status: {e}"))
                }
            };
            let _ = workers.progress(Progress {
                project_id: project_id.clone(),
                job_id: job.id.clone(),
                phase: phase.into(),
                processed: counts.0,
                total: counts.1,
                current_file: None,
                failed: counts.2,
                message: final_message,
            });
            if let Ok(mut jobs) = workers.jobs.lock() {
                jobs.remove(&project_id);
            }
            workers.updated(&project_id);
        });
        Ok(id)
    }
    fn consume(
        &self,
        project_id: &str,
        job: &Job,
        mut reader: impl BufRead,
    ) -> Result<(u64, u64, u64)> {
        let mut line = String::new();
        let mut total = 0;
        let mut processed = 0;
        let mut failed = 0;
        let mut complete = false;
        let mut last_update = Instant::now() - Duration::from_secs(1);
        loop {
            line.clear();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                break;
            }
            if job.cancelled.load(Ordering::SeqCst) {
                return Ok((processed, total, failed));
            }
            if line.trim().is_empty() {
                continue;
            }
            let record: Value =
                serde_json::from_str(&line).map_err(|e| format!("Invalid engine response: {e}"))?;
            let kind = record
                .get("type")
                .and_then(Value::as_str)
                .ok_or("Engine record has no type")?;
            let mut phase = None;
            let mut message = None;
            match kind {
                "scan" => {
                    total = number(&record, "total")?;
                    phase = Some("scan");
                }
                "photo" => {
                    let photo: Photo = serde_json::from_value(
                        record.get("photo").cloned().ok_or("Missing photo")?,
                    )
                    .map_err(|e| e.to_string())?;
                    self.store.ingest_photo(project_id, photo)?;
                }
                "progress" => {
                    processed = number(&record, "processed")?;
                    total = number(&record, "total")?;
                    failed = number(&record, "failed")?;
                    phase = Some("analysis");
                }
                "groups" => {
                    let groups: Vec<Group> = serde_json::from_value(
                        record.get("groups").cloned().ok_or("Missing groups")?,
                    )
                    .map_err(|e| e.to_string())?;
                    self.store.ingest_groups(project_id, groups)?;
                    phase = Some("grouping");
                }
                "complete" => {
                    complete = true;
                    processed = number(&record, "processed")?;
                    total = number(&record, "total")?;
                    failed = number(&record, "failed")?;
                }
                "error" => {
                    let reason = record
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("Unknown engine error");
                    let file = record
                        .get("path")
                        .and_then(Value::as_str)
                        .filter(|path| !path.trim().is_empty());
                    if file.is_none() {
                        return Err(reason.into());
                    }
                    // A file can disappear between discovery and stat. The engine continues the batch.
                    failed = failed.saturating_add(1);
                    phase = Some("analysis");
                    message = Some(reason.to_string());
                }
                _ => return Err(format!("Unknown engine record: {kind}")),
            }
            if let Some(phase) = phase {
                self.progress(Progress {
                    project_id: project_id.into(),
                    job_id: job.id.clone(),
                    phase: phase.into(),
                    processed,
                    total,
                    current_file: record
                        .get("currentFile")
                        .or_else(|| record.get("path"))
                        .and_then(Value::as_str)
                        .map(String::from),
                    failed,
                    message,
                })?;
            }
            if last_update.elapsed() > Duration::from_millis(350) {
                self.updated(project_id);
                last_update = Instant::now();
            }
        }
        if !complete && !job.cancelled.load(Ordering::SeqCst) {
            return Err("Analysis engine stopped before completing the import".into());
        }
        Ok((processed, total, failed))
    }
    pub fn is_running(&self, id: &str) -> bool {
        self.jobs.lock().map(|v| v.contains_key(id)).unwrap_or(true)
    }
    pub fn cancel(&self, id: &str) -> Result<()> {
        let jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        if let Some(job) = jobs.get(id) {
            job.cancelled.store(true, Ordering::SeqCst);
            let mut child = job.child.lock().map_err(|e| e.to_string())?;
            if child.try_wait().map_err(|e| e.to_string())?.is_none() {
                child.kill().map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
    pub fn cancel_all(&self) {
        if let Ok(jobs) = self.jobs.lock() {
            for job in jobs.values() {
                job.cancelled.store(true, Ordering::SeqCst);
                if let Ok(mut child) = job.child.lock() {
                    let _ = child.kill();
                }
            }
        }
    }
    pub fn detail(&self, project_id: &str, photo_id: &str) -> Result<Detail> {
        let key = format!("detail:{project_id}:{photo_id}");
        let mut jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        if jobs.contains_key(&key) {
            return Err("Detailed preview is already being generated".into());
        }
        let p = self.store.project(project_id)?;
        let photo = p
            .photos
            .iter()
            .find(|v| v.id == photo_id)
            .ok_or("Unknown photo")?;
        let output = PathBuf::from(path_string(
            &self
                .store
                .prepare_cache(project_id)?
                .join(format!("{photo_id}-detail.jpg")),
        ));
        crate::core::ensure_within(&output, &self.store.root)?;
        if output.exists()
            && fs::symlink_metadata(&output)
                .map_err(|e| e.to_string())?
                .file_type()
                .is_symlink()
        {
            return Err("Detailed preview cache cannot be a symbolic link".into());
        }
        // Output is always an app-owned cache path, never a user-supplied original destination.
        let mut command = self.engine.command("detail");
        command
            .arg("--source")
            .arg(&photo.path)
            .arg("--output")
            .arg(&output);
        let mut child = command
            .spawn()
            .map_err(|e| format!("Cannot start detailed preview: {e}"))?;
        let stdout = child.stdout.take().ok_or("Engine stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Engine stderr unavailable")?;
        let job = Arc::new(Job {
            id: Uuid::new_v4().to_string(),
            cancelled: AtomicBool::new(false),
            child: Mutex::new(child),
        });
        jobs.insert(key.clone(), job.clone());
        drop(jobs);
        let diagnostics = thread::spawn(move || read_diagnostics(stderr));
        let mut detail = None;
        let mut failure = None;
        for line in BufReader::new(stdout).lines() {
            let record = line
                .map_err(|e| e.to_string())
                .and_then(|v| serde_json::from_str::<Value>(&v).map_err(|e| e.to_string()));
            match record {
                Ok(v) if v.get("type").and_then(Value::as_str) == Some("detail") => {
                    match serde_json::from_value::<Detail>(v) {
                        Ok(v) => detail = Some(v),
                        Err(e) => {
                            failure = Some(e.to_string());
                            break;
                        }
                    }
                }
                Ok(v) => {
                    failure = Some(
                        v.get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("Unexpected detailed-preview response")
                            .to_string(),
                    );
                    break;
                }
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        if failure.is_some() {
            if let Ok(mut child) = job.child.lock() {
                let _ = child.kill();
            }
        }
        let status = wait_for_child(&job);
        let diagnostic = diagnostics.join().unwrap_or_default();
        let result = (|| {
            if let Some(e) = failure {
                return Err(attach_diagnostic(e, &diagnostic));
            }
            if !status?.success() {
                return Err(attach_diagnostic(
                    "Detailed preview generation failed".into(),
                    &diagnostic,
                ));
            }
            let detail = detail.ok_or("Engine did not return a detailed preview")?;
            if PathBuf::from(&detail.detail_path) != output
                || !output.is_file()
                || detail.width == 0
                || detail.height == 0
            {
                return Err("Invalid detailed preview output".into());
            }
            self.store
                .set_detail(project_id, photo_id, &detail.detail_path)?;
            Ok(detail)
        })();
        self.jobs.lock().map_err(|e| e.to_string())?.remove(&key);
        if result.is_ok() {
            self.updated(project_id);
        }
        result
    }
    pub fn updated(&self, id: &str) {
        (self.emit)("project-updated", serde_json::json!({"projectId":id}));
    }
    fn progress(&self, p: Progress) -> Result<()> {
        let value = serde_json::to_value(&p).map_err(|e| e.to_string())?;
        let saved = self.store.save_progress(&p.project_id, &value);
        (self.emit)("import-progress", value);
        saved
    }
}
fn number(record: &Value, key: &str) -> Result<u64> {
    record
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("Missing engine {key}"))
}
fn read_diagnostics(mut reader: impl Read) -> String {
    let mut output = Vec::new();
    let mut buffer = [0u8; 1024];
    while let Ok(count) = reader.read(&mut buffer) {
        if count == 0 {
            break;
        }
        output.extend_from_slice(&buffer[..count]);
        if output.len() > 8192 {
            output.drain(..output.len() - 8192);
        }
    }
    String::from_utf8_lossy(&output).trim().into()
}
fn attach_diagnostic(message: String, diagnostic: &str) -> String {
    if diagnostic.is_empty() {
        message
    } else {
        format!("{message}\n{diagnostic}")
    }
}

fn wait_for_child(job: &Job) -> Result<std::process::ExitStatus> {
    loop {
        if let Some(status) = job
            .child
            .lock()
            .map_err(|e| e.to_string())?
            .try_wait()
            .map_err(|e| e.to_string())?
        {
            return Ok(status);
        }
        thread::sleep(Duration::from_millis(20));
    }
}
