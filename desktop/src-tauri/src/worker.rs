use crate::core::{
    path_string, validate_selection_mode, Group, Photo, Project, Result, Store, Suggestion,
};
use crate::process::ManagedChild;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Command, Stdio},
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
        if !cfg!(debug_assertions) && std::env::var_os("PHOTO_SELECT_PYTHON").is_none() {
            return Self::executable(executable);
        }
        let python = std::env::var_os("PHOTO_SELECT_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }));
        Self {
            executable: python,
            python: true,
            working_dir: {
                let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
                repo.join("culling_engine").is_dir().then_some(repo)
            },
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
        self.probe(Duration::from_secs(30))
    }
    pub fn probe(&self, timeout: Duration) -> bool {
        let Ok(mut child) = ManagedChild::spawn(&mut self.command("self-test")) else {
            return false;
        };
        let Some(stdout) = child.stdout.take() else {
            return false;
        };
        let Some(stderr) = child.stderr.take() else {
            return false;
        };
        let output = thread::spawn(move || read_diagnostics(stdout));
        let diagnostics = thread::spawn(move || read_diagnostics(stderr));
        let started = Instant::now();
        let success = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status.success(),
                Err(_) => break false,
                Ok(None) => {}
            }
            if started.elapsed() > timeout {
                break false;
            }
            thread::sleep(Duration::from_millis(20));
        };
        // Close inherited handles held by descendants too, even when the parent has already exited.
        let _ = child.kill_tree();
        let _ = child.wait();
        let stdout = output.join().unwrap_or_default();
        let _ = diagnostics.join();
        let probes: Vec<Value> = stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|v| v["type"] == "self-test")
            .collect();
        success
            && probes.len() == 1
            && probes.iter().all(|v| {
                v["success"] == true
                    && v["version"] == env!("CARGO_PKG_VERSION")
                    && v["checks"].as_array().is_some_and(|checks| {
                        ["folder-scope", "automatic-selection", "raw-jpeg-pairs"]
                            .iter()
                            .all(|required| {
                                checks.iter().any(|check| check.as_str() == Some(required))
                            })
                    })
            })
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
struct ScanSummary {
    processed: u64,
    total: u64,
    failed: u64,
    seen: HashSet<String>,
    suggestions: Option<Vec<Suggestion>>,
    exclusions: Vec<String>,
}
struct Job {
    id: String,
    cancelled: AtomicBool,
    child: Mutex<ManagedChild>,
    finished: AtomicBool,
    timed_out: AtomicBool,
    last_activity: Mutex<Instant>,
}
impl Job {
    fn watchdog(self: &Arc<Self>, timeout: Duration) {
        let job = self.clone();
        thread::spawn(move || {
            while !job.finished.load(Ordering::SeqCst) {
                let expired = job
                    .last_activity
                    .lock()
                    .map(|v| v.elapsed() > timeout)
                    .unwrap_or(true);
                if expired {
                    job.timed_out.store(true, Ordering::SeqCst);
                    if let Ok(mut child) = job.child.lock() {
                        let _ = child.kill_tree();
                    }
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
        });
    }
    fn activity(&self) {
        if let Ok(mut last) = self.last_activity.lock() {
            *last = Instant::now();
        }
    }
}
#[derive(Clone)]
pub struct Workers {
    pub store: Store,
    pub engine: Engine,
    jobs: Arc<Mutex<HashMap<String, Arc<Job>>>>,
    emit: Emitter,
    inactivity_timeout: Duration,
}
impl Workers {
    pub fn new(store: Store, engine: Engine, emit: Emitter) -> Self {
        Self {
            store,
            engine,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            emit,
            inactivity_timeout: Duration::from_secs(300),
        }
    }
    pub fn with_inactivity_timeout(mut self, timeout: Duration) -> Self {
        self.inactivity_timeout = timeout;
        self
    }
    pub fn start(&self, project_id: &str) -> Result<String> {
        self.start_configured(project_id, None)
    }
    fn start_configured(&self, project_id: &str, selection_mode: Option<&str>) -> Result<String> {
        let mut jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        if jobs.contains_key(project_id) {
            return Err("An import is already running for this project".into());
        }
        if jobs
            .keys()
            .any(|key| key.starts_with(&format!("detail:{project_id}:")))
        {
            return Err(
                "Wait for detailed previews to finish before reimporting this project".into(),
            );
        }
        if let Some(mode) = selection_mode {
            self.store.configure_first_pass(project_id, mode)?;
        }
        let project = self.store.project(project_id)?;
        self.store.validate_source_dir(&project.source_dir)?;
        let cache = self.store.prepare_cache(project_id)?;
        let mut command = self.engine.command("scan");
        command
            .arg("--source")
            .arg(&project.source_dir)
            .arg("--cache")
            .arg(path_string(&cache))
            .arg("--selection-mode")
            .arg(&project.selection_mode);
        if !project.include_subfolders {
            command.arg("--no-subfolders");
        }
        if project.prefer_raw {
            command.arg("--prefer-raw");
        }
        let mut child = ManagedChild::spawn(&mut command)
            .map_err(|e| format!("Cannot start analysis engine: {e}"))?;
        let stdout = child.stdout.take().ok_or("Engine stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Engine stderr unavailable")?;
        if let Err(e) = self.store.set_status(project_id, "running", None) {
            let _ = child.kill_tree();
            let _ = child.wait();
            return Err(e);
        }
        let job = Arc::new(Job {
            id: Uuid::new_v4().to_string(),
            cancelled: AtomicBool::new(false),
            child: Mutex::new(child),
            finished: AtomicBool::new(false),
            timed_out: AtomicBool::new(false),
            last_activity: Mutex::new(Instant::now()),
        });
        jobs.insert(project_id.into(), job.clone());
        let id = job.id.clone();
        let workers = self.clone();
        let project_id = project_id.to_string();
        thread::spawn(move || {
            job.watchdog(workers.inactivity_timeout);
            let diagnostics = thread::spawn(move || read_diagnostics(stderr));
            let result = workers.consume(&project_id, &job, BufReader::new(stdout));
            // A malformed record must not leave a subprocess writing into a closed pipe indefinitely.
            if result.is_err() {
                if let Ok(mut child) = job.child.lock() {
                    let _ = child.kill_tree();
                }
            }
            let status = wait_for_child(&job);
            let diagnostic = diagnostics.join().unwrap_or_default();
            job.finished.store(true, Ordering::SeqCst);
            let result = if job.timed_out.load(Ordering::SeqCst) {
                Err("Analysis engine stopped responding; import timed out. You can retry the import.".into())
            } else {
                result
            };
            let counts = result
                .as_ref()
                .ok()
                .map(|summary| (summary.processed, summary.total, summary.failed))
                .unwrap_or((0, 0, 0));
            // Serialize cancellation with the successful completion transaction. Once committed,
            // the job is finished and later cancellation requests cannot relabel its decisions.
            let mut finishing_jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
            let cancelled = job.cancelled.load(Ordering::SeqCst);
            let (mut phase, reason) = if cancelled {
                ("cancelled", None)
            } else {
                match (result, status) {
                    (Ok(summary), Ok(s)) if s.success() => {
                        match workers.store.finish_scan_with_exclusions(
                            &project_id,
                            &summary.seen,
                            summary.suggestions.as_deref(),
                            &summary.exclusions,
                        ) {
                            Ok(()) => ("complete", None),
                            Err(error) => ("error", Some(error)),
                        }
                    }
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
            let persistence = if phase == "complete" {
                Ok(())
            } else {
                workers
                    .store
                    .set_status(&project_id, status, reason.clone())
            };
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
            finishing_jobs.remove(&project_id);
            drop(finishing_jobs);
            workers.updated(&project_id);
        });
        Ok(id)
    }
    fn consume(
        &self,
        project_id: &str,
        job: &Job,
        mut reader: impl BufRead,
    ) -> Result<ScanSummary> {
        let mut line;
        let mut total = 0;
        let mut processed = 0;
        let mut failed = 0;
        let mut complete = false;
        let mut seen = HashSet::new();
        let mut suggestions = None;
        let mut exclusions = None;
        let mut grouped = false;
        let mut last_update = Instant::now() - Duration::from_secs(1);
        loop {
            match read_record(&mut reader)? {
                Some(value) => line = value,
                None => break,
            }
            if job.cancelled.load(Ordering::SeqCst) {
                return Ok(ScanSummary {
                    processed,
                    total,
                    failed,
                    seen,
                    suggestions,
                    exclusions: exclusions.unwrap_or_default(),
                });
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
            job.activity();
            if complete {
                return Err("Engine sent records after completing the import".into());
            }
            let mut phase = None;
            let mut message = None;
            match kind {
                "scan" => {
                    total = number(&record, "total")?;
                    phase = Some("scan");
                }
                "photo" => {
                    if grouped || suggestions.is_some() || exclusions.is_some() {
                        return Err("Engine sent photos after grouping".into());
                    }
                    let photo: Photo = serde_json::from_value(
                        record.get("photo").cloned().ok_or("Missing photo")?,
                    )
                    .map_err(|e| e.to_string())?;
                    let photo_id = photo.id.clone();
                    self.store.ingest_photo(project_id, photo)?;
                    if !seen.insert(photo_id) {
                        return Err("Engine sent a duplicate photo".into());
                    }
                }
                "progress" => {
                    processed = number(&record, "processed")?;
                    total = number(&record, "total")?;
                    failed = number(&record, "failed")?;
                    phase = Some("analysis");
                }
                "groups" => {
                    if grouped || suggestions.is_some() {
                        return Err("Engine sent duplicate or misplaced grouping".into());
                    }
                    grouped = true;
                    let groups: Vec<Group> = serde_json::from_value(
                        record.get("groups").cloned().ok_or("Missing groups")?,
                    )
                    .map_err(|e| e.to_string())?;
                    self.store.ingest_groups(project_id, groups)?;
                    phase = Some("grouping");
                }
                "excluded" => {
                    if grouped || suggestions.is_some() || exclusions.is_some() {
                        return Err("Engine sent duplicate or misplaced RAW exclusions".into());
                    }
                    let paths: Vec<String> = serde_json::from_value(
                        record
                            .get("paths")
                            .cloned()
                            .ok_or("Missing excluded paths")?,
                    )
                    .map_err(|e| format!("Invalid excluded paths: {e}"))?;
                    if paths.iter().collect::<HashSet<_>>().len() != paths.len() {
                        return Err("Engine sent duplicate excluded paths".into());
                    }
                    exclusions = Some(paths);
                }
                "suggestions" => {
                    if !grouped || suggestions.is_some() {
                        return Err("Engine sent duplicate or misplaced suggestions".into());
                    }
                    suggestions = Some(
                        serde_json::from_value::<Vec<Suggestion>>(
                            record
                                .get("suggestions")
                                .cloned()
                                .ok_or("Missing automatic suggestions")?,
                        )
                        .map_err(|e| e.to_string())?,
                    );
                    phase = Some("selection");
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
        Ok(ScanSummary {
            processed,
            total,
            failed,
            seen,
            suggestions,
            exclusions: exclusions.unwrap_or_default(),
        })
    }
    pub fn automatic_first_pass(
        &self,
        project_id: &str,
        selection_mode: Option<&str>,
    ) -> Result<Project> {
        let jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        if jobs
            .keys()
            .any(|key| key == project_id || key.starts_with(&format!("detail:{project_id}:")))
        {
            return Err("Wait for this project's import or detailed previews to finish before automatic selection".into());
        }
        let project = self.store.project(project_id)?;
        let mode = selection_mode.unwrap_or(&project.selection_mode);
        validate_selection_mode(mode)?;
        if project.first_pass_ready && project.selection_mode == mode {
            let project = self.store.apply_cached_first_pass(project_id, mode)?;
            drop(jobs);
            self.updated(project_id);
            return Ok(project);
        }
        let mode = mode.to_string();
        drop(jobs);
        self.start_configured(project_id, Some(&mode))?;
        self.store.project(project_id)
    }
    pub fn clear_automatic_selection(&self, project_id: &str) -> Result<Project> {
        let jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        if jobs
            .keys()
            .any(|key| key == project_id || key.starts_with(&format!("detail:{project_id}:")))
        {
            return Err("Wait for this project's import or detailed previews to finish before clearing automatic selection".into());
        }
        let project = self.store.clear_automatic_selection(project_id)?;
        drop(jobs);
        self.updated(project_id);
        Ok(project)
    }
    pub fn open_project(&self, path: &str) -> Result<crate::core::Project> {
        let id = self.store.identify_project(path)?;
        let jobs = self.jobs.lock().map_err(|error| error.to_string())?;
        if jobs
            .keys()
            .any(|key| key == &id || key.starts_with(&format!("detail:{id}:")))
        {
            return Err("Wait for this project's import or detailed previews to finish before reopening its file".into());
        }
        self.store.open(path)
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
                child.kill_tree().map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
    pub fn cancel_all(&self) {
        if let Ok(jobs) = self.jobs.lock() {
            for job in jobs.values() {
                job.cancelled.store(true, Ordering::SeqCst);
                if let Ok(mut child) = job.child.lock() {
                    let _ = child.kill_tree();
                }
            }
        }
    }
    pub fn detail(&self, project_id: &str, photo_id: &str) -> Result<Detail> {
        let key = format!("detail:{project_id}:{photo_id}");
        let mut jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        if jobs.contains_key(project_id) {
            return Err(
                "Wait for this project's import to finish before generating detailed previews"
                    .into(),
            );
        }
        if jobs.contains_key(&key) {
            return Err("Detailed preview is already being generated".into());
        }
        let p = self.store.project(project_id)?;
        let photo = p
            .photos
            .iter()
            .find(|v| v.id == photo_id)
            .ok_or("Unknown photo")?;
        // Saved offline paths are validated lexically; before reading an available original,
        // re-check physical containment to reject a directory replaced by an external junction.
        crate::core::ensure_within(
            std::path::Path::new(&photo.path),
            std::path::Path::new(&p.source_dir),
        )?;
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
        let mut child = ManagedChild::spawn(&mut command)
            .map_err(|e| format!("Cannot start detailed preview: {e}"))?;
        let stdout = child.stdout.take().ok_or("Engine stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Engine stderr unavailable")?;
        let job = Arc::new(Job {
            id: Uuid::new_v4().to_string(),
            cancelled: AtomicBool::new(false),
            child: Mutex::new(child),
            finished: AtomicBool::new(false),
            timed_out: AtomicBool::new(false),
            last_activity: Mutex::new(Instant::now()),
        });
        jobs.insert(key.clone(), job.clone());
        drop(jobs);
        job.watchdog(self.inactivity_timeout);
        let diagnostics = thread::spawn(move || read_diagnostics(stderr));
        let mut detail = None;
        let mut failure = None;
        let mut reader = BufReader::new(stdout);
        loop {
            let line = match read_record(&mut reader) {
                Ok(Some(value)) => value,
                Ok(None) => break,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            };
            job.activity();
            let record = serde_json::from_str::<Value>(&line).map_err(|error| error.to_string());
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
                let _ = child.kill_tree();
            }
        }
        let status = wait_for_child(&job);
        let diagnostic = diagnostics.join().unwrap_or_default();
        job.finished.store(true, Ordering::SeqCst);
        let result = (|| {
            if job.timed_out.load(Ordering::SeqCst) {
                return Err("Detailed preview generation stopped responding and timed out".into());
            }
            if job.cancelled.load(Ordering::SeqCst) {
                return Err("Detailed preview generation was cancelled".into());
            }
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

fn read_record(reader: &mut impl BufRead) -> Result<Option<String>> {
    const MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf().map_err(|error| error.to_string())?;
        if available.is_empty() {
            if bytes.is_empty() {
                return Ok(None);
            }
            break;
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|position| position + 1)
            .unwrap_or(available.len());
        if bytes.len() + count > MAX_RECORD_BYTES {
            return Err("Engine response exceeded the maximum record size".into());
        }
        let ended = available[count - 1] == b'\n';
        bytes.extend_from_slice(&available[..count]);
        reader.consume(count);
        if ended {
            break;
        }
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|error| format!("Engine response was not valid UTF-8: {error}"))
}
