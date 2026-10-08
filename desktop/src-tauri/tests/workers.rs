#![cfg(unix)]
use photo_select::{
    core::Store,
    worker::{Engine, Workers},
};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tempfile::TempDir;
fn fake_engine(root: &Path, body: &str) -> Engine {
    let file = root.join("engine.sh");
    fs::write(&file, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(&file, fs::Permissions::from_mode(0o700)).unwrap();
    Engine::executable(file)
}
fn wait_done(worker: &Workers, id: &str) {
    let until = Instant::now() + Duration::from_secs(4);
    while worker.is_running(id) {
        assert!(Instant::now() < until, "worker failed to finish promptly");
        thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn cancellation_kills_process_and_records_durable_status() {
    let temp = TempDir::new().unwrap();
    fs::create_dir(temp.path().join("photos")).unwrap();
    let store = Store::new(temp.path().join("data")).unwrap();
    let p = store
        .create("Cancel", temp.path().join("photos").to_str().unwrap())
        .unwrap();
    let engine = fake_engine(
        temp.path(),
        "printf '%s\\n' '{\"type\":\"scan\",\"total\":10}'\nexec sleep 60",
    );
    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    let workers = Workers::new(
        store.clone(),
        engine,
        Arc::new(move |name, value| capture.lock().unwrap().push((name.to_string(), value))),
    );
    let job = workers.start(&p.id).unwrap();
    assert!(!job.is_empty());
    assert!(workers.start(&p.id).is_err());
    workers.cancel(&p.id).unwrap();
    wait_done(&workers, &p.id);
    assert_eq!(store.project(&p.id).unwrap().import_status, "cancelled");
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|(name, v)| name == "import-progress" && v["phase"] == "cancelled"));
}
#[test]
fn malformed_or_incomplete_engine_stream_fails_without_hanging() {
    for body in [
        "printf '%s\\n' 'bad json'\nexec sleep 60",
        "printf '%s\\n' '{\"type\":\"scan\",\"total\":0}'",
    ] {
        let temp = TempDir::new().unwrap();
        fs::create_dir(temp.path().join("photos")).unwrap();
        let store = Store::new(temp.path().join("data")).unwrap();
        let p = store
            .create("Broken", temp.path().join("photos").to_str().unwrap())
            .unwrap();
        let worker = Workers::new(
            store.clone(),
            fake_engine(temp.path(), body),
            Arc::new(|_, _| {}),
        );
        worker.start(&p.id).unwrap();
        wait_done(&worker, &p.id);
        let project = store.project(&p.id).unwrap();
        assert_eq!(project.import_status, "failed");
        assert!(project.import_error.is_some());
    }
}
#[test]
fn successful_import_emits_completion_and_is_repeatable() {
    let temp = TempDir::new().unwrap();
    fs::create_dir(temp.path().join("photos")).unwrap();
    let store = Store::new(temp.path().join("data")).unwrap();
    let p = store
        .create("Empty", temp.path().join("photos").to_str().unwrap())
        .unwrap();
    let engine=fake_engine(temp.path(),"printf '%s\\n' '{\"type\":\"scan\",\"total\":0}' '{\"type\":\"groups\",\"groups\":[]}' '{\"type\":\"complete\",\"processed\":0,\"total\":0,\"failed\":0}'");
    let worker = Workers::new(store.clone(), engine, Arc::new(|_, _| {}));
    for _ in 0..2 {
        worker.start(&p.id).unwrap();
        wait_done(&worker, &p.id);
        assert_eq!(store.project(&p.id).unwrap().import_status, "completed");
    }
}

#[test]
fn file_error_is_recoverable_while_batch_error_is_fatal() {
    for (error, expected) in [
        (
            r#"{"type":"error","message":"Photo disappeared","path":"/photos/missing.jpg"}"#,
            "completed",
        ),
        (
            r#"{"type":"error","message":"Cannot scan folder"}"#,
            "failed",
        ),
    ] {
        let temp = TempDir::new().unwrap();
        fs::create_dir(temp.path().join("photos")).unwrap();
        let store = Store::new(temp.path().join("data")).unwrap();
        let p = store
            .create("Errors", temp.path().join("photos").to_str().unwrap())
            .unwrap();
        let body = format!(
            r#"printf '%s\n' '{{"type":"scan","total":1}}' '{error}' '{{"type":"progress","processed":1,"total":1,"failed":1,"currentFile":"/photos/missing.jpg"}}' '{{"type":"complete","processed":1,"total":1,"failed":1}}'"#
        );
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = events.clone();
        let worker = Workers::new(
            store.clone(),
            fake_engine(temp.path(), &body),
            Arc::new(move |name, value| capture.lock().unwrap().push((name.to_string(), value))),
        );
        worker.start(&p.id).unwrap();
        wait_done(&worker, &p.id);
        let result = store.project(&p.id).unwrap();
        assert_eq!(result.import_status, expected);
        assert!(result.photos.is_empty());
        if expected == "completed" {
            assert!(result.import_error.is_none());
            let events = events.lock().unwrap();
            assert!(events.iter().any(|(name, v)| name == "import-progress"
                && v["phase"] == "analysis"
                && v["message"] == "Photo disappeared"));
            assert!(events.iter().any(|(name, v)| name == "import-progress"
                && v["phase"] == "complete"
                && v["failed"] == 1));
        } else {
            assert!(result.import_error.unwrap().contains("Cannot scan folder"));
        }
    }
}
