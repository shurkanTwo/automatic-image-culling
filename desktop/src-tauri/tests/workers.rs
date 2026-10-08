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

#[test]
fn probes_validate_the_protocol_drain_verbose_output_and_stop_hangs() {
    for (body, expected) in [
        ("exit 0", false),
        (
            r#"printf '%s\n' '{"type":"self-test","success":false}'"#,
            false,
        ),
        (
            r#"head -c 131072 /dev/zero >&2; printf '%s\n' '{"type":"self-test","success":true}'"#,
            true,
        ),
        ("exec sleep 60", false),
    ] {
        let temp = TempDir::new().unwrap();
        let engine = fake_engine(temp.path(), body);
        let started = Instant::now();
        assert_eq!(engine.probe(Duration::from_millis(150)), expected);
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}

#[test]
fn stalled_import_times_out_and_releases_the_project_for_retry() {
    let temp = TempDir::new().unwrap();
    fs::create_dir(temp.path().join("photos")).unwrap();
    let store = Store::new(temp.path().join("data")).unwrap();
    let p = store
        .create("Timeout", temp.path().join("photos").to_str().unwrap())
        .unwrap();
    let worker = Workers::new(
        store.clone(),
        fake_engine(temp.path(), "exec sleep 60"),
        Arc::new(|_, _| {}),
    )
    .with_inactivity_timeout(Duration::from_millis(100));
    for _ in 0..2 {
        worker.start(&p.id).unwrap();
        wait_done(&worker, &p.id);
        let project = store.project(&p.id).unwrap();
        assert_eq!(project.import_status, "failed");
        assert!(project.import_error.unwrap().contains("timed out"));
    }
}

#[test]
fn reopening_an_alias_of_an_importing_project_is_rejected() {
    let temp = TempDir::new().unwrap();
    fs::create_dir(temp.path().join("photos")).unwrap();
    let store = Store::new(temp.path().join("data")).unwrap();
    let p = store
        .create("Alias", temp.path().join("photos").to_str().unwrap())
        .unwrap();
    let alias = temp.path().join("alias.cullproj");
    std::os::unix::fs::symlink(&p.project_path, &alias).unwrap();
    let worker = Workers::new(
        store,
        fake_engine(temp.path(), "exec sleep 60"),
        Arc::new(|_, _| {}),
    );
    worker.start(&p.id).unwrap();
    assert!(worker.open_project(alias.to_str().unwrap()).is_err());
    worker.cancel(&p.id).unwrap();
    wait_done(&worker, &p.id);
}

#[cfg(target_os = "linux")]
#[test]
fn cancellation_also_terminates_a_worker_descendant_holding_the_stream_open() {
    let temp = TempDir::new().unwrap();
    fs::create_dir(temp.path().join("photos")).unwrap();
    let store = Store::new(temp.path().join("data")).unwrap();
    let p = store
        .create("Tree", temp.path().join("photos").to_str().unwrap())
        .unwrap();
    let pidfile = temp.path().join("descendant.pid");
    let body = format!("sleep 60 &\nprintf '%s' $! > '{}'\nwait", pidfile.display());
    let worker = Workers::new(store, fake_engine(temp.path(), &body), Arc::new(|_, _| {}));
    worker.start(&p.id).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !pidfile.exists() {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(10));
    }
    let pid = fs::read_to_string(pidfile).unwrap();
    worker.cancel(&p.id).unwrap();
    wait_done(&worker, &p.id);
    let state = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    assert!(
        state.is_empty()
            || state
                .split(')')
                .last()
                .unwrap_or("")
                .trim_start()
                .starts_with('Z'),
        "descendant remains active: {state}"
    );
}

#[test]
fn detail_timeout_prevents_overlapping_rescan_and_releases_its_reservation() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("photos");
    fs::create_dir(&source).unwrap();
    let original = source.join("a.jpg");
    fs::write(&original, b"original").unwrap();
    let store = Store::new(temp.path().join("data")).unwrap();
    let project = store.create("Detail", source.to_str().unwrap()).unwrap();
    let preview = store.cache(&project.id).join("a.jpg");
    fs::write(&preview, b"preview").unwrap();
    let photo:photo_select::core::Photo=serde_json::from_value(serde_json::json!({"id":"a00000000000000000000000","path":original,"filename":"a.jpg","previewPath":preview,"thumbnailPath":preview,"captureTime":"2026-01-01T10:00:00Z","width":100,"height":100,"qualityScore":50,"hints":[]})).unwrap();
    store.ingest_photo(&project.id, photo.clone()).unwrap();
    let worker = Workers::new(
        store.clone(),
        fake_engine(temp.path(), "exec sleep 60"),
        Arc::new(|_, _| {}),
    )
    .with_inactivity_timeout(Duration::from_millis(150));
    let detail_worker = worker.clone();
    let project_id = project.id.clone();
    let photo_id = photo.id.clone();
    let detail = thread::spawn(move || detail_worker.detail(&project_id, &photo_id));
    let key = format!("detail:{}:{}", project.id, photo.id);
    let deadline = Instant::now() + Duration::from_secs(2);
    while !worker.is_running(&key) {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert!(worker.start(&project.id).is_err());
    assert!(worker.detail(&project.id, &photo.id).is_err());
    assert!(detail.join().unwrap().unwrap_err().contains("timed out"));
    assert!(!worker.is_running(&key));
    assert_eq!(fs::read(&original).unwrap(), b"original");
    assert!(store.project(&project.id).unwrap().photos[0]
        .detail_path
        .is_none());
    worker.start(&project.id).unwrap();
    assert!(worker.detail(&project.id, &photo.id).is_err());
    worker.cancel(&project.id).unwrap();
    wait_done(&worker, &project.id);
}
