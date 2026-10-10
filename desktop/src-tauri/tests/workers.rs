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
            r#"head -c 131072 /dev/zero >&2; printf '%s\n' '{"type":"self-test","success":true,"version":"CURRENT","checks":["folder-scope","automatic-selection","raw-jpeg-pairs"]}'"#,
            true,
        ),
        (
            r#"printf '%s\n' '{"type":"self-test","success":true,"version":"0.2.2","checks":["folder-scope","automatic-selection","raw-jpeg-pairs"]}'"#,
            false,
        ),
        (
            r#"printf '%s\n' '{"type":"self-test","success":true,"version":"CURRENT","checks":["folder-scope"]}'"#,
            false,
        ),
        (
            r#"printf '%s\n' '{"type":"self-test","success":true,"version":"CURRENT","checks":["automatic-selection"]}'"#,
            false,
        ),
        (
            r#"printf '%s\n' '{"type":"self-test","success":true,"version":"CURRENT","checks":["folder-scope","automatic-selection"]}'"#,
            false,
        ),
        ("exec sleep 60", false),
    ] {
        let temp = TempDir::new().unwrap();
        let engine = fake_engine(
            temp.path(),
            &body.replace("CURRENT", env!("CARGO_PKG_VERSION")),
        );
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

fn first_pass_fixture() -> (TempDir, Store, String, serde_json::Value) {
    first_pass_fixture_with_raw(false)
}
fn first_pass_fixture_with_raw(prefer_raw: bool) -> (TempDir, Store, String, serde_json::Value) {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("photos");
    fs::create_dir(&source).unwrap();
    let original = source.join("a.jpg");
    fs::write(&original, b"original untouched").unwrap();
    let store = Store::new(temp.path().join("data")).unwrap();
    let p = store
        .create_configured(
            "Automatic",
            source.to_str().unwrap(),
            false,
            true,
            "cautious",
            prefer_raw,
        )
        .unwrap();
    let preview = store.cache(&p.id).join("a.jpg");
    fs::write(&preview, b"preview").unwrap();
    let photo = serde_json::json!({"id":"aaaaaaaaaaaaaaaaaaaaaaaa","path":original,"filename":"a.jpg","previewPath":preview,"thumbnailPath":preview,"captureTime":"2026-01-01T10:00:00Z","width":6000,"height":4000,"qualityScore":90.0});
    (temp, store, p.id, photo)
}
#[test]
fn raw_exclusions_commit_only_after_clean_success_and_never_modify_originals() {
    for ending in [
        "exit 1",
        "exec sleep 60",
        "printf '%s\\n' 'invalid json'",
        "complete-failed",
        "complete",
    ] {
        let (temp, store, id, jpeg) = first_pass_fixture_with_raw(true);
        store
            .ingest_photo(&id, serde_json::from_value(jpeg.clone()).unwrap())
            .unwrap();
        let raw_path = temp.path().join("photos/a.DNG");
        fs::write(&raw_path, b"raw original").unwrap();
        let mut raw = jpeg.clone();
        raw["id"] = serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbb");
        raw["path"] = serde_json::json!(raw_path);
        let photo = serde_json::json!({"type":"photo", "photo":raw});
        let excluded = serde_json::json!({"type":"excluded", "paths":[jpeg["path"]]});
        let finish = match ending {
            "complete" => FIRST_PASS_COMPLETE.to_string(),
            "complete-failed" => format!("{FIRST_PASS_COMPLETE}\nexit 1"),
            other => other.to_string(),
        };
        let worker = Workers::new(store.clone(), fake_engine(temp.path(), &format!("printf '%s\\n' '{{\"type\":\"scan\",\"total\":1}}' '{photo}' '{excluded}' '{{\"type\":\"groups\",\"groups\":[]}}'\n{finish}")), Arc::new(|_, _| {}));
        worker.start(&id).unwrap();
        if ending == "exec sleep 60" {
            let deadline = Instant::now() + Duration::from_secs(2);
            while store.project(&id).unwrap().photos.len() != 2 {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(10));
            }
            worker.cancel(&id).unwrap();
        }
        wait_done(&worker, &id);
        let result = store.project(&id).unwrap();
        assert_eq!(
            result.photos.iter().any(|p| p.filename == "a.jpg"),
            ending != "complete",
            "{ending}"
        );
        assert_eq!(
            result.import_status == "completed",
            ending == "complete",
            "{:?}",
            result.import_error
        );
        assert_eq!(
            fs::read(jpeg["path"].as_str().unwrap()).unwrap(),
            b"original untouched"
        );
        assert_eq!(fs::read(raw_path).unwrap(), b"raw original");
    }
}
#[test]
fn malformed_or_unmatched_raw_exclusions_fail_without_removing_jpeg_rows() {
    for paths in [
        serde_json::json!(["relative.jpg"]),
        serde_json::json!([42]),
        serde_json::json!([]),
        serde_json::json!(["duplicate"]),
        serde_json::json!(["unmatched"]),
        serde_json::json!(["failed-raw"]),
        serde_json::json!(["outside-source"]),
        serde_json::json!(["nested"]),
        serde_json::json!(["not-jpeg"]),
        serde_json::json!(["preference-off"]),
    ] {
        let preference_off = paths == serde_json::json!(["preference-off"]);
        let (temp, store, id, jpeg) = first_pass_fixture_with_raw(!preference_off);
        store
            .ingest_photo(&id, serde_json::from_value(jpeg.clone()).unwrap())
            .unwrap();
        let raw_path = temp.path().join("photos/a.DNG");
        fs::write(&raw_path, b"raw original").unwrap();
        let mut raw = jpeg.clone();
        raw["id"] = serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbb");
        raw["path"] = serde_json::json!(raw_path);
        let mut paths = paths;
        let repeated = paths == serde_json::json!([]);
        if paths == serde_json::json!(["duplicate"]) {
            paths = serde_json::json!([jpeg["path"], jpeg["path"]]);
        } else if paths == serde_json::json!(["unmatched"]) {
            let unrelated = temp.path().join("photos/unrelated.jpg");
            fs::write(&unrelated, b"unrelated original").unwrap();
            paths = serde_json::json!([unrelated]);
        } else if paths == serde_json::json!(["failed-raw"]) {
            raw["analysisError"] = serde_json::json!("Cannot decode RAW");
            paths = serde_json::json!([jpeg["path"]]);
        } else if paths == serde_json::json!(["outside-source"]) {
            let outside = temp.path().join("a.jpg");
            fs::write(&outside, b"outside original").unwrap();
            paths = serde_json::json!([outside]);
        } else if paths == serde_json::json!(["nested"]) {
            let nested = temp.path().join("photos/nested");
            fs::create_dir(&nested).unwrap();
            let jpeg = nested.join("a.jpg");
            let nested_raw = nested.join("a.DNG");
            fs::write(&jpeg, b"nested original").unwrap();
            fs::write(&nested_raw, b"nested raw").unwrap();
            raw["path"] = serde_json::json!(nested_raw);
            paths = serde_json::json!([jpeg]);
        } else if paths == serde_json::json!(["not-jpeg"]) {
            paths = serde_json::json!([raw_path]);
        } else if preference_off {
            paths = serde_json::json!([jpeg["path"]]);
        }
        let photo = serde_json::json!({"type":"photo", "photo":raw});
        let excluded = serde_json::json!({"type":"excluded", "paths":paths});
        let extra = if repeated {
            format!("'{excluded}'")
        } else {
            String::new()
        };
        let worker = Workers::new(store.clone(), fake_engine(temp.path(), &format!("printf '%s\\n' '{{\"type\":\"scan\",\"total\":1}}' '{photo}' '{excluded}' {extra} '{{\"type\":\"groups\",\"groups\":[]}}'\n{FIRST_PASS_COMPLETE}")), Arc::new(|_, _| {}));
        worker.start(&id).unwrap();
        wait_done(&worker, &id);
        let result = store.project(&id).unwrap();
        assert_eq!(result.import_status, "failed", "{:?}", result.import_error);
        assert!(result.photos.iter().any(|p| p.filename == "a.jpg"));
    }
}
fn first_pass_stream(photo: &serde_json::Value) -> String {
    let photo = serde_json::json!({"type":"photo","photo":photo});
    format!("printf '%s\\n' '{{\"type\":\"scan\",\"total\":1}}' '{photo}' '{{\"type\":\"groups\",\"groups\":[]}}' '{{\"type\":\"suggestions\",\"suggestions\":[{{\"photoId\":\"aaaaaaaaaaaaaaaaaaaaaaaa\",\"decision\":\"favorite\",\"reason\":\"Best frame\",\"confidence\":0.95}}]}}'\n")
}
const FIRST_PASS_COMPLETE: &str =
    "printf '%s\\n' '{\"type\":\"complete\",\"processed\":1,\"total\":1,\"failed\":0}'";

#[test]
fn automatic_decisions_wait_for_success_and_failed_or_cancelled_streams_never_apply() {
    for ending in [
        "exit 1".to_string(),
        format!("{FIRST_PASS_COMPLETE}\nexit 1"),
        "exec sleep 60".to_string(),
        "printf '%s\\n' 'invalid json'".to_string(),
    ] {
        let (temp, store, id, photo) = first_pass_fixture();
        let worker = Workers::new(
            store.clone(),
            fake_engine(
                temp.path(),
                &format!("{}{ending}", first_pass_stream(&photo)),
            ),
            Arc::new(|_, _| {}),
        );
        worker.start(&id).unwrap();
        if ending == "exec sleep 60" {
            let until = Instant::now() + Duration::from_secs(2);
            while store.project(&id).unwrap().photos.is_empty() {
                assert!(Instant::now() < until);
                thread::sleep(Duration::from_millis(10));
            }
            let p = store.project(&id).unwrap();
            assert_eq!(p.photos[0].decision, "undecided");
            assert!(p.photos[0].suggested_decision.is_none());
            worker.cancel(&id).unwrap();
        }
        wait_done(&worker, &id);
        let p = store.project(&id).unwrap();
        assert!(!p.first_pass_ready);
        assert_eq!(p.photos[0].decision, "undecided");
        assert!(p.photos[0].suggested_decision.is_none());
        assert_eq!(
            fs::read(photo["path"].as_str().unwrap()).unwrap(),
            b"original untouched"
        );
    }
}

#[test]
fn ready_cached_first_pass_applies_without_engine_and_changed_mode_runs_fresh_analysis() {
    let (temp, store, id, photo) = first_pass_fixture();
    let engine = fake_engine(
        temp.path(),
        &format!("{}{}", first_pass_stream(&photo), FIRST_PASS_COMPLETE),
    );
    let worker = Workers::new(store.clone(), engine, Arc::new(|_, _| {}));
    worker.start(&id).unwrap();
    wait_done(&worker, &id);
    assert_eq!(store.project(&id).unwrap().photos[0].decision, "favorite");
    worker.clear_automatic_selection(&id).unwrap();
    // Replacing the executable proves same-mode cached application starts no subprocess.
    let file = temp.path().join("engine.sh");
    fs::write(&file, "#!/bin/sh\nexit 19\n").unwrap();
    let p = worker.automatic_first_pass(&id, None).unwrap();
    assert_eq!(p.photos[0].decision, "favorite");
    assert!(!worker.is_running(&id));
    fs::write(
        &file,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n{}{}",
            temp.path().join("args.txt").display(),
            first_pass_stream(&photo),
            format!("sleep 0.15\n{FIRST_PASS_COMPLETE}")
        ),
    )
    .unwrap();
    let p = worker.automatic_first_pass(&id, Some("stronger")).unwrap();
    assert_eq!(p.selection_mode, "stronger");
    assert_eq!(p.import_status, "running");
    assert!(worker.automatic_first_pass(&id, None).is_err());
    assert!(worker.clear_automatic_selection(&id).is_err());
    wait_done(&worker, &id);
    let p = store.project(&id).unwrap();
    assert!(p.first_pass_ready);
    assert_eq!(p.import_status, "completed");
    assert!(fs::read_to_string(temp.path().join("args.txt"))
        .unwrap()
        .contains("--selection-mode\nstronger"));
}
