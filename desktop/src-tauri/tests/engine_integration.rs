use photo_select::{
    core::{path_string, PhotoPatch, Store},
    worker::{Engine, Workers},
};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use tempfile::TempDir;

#[test]
#[ignore = "Requires Python engine dependencies and PHOTO_SELECT_RAW_FIXTURE pointing to the pinned Canon CR2 fixture"]
fn real_raw_preference_replaces_unreviewed_jpegs_but_preserves_review_on_rescan() {
    let fixture = PathBuf::from(
        std::env::var("PHOTO_SELECT_RAW_FIXTURE")
            .expect("Set PHOTO_SELECT_RAW_FIXTURE to the pinned Canon CR2 fixture"),
    );
    assert!(fixture.is_file(), "RAW fixture is unavailable");
    let temporary = TempDir::new().unwrap();
    let source = temporary.path().join("旅の写真 with spaces");
    let nested = source.join("別の folder");
    fs::create_dir_all(&nested).unwrap();
    let python = std::env::var("PHOTO_SELECT_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) {
            "python".into()
        } else {
            "python3".into()
        }
    });
    let write_jpeg = |path: &std::path::Path| {
        assert!(Command::new(&python).args(["-c", "from PIL import Image; import sys; Image.new('RGB', (600,400), (45,90,180)).save(sys.argv[1])"]).arg(path).status().unwrap().success());
    };
    let untouched = source.join("Straße.JPG");
    let reviewed = source.join("東京の街.JPEG");
    let edited = source.join("Straße-edited.jpg");
    let other_folder = nested.join("Straße.jpg");
    for path in [&untouched, &reviewed, &edited, &other_folder] {
        write_jpeg(path);
    }
    let data_root = temporary.path().join("appdata");
    let store = Store::new(data_root.clone()).unwrap();
    let project = store
        .create_configured(
            "RAW preference",
            source.to_str().unwrap(),
            true,
            true,
            "cautious",
            true,
        )
        .unwrap();
    let scan = |store: &Store, id: &str| {
        let workers = Workers::new(
            store.clone(),
            Engine::discover(PathBuf::from("/no-bundled-engine")),
            Arc::new(|_, _| {}),
        );
        workers.start(id).unwrap();
        let deadline = Instant::now() + Duration::from_secs(120);
        while workers.is_running(id) {
            assert!(Instant::now() < deadline, "RAW analysis timed out");
            thread::sleep(Duration::from_millis(20));
        }
        let result = store.project(id).unwrap();
        assert_eq!(
            result.import_status, "completed",
            "{:?}",
            result.import_error
        );
        result
    };
    let initial = scan(&store, &project.id);
    assert_eq!(initial.photos.len(), 4);
    let jpeg = initial
        .photos
        .iter()
        .find(|p| p.filename == "東京の街.JPEG")
        .unwrap();
    let jpeg_id = jpeg.id.clone();
    store
        .update_photos(
            &project.id,
            &[jpeg_id.clone()],
            &PhotoPatch {
                rating: Some(4),
                decision: Some("favorite".into()),
                reviewed: Some(true),
                tags: Some(vec!["Album".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let collection = store.create_collection(&project.id, "Album").unwrap();
    store
        .update_collection(
            &project.id,
            &collection.id,
            None,
            Some(vec![jpeg_id.clone()]),
        )
        .unwrap();
    let raw = source.join("STRASSE.CR2");
    let reviewed_raw = source.join("東京の街.CR2");
    for path in [&raw, &reviewed_raw] {
        fs::copy(&fixture, path).unwrap();
    }
    let broken_raw = source.join("broken.CR2");
    fs::write(&broken_raw, b"invalid RAW").unwrap();
    let fallback = source.join("broken.JPG");
    write_jpeg(&fallback);
    let originals: Vec<_> = [
        &untouched,
        &reviewed,
        &edited,
        &other_folder,
        &raw,
        &reviewed_raw,
        &broken_raw,
        &fallback,
    ]
    .into_iter()
    .map(|path| (path.clone(), fs::read(path).unwrap()))
    .collect();
    let reopened = Store::new(data_root).unwrap();
    assert!(reopened.open(&project.project_path).unwrap().prefer_raw);
    let preferred = scan(&reopened, &project.id);
    assert!(preferred.first_pass_ready);
    assert_eq!(preferred.photos.len(), 7);
    assert!(!preferred
        .photos
        .iter()
        .any(|p| p.path == path_string(&untouched)));
    let retained = preferred.photos.iter().find(|p| p.id == jpeg_id).unwrap();
    assert!(retained.raw_companion_retained);
    assert!(retained.analysis_error.is_none());
    assert!(retained
        .hints
        .iter()
        .any(|h| h == "Matching RAW preferred; JPEG retained to preserve your review"));
    assert_eq!(retained.rating, 4);
    assert_eq!(retained.decision, "favorite");
    assert_eq!(retained.tags, vec!["Album"]);
    assert!(retained.reviewed);
    assert_eq!(preferred.collections[0].photo_ids, vec![jpeg_id.clone()]);
    assert!(preferred
        .photos
        .iter()
        .find(|p| p.filename == "broken.CR2")
        .unwrap()
        .analysis_error
        .is_some());
    assert!(preferred
        .photos
        .iter()
        .find(|p| p.filename == "broken.JPG")
        .unwrap()
        .analysis_error
        .is_none());
    assert!(preferred
        .photos
        .iter()
        .any(|p| p.path == path_string(&other_folder)));
    assert!(preferred
        .photos
        .iter()
        .any(|p| p.path == path_string(&edited)));
    // Retained JPEGs have no new suggestion, but do not invalidate fresh RAW suggestions.
    reopened
        .apply_cached_first_pass(&project.id, "cautious")
        .unwrap();
    let raw_photo = preferred
        .photos
        .iter()
        .find(|p| p.filename == "STRASSE.CR2")
        .unwrap();
    reopened
        .update_photos(
            &project.id,
            &[raw_photo.id.clone()],
            &PhotoPatch {
                decision: Some("favorite".into()),
                ..Default::default()
            },
        )
        .unwrap();
    let manifest = temporary.path().join("selection.json");
    reopened
        .export(&project.id, manifest.to_str().unwrap(), None, true)
        .unwrap();
    let exported: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    assert!(exported["photos"]
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p["path"] == raw_photo.path));
    let keep_all = reopened
        .create_configured(
            "Keep both formats",
            source.to_str().unwrap(),
            true,
            false,
            "cautious",
            false,
        )
        .unwrap();
    let all = scan(&reopened, &keep_all.id);
    assert_eq!(all.photos.len(), 8);
    assert!(all.photos.iter().any(|p| p.path == path_string(&untouched)));
    assert!(all.photos.iter().all(|p| !p.raw_companion_retained));
    for (path, bytes) in originals {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
    // Missing originals retain their existing unavailable status rather than a pairing claim.
    fs::remove_file(&reviewed).unwrap();
    let missing = scan(&reopened, &project.id);
    let retained = missing.photos.iter().find(|p| p.id == jpeg_id).unwrap();
    assert!(!retained.raw_companion_retained);
    assert!(!retained
        .hints
        .iter()
        .any(|hint| hint.contains("JPEG retained")));
    assert!(retained
        .analysis_error
        .as_ref()
        .unwrap()
        .contains("unavailable"));
    assert_eq!(retained.rating, 4);
    assert_eq!(retained.decision, "favorite");
}

#[test]
#[ignore = "Requires installed Python engine dependencies; set PHOTO_SELECT_PYTHON to their interpreter"]
fn real_engine_folder_scope_survives_reopening_and_rescan() {
    let temporary = TempDir::new().unwrap();
    let source = temporary.path().join("originals");
    let nested = source.join("album");
    fs::create_dir_all(&nested).unwrap();
    let python = std::env::var("PHOTO_SELECT_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) {
            "python".into()
        } else {
            "python3".into()
        }
    });
    let write_photo = |path: &std::path::Path| {
        let status = Command::new(&python)
            .args([
                "-c",
                "from PIL import Image; import sys; Image.new('RGB', (600,400), (45,90,180)).save(sys.argv[1])",
            ])
            .arg(path)
            .status()
            .unwrap();
        assert!(status.success());
    };
    write_photo(&source.join("root.jpg"));
    write_photo(&nested.join("nested.jpg"));
    let originals: Vec<_> = [source.join("root.jpg"), nested.join("nested.jpg")]
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    let data_root = temporary.path().join("appdata");
    let store = Store::new(data_root.clone()).unwrap();
    let root_only = store
        .create_with_options("Root only", source.to_str().unwrap(), false)
        .unwrap();
    let recursive = store
        .create_with_options("Recursive", source.to_str().unwrap(), true)
        .unwrap();
    let scan = |store: &Store, project_id: &str, expected: &[&str]| {
        let workers = Workers::new(
            store.clone(),
            Engine::discover(PathBuf::from("/no-bundled-engine")),
            Arc::new(|_, _| {}),
        );
        workers.start(project_id).unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        while workers.is_running(project_id) {
            assert!(Instant::now() < deadline, "analysis timed out");
            thread::sleep(Duration::from_millis(20));
        }
        let project = store.project(project_id).unwrap();
        assert_eq!(
            project.import_status, "completed",
            "{:?}",
            project.import_error
        );
        let mut filenames: Vec<_> = project.photos.iter().map(|p| p.filename.as_str()).collect();
        filenames.sort();
        let mut expected = expected.to_vec();
        expected.sort();
        assert_eq!(filenames, expected);
    };
    scan(&store, &root_only.id, &["root.jpg"]);
    scan(&store, &recursive.id, &["root.jpg", "nested.jpg"]);

    write_photo(&source.join("new-root.jpg"));
    write_photo(&nested.join("new-nested.jpg"));
    let reopened = Store::new(data_root).unwrap();
    assert!(
        !reopened
            .open(&root_only.project_path)
            .unwrap()
            .include_subfolders
    );
    assert!(
        reopened
            .open(&recursive.project_path)
            .unwrap()
            .include_subfolders
    );
    scan(&reopened, &root_only.id, &["root.jpg", "new-root.jpg"]);
    scan(
        &reopened,
        &recursive.id,
        &["root.jpg", "nested.jpg", "new-root.jpg", "new-nested.jpg"],
    );
    for (path, bytes) in originals {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
}

#[test]
#[ignore = "Requires installed Python engine dependencies; set PHOTO_SELECT_PYTHON to their interpreter"]
fn real_engine_import_detail_and_rescan_preserve_originals_and_review() {
    let temporary = TempDir::new().unwrap();
    let source = temporary.path().join("originals");
    fs::create_dir(&source).unwrap();
    let python = std::env::var("PHOTO_SELECT_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) {
            "python".into()
        } else {
            "python3".into()
        }
    });
    let status=Command::new(python).args(["-c","from PIL import Image; import sys; from pathlib import Path; p=Path(sys.argv[1]); Image.new('RGB',(2400,1600),(45,90,180)).save(p/'a.jpg'); Image.new('RGB',(2400,1600),(46,91,181)).save(p/'b.jpg'); (p/'broken.jpg').write_bytes(b'not an image')"]).arg(&source).status().unwrap();
    assert!(status.success());
    let originals: Vec<_> = ["a.jpg", "b.jpg", "broken.jpg"]
        .iter()
        .map(|name| (source.join(name), fs::read(source.join(name)).unwrap()))
        .collect();
    let store = Store::new(temporary.path().join("appdata")).unwrap();
    let project = store
        .create("Real engine", source.to_str().unwrap())
        .unwrap();
    let engine = Engine::discover(PathBuf::from("/no-bundled-engine"));
    assert!(engine.available());
    let workers = Workers::new(store.clone(), engine, Arc::new(|_, _| {}));
    let wait = || {
        let deadline = Instant::now() + Duration::from_secs(60);
        while workers.is_running(&project.id) {
            assert!(Instant::now() < deadline, "analysis timed out");
            thread::sleep(Duration::from_millis(20));
        }
    };
    workers.start(&project.id).unwrap();
    wait();
    let result = store.project(&project.id).unwrap();
    assert_eq!(
        result.import_status, "completed",
        "{:?}",
        result.import_error
    );
    assert_eq!(result.photos.len(), 3);
    assert_eq!(
        result
            .photos
            .iter()
            .filter(|p| p.analysis_error.is_some())
            .count(),
        1
    );
    let photo = result
        .photos
        .iter()
        .find(|p| p.filename == "a.jpg")
        .unwrap();
    let group_id = photo.group_id.clone();
    store
        .update_photos(
            &project.id,
            &[photo.id.clone()],
            &PhotoPatch {
                rating: Some(3),
                decision: Some("favorite".into()),
                reviewed: Some(true),
                tags: Some(vec!["Album".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let detail = workers.detail(&project.id, &photo.id).unwrap();
    assert_eq!((detail.width, detail.height), (2400, 1600));
    assert!(PathBuf::from(&detail.detail_path).is_file());
    let repeated = workers.detail(&project.id, &photo.id).unwrap();
    assert_eq!(repeated.detail_path, detail.detail_path);
    workers.start(&project.id).unwrap();
    wait();
    let rescanned = store.project(&project.id).unwrap();
    assert_eq!(rescanned.import_status, "completed");
    let reviewed = rescanned.photos.iter().find(|p| p.id == photo.id).unwrap();
    assert_eq!(reviewed.rating, 3);
    assert_eq!(reviewed.decision, "favorite");
    assert!(reviewed.reviewed);
    assert_eq!(reviewed.tags, vec!["Album"]);
    assert_eq!(reviewed.group_id, group_id);
    let destination = temporary.path().join("selection.json");
    assert_eq!(
        store
            .export(&project.id, destination.to_str().unwrap(), None, true)
            .unwrap()
            .count,
        1
    );
    for (path, bytes) in originals {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
    fs::remove_file(source.join("a.jpg")).unwrap();
    workers.start(&project.id).unwrap();
    wait();
    let unavailable = store.project(&project.id).unwrap();
    assert_eq!(unavailable.import_status, "completed");
    assert_eq!(unavailable.photos.len(), 3);
    let retained = unavailable
        .photos
        .iter()
        .find(|candidate| candidate.id == photo.id)
        .unwrap();
    assert_eq!(retained.rating, 3);
    assert_eq!(retained.decision, "favorite");
    assert!(retained
        .analysis_error
        .as_ref()
        .unwrap()
        .contains("unavailable"));
    assert!(retained.group_id.is_none());
    assert!(PathBuf::from(&retained.preview_path).is_file());
    fs::remove_dir_all(&source).unwrap();
    assert_eq!(
        store
            .export(&project.id, destination.to_str().unwrap(), None, true)
            .unwrap()
            .count,
        1
    );
}

#[test]
#[ignore = "Requires installed Python fixture dependencies; supports PHOTO_SELECT_ENGINE for the bundled worker"]
fn real_engine_automatic_first_pass_preserves_manual_undecided_and_exports_rejects() {
    let temporary = TempDir::new().unwrap();
    let source = temporary.path().join("originals");
    fs::create_dir(&source).unwrap();
    let python = std::env::var("PHOTO_SELECT_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) {
            "python".into()
        } else {
            "python3".into()
        }
    });
    let status = Command::new(python)
        .args([
            "-c",
            r#"
from PIL import Image
from pathlib import Path
import sys
source = Path(sys.argv[1])
sharp = Image.new('RGB', (1200, 800))
sharp.putdata([(v, v, v) for y in range(800) for x in range(1200)
               for v in [220 if ((x // 8) + (y // 8)) % 2 else 40]])
sharp.save(source / 'sharp.jpg', quality=98)
Image.new('RGB', (1200, 800), (0, 0, 0)).save(source / 'blank.jpg')
(source / 'corrupt.jpg').write_bytes(b'not an image')
"#,
        ])
        .arg(&source)
        .status()
        .unwrap();
    assert!(status.success());
    let originals: Vec<_> = ["sharp.jpg", "blank.jpg", "corrupt.jpg"]
        .into_iter()
        .map(|name| {
            let path = source.join(name);
            let bytes = fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    let store = Store::new(temporary.path().join("appdata")).unwrap();
    let project = store
        .create_configured(
            "Automatic real worker",
            source.to_str().unwrap(),
            false,
            true,
            "cautious",
            false,
        )
        .unwrap();
    let engine = Engine::discover(PathBuf::from("/no-bundled-engine"));
    assert!(engine.available());
    let workers = Workers::new(store.clone(), engine, Arc::new(|_, _| {}));
    let wait = || {
        let deadline = Instant::now() + Duration::from_secs(60);
        while workers.is_running(&project.id) {
            assert!(Instant::now() < deadline, "analysis timed out");
            thread::sleep(Duration::from_millis(20));
        }
        let result = store.project(&project.id).unwrap();
        assert_eq!(
            result.import_status, "completed",
            "{:?}",
            result.import_error
        );
        result
    };
    workers.start(&project.id).unwrap();
    let selected = wait();
    assert!(selected.first_pass_ready);
    assert!(selected.automatic_selection_enabled);
    assert_eq!(selected.photos.len(), 3);
    let sharp = selected
        .photos
        .iter()
        .find(|p| p.filename == "sharp.jpg")
        .unwrap();
    let blank = selected
        .photos
        .iter()
        .find(|p| p.filename == "blank.jpg")
        .unwrap();
    let corrupt = selected
        .photos
        .iter()
        .find(|p| p.filename == "corrupt.jpg")
        .unwrap();
    assert_eq!(sharp.decision, "favorite", "{:?}", sharp);
    assert_eq!(blank.decision, "pass", "{:?}", blank);
    for photo in [sharp, blank] {
        assert_eq!(photo.decision_source, "automatic");
        assert_eq!(
            photo.suggested_decision.as_deref(),
            Some(photo.decision.as_str())
        );
        assert!(photo
            .suggestion_reason
            .as_ref()
            .is_some_and(|s| !s.is_empty()));
        assert!(photo.suggestion_confidence.is_some());
        assert!(!photo.reviewed);
        assert_eq!(photo.rating, 0);
        assert!(!photo.rating_touched);
    }
    assert!(corrupt.analysis_error.is_some());
    assert_eq!(corrupt.decision, "undecided");
    assert_eq!(corrupt.suggested_decision.as_deref(), Some("undecided"));
    let sharp_id = sharp.id.clone();
    let blank_id = blank.id.clone();
    // An explicit manual Undecided is protected even when reviewed remains false.
    store
        .update_photos(
            &project.id,
            &[sharp_id.clone()],
            &PhotoPatch {
                decision: Some("undecided".into()),
                reviewed: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
    workers
        .automatic_first_pass(&project.id, Some("stronger"))
        .unwrap();
    let rescanned = wait();
    assert_eq!(rescanned.selection_mode, "stronger");
    let manual = rescanned.photos.iter().find(|p| p.id == sharp_id).unwrap();
    assert_eq!(manual.decision, "undecided");
    assert_eq!(manual.decision_source, "manual");
    assert!(manual.decision_touched);
    assert_eq!(manual.suggested_decision.as_deref(), Some("favorite"));
    assert_eq!(
        rescanned
            .photos
            .iter()
            .find(|p| p.id == blank_id)
            .unwrap()
            .decision,
        "pass"
    );
    let destination = temporary.path().join("rejects.json");
    let exported = store
        .export_with_discards(
            &project.id,
            destination.to_str().unwrap(),
            None,
            false,
            true,
        )
        .unwrap();
    assert_eq!(
        (
            exported.count,
            exported.selected_count,
            exported.discard_count
        ),
        (1, 0, 1)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
    assert_eq!(manifest["schemaVersion"], 2);
    assert_eq!(manifest["photos"][0]["path"], blank.path);
    assert_eq!(manifest["photos"][0]["catalogFlag"], "reject");
    assert_eq!(manifest["photos"][0]["addToCollection"], false);
    assert!(manifest["photos"][0]["rating"].is_null());
    let cleared = workers.clear_automatic_selection(&project.id).unwrap();
    assert!(!cleared.automatic_selection_enabled);
    assert!(cleared.photos.iter().all(|p| p.decision == "undecided"));
    workers.start(&project.id).unwrap();
    let disabled = wait();
    assert!(!disabled.automatic_selection_enabled);
    assert!(disabled.first_pass_ready);
    assert!(disabled.photos.iter().all(|p| p.decision == "undecided"));
    assert_eq!(
        disabled
            .photos
            .iter()
            .find(|p| p.id == blank_id)
            .unwrap()
            .suggested_decision
            .as_deref(),
        Some("pass")
    );
    for (path, bytes) in originals {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
}
