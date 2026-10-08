use photo_select::{
    core::{PhotoPatch, Store},
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
