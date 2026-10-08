use photo_select::core::{Collection, Group, Photo, PhotoPatch, Store};
use std::{fs, path::Path};
use tempfile::TempDir;

struct Fixture {
    temp: TempDir,
    store: Store,
    id: String,
    source: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("originals");
        fs::create_dir(&source).unwrap();
        let store = Store::new(temp.path().join("appdata")).unwrap();
        let p = store.create("Wedding", source.to_str().unwrap()).unwrap();
        Self {
            temp,
            store,
            id: p.id,
            source: source.to_str().unwrap().into(),
        }
    }
    fn photo(&self, key: &str) -> Photo {
        let path = Path::new(&self.source).join(format!("{key}.jpg"));
        fs::write(&path, b"original bytes untouched").unwrap();
        let preview = self.store.cache(&self.id).join(format!("{key}.jpg"));
        fs::write(&preview, b"preview").unwrap();
        Photo {
            id: format!("{key:0<24}"),
            path: path.to_str().unwrap().into(),
            filename: format!("{key}.jpg"),
            preview_path: preview.to_str().unwrap().into(),
            thumbnail_path: preview.to_str().unwrap().into(),
            detail_path: None,
            capture_time: "2026-01-01T10:00:00Z".into(),
            width: 6000,
            height: 4000,
            camera: Some("Test Camera".into()),
            group_id: None,
            quality_score: 92.,
            hints: vec![],
            rating: 0,
            rating_touched: false,
            decision: "undecided".into(),
            reviewed: false,
            tags: vec![],
            analysis_error: None,
        }
    }
    fn add(&self, key: &str) -> Photo {
        let p = self.photo(key);
        self.store.ingest_photo(&self.id, p.clone()).unwrap();
        p
    }
}
#[test]
fn projects_and_manual_review_survive_restart() {
    let f = Fixture::new();
    let p = f.add("a");
    f.store
        .update_photos(
            &f.id,
            &[p.id.clone()],
            &PhotoPatch {
                rating: Some(4),
                decision: Some("favorite".into()),
                reviewed: Some(true),
                tags: Some(vec![" print ".into(), "print".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let collection = f.store.create_collection(&f.id, "Album").unwrap();
    f.store
        .update_collection(&f.id, &collection.id, None, Some(vec![p.id.clone()]))
        .unwrap();
    let store = Store::new(f.temp.path().join("appdata")).unwrap();
    let project = store.project(&f.id).unwrap();
    assert_eq!(project.photos[0].rating, 4);
    assert!(project.photos[0].rating_touched);
    assert!(project.photos[0].reviewed);
    assert_eq!(project.photos[0].tags, vec!["print"]);
    assert_eq!(project.collections[0].photo_ids, vec![p.id]);
    assert_eq!(store.summaries().unwrap()[0].favorite_count, 1);
    assert_eq!(fs::read(p.path).unwrap(), b"original bytes untouched");
}
#[test]
fn rescan_preserves_decisions_and_group_identity() {
    let f = Fixture::new();
    let p = f.add("a");
    let second = f.add("b");
    f.store
        .ingest_groups(
            &f.id,
            vec![Group {
                id: "original-group".into(),
                label: "Burst 1".into(),
                photo_ids: vec![p.id.clone(), second.id.clone()],
                recommended_photo_ids: vec![p.id.clone()],
            }],
        )
        .unwrap();
    f.store
        .update_photos(
            &f.id,
            &[p.id.clone()],
            &PhotoPatch {
                rating: Some(2),
                decision: Some("pass".into()),
                tags: Some(vec!["story".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let mut rescan = p.clone();
    rescan.quality_score = 55.;
    rescan.rating = 5;
    rescan.decision = "favorite".into();
    f.store.ingest_photo(&f.id, rescan).unwrap();
    let third = f.add("c");
    f.store
        .ingest_groups(
            &f.id,
            vec![Group {
                id: "new-engine-id".into(),
                label: "Burst 1".into(),
                photo_ids: vec![p.id.clone(), second.id, third.id],
                recommended_photo_ids: vec![p.id.clone()],
            }],
        )
        .unwrap();
    let result = f.store.project(&f.id).unwrap();
    let photo = result.photos.iter().find(|v| v.id == p.id).unwrap();
    assert_eq!(photo.rating, 2);
    assert_eq!(photo.decision, "pass");
    assert_eq!(photo.tags, vec!["story"]);
    assert_eq!(photo.quality_score, 55.);
    assert_eq!(photo.group_id.as_deref(), Some("original-group"));
    assert_eq!(result.groups[0].id, "original-group");
}
#[test]
fn batch_updates_are_atomic_and_invalid_input_is_rejected() {
    let f = Fixture::new();
    let p = f.add("a");
    assert!(f
        .store
        .update_photos(
            &f.id,
            &[p.id.clone(), "missing".into()],
            &PhotoPatch {
                rating: Some(5),
                ..Default::default()
            }
        )
        .is_err());
    assert_eq!(f.store.project(&f.id).unwrap().photos[0].rating, 0);
    for patch in [
        PhotoPatch {
            rating: Some(6),
            ..Default::default()
        },
        PhotoPatch {
            decision: Some("delete".into()),
            ..Default::default()
        },
        PhotoPatch {
            tags: Some(vec!["x".repeat(101)]),
            ..Default::default()
        },
    ] {
        assert!(f
            .store
            .update_photos(&f.id, &[p.id.clone()], &patch)
            .is_err());
    }
    assert!(f
        .store
        .update_collection(&f.id, "missing", None, None)
        .is_err());
    assert!(f.store.create_collection(&f.id, "   ").is_err());
}
#[test]
fn export_uses_explicit_collection_and_does_not_invent_ratings() {
    let f = Fixture::new();
    let a = f.add("a");
    let b = f.add("b");
    let c = f.add("c");
    f.store
        .update_photos(
            &f.id,
            &[a.id.clone(), b.id.clone()],
            &PhotoPatch {
                decision: Some("favorite".into()),
                ..Default::default()
            },
        )
        .unwrap();
    f.store
        .update_photos(
            &f.id,
            &[a.id.clone()],
            &PhotoPatch {
                rating: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    let collection = f.store.create_collection(&f.id, "Story").unwrap();
    f.store
        .update_collection(
            &f.id,
            &collection.id,
            None,
            Some(vec![b.id.clone(), c.id.clone()]),
        )
        .unwrap();
    let dest = f.temp.path().join("selection.json");
    let result = f
        .store
        .export(&f.id, dest.to_str().unwrap(), Some(&collection.id), false)
        .unwrap();
    assert_eq!(result.count, 2);
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(&dest).unwrap()).unwrap();
    assert_eq!(manifest["schemaVersion"], 1);
    assert_eq!(manifest["application"], "Photo Select");
    assert_eq!(manifest["collectionName"], "Story");
    assert!(manifest["photos"][0]["rating"].is_null());
    assert_eq!(
        f.store
            .export(&f.id, dest.to_str().unwrap(), Some(&collection.id), true)
            .unwrap()
            .count,
        1
    );
    assert_eq!(
        f.store
            .export(&f.id, dest.to_str().unwrap(), None, false)
            .unwrap()
            .count,
        2
    );
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(dest).unwrap()).unwrap();
    assert_eq!(manifest["photos"][0]["rating"], 0);
    assert!(manifest["photos"][1]["rating"].is_null());
    assert!(f
        .store
        .export(&f.id, &format!("{}/a.jpg", f.source), None, false)
        .is_err());
    assert!(f
        .store
        .export(&f.id, &format!("{}/selection.json", f.source), None, false)
        .is_err());
    for p in [a, b, c] {
        assert_eq!(fs::read(p.path).unwrap(), b"original bytes untouched");
    }
}
#[test]
fn interrupted_import_is_recoverable_and_corrupt_photo_is_visible() {
    let f = Fixture::new();
    let mut photo = f.photo("d");
    photo.preview_path.clear();
    photo.thumbnail_path.clear();
    photo.analysis_error = Some("Damaged JPEG".into());
    photo.quality_score = 0.;
    f.store.ingest_photo(&f.id, photo).unwrap();
    f.store.set_status(&f.id, "running", None).unwrap();
    let restarted = Store::new(f.temp.path().join("appdata")).unwrap();
    let p = restarted.project(&f.id).unwrap();
    assert_eq!(p.import_status, "cancelled");
    assert!(p.import_error.unwrap().contains("interrupted"));
    assert_eq!(p.photos[0].analysis_error.as_deref(), Some("Damaged JPEG"));
}
#[test]
fn worker_cannot_ingest_outside_source_or_cache() {
    let f = Fixture::new();
    let mut p = f.photo("e");
    let outside = f.temp.path().join("outside.jpg");
    fs::write(&outside, b"outside").unwrap();
    p.preview_path = outside.to_str().unwrap().into();
    assert!(f.store.ingest_photo(&f.id, p).is_err());
    let mut p = f.photo("f");
    p.path = outside.to_str().unwrap().into();
    assert!(f.store.ingest_photo(&f.id, p).is_err());
    assert!(f
        .store
        .set_detail(&f.id, "unknown", outside.to_str().unwrap())
        .is_err());
}
#[test]
fn undo_restores_untouched_rating_and_collection_errors_do_not_mutate() {
    let f = Fixture::new();
    let p = f.add("a");
    f.store
        .update_photos(
            &f.id,
            &[p.id.clone()],
            &PhotoPatch {
                rating: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
    f.store
        .update_photos(
            &f.id,
            &[p.id.clone()],
            &PhotoPatch {
                rating: Some(0),
                rating_touched: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!f.store.project(&f.id).unwrap().photos[0].rating_touched);
    let c = f.store.create_collection(&f.id, "Album").unwrap();
    assert!(f
        .store
        .update_collection(
            &f.id,
            &c.id,
            Some("Changed".into()),
            Some(vec!["missing".into()])
        )
        .is_err());
    let p = f.store.project(&f.id).unwrap();
    assert_eq!(p.collections[0].name, "Album");
    let _: Collection = p.collections[0].clone();
    f.store.delete_collection(&f.id, &c.id).unwrap();
    assert!(f.store.project(&f.id).unwrap().collections.is_empty());
}
#[test]
fn opening_project_retains_its_identity_and_wal_edits() {
    let f = Fixture::new();
    let photo = f.add("a");
    f.store
        .update_photos(
            &f.id,
            &[photo.id],
            &PhotoPatch {
                decision: Some("favorite".into()),
                ..Default::default()
            },
        )
        .unwrap();
    let project = f.store.project(&f.id).unwrap();
    let other = Store::new(f.temp.path().join("second-appdata")).unwrap();
    let reopened = other.open(&project.project_path).unwrap();
    assert_eq!(reopened.id, f.id);
    assert_eq!(reopened.photos[0].decision, "favorite");
    assert_eq!(other.summaries().unwrap().len(), 1);
    assert!(other
        .open(&photo_select::core::path_string(
            &f.temp.path().join("missing.cullproj")
        ))
        .is_err());
}

#[test]
fn missing_or_damaged_recent_index_recovers_authoritative_project_databases() {
    let f = Fixture::new();
    let photo = f.add("a");
    f.store
        .update_photos(
            &f.id,
            &[photo.id],
            &PhotoPatch {
                rating: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
    fs::write(f.store.root.join("recent-projects.json"), b"damaged json").unwrap();
    let recovered = Store::new(f.store.root.clone()).unwrap();
    assert_eq!(recovered.summaries().unwrap().len(), 1);
    assert_eq!(recovered.project(&f.id).unwrap().photos[0].rating, 5);
    assert!(fs::read_dir(&f.store.root).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with("recent-projects-damaged-")));
    fs::remove_file(f.store.root.join("recent-projects.json")).unwrap();
    let recovered = Store::new(f.store.root.clone()).unwrap();
    assert_eq!(recovered.summaries().unwrap().len(), 1);
}

#[cfg(unix)]
#[test]
fn symlinked_cache_cannot_direct_worker_writes_into_originals() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let cache = f.store.cache(&f.id);
    fs::remove_dir(&cache).unwrap();
    symlink(&f.source, &cache).unwrap();
    assert!(f.store.prepare_cache(&f.id).is_err());
}

#[test]
fn review_autosave_and_analysis_can_write_concurrently_without_lost_decisions() {
    use std::sync::{Arc, Barrier};
    let f = Fixture::new();
    let photo = f.add("a");
    let barrier = Arc::new(Barrier::new(2));
    let review_store = f.store.clone();
    let review_id = f.id.clone();
    let photo_id = photo.id.clone();
    let review_barrier = barrier.clone();
    let review = std::thread::spawn(move || {
        review_barrier.wait();
        for _ in 0..50 {
            review_store
                .update_photos(
                    &review_id,
                    &[photo_id.clone()],
                    &PhotoPatch {
                        rating: Some(5),
                        decision: Some("favorite".into()),
                        reviewed: Some(true),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    });
    barrier.wait();
    for _ in 0..50 {
        f.store.ingest_photo(&f.id, photo.clone()).unwrap();
    }
    review.join().unwrap();
    let project = f.store.project(&f.id).unwrap();
    assert_eq!(project.photos[0].rating, 5);
    assert_eq!(project.photos[0].decision, "favorite");
    assert!(project.photos[0].rating_touched);
    assert!(project.photos[0].reviewed);
}

#[test]
fn saved_tags_are_safe_for_the_lightroom_keyword_api() {
    let f = Fixture::new();
    let photo = f.add("a");
    for tag in ["a,b", "a;b", "a|b", "a<b", "a>b", "a\nb", "a\0b"] {
        assert!(f
            .store
            .update_photos(
                &f.id,
                &[photo.id.clone()],
                &PhotoPatch {
                    tags: Some(vec![tag.into()]),
                    ..Default::default()
                }
            )
            .is_err());
    }
    f.store
        .update_photos(
            &f.id,
            &[photo.id],
            &PhotoPatch {
                tags: Some(vec!["Été 2026".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        f.store.project(&f.id).unwrap().photos[0].tags,
        vec!["Été 2026"]
    );
}

#[cfg(windows)]
#[test]
fn canonical_windows_paths_are_emitted_in_lightroom_compatible_notation() {
    use photo_select::core::path_string;
    assert_eq!(
        path_string(Path::new(r"\\?\C:\Photos\image.jpg")),
        r"C:\Photos\image.jpg"
    );
    assert_eq!(
        path_string(Path::new(r"\\?\UNC\server\share\image.jpg")),
        r"\\server\share\image.jpg"
    );
}
