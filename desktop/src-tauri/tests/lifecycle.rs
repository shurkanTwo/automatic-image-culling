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

#[test]
fn heterogeneous_review_updates_commit_or_roll_back_as_one_unit() {
    use photo_select::core::PhotoUpdate;
    let f = Fixture::new();
    let a = f.add("a");
    let b = f.add("b");
    let updates = vec![
        PhotoUpdate {
            photo_id: a.id.clone(),
            patch: PhotoPatch {
                rating: Some(5),
                tags: Some(vec!["Album".into()]),
                ..Default::default()
            },
        },
        PhotoUpdate {
            photo_id: b.id.clone(),
            patch: PhotoPatch {
                decision: Some("pass".into()),
                ..Default::default()
            },
        },
    ];
    f.store.update_photo_patches(&f.id, &updates).unwrap();
    let before = f.store.project(&f.id).unwrap();
    assert_eq!(before.photos[0].rating, 5);
    assert_eq!(before.photos[1].decision, "pass");
    let invalid = vec![
        PhotoUpdate {
            photo_id: a.id.clone(),
            patch: PhotoPatch {
                rating: Some(0),
                rating_touched: Some(false),
                ..Default::default()
            },
        },
        PhotoUpdate {
            photo_id: "missing".into(),
            patch: PhotoPatch::default(),
        },
    ];
    assert!(f.store.update_photo_patches(&f.id, &invalid).is_err());
    assert_eq!(f.store.project(&f.id).unwrap().photos[0].rating, 5);
    assert!(f
        .store
        .update_photo_patches(&f.id, &[updates[0].clone(), updates[0].clone()])
        .is_err());
    let undo = vec![
        PhotoUpdate {
            photo_id: a.id.clone(),
            patch: PhotoPatch {
                rating: Some(0),
                rating_touched: Some(false),
                tags: Some(vec![]),
                ..Default::default()
            },
        },
        PhotoUpdate {
            photo_id: b.id,
            patch: PhotoPatch {
                decision: Some("undecided".into()),
                ..Default::default()
            },
        },
    ];
    f.store.update_photo_patches(&f.id, &undo).unwrap();
    let result = f.store.project(&f.id).unwrap();
    assert!(!result.photos[0].rating_touched);
    assert!(result.photos[0].tags.is_empty());
    assert_eq!(result.photos[1].decision, "undecided");
}

#[test]
fn unavailable_originals_keep_review_and_cached_previews_and_export_offline() {
    let f = Fixture::new();
    let photo = f.add("a");
    f.store
        .update_photos(
            &f.id,
            &[photo.id.clone()],
            &PhotoPatch {
                decision: Some("favorite".into()),
                rating: Some(4),
                tags: Some(vec!["Album".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let collection = f.store.create_collection(&f.id, "Album").unwrap();
    f.store
        .update_collection(&f.id, &collection.id, None, Some(vec![photo.id.clone()]))
        .unwrap();
    f.store
        .ingest_groups(
            &f.id,
            vec![Group {
                id: "burst".into(),
                label: "Burst".into(),
                photo_ids: vec![photo.id.clone()],
                recommended_photo_ids: vec![photo.id.clone()],
            }],
        )
        .unwrap();
    let detail = f.store.cache(&f.id).join("offline-detail.jpg");
    fs::write(&detail, b"full-resolution cache").unwrap();
    f.store
        .set_detail(&f.id, &photo.id, detail.to_str().unwrap())
        .unwrap();
    fs::remove_dir_all(&f.source).unwrap();
    f.store.finish_scan(&f.id, &Default::default()).unwrap();
    let project = f.store.project(&f.id).unwrap();
    let retained = &project.photos[0];
    assert_eq!(retained.rating, 4);
    assert_eq!(retained.decision, "favorite");
    assert_eq!(retained.tags, vec!["Album"]);
    assert!(Path::new(&retained.preview_path).is_file());
    assert!(retained
        .analysis_error
        .as_ref()
        .unwrap()
        .contains("unavailable"));
    assert_eq!(retained.detail_path.as_deref(), detail.to_str());
    assert_eq!(fs::read(&detail).unwrap(), b"full-resolution cache");
    assert!(retained.group_id.is_none());
    assert!(project.groups.is_empty());
    assert_eq!(project.collections[0].photo_ids, vec![photo.id]);
    let destination = f.temp.path().join("offline.json");
    assert_eq!(
        f.store
            .export(
                &f.id,
                destination.to_str().unwrap(),
                Some(&collection.id),
                false
            )
            .unwrap()
            .count,
        1
    );
}

#[test]
fn manifest_cannot_overwrite_application_state_or_dangling_symlinks() {
    let f = Fixture::new();
    let _ = f.add("a");
    let index = f.store.root.join("recent-projects.json");
    let before = fs::read(&index).unwrap();
    assert!(f
        .store
        .export(&f.id, index.to_str().unwrap(), None, false)
        .is_err());
    assert_eq!(fs::read(index).unwrap(), before);
    #[cfg(unix)]
    {
        let destination = f.temp.path().join("linked.json");
        let missing = f.temp.path().join("missing.json");
        std::os::unix::fs::symlink(&missing, &destination).unwrap();
        assert!(f
            .store
            .export(&f.id, destination.to_str().unwrap(), None, false)
            .is_err());
        assert!(fs::symlink_metadata(destination)
            .unwrap()
            .file_type()
            .is_symlink());
    }
}

#[test]
fn forged_registry_identity_and_mismatched_database_rows_are_rejected() {
    let f = Fixture::new();
    let photo = f.add("a");
    let project = f.store.project(&f.id).unwrap();
    let db = rusqlite::Connection::open(&project.project_path).unwrap();
    let mut row = serde_json::to_value(&photo).unwrap();
    row["id"] = serde_json::json!("b00000000000000000000000");
    db.execute(
        "UPDATE photos SET data=?1 WHERE id=?2",
        rusqlite::params![row.to_string(), photo.id],
    )
    .unwrap();
    assert!(f.store.project(&f.id).is_err());
    assert!(f.store.open(&project.project_path).is_err());
    db.execute("DELETE FROM photos", []).unwrap();
    let mut metadata: serde_json::Value = serde_json::from_str(
        &db.query_row::<String, _, _>("SELECT data FROM meta", [], |row| row.get(0))
            .unwrap(),
    )
    .unwrap();
    metadata["id"] = serde_json::json!(uuid::Uuid::new_v4().to_string());
    db.execute("UPDATE meta SET data=?1", [metadata.to_string()])
        .unwrap();
    assert!(f.store.project(&f.id).is_err());
    let restarted = Store::new(f.store.root.clone()).unwrap();
    assert!(restarted.summaries().unwrap().is_empty());
    assert!(restarted.project("../../outside").is_err());
}

#[test]
fn reanalysis_cancellation_invalidates_recommendations_without_losing_review_or_moments() {
    for failed in [false, true] {
        let f = Fixture::new();
        let a = f.add("a");
        let b = f.add("b");
        let group = Group {
            id: "stable-moment".into(),
            label: "Moment".into(),
            photo_ids: vec![a.id.clone(), b.id.clone()],
            recommended_photo_ids: vec![a.id.clone(), b.id.clone()],
        };
        f.store.ingest_groups(&f.id, vec![group.clone()]).unwrap();
        f.store
            .update_photos(
                &f.id,
                &[a.id.clone()],
                &PhotoPatch {
                    rating: Some(4),
                    decision: Some("favorite".into()),
                    reviewed: Some(true),
                    tags: Some(vec!["Album".into()]),
                    ..Default::default()
                },
            )
            .unwrap();
        f.store.set_status(&f.id, "running", None).unwrap();
        let mut changed = a.clone();
        changed.quality_score = 20.;
        if failed {
            changed.analysis_error = Some("Cannot decode changed original".into());
            changed.preview_path.clear();
            changed.thumbnail_path.clear();
        }
        f.store.ingest_photo(&f.id, changed).unwrap();
        f.store.set_status(&f.id, "cancelled", None).unwrap();
        let reopened = Store::new(f.store.root.clone()).unwrap();
        let cancelled = reopened.project(&f.id).unwrap();
        let reviewed = cancelled
            .photos
            .iter()
            .find(|photo| photo.id == a.id)
            .unwrap();
        assert_eq!(cancelled.import_status, "cancelled");
        assert_eq!(cancelled.groups[0].id, group.id);
        assert_eq!(cancelled.groups[0].photo_ids, group.photo_ids);
        assert_eq!(
            cancelled.groups[0].recommended_photo_ids,
            vec![b.id.clone()]
        );
        assert_eq!(reviewed.group_id.as_deref(), Some("stable-moment"));
        assert_eq!(reviewed.rating, 4);
        assert_eq!(reviewed.decision, "favorite");
        assert!(reviewed.reviewed);
        assert_eq!(reviewed.tags, vec!["Album"]);
        reopened
            .ingest_groups(
                &f.id,
                vec![Group {
                    id: "new-group-id".into(),
                    ..group.clone()
                }],
            )
            .unwrap();
        let regrouped = reopened.project(&f.id).unwrap();
        let expected = if failed {
            vec![b.id.clone()]
        } else {
            vec![a.id.clone(), b.id.clone()]
        };
        assert_eq!(regrouped.groups[0].recommended_photo_ids, expected);
        assert_eq!(regrouped.groups[0].id, "stable-moment");
        if failed {
            reopened.ingest_photo(&f.id, a.clone()).unwrap();
            reopened.ingest_groups(&f.id, vec![group.clone()]).unwrap();
            let recovered = reopened.project(&f.id).unwrap();
            assert_eq!(
                recovered.groups[0].recommended_photo_ids,
                vec![a.id.clone(), b.id.clone()]
            );
            assert_eq!(
                recovered
                    .photos
                    .iter()
                    .find(|photo| photo.id == a.id)
                    .unwrap()
                    .rating,
                4
            );
        }
    }
}

#[cfg(windows)]
#[test]
fn windows_aliases_are_normalized_on_ingestion_and_legacy_alias_rows_remain_readable() {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetShortPathNameW(long: *const u16, short: *mut u16, length: u32) -> u32;
    }
    fn short_path(path: &Path) -> String {
        let input: Vec<_> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let length = unsafe { GetShortPathNameW(input.as_ptr(), std::ptr::null_mut(), 0) };
        assert!(length > 0, "{}", std::io::Error::last_os_error());
        let mut output = vec![0u16; length as usize];
        let written = unsafe { GetShortPathNameW(input.as_ptr(), output.as_mut_ptr(), length) };
        assert!(written > 0 && written < length);
        String::from_utf16(&output[..written as usize]).unwrap()
    }
    let f = Fixture::new();
    let mut photo = f.photo("a");
    let long = Path::new(&f.source).join("photo with spaces.jpg");
    fs::rename(&photo.path, &long).unwrap();
    let alias = short_path(&long);
    photo.path = alias.clone();
    photo.filename = "photo with spaces.jpg".into();
    photo.preview_path = short_path(Path::new(&photo.preview_path));
    photo.thumbnail_path = photo.preview_path.clone();
    f.store.ingest_photo(&f.id, photo.clone()).unwrap();
    let project = f.store.project(&f.id).unwrap();
    let stored = &project.photos[0];
    assert_eq!(
        stored.path,
        photo_select::core::path_string(&fs::canonicalize(&long).unwrap())
    );
    assert_eq!(stored.filename, "photo with spaces.jpg");
    assert_eq!(
        stored.preview_path,
        photo_select::core::path_string(&fs::canonicalize(&stored.preview_path).unwrap())
    );
    let mut legacy = stored.clone();
    legacy.path = alias;
    let db = f.store.connection(&f.id).unwrap();
    db.execute(
        "UPDATE photos SET data=?1 WHERE id=?2",
        rusqlite::params![serde_json::to_string(&legacy).unwrap(), legacy.id],
    )
    .unwrap();
    assert!(f.store.project(&f.id).is_ok());
}

#[cfg(windows)]
#[test]
fn wholly_disconnected_source_drive_keeps_cached_review_reopen_and_export_available() {
    let unavailable = (b'D'..=b'Z')
        .rev()
        .map(|letter| format!("{}:\\", letter as char))
        .find(|drive| !Path::new(drive).exists())
        .expect("Windows fixture needs an unmounted drive letter");
    let f = Fixture::new();
    let photo = f.add("a");
    f.store
        .update_photos(
            &f.id,
            &[photo.id.clone()],
            &PhotoPatch {
                rating: Some(4),
                decision: Some("favorite".into()),
                ..Default::default()
            },
        )
        .unwrap();
    let project = f.store.project(&f.id).unwrap();
    let mut offline = project.photos[0].clone();
    let source = Path::new(&unavailable).join("Photo Select offline originals");
    offline.path = photo_select::core::path_string(&source.join("a.jpg"));
    let db = f.store.connection(&f.id).unwrap();
    let mut metadata: serde_json::Value = serde_json::from_str(
        &db.query_row::<String, _, _>("SELECT data FROM meta", [], |row| row.get(0))
            .unwrap(),
    )
    .unwrap();
    metadata["sourceDir"] = serde_json::json!(source);
    db.execute("UPDATE meta SET data=?1", [metadata.to_string()])
        .unwrap();
    db.execute(
        "UPDATE photos SET data=?1 WHERE id=?2",
        rusqlite::params![serde_json::to_string(&offline).unwrap(), offline.id],
    )
    .unwrap();
    let loaded = f.store.project(&f.id).unwrap();
    assert_eq!(loaded.photos[0].rating, 4);
    assert!(Path::new(&loaded.photos[0].preview_path).is_file());
    f.store.open(&project.project_path).unwrap();
    let destination = f.temp.path().join("offline-drive.json");
    assert_eq!(
        f.store
            .export(&f.id, destination.to_str().unwrap(), None, true)
            .unwrap()
            .count,
        1
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(destination).unwrap()).unwrap();
    assert_eq!(manifest["photos"][0]["path"], offline.path);
}
