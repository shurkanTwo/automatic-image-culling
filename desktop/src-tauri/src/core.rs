use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};
use uuid::Uuid;

pub type Result<T> = std::result::Result<T, String>;
pub fn now() -> String {
    Utc::now().to_rfc3339()
}
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Photo {
    pub id: String,
    pub path: String,
    pub filename: String,
    pub preview_path: String,
    pub thumbnail_path: String,
    #[serde(default)]
    pub detail_path: Option<String>,
    pub capture_time: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub camera: Option<String>,
    #[serde(default)]
    pub group_id: Option<String>,
    pub quality_score: f64,
    #[serde(default)]
    pub hints: Vec<String>,
    #[serde(default)]
    pub rating: u8,
    #[serde(default)]
    pub rating_touched: bool,
    #[serde(default = "undecided")]
    pub decision: String,
    #[serde(default)]
    pub reviewed: bool,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub analysis_error: Option<String>,
}
fn undecided() -> String {
    "undecided".into()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Group {
    pub id: String,
    pub label: String,
    pub photo_ids: Vec<String>,
    pub recommended_photo_ids: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub photo_ids: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub id: String,
    pub name: String,
    pub source_dir: String,
    pub project_path: String,
    pub created_at: String,
    pub updated_at: String,
    pub photos: Vec<Photo>,
    pub groups: Vec<Group>,
    pub collections: Vec<Collection>,
    pub import_status: String,
    pub import_error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub source_dir: String,
    pub project_path: String,
    pub photo_count: usize,
    pub favorite_count: usize,
    pub updated_at: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhotoPatch {
    pub rating: Option<u8>,
    pub rating_touched: Option<bool>,
    pub decision: Option<String>,
    pub reviewed: Option<bool>,
    pub tags: Option<Vec<String>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhotoUpdate {
    pub photo_id: String,
    pub patch: PhotoPatch,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub path: String,
    pub count: usize,
}

#[derive(Clone)]
pub struct Store {
    pub root: PathBuf,
    paths: Arc<Mutex<HashMap<String, PathBuf>>>,
}
impl Store {
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(root.join("projects")).map_err(error)?;
        let root = fs::canonicalize(root).map_err(error)?;
        let index_path = root.join("recent-projects.json");
        let mut paths: HashMap<String, PathBuf> = if index_path.exists() {
            match serde_json::from_slice(&fs::read(&index_path).map_err(error)?) {
                Ok(paths) => paths,
                Err(_) => {
                    // The index is only a convenience; the project databases remain authoritative.
                    fs::rename(
                        &index_path,
                        root.join(format!("recent-projects-damaged-{}.json", Uuid::new_v4())),
                    )
                    .map_err(error)?;
                    HashMap::new()
                }
            }
        } else {
            HashMap::new()
        };
        paths.retain(|id, path| {
            valid_project_id(id).is_ok()
                && path.is_absolute()
                && Self::connection_at(path)
                    .and_then(|db| read_meta(&db))
                    .is_ok_and(|project| project.id == *id)
        });
        for entry in fs::read_dir(root.join("projects")).map_err(error)? {
            let entry = entry.map_err(error)?;
            let Some(id) = entry.file_name().to_str().map(String::from) else {
                continue;
            };
            if Uuid::parse_str(&id).is_err() || paths.contains_key(&id) {
                continue;
            }
            let project_path = entry.path().join("project.cullproj");
            if !project_path.is_file() {
                continue;
            }
            if let Ok(db) = Self::connection_at(&project_path) {
                if let Ok(project) = read_meta(&db) {
                    if project.id == id {
                        paths.insert(id, project_path);
                    }
                }
            }
        }
        atomic_write(
            &index_path,
            &serde_json::to_vec_pretty(&paths).map_err(error)?,
        )?;
        let store = Self {
            root,
            paths: Arc::new(Mutex::new(paths)),
        };
        // A prior process cannot still own these jobs. Partial analysis and every manual edit remain durable.
        let ids: Vec<String> = store.paths.lock().map_err(error)?.keys().cloned().collect();
        for id in ids {
            if let Ok(p) = store.project(&id) {
                if p.import_status == "running" {
                    store.set_status(
                        &id,
                        "cancelled",
                        Some("Import interrupted when the application closed.".into()),
                    )?;
                }
            }
        }
        Ok(store)
    }
    fn connection_at(path: &Path) -> Result<Connection> {
        let db = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(error)?;
        db.busy_timeout(Duration::from_secs(10)).map_err(error)?;
        db.pragma_update(None, "foreign_keys", "ON")
            .map_err(error)?;
        let version: i64 = db
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .map_err(error)?;
        if version != 1 {
            return Err("Unsupported project format".into());
        }
        Ok(db)
    }
    pub fn connection(&self, id: &str) -> Result<Connection> {
        valid_project_id(id)?;
        let path = self
            .paths
            .lock()
            .map_err(error)?
            .get(id)
            .cloned()
            .ok_or("Unknown project")?;
        let db = Self::connection_at(&path)?;
        if read_meta(&db)?.id != id {
            return Err("Project identity does not match its registered file".into());
        }
        Ok(db)
    }
    fn register(&self, id: String, path: PathBuf) -> Result<()> {
        let mut paths = self.paths.lock().map_err(error)?;
        let mut updated = paths.clone();
        updated.insert(id, path);
        atomic_write(
            &self.root.join("recent-projects.json"),
            &serde_json::to_vec_pretty(&updated).map_err(error)?,
        )?;
        *paths = updated;
        Ok(())
    }
    pub fn create(&self, name: &str, source: &str) -> Result<Project> {
        let name = valid_name(name)?;
        let source =
            fs::canonicalize(source).map_err(|e| format!("Cannot open source folder: {e}"))?;
        if !source.is_dir() {
            return Err("Source must be a folder".into());
        }
        let data_root = fs::canonicalize(&self.root).map_err(error)?;
        if source.starts_with(&data_root) || data_root.starts_with(&source) {
            return Err(
                "Choose a photo folder separate from the application's project storage".into(),
            );
        }
        let id = Uuid::new_v4().to_string();
        let dir = self.root.join("projects").join(&id);
        fs::create_dir_all(dir.join("cache")).map_err(error)?;
        let path = dir.join("project.cullproj");
        let db = Connection::open(&path).map_err(error)?;
        db.pragma_update(None, "journal_mode", "WAL")
            .map_err(error)?;
        db.execute_batch("CREATE TABLE meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), data TEXT NOT NULL); CREATE TABLE photos (id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE groups_data (id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE collections (id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE import_progress (singleton INTEGER PRIMARY KEY CHECK(singleton=1), data TEXT NOT NULL); PRAGMA user_version=1;").map_err(error)?;
        let p = Project {
            id: id.clone(),
            name,
            source_dir: path_string(&source),
            project_path: path_string(&path),
            created_at: now(),
            updated_at: now(),
            photos: vec![],
            groups: vec![],
            collections: vec![],
            import_status: "idle".into(),
            import_error: None,
        };
        write_meta(&db, &p)?;
        self.register(id, path)?;
        Ok(p)
    }
    pub fn identify_project(&self, path: &str) -> Result<String> {
        let path = fs::canonicalize(path).map_err(error)?;
        if path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| !value.eq_ignore_ascii_case("cullproj"))
            .unwrap_or(true)
        {
            return Err("Choose a .cullproj project file".into());
        }
        read_meta(&Self::connection_at(&path)?).map(|project| project.id)
    }
    pub fn validate_source_dir(&self, source: &str) -> Result<()> {
        let source = canonical_or_lexical(Path::new(source))?;
        if path_is_within(&source, &self.root) || path_is_within(&self.root, &source) {
            return Err(
                "Choose a photo folder separate from the application's project storage".into(),
            );
        }
        Ok(())
    }
    pub fn open(&self, path: &str) -> Result<Project> {
        let path = fs::canonicalize(path).map_err(error)?;
        if path
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| !v.eq_ignore_ascii_case("cullproj"))
            .unwrap_or(true)
        {
            return Err("Choose a .cullproj project file".into());
        }
        let mut db = Self::connection_at(&path)?;
        let version: i64 = db
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .map_err(error)?;
        if version != 1 {
            return Err("Unsupported project format".into());
        }
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let mut p = read_meta(&tx)?;
        valid_project_id(&p.id)?;
        validate_rows(&tx, &p)?;
        self.validate_source_dir(&p.source_dir)?;
        let registered = self.paths.lock().map_err(error)?.get(&p.id).cloned();
        if registered
            .as_ref()
            .is_some_and(|existing| existing != &path && existing.exists())
        {
            return Err("Another file with this project identity is already registered".into());
        }
        p.project_path = path_string(&path);
        if registered.is_none() && p.import_status == "running" {
            p.import_status = "cancelled".into();
            p.import_error = Some("Import interrupted when the application closed.".into());
        }
        write_meta(&tx, &p)?;
        tx.commit().map_err(error)?;
        self.prepare_cache(&p.id)?;
        self.register(p.id.clone(), path)?;
        self.project(&p.id)
    }
    pub fn project(&self, id: &str) -> Result<Project> {
        let mut db = self.connection(id)?;
        let tx = db.transaction().map_err(error)?;
        let mut p = read_meta(&tx)?;
        p.project_path = path_string(Path::new(
            tx.path().ok_or("Project database has no file path")?,
        ));
        p.photos = read_items(&tx, "photos")?;
        p.photos.sort_by(|a, b| {
            a.capture_time
                .cmp(&b.capture_time)
                .then(a.filename.cmp(&b.filename))
        });
        p.groups = read_items(&tx, "groups_data")?;
        p.collections = read_items(&tx, "collections")?;
        validate_project_rows(&p)?;
        tx.commit().map_err(error)?;
        Ok(p)
    }
    pub fn summaries(&self) -> Result<Vec<ProjectSummary>> {
        let ids: Vec<_> = self.paths.lock().map_err(error)?.keys().cloned().collect();
        let mut output = Vec::new();
        for id in ids {
            if let Ok(p) = self.project(&id) {
                output.push(ProjectSummary {
                    id: p.id,
                    name: p.name,
                    source_dir: p.source_dir,
                    project_path: p.project_path,
                    photo_count: p.photos.len(),
                    favorite_count: p.photos.iter().filter(|v| v.decision == "favorite").count(),
                    updated_at: p.updated_at,
                });
            }
        }
        output.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(output)
    }
    pub fn cache(&self, id: &str) -> PathBuf {
        self.root.join("projects").join(id).join("cache")
    }
    pub fn prepare_cache(&self, id: &str) -> Result<PathBuf> {
        valid_project_id(id)?;
        let dir = self.root.join("projects").join(id);
        ensure_within(&dir, &self.root)?;
        fs::create_dir_all(&dir).map_err(error)?;
        let cache = dir.join("cache");
        ensure_within(&cache, &self.root)?;
        fs::create_dir_all(&cache).map_err(error)?;
        Ok(cache)
    }
    pub fn set_status(&self, id: &str, status: &str, reason: Option<String>) -> Result<()> {
        if !["idle", "running", "completed", "cancelled", "failed"].contains(&status) {
            return Err("Invalid import status".into());
        }
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let mut p = read_meta(&tx)?;
        p.import_status = status.into();
        p.import_error = reason;
        p.updated_at = now();
        write_meta(&tx, &p)?;
        tx.commit().map_err(error)
    }
    pub fn save_progress(&self, id: &str, value: &serde_json::Value) -> Result<()> {
        let db = self.connection(id)?;
        db.execute_batch("CREATE TABLE IF NOT EXISTS import_progress (singleton INTEGER PRIMARY KEY CHECK(singleton=1),data TEXT NOT NULL)").map_err(error)?;
        db.execute("INSERT INTO import_progress(singleton,data) VALUES(1,?1) ON CONFLICT(singleton) DO UPDATE SET data=excluded.data",[serde_json::to_string(value).map_err(error)?]).map_err(error)?;
        Ok(())
    }
    pub fn ingest_photo(&self, id: &str, mut photo: Photo) -> Result<()> {
        photo.path = path_string(Path::new(&photo.path));
        photo.preview_path = path_string(Path::new(&photo.preview_path));
        photo.thumbnail_path = path_string(Path::new(&photo.thumbnail_path));
        if photo.id.len() != 24 || !photo.id.bytes().all(|v| v.is_ascii_hexdigit()) {
            return Err("Invalid engine photo identity".into());
        }
        if !photo.quality_score.is_finite() || !(0.0..=100.0).contains(&photo.quality_score) {
            return Err("Invalid technical score".into());
        }
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let project = read_meta(&tx)?;
        ensure_within(Path::new(&photo.path), Path::new(&project.source_dir))?;
        for image in [&photo.preview_path, &photo.thumbnail_path] {
            if image.is_empty() && photo.analysis_error.is_some() {
                continue;
            }
            ensure_within(Path::new(image), &self.cache(id))?;
        }
        let previous: Option<Photo> = read_item(&tx, "photos", &photo.id)?;
        photo.rating = 0;
        photo.rating_touched = false;
        photo.decision = undecided();
        photo.reviewed = false;
        photo.tags.clear();
        photo.detail_path = None;
        photo.group_id = None;
        if let Some(old) = previous {
            photo.rating = old.rating;
            photo.rating_touched = old.rating_touched;
            photo.decision = old.decision;
            photo.reviewed = old.reviewed;
            photo.tags = old.tags;
            // Revalidate full-resolution caches against the source through the engine on demand.
            photo.detail_path = None;
            photo.group_id = old.group_id;
        }
        write_item(&tx, "photos", &photo.id, &photo)?;
        touch(&tx)?;
        tx.commit().map_err(error)
    }
    pub fn ingest_groups(&self, id: &str, groups: Vec<Group>) -> Result<()> {
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let mut photos: Vec<Photo> = read_items(&tx, "photos")?;
        let old_groups: Vec<Group> = read_items(&tx, "groups_data")?;
        let known: HashSet<_> = photos.iter().map(|v| v.id.clone()).collect();
        let mut assigned = HashSet::new();
        let mut group_ids = HashSet::new();
        let mut prepared = Vec::new();
        for mut group in groups {
            if group.photo_ids.is_empty()
                || group
                    .photo_ids
                    .iter()
                    .any(|v| !known.contains(v) || !assigned.insert(v.clone()))
            {
                return Err("Invalid group membership".into());
            }
            if group
                .recommended_photo_ids
                .iter()
                .any(|v| !group.photo_ids.contains(v))
            {
                return Err("Recommendation outside group".into());
            }
            // Reuse the group with the greatest overlap, so an added photo does not destroy stable UI identities.
            if let Some(old) = old_groups
                .iter()
                .filter(|v| !group_ids.contains(&v.id))
                .max_by_key(|v| {
                    v.photo_ids
                        .iter()
                        .filter(|id| group.photo_ids.contains(id))
                        .count()
                })
            {
                if old.photo_ids.iter().any(|v| group.photo_ids.contains(v)) {
                    group.id = old.id.clone();
                }
            }
            if group.id.is_empty() || !group_ids.insert(group.id.clone()) {
                return Err("Duplicate group identity".into());
            }
            for photo in &mut photos {
                if group.photo_ids.contains(&photo.id) {
                    photo.group_id = Some(group.id.clone());
                }
            }
            prepared.push(group);
        }
        for photo in &mut photos {
            if !assigned.contains(&photo.id) {
                photo.group_id = None;
            }
            write_item(&tx, "photos", &photo.id, photo)?;
        }
        tx.execute("DELETE FROM groups_data", []).map_err(error)?;
        for group in prepared {
            write_item(&tx, "groups_data", &group.id, &group)?;
        }
        touch(&tx)?;
        tx.commit().map_err(error)
    }
    pub fn update_photos(
        &self,
        id: &str,
        photo_ids: &[String],
        patch: &PhotoPatch,
    ) -> Result<Vec<Photo>> {
        let mut seen = HashSet::new();
        let updates: Vec<_> = photo_ids
            .iter()
            .filter(|photo_id| seen.insert(*photo_id))
            .map(|photo_id| PhotoUpdate {
                photo_id: photo_id.clone(),
                patch: patch.clone(),
            })
            .collect();
        self.update_photo_patches(id, &updates)
    }
    pub fn update_photo_patches(&self, id: &str, updates: &[PhotoUpdate]) -> Result<Vec<Photo>> {
        if updates.is_empty() {
            return Err("Choose at least one photo".into());
        }
        let mut unique = HashSet::new();
        for update in updates {
            validate_patch(&update.patch)?;
            if !unique.insert(&update.photo_id) {
                return Err("Each photo may appear only once in a batch update".into());
            }
        }
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let mut output = Vec::new();
        for update in updates {
            let photo_id = &update.photo_id;
            let patch = &update.patch;
            let mut photo: Photo = read_item(&tx, "photos", photo_id)?.ok_or("Unknown photo")?;
            if let Some(rating) = patch.rating {
                photo.rating = rating;
                photo.rating_touched = true;
            }
            if let Some(touched) = patch.rating_touched {
                photo.rating_touched = touched;
            }
            if let Some(decision) = &patch.decision {
                photo.decision = decision.clone();
            }
            if let Some(reviewed) = patch.reviewed {
                photo.reviewed = reviewed;
            }
            if let Some(tags) = &patch.tags {
                photo.tags = tags
                    .iter()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .collect();
                photo.tags.sort();
                photo.tags.dedup();
            }
            write_item(&tx, "photos", photo_id, &photo)?;
            output.push(photo);
        }
        touch(&tx)?;
        tx.commit().map_err(error)?;
        Ok(output)
    }
    pub fn finish_scan(&self, id: &str, seen: &HashSet<String>) -> Result<()> {
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let mut photos: Vec<Photo> = read_items(&tx, "photos")?;
        for photo in &mut photos {
            if !seen.contains(&photo.id) {
                photo.analysis_error=Some("Original file was unavailable during the latest import (missing, unreadable, or unsupported). Cached previews and review decisions are preserved.".into());
                photo.group_id = None;
                write_item(&tx, "photos", &photo.id, photo)?;
            }
        }
        let groups: Vec<Group> = read_items(&tx, "groups_data")?;
        for mut group in groups {
            group.photo_ids.retain(|photo_id| seen.contains(photo_id));
            group
                .recommended_photo_ids
                .retain(|photo_id| seen.contains(photo_id));
            if group.photo_ids.is_empty() {
                tx.execute("DELETE FROM groups_data WHERE id=?1", [&group.id])
                    .map_err(error)?;
            } else {
                write_item(&tx, "groups_data", &group.id, &group)?;
            }
        }
        touch(&tx)?;
        tx.commit().map_err(error)
    }
    pub fn create_collection(&self, id: &str, name: &str) -> Result<Collection> {
        let collection = Collection {
            id: Uuid::new_v4().to_string(),
            name: valid_name(name)?,
            photo_ids: vec![],
        };
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        write_item(&tx, "collections", &collection.id, &collection)?;
        touch(&tx)?;
        tx.commit().map_err(error)?;
        Ok(collection)
    }
    pub fn update_collection(
        &self,
        id: &str,
        collection_id: &str,
        name: Option<String>,
        photo_ids: Option<Vec<String>>,
    ) -> Result<Collection> {
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let mut collection: Collection =
            read_item(&tx, "collections", collection_id)?.ok_or("Unknown collection")?;
        if let Some(name) = name {
            collection.name = valid_name(&name)?;
        }
        if let Some(ids) = photo_ids {
            let mut unique = Vec::new();
            let mut seen = HashSet::new();
            for photo_id in ids {
                if read_item::<Photo>(&tx, "photos", &photo_id)?.is_none() {
                    return Err("Collection contains an unknown photo".into());
                }
                if seen.insert(photo_id.clone()) {
                    unique.push(photo_id);
                }
            }
            collection.photo_ids = unique;
        }
        write_item(&tx, "collections", collection_id, &collection)?;
        touch(&tx)?;
        tx.commit().map_err(error)?;
        Ok(collection)
    }
    pub fn delete_collection(&self, id: &str, collection_id: &str) -> Result<()> {
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        if tx
            .execute("DELETE FROM collections WHERE id=?1", [collection_id])
            .map_err(error)?
            == 0
        {
            return Err("Unknown collection".into());
        }
        touch(&tx)?;
        tx.commit().map_err(error)
    }
    pub fn set_detail(&self, id: &str, photo_id: &str, path: &str) -> Result<()> {
        ensure_within(Path::new(path), &self.cache(id))?;
        let mut db = self.connection(id)?;
        let tx = db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(error)?;
        let mut photo: Photo = read_item(&tx, "photos", photo_id)?.ok_or("Unknown photo")?;
        photo.detail_path = Some(path.into());
        write_item(&tx, "photos", photo_id, &photo)?;
        touch(&tx)?;
        tx.commit().map_err(error)
    }
    pub fn export(
        &self,
        id: &str,
        destination: &str,
        collection_id: Option<&str>,
        only_favorites: bool,
    ) -> Result<ExportResult> {
        let p = self.project(id)?;
        let destination = Path::new(destination);
        if !destination.is_absolute()
            || destination.extension().and_then(|v| v.to_str()) != Some("json")
        {
            return Err("Export destination must be an absolute .json path".into());
        }
        if destination.exists() && destination.is_dir() {
            return Err("Export destination is a folder".into());
        }
        let parent =
            fs::canonicalize(destination.parent().ok_or("Invalid export path")?).map_err(error)?;
        let resolved = parent.join(destination.file_name().ok_or("Invalid export path")?);
        let source = canonical_or_lexical(Path::new(&p.source_dir))?;
        if path_is_within(&resolved, &source)
            || path_is_within(destination, Path::new(&p.source_dir))
        {
            return Err("Save the manifest outside the original photo folder".into());
        }
        if path_is_within(&resolved, &self.root) {
            return Err("Save the manifest outside the application's project storage".into());
        }
        if fs::symlink_metadata(destination).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err("Export destination cannot be a symbolic link".into());
        }
        let collection = collection_id
            .map(|v| {
                p.collections
                    .iter()
                    .find(|c| c.id == v)
                    .ok_or("Unknown collection")
            })
            .transpose()?;
        let photos:Vec<_>=p.photos.iter().filter(|photo|collection.map(|c|c.photo_ids.contains(&photo.id)).unwrap_or(photo.decision=="favorite")).filter(|photo|!only_favorites || photo.decision=="favorite").map(|photo|serde_json::json!({"path":photo.path,"rating":if photo.rating_touched {Some(photo.rating)} else {None},"decision":photo.decision,"tags":photo.tags})).collect();
        let count = photos.len();
        let manifest = serde_json::json!({"schemaVersion":1,"application":"Photo Select","projectName":p.name,"collectionName":collection.map(|v|v.name.as_str()).unwrap_or("Favorites"),"exportedAt":now(),"photos":photos});
        atomic_write(
            &resolved,
            &serde_json::to_vec_pretty(&manifest).map_err(error)?,
        )?;
        Ok(ExportResult {
            path: path_string(&resolved),
            count,
        })
    }
}
fn valid_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 200 {
        return Err("Name must contain 1 to 200 characters".into());
    }
    Ok(name.into())
}
fn validate_patch(p: &PhotoPatch) -> Result<()> {
    if p.rating.is_some_and(|v| v > 5) {
        return Err("Rating must be between 0 and 5".into());
    }
    if p.decision
        .as_ref()
        .is_some_and(|v| !["undecided", "favorite", "pass"].contains(&v.as_str()))
    {
        return Err("Invalid review decision".into());
    }
    if p.tags
        .as_ref()
        .is_some_and(|v| v.len() > 100 || v.iter().any(|s| s.chars().count() > 100))
    {
        return Err("Tags are limited to 100 tags of 100 characters".into());
    }
    if p.tags.as_ref().is_some_and(|tags| {
        tags.iter().any(|tag| {
            tag.chars()
                .any(|ch| ch.is_control() || ",;|<>".contains(ch))
        })
    }) {
        return Err(
            "Tags cannot contain control characters, comma, semicolon, pipe, or angle brackets"
                .into(),
        );
    }
    Ok(())
}
fn read_meta(db: &Connection) -> Result<Project> {
    let data: String = db
        .query_row("SELECT data FROM meta WHERE singleton=1", [], |r| r.get(0))
        .map_err(error)?;
    let project: Project = serde_json::from_str(&data).map_err(error)?;
    valid_project_id(&project.id)?;
    valid_name(&project.name)?;
    chrono::DateTime::parse_from_rfc3339(&project.created_at)
        .map_err(|_| "Invalid saved project creation time".to_string())?;
    chrono::DateTime::parse_from_rfc3339(&project.updated_at)
        .map_err(|_| "Invalid saved project update time".to_string())?;
    valid_absolute_path(Path::new(&project.source_dir))?;
    if !["idle", "running", "completed", "cancelled", "failed"]
        .contains(&project.import_status.as_str())
    {
        return Err("Invalid saved import status".into());
    }
    Ok(project)
}
fn write_meta(db: &Connection, p: &Project) -> Result<()> {
    db.execute("INSERT INTO meta(singleton,data) VALUES(1,?1) ON CONFLICT(singleton) DO UPDATE SET data=excluded.data",[serde_json::to_string(p).map_err(error)?]).map_err(error)?;
    Ok(())
}
fn touch(db: &Connection) -> Result<()> {
    let mut p = read_meta(db)?;
    p.updated_at = now();
    write_meta(db, &p)
}
fn read_items<T: serde::de::DeserializeOwned>(db: &Connection, table: &str) -> Result<Vec<T>> {
    let mut statement = db
        .prepare(&format!("SELECT id,data FROM {table} ORDER BY id"))
        .map_err(error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(error)?;
    rows.map(|row| {
        let (id, data) = row.map_err(error)?;
        decode_item(&id, &data)
    })
    .collect()
}
fn read_item<T: serde::de::DeserializeOwned>(
    db: &Connection,
    table: &str,
    id: &str,
) -> Result<Option<T>> {
    let data: Option<String> = db
        .query_row(
            &format!("SELECT data FROM {table} WHERE id=?1"),
            [id],
            |row| row.get(0),
        )
        .optional()
        .map_err(error)?;
    data.map(|data| decode_item(id, &data)).transpose()
}
fn decode_item<T: serde::de::DeserializeOwned>(id: &str, data: &str) -> Result<T> {
    let value: serde_json::Value = serde_json::from_str(data).map_err(error)?;
    if value.get("id").and_then(serde_json::Value::as_str) != Some(id) {
        return Err("Saved row identity does not match its database key".into());
    }
    serde_json::from_value(value).map_err(error)
}
fn validate_rows(db: &Connection, meta: &Project) -> Result<()> {
    let mut project = meta.clone();
    project.photos = read_items(db, "photos")?;
    project.groups = read_items(db, "groups_data")?;
    project.collections = read_items(db, "collections")?;
    validate_project_rows(&project)
}
fn validate_project_rows(project: &Project) -> Result<()> {
    let mut photos = HashSet::new();
    let mut original_paths = HashSet::new();
    for photo in &project.photos {
        if photo.id.len() != 24
            || !photo.id.bytes().all(|ch| ch.is_ascii_hexdigit())
            || !photos.insert(&photo.id)
        {
            return Err("Invalid saved photo identity".into());
        }
        valid_absolute_path(Path::new(&photo.path))?;
        let original = if cfg!(windows) {
            path_string(Path::new(&photo.path)).to_lowercase()
        } else {
            photo.path.clone()
        };
        if !original_paths.insert(original) {
            return Err("Saved project contains duplicate original paths".into());
        }
        if Path::new(&photo.path)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(photo.filename.as_str())
        {
            return Err("Saved photo filename does not match its original path".into());
        }
        if !path_is_within(Path::new(&photo.path), Path::new(&project.source_dir)) {
            return Err("Saved photo lies outside its source folder".into());
        }
        if !photo.quality_score.is_finite() || !(0.0..=100.0).contains(&photo.quality_score) {
            return Err("Invalid saved technical score".into());
        }
        validate_patch(&PhotoPatch {
            rating: Some(photo.rating),
            decision: Some(photo.decision.clone()),
            tags: Some(photo.tags.clone()),
            ..Default::default()
        })?;
        for cached in [&photo.preview_path, &photo.thumbnail_path] {
            if !cached.is_empty() {
                valid_absolute_path(Path::new(cached))?;
            }
        }
        if let Some(detail) = &photo.detail_path {
            valid_absolute_path(Path::new(detail))?;
        }
    }
    let mut grouped = HashSet::new();
    let group_ids: HashSet<_> = project
        .groups
        .iter()
        .map(|group| group.id.as_str())
        .collect();
    for group in &project.groups {
        if group.id.is_empty()
            || group.photo_ids.is_empty()
            || group
                .photo_ids
                .iter()
                .any(|id| !photos.contains(id) || !grouped.insert(id))
            || group
                .recommended_photo_ids
                .iter()
                .any(|id| !group.photo_ids.contains(id))
        {
            return Err("Invalid saved group membership".into());
        }
    }
    for photo in &project.photos {
        if let Some(group_id) = &photo.group_id {
            if !group_ids.contains(group_id.as_str())
                || !project
                    .groups
                    .iter()
                    .any(|group| group.id == *group_id && group.photo_ids.contains(&photo.id))
            {
                return Err("Saved photo references an invalid group".into());
            }
        }
    }
    for collection in &project.collections {
        valid_name(&collection.name)?;
        let mut members = HashSet::new();
        if collection.id.is_empty()
            || collection
                .photo_ids
                .iter()
                .any(|id| !photos.contains(id) || !members.insert(id))
        {
            return Err("Invalid saved collection membership".into());
        }
    }
    Ok(())
}
fn valid_project_id(id: &str) -> Result<()> {
    let parsed = Uuid::parse_str(id).map_err(|_| "Invalid project identity".to_string())?;
    if parsed.to_string() != id {
        return Err("Project identity must use its canonical UUID form".into());
    }
    Ok(())
}
fn valid_absolute_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("Saved paths must be absolute and cannot contain parent traversal".into());
    }
    Ok(())
}
fn path_is_within(path: &Path, root: &Path) -> bool {
    let path = path_string(path);
    let root = path_string(root);
    if cfg!(windows) {
        Path::new(&path.to_lowercase()).starts_with(Path::new(&root.to_lowercase()))
    } else {
        Path::new(&path).starts_with(Path::new(&root))
    }
}
fn canonical_or_lexical(path: &Path) -> Result<PathBuf> {
    valid_absolute_path(path)?;
    let mut ancestor = path.to_path_buf();
    let mut missing = Vec::new();
    while !ancestor.exists() {
        missing.push(
            ancestor
                .file_name()
                .ok_or("Invalid source folder path")?
                .to_owned(),
        );
        ancestor = ancestor
            .parent()
            .ok_or("Invalid source folder path")?
            .to_path_buf();
    }
    let mut resolved = fs::canonicalize(ancestor).map_err(error)?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}
fn write_item<T: Serialize>(db: &Connection, table: &str, id: &str, item: &T) -> Result<()> {
    db.execute(&format!("INSERT INTO {table}(id,data) VALUES(?1,?2) ON CONFLICT(id) DO UPDATE SET data=excluded.data"),params![id,serde_json::to_string(item).map_err(error)?]).map_err(error)?;
    Ok(())
}
pub fn path_string(path: &Path) -> String {
    let value = path.to_string_lossy().to_string();
    if cfg!(windows) {
        if let Some(unc) = value.strip_prefix(r"\\?\UNC\") {
            return format!(r"\\{unc}");
        }
        if let Some(normal) = value.strip_prefix(r"\\?\") {
            return normal.to_string();
        }
    }
    value
}
pub fn ensure_within(path: &Path, root: &Path) -> Result<()> {
    if !path.is_absolute() || path.components().any(|v| matches!(v, Component::ParentDir)) {
        return Err("Unsafe worker path".into());
    }
    let root = fs::canonicalize(root).map_err(error)?;
    let resolved = if path.exists() {
        fs::canonicalize(path).map_err(error)?
    } else {
        fs::canonicalize(path.parent().ok_or("Invalid cache path")?)
            .map_err(error)?
            .join(path.file_name().ok_or("Invalid cache path")?)
    };
    if !resolved.starts_with(root) {
        return Err("Worker path lies outside its allowed folder".into());
    }
    Ok(())
}
pub fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    let temporary = path.with_extension(format!("{}.tmp", Uuid::new_v4()));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(error)?;
        file.write_all(data).map_err(error)?;
        file.sync_all().map_err(error)?;
        drop(file);
        // Windows rename cannot replace an existing destination. Use the native replace operation there.
        replace_file(&temporary, path)?;
        #[cfg(unix)]
        if let Some(parent) = path.parent() {
            fs::File::open(parent)
                .map_err(error)?
                .sync_all()
                .map_err(error)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
#[cfg(not(windows))]
fn replace_file(from: &Path, to: &Path) -> Result<()> {
    fs::rename(from, to).map_err(error)
}
#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }
    let from: Vec<u16> = from.as_os_str().encode_wide().chain(Some(0)).collect();
    let to: Vec<u16> = to.as_os_str().encode_wide().chain(Some(0)).collect();
    if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 0x1 | 0x8) } == 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(())
}
