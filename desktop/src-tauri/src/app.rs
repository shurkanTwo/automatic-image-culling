use crate::{
    core::{
        Collection, ExportResult, Photo, PhotoPatch, PhotoUpdate, Project, ProjectSummary, Result,
        Store,
    },
    worker::{Detail, Engine, Workers},
};
use serde::Serialize;
use std::sync::Arc;
use tauri::{Emitter, Manager, State};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AppState {
    projects: Vec<ProjectSummary>,
    version: String,
    engine_available: bool,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Started {
    job_id: String,
}
#[tauri::command]
async fn get_app_state(state: State<'_, Workers>) -> Result<AppState> {
    let worker = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        Ok(AppState {
            projects: worker.store.summaries()?,
            version: env!("CARGO_PKG_VERSION").into(),
            engine_available: worker.engine.available(),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn create_project(
    state: State<'_, Workers>,
    name: String,
    source_dir: String,
    include_subfolders: Option<bool>,
    automatic_selection_enabled: Option<bool>,
    selection_mode: Option<String>,
    prefer_raw: Option<bool>,
) -> Result<Project> {
    state.store.create_configured(
        &name,
        &source_dir,
        include_subfolders.unwrap_or(false),
        automatic_selection_enabled.unwrap_or(true),
        selection_mode.as_deref().unwrap_or("cautious"),
        prefer_raw.unwrap_or(false),
    )
}
#[tauri::command]
fn open_project(state: State<'_, Workers>, project_path: String) -> Result<Project> {
    state.open_project(&project_path)
}
#[tauri::command]
fn get_project(state: State<'_, Workers>, project_id: String) -> Result<Project> {
    state.store.project(&project_id)
}
#[tauri::command]
fn start_import(state: State<'_, Workers>, project_id: String) -> Result<Started> {
    Ok(Started {
        job_id: state.start(&project_id)?,
    })
}
#[tauri::command]
fn automatic_first_pass(
    state: State<'_, Workers>,
    project_id: String,
    selection_mode: Option<String>,
) -> Result<Project> {
    state.automatic_first_pass(&project_id, selection_mode.as_deref())
}
#[tauri::command]
fn clear_automatic_selection(state: State<'_, Workers>, project_id: String) -> Result<Project> {
    state.clear_automatic_selection(&project_id)
}
#[tauri::command]
fn cancel_import(state: State<'_, Workers>, project_id: String) -> Result<()> {
    state.cancel(&project_id)
}
#[tauri::command]
fn update_photo(
    state: State<'_, Workers>,
    project_id: String,
    photo_id: String,
    patch: PhotoPatch,
) -> Result<Photo> {
    let photo = state
        .store
        .update_photos(&project_id, &[photo_id], &patch)?
        .remove(0);
    state.updated(&project_id);
    Ok(photo)
}
#[tauri::command]
fn update_photos(
    state: State<'_, Workers>,
    project_id: String,
    photo_ids: Vec<String>,
    patch: PhotoPatch,
) -> Result<Vec<Photo>> {
    let photos = state.store.update_photos(&project_id, &photo_ids, &patch)?;
    state.updated(&project_id);
    Ok(photos)
}
#[tauri::command]
fn update_photo_patches(
    state: State<'_, Workers>,
    project_id: String,
    updates: Vec<PhotoUpdate>,
) -> Result<Vec<Photo>> {
    let photos = state.store.update_photo_patches(&project_id, &updates)?;
    state.updated(&project_id);
    Ok(photos)
}
#[tauri::command]
fn create_collection(
    state: State<'_, Workers>,
    project_id: String,
    name: String,
) -> Result<Collection> {
    let collection = state.store.create_collection(&project_id, &name)?;
    state.updated(&project_id);
    Ok(collection)
}
#[tauri::command]
fn update_collection(
    state: State<'_, Workers>,
    project_id: String,
    collection_id: String,
    name: Option<String>,
    photo_ids: Option<Vec<String>>,
) -> Result<Collection> {
    let collection = state
        .store
        .update_collection(&project_id, &collection_id, name, photo_ids)?;
    state.updated(&project_id);
    Ok(collection)
}
#[tauri::command]
fn delete_collection(
    state: State<'_, Workers>,
    project_id: String,
    collection_id: String,
) -> Result<()> {
    state.store.delete_collection(&project_id, &collection_id)?;
    state.updated(&project_id);
    Ok(())
}
#[tauri::command]
async fn generate_detail(
    state: State<'_, Workers>,
    project_id: String,
    photo_id: String,
) -> Result<Detail> {
    let worker = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || worker.detail(&project_id, &photo_id))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
fn export_selection(
    state: State<'_, Workers>,
    project_id: String,
    destination: String,
    collection_id: Option<String>,
    only_favorites: bool,
    include_discards: Option<bool>,
) -> Result<ExportResult> {
    state.store.export_with_discards(
        &project_id,
        &destination,
        collection_id.as_deref(),
        only_favorites,
        include_discards.unwrap_or(false),
    )
}
#[tauri::command]
fn get_lightroom_plugin_path(app: tauri::AppHandle) -> Result<String> {
    let path = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("resources/lightroom/PhotoSelect.lrplugin");
    if !path.is_dir() {
        return Err("Bundled Lightroom plugin is missing".into());
    }
    Ok(crate::core::path_string(&path))
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let store =
                Store::new(app.path().app_local_data_dir()?).map_err(std::io::Error::other)?;
            let resources = app.path().resource_dir()?;
            let engine = Engine::discover(resources.clone());
            let handle = app.handle().clone();
            let emit = Arc::new(move |name: &str, payload: serde_json::Value| {
                let _ = handle.emit(name, payload);
            });
            app.manage(Workers::new(store, engine, emit));
            if let Some(report_path) = std::env::var_os("PHOTO_SELECT_SMOKE_TEST_OUTPUT") {
                let plugin = resources.join("resources/lightroom/PhotoSelect.lrplugin");
                let packaged_engine = resources.join("resources/engine").join(if cfg!(windows) {
                    "photo-select-engine.exe"
                } else {
                    "photo-select-engine"
                });
                let available =
                    packaged_engine.is_file() && Engine::executable(packaged_engine).available();
                let projects = app
                    .state::<Workers>()
                    .store
                    .summaries()
                    .map_err(std::io::Error::other)?;
                let report = serde_json::json!({
                    "version":env!("CARGO_PKG_VERSION"),
                    "engineAvailable":available,
                    "lightroomPluginPath":crate::core::path_string(&plugin),
                    "resourceDir":crate::core::path_string(&resources),
                    "projects":projects
                });
                // Persist and flush the report before requesting normal event-loop shutdown.
                let bytes = serde_json::to_vec_pretty(&report)?;
                crate::core::atomic_write(&std::path::PathBuf::from(report_path), &bytes)
                    .map_err(std::io::Error::other)?;
                app.handle()
                    .exit(if available && plugin.join("Info.lua").is_file() {
                        0
                    } else {
                        1
                    });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_app_state,
            create_project,
            open_project,
            get_project,
            start_import,
            cancel_import,
            automatic_first_pass,
            clear_automatic_selection,
            update_photo,
            update_photos,
            update_photo_patches,
            create_collection,
            update_collection,
            delete_collection,
            generate_detail,
            export_selection,
            get_lightroom_plugin_path
        ])
        .build(tauri::generate_context!())
        .expect("Unable to start Photo Select");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            app.state::<Workers>().cancel_all();
        }
    });
}
