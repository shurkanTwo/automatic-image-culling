import type { AppState, Photo, PhotoPatch, Project } from "./types";
const sampleImages = [
  "photo-1500530855697-b586d89ba3ee",
  "photo-1476514525535-07fb3b4ae5f1",
  "photo-1464822759023-fed622ff2c3b",
  "photo-1501785888041-af3ef285b470",
  "photo-1441974231531-c6227db76b6e",
  "photo-1519681393784-d120267933ba",
  "photo-1469474968028-56623f02e42e",
  "photo-1433086966358-54859d0ed716",
];
const key = "photo-select-explicit-demo-v1";
function initialProject(): Project {
  const photos: Photo[] = Array.from({ length: 32 }, (_, index) => ({
    id: `demo-${index}`,
    path: `/Demo/Alpine weekend/DSC_${String(2401 + index)}.NEF`,
    filename: `DSC_${2401 + index}.NEF`,
    thumbnailPath: `https://images.unsplash.com/${sampleImages[index % 8]}?auto=format&fit=crop&w=600&q=80`,
    previewPath: `https://images.unsplash.com/${sampleImages[index % 8]}?auto=format&fit=max&w=1800&q=85`,
    detailPath: null,
    captureTime: new Date(
      Date.UTC(2026, 8, 19, 8 + Math.floor(index / 8), (index % 8) * 4),
    ).toISOString(),
    width: 6000,
    height: 4000,
    camera: "Nikon Z 6II",
    groupId: `moment-${Math.floor(index / 8)}`,
    qualityScore: 65 + ((index * 7) % 35),
    hints: index % 5 === 0 ? ["Check fine detail"] : ["Good exposure"],
    rating: index < 3 ? 4 : 0,
    ratingTouched: index < 3,
    decision: index < 3 ? "favorite" : index === 5 ? "pass" : "undecided",
    reviewed: index < 6,
    tags: index < 3 ? ["landscape"] : [],
    analysisError: null,
  }));
  return {
    id: "demo-alpine",
    name: "Alpine weekend",
    sourceDir: "/Demo/Alpine weekend",
    projectPath: "/Demo/alpine.photoselect",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    photos,
    groups: [
      "Morning by the lake",
      "Into the mountains",
      "Forest walk",
      "Last light",
    ].map((label, index) => ({
      id: `moment-${index}`,
      label,
      photoIds: photos.slice(index * 8, index * 8 + 8).map((photo) => photo.id),
      recommendedPhotoIds: [`demo-${index * 8 + 2}`],
    })),
    collections: [
      {
        id: "collection-print",
        name: "For the wall",
        photoIds: ["demo-0", "demo-2"],
      },
    ],
    importStatus: "completed",
    importError: null,
  };
}
function readProject(): Project {
  try {
    const stored = localStorage.getItem(key);
    if (stored) return JSON.parse(stored);
  } catch {
    /* Start a clean explicit demo if its stored data is invalid. */
  }
  return initialProject();
}
let project = readProject();
export async function demoInvoke<T>(
  command: string,
  args: Record<string, unknown> = {},
): Promise<T> {
  let result: unknown;
  const save = () => {
    project.updatedAt = new Date().toISOString();
    localStorage.setItem(key, JSON.stringify(project));
  };
  switch (command) {
    case "get_app_state":
      result = {
        version: "0.2.0",
        engineAvailable: true,
        projects: [
          {
            id: project.id,
            name: project.name,
            sourceDir: project.sourceDir,
            projectPath: project.projectPath,
            photoCount: project.photos.length,
            favoriteCount: project.photos.filter(
              (photo) => photo.decision === "favorite",
            ).length,
            updatedAt: project.updatedAt,
          },
        ],
      } satisfies AppState;
      break;
    case "open_project":
    case "get_project":
      result = project;
      break;
    case "create_project":
      project = { ...initialProject(), name: String(args.name) };
      save();
      result = project;
      break;
    case "start_import":
      result = { jobId: "demo-preview" };
      break;
    case "cancel_import":
      result = undefined;
      break;
    case "update_photo":
    case "update_photos": {
      const ids =
        command === "update_photo"
          ? [String(args.photoId)]
          : (args.photoIds as string[]);
      const patch = args.patch as PhotoPatch;
      project = {
        ...project,
        photos: project.photos.map((photo) =>
          ids.includes(photo.id)
            ? {
                ...photo,
                ...patch,
                ratingTouched:
                  patch.ratingTouched ??
                  (patch.rating !== undefined ? true : photo.ratingTouched),
              }
            : photo,
        ),
      };
      save();
      result =
        command === "update_photo"
          ? project.photos.find((photo) => photo.id === args.photoId)
          : project.photos.filter((photo) => ids.includes(photo.id));
      break;
    }
    case "create_collection": {
      const collection = {
        id: crypto.randomUUID(),
        name: String(args.name),
        photoIds: [],
      };
      project = {
        ...project,
        collections: [...project.collections, collection],
      };
      save();
      result = collection;
      break;
    }
    case "update_collection": {
      project = {
        ...project,
        collections: project.collections.map((collection) =>
          collection.id === args.collectionId
            ? {
                ...collection,
                name: args.name == null ? collection.name : String(args.name),
                photoIds:
                  args.photoIds == null
                    ? collection.photoIds
                    : (args.photoIds as string[]),
              }
            : collection,
        ),
      };
      save();
      result = project.collections.find(
        (collection) => collection.id === args.collectionId,
      );
      break;
    }
    case "delete_collection":
      project = {
        ...project,
        collections: project.collections.filter(
          (collection) => collection.id !== args.collectionId,
        ),
      };
      save();
      break;
    case "generate_detail":
      result = {
        detailPath: project.photos.find((photo) => photo.id === args.photoId)!
          .previewPath,
        width: 6000,
        height: 4000,
      };
      break;
    case "export_selection": {
      const collection = args.collectionId
        ? project.collections.find((value) => value.id === args.collectionId)
        : null;
      const photos = project.photos.filter(
        (photo) =>
          (!collection || collection.photoIds.includes(photo.id)) &&
          (!args.onlyFavorites || photo.decision === "favorite"),
      );
      const manifest = {
        schemaVersion: 1,
        application: "Photo Select",
        projectName: project.name,
        collectionName: collection?.name ?? "Favorites",
        exportedAt: new Date().toISOString(),
        photos: photos.map((photo) => ({
          path: photo.path,
          rating: photo.ratingTouched ? photo.rating : null,
          decision: photo.decision,
          tags: photo.tags,
        })),
      };
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(manifest, null, 2)], {
          type: "application/json",
        }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = "photo-select-demo.json";
      link.click();
      URL.revokeObjectURL(url);
      result = { path: "photo-select-demo.json", count: photos.length };
      break;
    }
    case "get_lightroom_plugin_path":
      throw new Error(
        "The Lightroom plugin is included in the desktop installation.",
      );
    default:
      throw new Error(`Unsupported demo command: ${command}`);
  }
  return structuredClone(result) as T;
}
