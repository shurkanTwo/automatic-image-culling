import type { Photo, Project } from "../types";
export function testPhoto(index: number): Photo {
  return {
    id: `photo-${index}`,
    path: `/photos/DSC_${index}.jpg`,
    filename: `DSC_${index}.jpg`,
    thumbnailPath: `/thumbnail-${index}.jpg`,
    previewPath: `/preview-${index}.jpg`,
    detailPath: null,
    captureTime: new Date(Date.UTC(2026, 8, 1, 8, index)).toISOString(),
    width: 6000,
    height: 4000,
    camera: "Nikon Z6",
    groupId: null,
    qualityScore: 70,
    hints: ["Good exposure"],
    rating: 0,
    ratingTouched: false,
    decision: "undecided",
    reviewed: false,
    tags: index === 0 ? ["family"] : ["travel"],
    analysisError: null,
  };
}
export function testProject(count = 4): Project {
  const photos = Array.from({ length: count }, (_, index) => testPhoto(index));
  return {
    id: "test-project",
    name: "Review trip",
    sourceDir: "/photos",
    projectPath: "/project.cullproj",
    photos,
    groups: [],
    collections: [{ id: "book", name: "Photo book", photoIds: [] }],
    createdAt: "",
    updatedAt: "",
    importStatus: "completed",
    importError: null,
  };
}
