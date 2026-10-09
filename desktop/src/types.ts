export type Decision = "undecided" | "favorite" | "pass";
export type SelectionMode = "cautious" | "stronger";
export type DecisionSource = "manual" | "automatic";
export interface Photo {
  id: string;
  path: string;
  filename: string;
  previewPath: string;
  thumbnailPath: string;
  detailPath: string | null;
  captureTime: string;
  width: number;
  height: number;
  camera: string | null;
  groupId: string | null;
  qualityScore: number;
  hints: string[];
  rating: number;
  ratingTouched: boolean;
  decision: Decision;
  decisionSource: DecisionSource;
  decisionTouched?: boolean;
  suggestedDecision: Decision | null;
  suggestionReason: string | null;
  suggestionConfidence: number | null;
  reviewed: boolean;
  tags: string[];
  analysisError: string | null;
}
export interface Group {
  id: string;
  label: string;
  photoIds: string[];
  recommendedPhotoIds: string[];
}
export interface Collection {
  id: string;
  name: string;
  photoIds: string[];
}
export interface Project {
  id: string;
  name: string;
  sourceDir: string;
  includeSubfolders: boolean;
  automaticSelectionEnabled: boolean;
  selectionMode: SelectionMode;
  firstPassReady: boolean;
  projectPath: string;
  createdAt: string;
  updatedAt: string;
  photos: Photo[];
  groups: Group[];
  collections: Collection[];
  importStatus: "idle" | "running" | "completed" | "cancelled" | "failed";
  importError: string | null;
}
export interface ProjectSummary {
  id: string;
  name: string;
  sourceDir: string;
  includeSubfolders: boolean;
  projectPath: string;
  photoCount: number;
  favoriteCount: number;
  updatedAt: string;
}
export interface AppState {
  projects: ProjectSummary[];
  version: string;
  engineAvailable: boolean;
}
export interface ImportProgress {
  projectId: string;
  jobId: string;
  phase:
    | "scan"
    | "analysis"
    | "grouping"
    | "selection"
    | "complete"
    | "cancelled"
    | "error";
  processed: number;
  total: number;
  currentFile: string | null;
  failed: number;
  message: string | null;
}
export interface PhotoPatch {
  ratingTouched?: boolean;
  rating?: number;
  decision?: Decision;
  decisionSource?: DecisionSource;
  decisionTouched?: boolean;
  reviewed?: boolean;
  tags?: string[];
}
export type Filter = "all" | "favorite" | "unreviewed" | "pass";
export type Sort = "time" | "quality" | "rating";
export type Scope =
  | { type: "all" }
  | { type: "group" | "collection"; id: string };
export interface BrowseOptions {
  filter: Filter;
  sort: Sort;
  query: string;
  scope: Scope;
}
