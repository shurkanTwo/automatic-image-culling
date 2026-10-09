"""Fetch the latest available older package from successful same-branch CI runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_PATTERN = re.compile(r"Photo-Select-(\d+\.\d+\.\d+)-Windows-x64$")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def version_tuple(version: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"Unsupported package version: {version}")
    return tuple(int(part) for part in version.split("."))


def api_request(endpoint: str) -> urllib.request.Request:
    return urllib.request.Request(
        f"https://api.github.com/{endpoint}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Photo-Select-upgrade-verification",
        },
    )


def api_json(endpoint: str) -> dict:
    with urllib.request.urlopen(api_request(endpoint), timeout=60) as response:
        return json.load(response)


def download_artifact(endpoint: str, destination: Path) -> None:
    # Do not forward the repository token to the signed blob-storage redirect.
    opener = urllib.request.build_opener(NoRedirect)
    try:
        response = opener.open(api_request(endpoint), timeout=60)
    except urllib.error.HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise
        location = error.headers["Location"]
        if urllib.parse.urlparse(location).scheme != "https":
            raise RuntimeError("Artifact download redirected to a non-HTTPS URL")
        response = urllib.request.urlopen(location, timeout=120)
    with response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def extract_and_verify(archive: Path, destination: Path, version: str) -> Path:
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            path = Path(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise RuntimeError(f"Unsafe artifact member: {entry.filename}")
        if destination.exists():
            shutil.rmtree(destination)
        package.extractall(destination)
    manifests = list(destination.rglob("manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("Older artifact must contain exactly one integrity manifest")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["version"] != version or manifest["platform"] != "windows-x64":
        raise RuntimeError(
            "Older artifact manifest does not match its advertised version"
        )
    installers = []
    for name, metadata in manifest["files"].items():
        if Path(name).name != name or "\\" in name:
            raise RuntimeError("Unsafe package filename in integrity manifest")
        path = manifest_path.parent / name
        content = path.read_bytes()
        if (
            len(content) != metadata["bytes"]
            or hashlib.sha256(content).hexdigest() != metadata["sha256"]
        ):
            raise RuntimeError(f"Older package integrity check failed: {name}")
        if name.endswith("-Setup.exe"):
            installers.append(path)
    if len(installers) != 1:
        raise RuntimeError("Older artifact must contain exactly one verified installer")
    return installers[0]


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ["GITHUB_REF_NAME"]
    current_run = int(os.environ["GITHUB_RUN_ID"])
    version = json.loads((ROOT / "desktop/package.json").read_text())["version"]
    current_version = version_tuple(version)
    output = Path(os.environ["RUNNER_TEMP"]) / "photo-select-upgrade-baseline"
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "skipped",
        "reason": "No unexpired older package from a successful same-branch run is available",
        "branch": branch,
        "newVersion": version,
    }
    # The run API is ordered newest first. Paginate until an older package is found.
    for page in range(1, 11):
        query = urllib.parse.urlencode(
            {"branch": branch, "status": "success", "per_page": 100, "page": page}
        )
        runs = api_json(
            f"repos/{repository}/actions/workflows/build-desktop.yml/runs?{query}"
        )["workflow_runs"]
        for run in runs:
            if run["id"] == current_run:
                continue
            artifacts = api_json(
                f"repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100"
            )["artifacts"]
            for artifact in artifacts:
                match = VERSION_PATTERN.fullmatch(artifact["name"])
                if (
                    artifact["expired"]
                    or not match
                    or version_tuple(match[1]) >= current_version
                ):
                    continue
                archive = output / f"{artifact['id']}.zip"
                try:
                    download_artifact(
                        f"repos/{repository}/actions/artifacts/{artifact['id']}/zip",
                        archive,
                    )
                except urllib.error.HTTPError as error:
                    if error.code == 410:
                        continue  # It expired between listing and downloading.
                    raise
                installer = extract_and_verify(archive, output / "package", match[1])
                record = {
                    "status": "available",
                    "version": match[1],
                    "newVersion": version,
                    "branch": branch,
                    "runId": run["id"],
                    "runUrl": run["html_url"],
                    "artifactId": artifact["id"],
                    "installer": str(installer.resolve()),
                    "manifestVerified": True,
                }
                break
            if record["status"] == "available":
                break
        if record["status"] == "available" or len(runs) < 100:
            break
    (output / "baseline.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
