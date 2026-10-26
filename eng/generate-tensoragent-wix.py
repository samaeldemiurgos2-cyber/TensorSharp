#!/usr/bin/env python3
"""Validate a self-contained TensorAgent payload and emit WiX 6 MSI authoring.

This helper uses only the Python standard library so its payload and upgrade
contract can also be checked on non-Windows machines. Building or installing the
MSI still requires WiX and Windows; XML validation is not an installation test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import uuid
import xml.etree.ElementTree as ET


WIX_NAMESPACE = "http://wixtoolset.org/schemas/v4/wxs"
# Keep this stable across releases and CPU/CUDA variants. Both install to the
# same directory and must replace one another, including at the same version.
UPGRADE_CODE = "7C7A995D-248E-4B72-9B7A-D736C82FA4B4"
COMPONENT_NAMESPACE = uuid.UUID("c73c2cde-f5f9-4e10-bf9b-941c8816f7d5")
GUIDE_URL = "https://tensorsharp.ai/tensoragent.html"
VARIANTS = ("win-x64-cpu", "win-x64-cuda")
REQUIRED_FILES = (
    "TensorAgent.Maui.exe",
    "TensorAgent.Maui.dll",
    "TensorAgent.Maui.deps.json",
    "TensorAgent.Maui.runtimeconfig.json",
    "GgmlOps.dll",
    "coreclr.dll",
    "hostfxr.dll",
    "hostpolicy.dll",
    "System.Private.CoreLib.dll",
    "Microsoft.UI.Xaml.dll",
    "Microsoft.WindowsAppRuntime.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "webui/index.html",
)
CUDA_FILES = (
    "cudart64_13.dll",
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "cuda_kernels/tensorsharp_kernels.ptx",
    # SASS for the release GPUs: a CUDA 13 driver older than the 13.4 toolkit
    # cannot JIT the PTX alone. The sidecar pairs the fatbin with that PTX.
    "cuda_kernels/tensorsharp_kernels.fatbin",
    "cuda_kernels/tensorsharp_kernels.fatbin.sha256",
)
VERSION_PATTERN = re.compile(
    r"([0-9]+)\.([0-9]+)\.([0-9]+)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def msi_version(version: str) -> str:
    """Map a release triplet to Windows Installer's numeric version limits.

    Calendar releases 2000..2255 use year minus 2000; ordinary 0..255 major
    versions retain their triplet. Prerelease/build labels remain in the asset
    and display names but MSI compares only the numeric triplet.
    """
    match = VERSION_PATTERN.fullmatch(version)
    if not match:
        raise ValueError("Version must be a numeric major.minor.patch with optional prerelease/build labels")
    major, minor, patch = (int(value) for value in match.groups())
    if 2000 <= major <= 2255:
        major -= 2000
    if major > 255 or minor > 255 or patch > 65535:
        raise ValueError("MSI version requires major/minor <= 255 and patch <= 65535 (calendar years 2000..2255 are mapped)")
    return f"{major}.{minor}.{patch}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_windows_relative_path(relative: str) -> None:
    """Validate an unnormalized, slash-separated path inside the payload.

    Work on the raw string rather than Path/filesystem operations: Windows can
    turn colons into alternate data streams and remove trailing dots/spaces
    before a directory scan ever sees the requested filename. The same lexical
    policy must apply to payloads and contract tests on every build platform.
    """
    for part in relative.split("/"):
        if (not part or part in (".", "..")
                or re.search(r'[<>:"\\|?*\x00-\x1f]', part)
                or part.endswith((".", " "))
                # Win32 also reserves the ISO-8859-1 superscript digits in
                # COM/LPT device names, including names with extensions.
                or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]",
                                part.split(".")[0].rstrip(" "), re.IGNORECASE)):
            raise ValueError(f"Invalid Windows payload filename: {relative}")


def payload_manifest(publish_directory: Path, variant: str) -> list[dict[str, object]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported Windows variant: {variant}")
    root = publish_directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Publish directory is not a directory: {root}")
    required = REQUIRED_FILES + (CUDA_FILES if variant.endswith("-cuda") else ())
    missing = [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        raise ValueError("Incomplete TensorAgent publish payload; missing/empty: " + ", ".join(missing))
    if not any(path.is_file() and path.stat().st_size for path in (root / "skills").glob("*/SKILL.md")):
        raise ValueError("Incomplete TensorAgent publish payload; no bundled skills/*/SKILL.md")
    runtime_config = json.loads((root / "TensorAgent.Maui.runtimeconfig.json").read_text(encoding="utf-8-sig"))
    if not isinstance(runtime_config, dict):
        raise ValueError("TensorAgent runtimeconfig must contain a JSON object")
    runtime_options = runtime_config.get("runtimeOptions", {})
    if not isinstance(runtime_options, dict):
        raise ValueError("TensorAgent runtimeconfig must contain a runtimeOptions object")
    if "framework" in runtime_options or "frameworks" in runtime_options:
        raise ValueError("TensorAgent runtimeconfig is framework-dependent; publish with --self-contained true")
    if not runtime_options.get("includedFrameworks"):
        raise ValueError("TensorAgent runtimeconfig does not declare includedFrameworks for its self-contained runtime")

    files: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix().casefold()):
        relative = path.relative_to(root).as_posix()
        validate_windows_relative_path(relative)
        if path.is_symlink():
            raise ValueError(f"Windows payload cannot contain symbolic links: {path.relative_to(root)}")
        # Windows paths are case-insensitive; duplicate names can be created on
        # another OS but cannot be represented safely in a Windows installer.
        if relative.casefold() in seen:
            raise ValueError(f"Duplicate case-insensitive Windows payload path: {relative}")
        seen.add(relative.casefold())
        if not path.is_file():
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    return files


def identifier(prefix: str, path: str) -> str:
    return prefix + hashlib.sha256(path.casefold().encode("utf-8")).hexdigest()[:32]


def component_guid(path: str) -> str:
    return str(uuid.uuid5(COMPONENT_NAMESPACE, path.casefold())).upper()


def generate_wix(publish_directory: Path, version: str, variant: str, files: list[dict[str, object]]) -> ET.ElementTree:
    numeric_version = msi_version(version)
    ET.register_namespace("", WIX_NAMESPACE)

    def add(parent: ET.Element, tag: str, **attributes: str) -> ET.Element:
        return ET.SubElement(parent, f"{{{WIX_NAMESPACE}}}{tag}", attributes)

    root = ET.Element(f"{{{WIX_NAMESPACE}}}Wix")
    package = add(root, "Package", Name=f"TensorAgent Desktop {version} ({'CUDA' if variant.endswith('-cuda') else 'CPU'})",
                  Manufacturer="TensorSharp", Version=numeric_version, Language="1033",
                  UpgradeCode=UPGRADE_CODE, Scope="perUser", InstallerVersion="500", Compressed="yes")
    add(package, "MajorUpgrade", AllowSameVersionUpgrades="yes", Schedule="afterInstallInitialize",
        DowngradeErrorMessage="A newer TensorAgent Desktop version is already installed. Uninstall it before installing an older release.")
    add(package, "MediaTemplate", EmbedCab="yes", CompressionLevel="high")
    add(package, "Property", Id="ARPNOMODIFY", Value="1")
    add(package, "Property", Id="ARPHELPLINK", Value=GUIDE_URL)
    add(package, "Property", Id="ARPURLINFOABOUT", Value="https://github.com/zhongkaifu/TensorSharp")
    add(package, "SetProperty", Id="ARPINSTALLLOCATION", Value="[INSTALLFOLDER]", After="CostFinalize")
    add(package, "Icon", Id="TensorAgent.exe", SourceFile=str(publish_directory.resolve() / "TensorAgent.Maui.exe"))
    add(package, "Property", Id="ARPPRODUCTICON", Value="TensorAgent.exe")

    app_data = add(package, "StandardDirectory", Id="LocalAppDataFolder")
    programs = add(app_data, "Directory", Id="UserProgramsFolder", Name="Programs")
    install = add(programs, "Directory", Id="INSTALLFOLDER", Name="TensorAgent")
    directories: dict[str, ET.Element] = {"": install}
    directory_ids: dict[str, str] = {"": "INSTALLFOLDER"}
    paths = sorted({parent.as_posix() for item in files for parent in Path(str(item["path"])).parents if parent.as_posix() != "."}, key=lambda path: (path.count("/"), path.casefold()))
    for path in paths:
        parent = Path(path).parent.as_posix()
        parent = "" if parent == "." else parent
        directory_id = identifier("Dir_", path)
        directories[path] = add(directories[parent], "Directory", Id=directory_id, Name=Path(path).name)
        directory_ids[path] = directory_id

    feature = add(package, "Feature", Id="Desktop", Title="TensorAgent Desktop", Level="1")

    def component(directory: ET.Element, name: str) -> ET.Element:
        component_id = identifier("Cmp_", name)
        item = add(directory, "Component", Id=component_id, Guid=component_guid(name), Bitness="always64")
        # All components installed in a user profile need HKCU registry keypaths
        # for MSI repair/validation. Explicit stable GUIDs support file components
        # with registry keypaths, for which WiX cannot auto-generate a GUID.
        add(item, "RegistryValue", Root="HKCU", Key=r"Software\TensorSharp\TensorAgent\Installer\Components",
            Name=component_id, Type="integer", Value="1", KeyPath="yes")
        add(feature, "ComponentRef", Id=component_id)
        return item

    for item in files:
        relative = str(item["path"])
        parent = Path(relative).parent.as_posix()
        parent = "" if parent == "." else parent
        item_component = component(directories[parent], "file:" + relative)
        add(item_component, "File", Id=identifier("File_", relative), Source=str(publish_directory.resolve() / relative),
            Name=Path(relative).name, KeyPath="no")

    # Remove only empty application-owned directories. Models, configuration,
    # conversations, and files outside the payload are never MSI resources.
    for path, directory in directories.items():
        cleanup = component(directory, "directory:" + path)
        add(cleanup, "RemoveFolder", Id=identifier("Remove_", path), Directory=directory_ids[path], On="uninstall")

    menu = add(package, "StandardDirectory", Id="ProgramMenuFolder")
    app_menu = add(menu, "Directory", Id="TensorAgentMenuFolder", Name="TensorAgent")
    shortcut = component(app_menu, "shortcut:start-menu")
    add(shortcut, "Shortcut", Id="TensorAgentStartMenuShortcut", Name="TensorAgent", Description="TensorAgent Desktop",
        Target="[INSTALLFOLDER]TensorAgent.Maui.exe", WorkingDirectory="INSTALLFOLDER", Icon="TensorAgent.exe")
    add(shortcut, "RemoveFolder", Id="RemoveTensorAgentMenu", On="uninstall")
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-directory", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        numeric_version = msi_version(args.version)
        payload_root = args.publish_directory.resolve(strict=True)
        if any(path.resolve().is_relative_to(payload_root) for path in (args.output, args.manifest)):
            raise ValueError("Generated installer files must be outside the publish directory")
        files = payload_manifest(payload_root, args.variant)
        tree = generate_wix(payload_root, args.version, args.variant, files)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        tree.write(args.output, encoding="utf-8", xml_declaration=True)
        args.manifest.write_text(json.dumps({"version": args.version, "msi_version": numeric_version,
                                             "variant": args.variant, "files": files}, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"TensorAgent packaging: {error}\n")
    print(f"Validated {len(files)} payload files; release {args.version}, MSI {numeric_version}, {args.variant}")


if __name__ == "__main__":
    main()
