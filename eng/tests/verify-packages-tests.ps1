# Pester-free contracts for the release verifier. Loading only the manifest and
# function ASTs keeps these tests independent of dotnet pack and native builds.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$source = Join-Path $PSScriptRoot "../verify-packages.ps1"
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($source, [ref] $tokens, [ref] $parseErrors)
if ($parseErrors.Count -gt 0) {
    throw "Cannot parse package verifier: $($parseErrors -join '; ')"
}

$manifest = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $node.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
        $node.Left.VariablePath.UserPath -eq "PublicPackages"
}, $false))
if ($manifest.Count -ne 1) { throw "Expected exactly one PublicPackages manifest." }
. ([scriptblock]::Create($manifest[0].Extent.Text))

foreach ($name in @("Read-Nupkg", "Assert-SameSet", "Assert-PackageManifest", "Assert-PackageClosure", "Get-NormalizedNuGetVersion")) {
    $definitions = @($ast.FindAll({
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }, $false))
    if ($definitions.Count -ne 1) { throw "Expected exactly one $name function." }
    . ([scriptblock]::Create($definitions[0].Extent.Text))
}

$script:Passed = 0
$script:Failures = @()

function Test-Case {
    param([string] $Name, [scriptblock] $Action)
    try {
        & $Action
        $script:Passed++
        Write-Host "PASS $Name"
    }
    catch {
        $script:Failures += "$Name`: $($_.Exception.Message)"
        Write-Host "FAIL $Name`: $($_.Exception.Message)"
    }
}

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Throws {
    param([scriptblock] $Action)
    $caught = $false
    try { & $Action }
    catch { $caught = $true }
    if (-not $caught) { throw "Expected the verifier to reject this invalid package graph." }
}

function New-FixturePackage {
    param([string] $Id, [string] $Version = "2026.10.23", [object[]] $Dependencies = @())

    $path = Join-Path $FixtureRoot "$Id.$Version.$([guid]::NewGuid().ToString('N')).nupkg"
    $dependencyXml = @($Dependencies | ForEach-Object {
        '<dependency id="{0}" version="{1}" />' -f
            [System.Security.SecurityElement]::Escape($_.Id),
            [System.Security.SecurityElement]::Escape($_.Version)
    }) -join ""
    $xml = @"
<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
  <metadata>
    <id>$Id</id><version>$Version</version><authors>fixture</authors><description>fixture</description>
    <dependencies><group targetFramework="net10.0">$dependencyXml</group></dependencies>
  </metadata>
</package>
"@
    $zip = [System.IO.Compression.ZipFile]::Open($path, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $entry = $zip.CreateEntry("$Id.nuspec")
        $writer = [System.IO.StreamWriter]::new($entry.Open())
        try { $writer.Write($xml) }
        finally { $writer.Dispose() }
        $zip.CreateEntry("lib/net10.0/$Id.dll") | Out-Null
    }
    finally { $zip.Dispose() }

    $package = Read-Nupkg $path
    return [pscustomobject] @{
        Path = $path
        Id = $package.Id
        Version = $package.Version
        Dependencies = $package.Dependencies
    }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$artifactRoot = Join-Path $PSScriptRoot "../../artifacts"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$FixtureRoot = Join-Path $artifactRoot "package-verifier-tests-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $FixtureRoot | Out-Null

try {
    Test-Case "public manifest ships Studio and declares Chat's real dependency" {
        $studio = @($PublicPackages | Where-Object { $_.Id -eq "TensorSharp.Studio" })
        $chat = @($PublicPackages | Where-Object { $_.Id -eq "TensorSharp.Chat" })
        Assert-True ($studio.Count -eq 1) "Studio must be published exactly once."
        Assert-True ($chat.Count -eq 1) "Chat must be published exactly once."
        Assert-True ($chat[0].TensorSharpDependencies -contains "TensorSharp.Studio") "Chat must retain its Studio dependency."
        Assert-PackageManifest -Packages $PublicPackages
    }

    Test-Case "an allowed dependency missing from the release manifest is rejected" {
        Assert-Throws { Assert-PackageManifest -Packages @(
            @{ Id = "TensorSharp.Chat"; TensorSharpDependencies = @("TensorSharp.Studio") }
        ) }
    }

    Test-Case "duplicate package IDs are rejected" {
        Assert-Throws { Assert-PackageManifest -Packages @(
            @{ Id = "TensorSharp.Studio"; TensorSharpDependencies = @() },
            @{ Id = "TensorSharp.Studio"; TensorSharpDependencies = @() }
        ) }
    }

    Test-Case "unexpected actual dependencies still fail the exact-set guard" {
        Assert-Throws {
            Assert-SameSet -PackageId "TensorSharp.Chat" -Actual @("TensorSharp.Studio", "TensorSharp.Unlisted") -Expected @("TensorSharp.Studio")
        }
    }

    Test-Case "missing actual dependencies still fail the exact-set guard" {
        Assert-Throws {
            Assert-SameSet -PackageId "TensorSharp.Chat" -Actual @() -Expected @("TensorSharp.Studio")
        }
    }

    Test-Case "nuspec metadata and a closed Chat-to-Studio graph are accepted" {
        $studio = New-FixturePackage -Id "TensorSharp.Studio"
        $chat = New-FixturePackage -Id "TensorSharp.Chat" -Dependencies @(
            @{ Id = "TensorSharp.Studio"; Version = "2026.10.23" },
            @{ Id = "Microsoft.Extensions.Logging"; Version = "10.0.6" }
        )
        Assert-True ($chat.Id -eq "TensorSharp.Chat") "Read-Nupkg must read the nuspec package ID."
        Assert-True ($chat.Version -eq "2026.10.23") "Read-Nupkg must read the nuspec version."
        Assert-True ($chat.Dependencies.Count -eq 2) "Read-Nupkg must retain internal and external dependencies."
        Assert-PackageClosure -PackagesById @{ "TensorSharp.Chat" = $chat; "TensorSharp.Studio" = $studio }
    }

    Test-Case "NuGet's exact and minimum-version dependency forms are accepted" {
        foreach ($range in @("[2026.10.23]", "[2026.10.23, )")) {
            $studio = New-FixturePackage -Id "TensorSharp.Studio"
            $chat = New-FixturePackage -Id "TensorSharp.Chat" -Dependencies @(
                @{ Id = "TensorSharp.Studio"; Version = $range }
            )
            Assert-PackageClosure -PackagesById @{ "TensorSharp.Chat" = $chat; "TensorSharp.Studio" = $studio }
        }
    }

    Test-Case "an actual dependency without a produced package is rejected" {
        $chat = New-FixturePackage -Id "TensorSharp.Chat" -Dependencies @(
            @{ Id = "TensorSharp.Studio"; Version = "2026.10.23" }
        )
        Assert-Throws { Assert-PackageClosure -PackagesById @{ "TensorSharp.Chat" = $chat } }
    }

    Test-Case "a manifest entry cannot substitute for a missing package artifact" {
        $studio = New-FixturePackage -Id "TensorSharp.Studio"
        $chat = New-FixturePackage -Id "TensorSharp.Chat" -Dependencies @(
            @{ Id = "TensorSharp.Studio"; Version = "2026.10.23" }
        )
        Remove-Item -LiteralPath $studio.Path
        Assert-Throws { Assert-PackageClosure -PackagesById @{ "TensorSharp.Chat" = $chat; "TensorSharp.Studio" = $studio } }
    }

    Test-Case "a dependency on a different release version is rejected" {
        $studio = New-FixturePackage -Id "TensorSharp.Studio"
        $chat = New-FixturePackage -Id "TensorSharp.Chat" -Dependencies @(
            @{ Id = "TensorSharp.Studio"; Version = "2026.10.22" }
        )
        Assert-Throws { Assert-PackageClosure -PackagesById @{ "TensorSharp.Chat" = $chat; "TensorSharp.Studio" = $studio } }
    }

    Test-Case "package filename versions follow NuGet normalization" {
        foreach ($pair in @(
            @("2026.10.23+build.4", "2026.10.23"),
            @("20260618", "20260618.0.0"),
            @("2.8.6.0", "2.8.6"),
            @("2.8.06-preview.1+build.4", "2.8.6-preview.1")
        )) {
            Assert-True ((Get-NormalizedNuGetVersion $pair[0]) -eq $pair[1]) "Wrong NuGet normalization for $($pair[0])."
        }
    }
}
finally {
    Remove-Item -LiteralPath $FixtureRoot -Recurse -Force
}

if ($script:Failures.Count -gt 0) {
    throw "$($script:Failures.Count) package verifier test(s) failed: $($script:Failures -join '; ')"
}
Write-Host "$script:Passed package verifier tests passed."
