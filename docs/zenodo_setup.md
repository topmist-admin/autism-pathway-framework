# Zenodo DOI Setup Guide

This document describes how to set up Zenodo integration for the Autism Pathway Framework to obtain a citable DOI.

## Overview

[Zenodo](https://zenodo.org) is a general-purpose open repository that allows researchers to deposit datasets, software, and other research outputs. When linked to GitHub, Zenodo automatically archives each release and assigns a DOI.

## Setup Steps

### 1. Link GitHub Repository to Zenodo

1. Go to [Zenodo](https://zenodo.org) and log in with your GitHub account
2. Navigate to **Settings** → **GitHub**
3. Find `topmist-admin/autism-pathway-framework` in the repository list
4. Toggle the switch to **ON** to enable automatic archiving

### 2. Verify Metadata

The repository includes a `.zenodo.json` file that defines metadata for Zenodo:

```json
{
    "title": "Autism Pathway Framework: A Research Tool for Pathway-Based Analysis of Genetic Heterogeneity in ASD",
    "upload_type": "software",
    "license": "MIT",
    "version": "0.1.0",
    "keywords": ["autism", "genetics", "pathway analysis", ...],
    "creators": [{"name": "Chauhan, Rohit"}]
}
```

This metadata will be used when Zenodo creates the DOI record.

### 3. Create a Release

When you create a GitHub release, Zenodo will automatically:

1. Archive the release
2. Assign a DOI
3. Create a citable record

#### Release Candidate (Dry Run)

For testing, create a release candidate first:

```bash
# Tag the release candidate
git tag -a v0.1.0-rc1 -m "Release candidate 1 for v0.1.0"
git push origin v0.1.0-rc1

# Create GitHub release
gh release create v0.1.0-rc1 \
    --title "v0.1.0-rc1: Release Candidate" \
    --notes "Release candidate for testing Zenodo integration" \
    --prerelease
```

#### Final Release

For the final release:

```bash
# Tag the release
git tag -a v0.1.0 -m "v0.1.0: First reproducibility release"
git push origin v0.1.0

# Create GitHub release
gh release create v0.1.0 \
    --title "v0.1.0: First Reproducibility Release" \
    --notes-file RELEASE_NOTES.md
```

### 4. Retrieve the DOI

After creating a release:

1. Go to [Zenodo](https://zenodo.org)
2. Navigate to **Upload** → **GitHub**
3. Find the release in the list
4. Copy the DOI badge and add it to README.md

Example badge format:
```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

## Using the Zenodo Sandbox (Recommended for Testing)

For dry-run testing without creating a real DOI:

1. Go to [Zenodo Sandbox](https://sandbox.zenodo.org)
2. Log in with GitHub
3. Link your repository
4. Create a pre-release tag
5. Verify the metadata appears correctly

The sandbox DOIs are not permanent and won't pollute the DOI namespace.

## Metadata Fields

| Field | Value | Source |
|-------|-------|--------|
| Title | Autism Pathway Framework: A Research Tool... | `.zenodo.json` |
| Type | Software | `.zenodo.json` |
| License | MIT | `.zenodo.json`, `LICENSE` |
| Version | 0.1.0 | `.zenodo.json`, `pyproject.toml` |
| Keywords | autism, genetics, pathway analysis, ... | `.zenodo.json` |
| Creators | Chauhan, Rohit | `.zenodo.json`, `CITATION.cff` |

## Updating Metadata

To update metadata for future releases:

1. Edit `.zenodo.json` with new version number and any changes
2. Update `CITATION.cff` to match
3. Update `pyproject.toml` version
4. Create a new release

## Citation

Once a DOI is assigned, users can cite the software as:

```
Chauhan, R. (2026). Autism Pathway Framework: A Research Tool for
Pathway-Based Analysis of Genetic Heterogeneity in ASD (v0.1.0).
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

Or use the CITATION.cff file for automatic citation formatting.

## Troubleshooting

### Zenodo doesn't detect the release

- Ensure the repository is linked in Zenodo settings
- Check that the release is not a draft
- Wait a few minutes for Zenodo to process

### Metadata is incorrect

- Update `.zenodo.json` and create a new release
- Zenodo reads metadata at release time, not retroactively

### DOI not showing

- DOI assignment can take a few minutes
- Check Zenodo upload page for status

## References

- [Zenodo Documentation](https://help.zenodo.org/)
- [GitHub-Zenodo Integration Guide](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
- [Making Your Code Citable](https://guides.github.com/activities/citable-code/)
