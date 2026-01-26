#!/bin/bash
# Create all 90-day plan issues using GitHub CLI
# Usage: ./.github/create_issues.sh

set -e

echo "Creating labels..."
gh label create milestone --color "6f42c1" --description "Weekly milestone issue" 2>/dev/null || echo "Label 'milestone' exists"
gh label create v0.1 --color "0366d6" --description "Part of v0.1 release" 2>/dev/null || echo "Label 'v0.1' exists"
gh label create infrastructure --color "1d76db" --description "Environment, CI, tooling" 2>/dev/null || echo "Label 'infrastructure' exists"
gh label create documentation --color "0075ca" --description "Docs, guides, notebooks" 2>/dev/null || echo "Label 'documentation' exists"
gh label create validation --color "d93f0b" --description "Tests, negative controls, stability" 2>/dev/null || echo "Label 'validation' exists"
gh label create release --color "28a745" --description "Versioning, changelog, DOI" 2>/dev/null || echo "Label 'release' exists"
gh label create outreach --color "e99695" --description "External collaboration" 2>/dev/null || echo "Label 'outreach' exists"
gh label create blocked --color "b60205" --description "Waiting on something" 2>/dev/null || echo "Label 'blocked' exists"

echo ""
echo "Creating milestones..."
gh api repos/:owner/:repo/milestones -f title="Phase 1: Foundation (Weeks 1-4)" -f description="Environment, demo data, golden path" 2>/dev/null || echo "Milestone exists"
gh api repos/:owner/:repo/milestones -f title="Phase 2: Validation (Weeks 5-7)" -f description="Outputs dictionary, validation gates, CI" 2>/dev/null || echo "Milestone exists"
gh api repos/:owner/:repo/milestones -f title="Phase 3: Documentation (Weeks 8-10)" -f description="Docs, notebook, release prep" 2>/dev/null || echo "Milestone exists"
gh api repos/:owner/:repo/milestones -f title="Phase 4: Release (Weeks 11-13)" -f description="Zenodo, v0.1 release, outreach" 2>/dev/null || echo "Milestone exists"

echo ""
echo "Creating issues..."

# Week 1
gh issue create \
  --title "[Week 1] Define the v0.1 Contract" \
  --label "milestone,v0.1,infrastructure" \
  --body "## Theme
Define the v0.1 contract

## Deliverables
- [ ] v0.1 scope doc (1 page): what's in/out, \"research-only\" boundary, supported OS/Python, expected runtime
- [ ] Define \"golden path\" target: end-to-end demo in ≤60 minutes
- [ ] Repo cleanup: define \`/docs\`, \`/configs\`, \`/examples\`, \`/outputs\` structure

## Acceptance Criteria
- Scope doc exists at \`docs/v0.1_scope.md\`
- Folder structure matches spec
- Team aligned on boundaries

## Dependencies
None (Week 0 pre-flight complete)"

# Week 2
gh issue create \
  --title "[Week 2] Freeze Dependencies + Environment" \
  --label "milestone,v0.1,infrastructure" \
  --body "## Theme
Freeze dependencies + environment

## Deliverables
- [ ] One reproducible environment: Dockerfile or conda-lock / uv.lock
- [ ] \`make setup\` and \`make test\` working
- [ ] Pinned versions for biology resources documented

## Acceptance Criteria
- \`make setup && make test\` passes on clean machine
- Environment file committed
- \`docs/data_versions.md\` documents all external data sources

## Dependencies
Week 1 complete"

# Week 3
gh issue create \
  --title "[Week 3] Minimal Runnable Demo Dataset" \
  --label "milestone,v0.1,infrastructure" \
  --body "## Theme
Minimal runnable demo dataset

## Deliverables
- [ ] Synthetic or tiny public-safe demo dataset at \`/examples/demo_data/\`
- [ ] Data schema docs: input → intermediate → output
- [ ] Deterministic seed strategy for demo runs

## Acceptance Criteria
- Demo data loads without errors
- Schema documented at \`docs/data_schema.md\`
- Same seed produces identical outputs

## Risks
- Synthetic genomics data that's realistic but privacy-safe is non-trivial
- May need domain expert review

## Dependencies
Week 2 complete"

# Week 4
gh issue create \
  --title "[Week 4] One-Command Pipeline Run (Golden Path)" \
  --label "milestone,v0.1,infrastructure" \
  --body "## Theme
One-command pipeline run (\"golden path\")

## Deliverables
- [ ] CLI or script: \`./run_demo.sh\` or \`python -m autism_pathway_framework --config configs/demo.yaml\`
- [ ] Generates consistent output folder: \`/outputs/demo_run/\`
- [ ] Output includes: pathway score table + summary figure + JSON/MD report

## Acceptance Criteria
- Single command runs end-to-end in ≤60 minutes
- Output folder structure is predictable
- Report is human-readable

## Dependencies
Week 3 complete"

# Week 5
gh issue create \
  --title "[Week 5] Outputs Dictionary + Interpretation Guardrails" \
  --label "milestone,v0.1,documentation" \
  --body "## Theme
Outputs dictionary + interpretation guardrails

## Deliverables
- [ ] \`docs/outputs_dictionary.md\`: each artifact + how to interpret + what NOT to infer
- [ ] \"Research-only / not clinical\" banner in README + docs + demo report

## Acceptance Criteria
- Every output file type documented
- Guardrails visible in all user-facing outputs

## Dependencies
Week 4 complete"

# Week 6
gh issue create \
  --title "[Week 6] Validation Gates v1 (Stability + Null Tests)" \
  --label "milestone,v0.1,validation" \
  --body "## Theme
Validation gates v1 (stability + null tests)

## Deliverables
- [ ] At least 2 mandatory negative controls (e.g., label shuffle; random gene sets)
- [ ] At least 1 stability test (bootstrap/resample + stability metric)
- [ ] Pass/fail summary in demo report: \"Validation Gates: PASS/FAIL with reasons\"

## Acceptance Criteria
- Negative controls run automatically in pipeline
- Stability metric defined and threshold documented (e.g., ARI ≥ 0.8)
- Report clearly shows pass/fail status

## Dependencies
Week 5 complete"

# Week 7
gh issue create \
  --title "[Week 7] Make Results Reproducible Across Machines" \
  --label "milestone,v0.1,validation,infrastructure" \
  --body "## Theme
Make results reproducible across machines

## Deliverables
- [ ] \"Golden outputs\" committed or stored as release artifacts
- [ ] CI workflow (GitHub Actions) runs demo + checks outputs
- [ ] Troubleshooting guide: common failures

## Acceptance Criteria
- CI passes on push to main
- Hash/metric check validates reproducibility
- \`docs/troubleshooting.md\` exists

## Dependencies
Week 6 complete"

# Week 8
gh issue create \
  --title "[Week 8] Documentation and Site Alignment" \
  --label "milestone,v0.1,documentation" \
  --body "## Theme
Documentation and site alignment

## Deliverables
- [ ] \"Start Here for Researchers\" page mirrors golden path
- [ ] \`docs/quickstart.md\` is the canonical entry point
- [ ] 1 diagram: pipeline architecture

## Acceptance Criteria
- Substack and repo docs are consistent
- Quickstart tested by external person
- Architecture diagram committed

## Dependencies
Week 7 complete"

# Week 9
gh issue create \
  --title "[Week 9] Create the Demo Notebook (Colab-Ready)" \
  --label "milestone,v0.1,documentation" \
  --body "## Theme
Create the demo notebook (Colab-ready)

## Deliverables
- [ ] \`notebooks/01_demo_end_to_end.ipynb\`:
  - Load demo data
  - Run pipeline
  - Inspect outputs
  - View validation gates
  - Show how to swap in cohort data
- [ ] Test on Google Colab

## Acceptance Criteria
- Notebook runs end-to-end without errors
- Works in both local Jupyter and Colab
- Clear \"swap your data here\" sections

## Dependencies
Week 8 complete"

# Week 10
gh issue create \
  --title "[Week 10] Release Prep (Versioning + Changelog + Licensing)" \
  --label "milestone,v0.1,release" \
  --body "## Theme
Release prep: versioning + changelog + licensing

## Deliverables
- [ ] \`CHANGELOG.md\` with v0.1 entry (features + known limitations)
- [ ] Confirm license (repo + notebook + docs)
- [ ] Add \`CITATION.cff\` (author, title, version, repo URL)

## Acceptance Criteria
- CHANGELOG follows Keep a Changelog format
- License file present and appropriate
- CITATION.cff validates

## Dependencies
Week 9 complete"

# Week 11
gh issue create \
  --title "[Week 11] Zenodo DOI Setup (Dry Run)" \
  --label "milestone,v0.1,release" \
  --body "## Theme
Zenodo DOI setup (dry run)

## Deliverables
- [ ] Link GitHub repo to Zenodo
- [ ] Prepare metadata: title, description, keywords, communities
- [ ] Tag \`v0.1.0-rc1\` and validate Zenodo capture

## Acceptance Criteria
- Zenodo sandbox shows correct metadata
- Release candidate tag exists
- DOI preview looks correct

## Dependencies
Week 10 complete"

# Week 12
gh issue create \
  --title "[Week 12] Final v0.1 Reproducibility Release" \
  --label "milestone,v0.1,release" \
  --body "## Theme
Final v0.1 reproducibility release

## Deliverables
- [ ] Tag \`v0.1.0\` on GitHub
- [ ] Release notes: quickstart, runtime, outputs, validation gates
- [ ] Zenodo DOI minted and added to README, Substack, ResearchGate

## Acceptance Criteria
- GitHub release page complete
- DOI resolves correctly
- All docs reference the DOI

## Dependencies
Week 11 complete"

# Week 13
gh issue create \
  --title "[Week 13] External Collaborator Outreach Package" \
  --label "milestone,v0.1,outreach" \
  --body "## Theme
External collaborator outreach package

## Deliverables
- [ ] One-pager PDF/MD for labs/biobanks
- [ ] Email template (short, credible)
- [ ] Demo notebook + \"How to run on your cohort\" appendix
- [ ] Target list of 5–10 labs
- [ ] Send to at least 1 external collaborator

## Acceptance Criteria
- Outreach bundle complete and reviewed
- At least 1 email sent
- Follow-up scheduled for +7 days

## Dependencies
Week 12 complete"

echo ""
echo "Done! Created 13 weekly milestone issues."
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/YOUR_ORG/YOUR_REPO/projects"
echo "2. Create a new project board"
echo "3. Add the issues to the board"
