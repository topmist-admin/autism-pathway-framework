# v0.1 Release: 90-Day Project Board

## Board Structure

Create a GitHub Project with these columns:

| Column | Description |
|--------|-------------|
| **Backlog** | Future weeks not yet started |
| **This Week** | Current week's deliverables |
| **In Progress** | Actively being worked on |
| **In Review** | PR open, awaiting review |
| **Done** | Merged and verified |
| **Blocked** | Waiting on external dependency |

---

## Labels

Create these labels in your repository:

| Label | Color | Description |
|-------|-------|-------------|
| `milestone` | `#6f42c1` | Weekly milestone issue |
| `v0.1` | `#0366d6` | Part of v0.1 release |
| `infrastructure` | `#1d76db` | Environment, CI, tooling |
| `documentation` | `#0075ca` | Docs, guides, notebooks |
| `validation` | `#d93f0b` | Tests, negative controls, stability |
| `release` | `#28a745` | Versioning, changelog, DOI |
| `outreach` | `#e99695` | External collaboration |
| `blocked` | `#b60205` | Waiting on something |
| `good first issue` | `#7057ff` | Good for new contributors |

---

## Milestones

Create these GitHub Milestones:

1. **Phase 1: Foundation** (Weeks 1-4) - Due: +4 weeks
2. **Phase 2: Validation** (Weeks 5-7) - Due: +7 weeks
3. **Phase 3: Documentation** (Weeks 8-10) - Due: +10 weeks
4. **Phase 4: Release** (Weeks 11-13) - Due: +13 weeks

---

## Pre-Populated Issues

Copy and create these issues in your repository:

---

### Week 1: Define the v0.1 Contract

**Labels:** `milestone`, `v0.1`, `infrastructure`
**Milestone:** Phase 1: Foundation

#### Deliverables
- [ ] v0.1 scope doc (1 page): what's in/out, "research-only" boundary, supported OS/Python, expected runtime
- [ ] Define "golden path" target: end-to-end demo in ≤60 minutes
- [ ] Repo cleanup: define `/docs`, `/configs`, `/examples`, `/outputs` structure

#### Acceptance Criteria
- Scope doc exists at `docs/v0.1_scope.md`
- Folder structure matches spec
- Team aligned on boundaries

---

### Week 2: Freeze Dependencies + Environment

**Labels:** `milestone`, `v0.1`, `infrastructure`
**Milestone:** Phase 1: Foundation

#### Deliverables
- [ ] One reproducible environment: Dockerfile or conda-lock / uv.lock
- [ ] `make setup` and `make test` working
- [ ] Pinned versions for biology resources (gene IDs, pathway DB versions, KG source versions) documented

#### Acceptance Criteria
- `make setup && make test` passes on clean machine
- Environment file committed
- `docs/data_versions.md` documents all external data sources

---

### Week 3: Minimal Runnable Demo Dataset

**Labels:** `milestone`, `v0.1`, `infrastructure`
**Milestone:** Phase 1: Foundation

#### Deliverables
- [ ] Synthetic or tiny public-safe demo dataset at `/examples/demo_data/`
- [ ] Data schema docs: input → intermediate → output (tables + column definitions)
- [ ] Deterministic seed strategy for demo runs

#### Acceptance Criteria
- Demo data loads without errors
- Schema documented at `docs/data_schema.md`
- Same seed produces identical outputs

#### Risks
- Synthetic genomics data that's realistic but privacy-safe is non-trivial
- May need domain expert review

---

### Week 4: One-Command Pipeline Run ("Golden Path")

**Labels:** `milestone`, `v0.1`, `infrastructure`
**Milestone:** Phase 1: Foundation

#### Deliverables
- [ ] CLI or script: `./run_demo.sh` or `python -m autism_pathway_framework --config configs/demo.yaml`
- [ ] Generates consistent output folder: `/outputs/demo_run/`
- [ ] Output includes: pathway score table + one summary figure + short JSON/MD report

#### Acceptance Criteria
- Single command runs end-to-end in ≤60 minutes
- Output folder structure is predictable
- Report is human-readable

---

### Week 5: Outputs Dictionary + Interpretation Guardrails

**Labels:** `milestone`, `v0.1`, `documentation`
**Milestone:** Phase 2: Validation

#### Deliverables
- [ ] `docs/outputs_dictionary.md`: each artifact + how to interpret + what NOT to infer
- [ ] "Research-only / not clinical" banner in README + docs + demo report template

#### Acceptance Criteria
- Every output file type documented
- Guardrails visible in all user-facing outputs

---

### Week 6: Validation Gates v1 (Stability + Null Tests)

**Labels:** `milestone`, `v0.1`, `validation`
**Milestone:** Phase 2: Validation

#### Deliverables
- [ ] At least 2 mandatory negative controls (e.g., label shuffle; random gene sets)
- [ ] At least 1 stability test (bootstrap/resample + stability metric like ARI ≥ 0.8)
- [ ] Pass/fail summary in demo report: "Validation Gates: PASS/FAIL with reasons"

#### Acceptance Criteria
- Negative controls run automatically in pipeline
- Stability metric defined and threshold documented
- Report clearly shows pass/fail status

---

### Week 7: Make Results Reproducible Across Machines

**Labels:** `milestone`, `v0.1`, `validation`, `infrastructure`
**Milestone:** Phase 2: Validation

#### Deliverables
- [ ] "Golden outputs" committed (small files) or stored as release artifacts: expected hashes/metrics
- [ ] CI workflow (GitHub Actions) that runs demo + checks key outputs
- [ ] Troubleshooting guide: common failures

#### Acceptance Criteria
- CI passes on push to main
- Hash/metric check validates reproducibility
- `docs/troubleshooting.md` exists

---

### Week 8: Documentation and Site Alignment

**Labels:** `milestone`, `v0.1`, `documentation`
**Milestone:** Phase 3: Documentation

#### Deliverables
- [ ] "Start Here for Researchers" page (Substack +/or repo docs) mirrors golden path
- [ ] `docs/quickstart.md` is the canonical entry point
- [ ] 1 diagram: pipeline architecture (variant→gene→pathway→validation→optional KG)

#### Acceptance Criteria
- Substack and repo docs are consistent
- Quickstart tested by external person (friend/colleague)
- Architecture diagram committed

---

### Week 9: Create the Demo Notebook (Colab-Ready)

**Labels:** `milestone`, `v0.1`, `documentation`
**Milestone:** Phase 3: Documentation

#### Deliverables
- [ ] `notebooks/01_demo_end_to_end.ipynb`:
  - Load demo data
  - Run pipeline
  - Inspect outputs
  - View validation gates
  - Show how to swap in their cohort's formats
- [ ] Test on Google Colab

#### Acceptance Criteria
- Notebook runs end-to-end without errors
- Works in both local Jupyter and Colab
- Clear "swap your data here" sections

---

### Week 10: Release Prep (Versioning + Changelog + Licensing)

**Labels:** `milestone`, `v0.1`, `release`
**Milestone:** Phase 3: Documentation

#### Deliverables
- [ ] `CHANGELOG.md` with v0.1 entry (features + known limitations)
- [ ] Confirm license (repo + notebook + docs)
- [ ] Add `CITATION.cff` (author, title, version, repo URL)

#### Acceptance Criteria
- CHANGELOG follows Keep a Changelog format
- License file present and appropriate
- CITATION.cff validates with cff-validator

---

### Week 11: Zenodo DOI Setup (Dry Run)

**Labels:** `milestone`, `v0.1`, `release`
**Milestone:** Phase 4: Release

#### Deliverables
- [ ] Link GitHub repo to Zenodo
- [ ] Prepare metadata: title, description, keywords, communities, license
- [ ] Tag candidate release `v0.1.0-rc1` and validate Zenodo capture

#### Acceptance Criteria
- Zenodo sandbox shows correct metadata
- Release candidate tag exists
- DOI preview looks correct

---

### Week 12: Final v0.1 Reproducibility Release

**Labels:** `milestone`, `v0.1`, `release`
**Milestone:** Phase 4: Release

#### Deliverables
- [ ] Tag `v0.1.0` on GitHub
- [ ] Release notes include: quickstart, runtime, outputs, validation gates
- [ ] Zenodo DOI minted and added to:
  - README
  - Substack "For Researchers" page
  - ResearchGate (if applicable)

#### Acceptance Criteria
- GitHub release page complete
- DOI resolves correctly
- All docs reference the DOI

---

### Week 13: External Collaborator Outreach Package

**Labels:** `milestone`, `v0.1`, `outreach`
**Milestone:** Phase 4: Release

#### Deliverables
- [ ] One-pager PDF/MD for labs/biobanks:
  - What it is / isn't
  - What problem it solves
  - What you need from them (data formats, cohort characteristics)
  - What they get (reproducible pipeline + stability/replication report)
  - Timeline: "2–4 week pilot"
- [ ] Email template (short, credible)
- [ ] Demo notebook + "How to run on your cohort" appendix
- [ ] Target list of 5–10 labs
- [ ] Send to at least 1 external collaborator

#### Acceptance Criteria
- Outreach bundle complete and reviewed
- At least 1 email sent
- Follow-up scheduled for +7 days if no response

---

## Quick Setup Commands

```bash
# Create labels using GitHub CLI
gh label create milestone --color "6f42c1" --description "Weekly milestone issue"
gh label create v0.1 --color "0366d6" --description "Part of v0.1 release"
gh label create infrastructure --color "1d76db" --description "Environment, CI, tooling"
gh label create documentation --color "0075ca" --description "Docs, guides, notebooks"
gh label create validation --color "d93f0b" --description "Tests, negative controls, stability"
gh label create release --color "28a745" --description "Versioning, changelog, DOI"
gh label create outreach --color "e99695" --description "External collaboration"
gh label create blocked --color "b60205" --description "Waiting on something"

# Create milestones
gh api repos/:owner/:repo/milestones -f title="Phase 1: Foundation" -f due_on="$(date -v+4w +%Y-%m-%dT%H:%M:%SZ)"
gh api repos/:owner/:repo/milestones -f title="Phase 2: Validation" -f due_on="$(date -v+7w +%Y-%m-%dT%H:%M:%SZ)"
gh api repos/:owner/:repo/milestones -f title="Phase 3: Documentation" -f due_on="$(date -v+10w +%Y-%m-%dT%H:%M:%SZ)"
gh api repos/:owner/:repo/milestones -f title="Phase 4: Release" -f due_on="$(date -v+13w +%Y-%m-%dT%H:%M:%SZ)"
```

---

## Day 90 Acceptance Criteria

- [ ] A new user can run one command and reproduce the same outputs (within defined tolerances)
- [ ] Demo report includes validation gates and clearly states limitations
- [ ] GitHub release v0.1.0 exists + Zenodo DOI resolves and is cited in README
- [ ] Complete outreach bundle: email + one-pager + notebook
- [ ] At least one email sent to external collaborator
