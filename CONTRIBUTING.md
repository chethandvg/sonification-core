# Contributing

Thank you for helping. In this project, being right matters more than being fast.

## Good first contributions

- **Report a wrong or outdated claim** with the "Claim correction" issue form. Give the claim ID and a source (DOI or PMID).
- **Report a bug** with the "Bug report" form.
- **Ask a question or suggest an idea** in Discussions.

## Code contributions

Please open an issue before writing code, so the approach can be agreed first.

The author keeps the right to offer this work under other licences in future. Outside code contributions can therefore be merged only after the contributor signs a Contributor Licence Agreement (CLA), which will be linked here before the first outside pull request is accepted.

## Standards

- Every new claim goes in `claims.yaml` with its level: **P** (Lean proof), **V** (verified numerics), **T** (tests), **R** (reference with DOI or PMID) or **M** (stated assumption).
- A referenced fact or a model assumption is never presented as proved.
- Changes keep CI green: formatting, lints, tests on every platform, proofs and the claims check.
- New dependencies must use a permissive licence (MIT, BSD, Apache-2.0, ISC or similar). GPL and AGPL dependencies can't be accepted.
