# Contributing to OptiFusionNet

Thank you for your interest in contributing! Please follow the guidelines below to keep the project clean and collaborative.

---

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

---

## How to Contribute

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/OptiFusionNet.git
cd OptiFusionNet
```

### 2. Create a Branch

Use the naming convention below:

| Type | Branch name |
|---|---|
| New feature | `feature/<short-description>` |
| Bug fix | `fix/<short-description>` |
| Documentation | `docs/<short-description>` |
| Experiment / research | `exp/<short-description>` |

```bash
git checkout -b feature/add-attention-block
```

### 3. Set Up Your Environment

```bash
bash setup.sh
```

### 4. Make Your Changes

- **Do not modify the core model logic** (src/model.py, src/train.py, src/loss.py, src/meta_opt.py) unless you are proposing a research improvement and have documented results.
- Keep changes focused — one feature / fix per pull request.
- Add or update tests where applicable.

### 5. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

Examples:

```
feat(inference): add batch inference support
fix(dataset): handle missing paired images gracefully
docs(readme): update dataset setup instructions
```

### 6. Push & Open a Pull Request

```bash
git push origin feature/add-attention-block
```

Then open a Pull Request against `main` on GitHub. Fill in the PR template:

- **What** does this PR do?
- **Why** is this change needed?
- **How** was it tested?

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, release-ready code |
| `dev` | Integration branch for in-progress work |
| `feature/*` | Individual features |
| `fix/*` | Bug fixes |
| `exp/*` | Experimental / research changes |

---

## Versioning

This project uses [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

- **MAJOR** — breaking API or architecture changes
- **MINOR** — new features, backward-compatible
- **PATCH** — bug fixes, documentation updates

---

## Reporting Issues

Open a [GitHub Issue](../../issues) with:

1. A clear title.
2. Steps to reproduce (if it's a bug).
3. Your environment (OS, Python version, PyTorch version, GPU).
4. Expected vs. actual behaviour.

---

Thank you for helping make OptiFusionNet better!
