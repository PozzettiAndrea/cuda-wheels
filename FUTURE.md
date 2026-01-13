# Future Vision: Distributed Wheel Building Network

> This document captures ideas for a future community-driven wheel building system.
> Current status: **Just ideas** - we're using centralized GitHub Actions builds for now.

## The Dream

Users running `comfy-env install sageattention` could:

1. Check the central cuda-wheels index (current approach)
2. Check a community index of user-contributed wheels
3. If not found, build locally and upload for others

```
comfy-env install sageattention
         │
         ▼
  ┌─────────────────┐
  │ Resolve CUDA/   │
  │ PyTorch version │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐     Found?
  │ Check official  │────────────► Install
  │ cuda-wheels     │
  └────────┬────────┘
           │ Not found
           ▼
  ┌─────────────────┐     Found + Verified?
  │ Check community │────────────────────────► Install
  │ index           │
  └────────┬────────┘
           │ Not found
           ▼
  ┌─────────────────┐
  │ Build locally   │
  │ (with consent)  │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Upload to       │
  │ community index │
  └─────────────────┘
```

## The Trust Problem

How do you trust a wheel compiled by a random user?

### Solution: Hash-Based Verification

Every build produces a **manifest** with hashes at each layer:

```
                    ┌─────────────────┐
                    │  FINAL HASH     │  ← Published with wheel
                    │  (wheel hash)   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
   ┌─────┴─────┐      ┌──────┴──────┐     ┌─────┴─────┐
   │  SOURCE   │      │    BUILD    │     │  PACKAGE  │
   │   HASH    │      │    HASH     │     │   HASH    │
   └─────┬─────┘      └──────┬──────┘     └─────┬─────┘
         │                   │                   │
    git repo            env + flags          wheel creation
    + commit
```

### Trust Rules

```python
derivation_hash = hash(source + environment + build_config)
output_hash = hash(wheel_contents)

# If 3+ independent users produce the same output_hash
# for the same derivation_hash → VERIFIED
```

This works because:
- Malicious actors would need to coordinate to produce the same malicious binary
- Legitimate builds converge on the same output (or close to it)
- Non-determinism is detectable (multiple outputs for same inputs)

## Manifest Schema

```json
{
  "version": "1.0",
  "package": "sageattention",
  "wheel": "sageattention-2.2.0+cu124torch24-cp311-cp311-linux_x86_64.whl",

  "derivation": {
    "hash": "7e8f9a0b1c2d3e4f",
    "source": {
      "hash": "3f8a2b1c",
      "repo": "thu-ml/SageAttention",
      "commit": "abc123def456789",
      "setup_py_sha256": "..."
    },
    "environment": {
      "hash": "9c2d4e5f",
      "cuda": "12.4.1",
      "pytorch": "2.4.0+cu124",
      "pytorch_sha256": "...",
      "python": "3.11.9",
      "compiler": "gcc-11.4.0",
      "platform": "linux-x86_64"
    },
    "config": {
      "hash": "1a2b3c4d",
      "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0",
      "build_command": "python setup.py bdist_wheel"
    }
  },

  "output": {
    "sha256": "d4e5f6a7b8c9...",
    "size": 12345678
  },

  "builder": {
    "id": "user@example.com",
    "timestamp": "2024-01-15T10:30:00Z",
    "signature": "..."
  }
}
```

## Build Environment Requirements

### Linux
```bash
# Minimal requirements
apt-get install build-essential  # gcc, make, etc.
# CUDA toolkit downloaded automatically by comfy-env
```

### Windows
```powershell
# The hard part - MSVC Build Tools (~2-4GB)
winget install Microsoft.VisualStudio.2022.BuildTools
# Or guide user through manual install

# CUDA toolkit downloaded automatically
```

## Community Index Storage Options

| Option | Pros | Cons |
|--------|------|------|
| GitHub Releases | Free, familiar | 2GB per release limit |
| R2/S3 bucket | Cheap (~$0.015/GB/mo) | Needs hosting |
| IPFS | Decentralized, content-addressed | Slow, complex |
| Simple Flask + S3 | Full control | Maintenance burden |

## Prior Art

| System | User Contributed? | Verification |
|--------|-------------------|--------------|
| [piwheels](https://piwheels.org) | No (central) | N/A |
| [conda-forge](https://conda-forge.org) | Recipes only | Central CI builds |
| [Nix](https://nixos.org) | Yes | Content-addressed |
| [Reproducible Builds](https://reproducible-builds.org) | Yes | Multi-party verification |

**Nobody has done this for Python CUDA wheels specifically.**

## Phases

### Phase 0: Current State
- Central GitHub Actions builds
- Single cuda-wheels index
- Manual package additions

### Phase 1: Local Build Fallback
- comfy-env can build from source if wheel not found
- Auto-download CUDA toolkit
- Guide through MSVC install on Windows

### Phase 2: Upload to Community Index
- After successful build, offer to upload
- Store manifest with derivation/output hashes
- Basic trust: show "uploaded by N users"

### Phase 3: Verification Network
- Track if multiple users produce same output
- "Verified by 5 independent builders" badge
- Flag suspicious divergence

### Phase 4: Full Trust Network
- Reputation scores for builders
- Automatic verification builds
- Web of trust for signing keys

## Open Questions

1. **Non-determinism**: CUDA compilation often produces different binaries. How much variance is acceptable?

2. **Incentives**: Why would users upload? Altruism? Gamification? Token rewards?

3. **Storage costs**: Who pays for hosting community wheels?

4. **Liability**: What happens if a malicious wheel slips through?

5. **Bootstrapping**: Need critical mass of builders for verification to work.

## References

- [Reproducible Builds](https://reproducible-builds.org/) - Philosophy and techniques
- [Nix Pills](https://nixos.org/guides/nix-pills/) - Content-addressed builds
- [Agora Paper](https://arxiv.org/html/2407.15062v1) - Academic work on crowdsourced verification
- [Sigstore](https://sigstore.dev/) - Keyless signing for open source

---

*Last updated: January 2026*
*Status: Future vision, not currently implemented*
