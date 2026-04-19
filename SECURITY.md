# Security policy

## Supported versions

The most recent minor release receives security updates. Older versions receive
critical fixes only on a best-effort basis.

## Reporting a vulnerability

**Do not file a public issue.** Instead, use GitHub's private vulnerability
reporting:

1. Go to https://github.com/ksk5429/polymarket-oracle-risk/security/advisories/new
2. File a draft advisory describing the issue, reproduction, and impact.

We acknowledge within 72 h and aim to triage within 7 days.

## Scope

In scope:

- RCE, path traversal, unsafe deserialization, SSRF via subgraph URLs
- Subgraph injection that bypasses signature validation
- Model-level exploits that cause the scorer to emit inverted risk for
  adversarial input (this matters — it could grant false green lights)

Out of scope:

- Model accuracy / calibration complaints (file a regular issue)
- Rate-limit exhaustion against third-party APIs
- Social-engineering, phishing

## Safe harbour

Good-faith security research is welcomed. We will not pursue legal action
against researchers who follow this policy.
