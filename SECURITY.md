# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Noctilux, please report it privately by opening a GitHub Security Advisory at:

https://github.com/yelikour/noctilux/security/advisories/new

Do not open a public issue for security vulnerabilities.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.3.x   | Yes       |
| < 0.3.0 | No        |

## Scope

Noctilux is an offline image processing tool. It reads local image files and writes processed outputs to local disk. It does not make network requests or expose any services.

Security concerns in scope:

- Path traversal or arbitrary file write via config
- Unsafe deserialization
- Dependency vulnerabilities

Out of scope:

- Performance issues
- Feature requests
- Issues in dependencies not triggered by Noctilux usage
