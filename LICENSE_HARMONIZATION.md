# License Harmonization - Nexus Guardian D7D

**Project License**: MIT with Ethical Addendum  
**Last Updated**: 2026-02-10  
**Status**: ✅ All licenses compatible

## Overview

This document explains the licensing structure of the Nexus Guardian D7D project, which is a fusion of multiple repositories and components. We ensure full compliance with all open-source licenses while maintaining the project's ethical child protection mission.

## Primary Project License

**MIT License with Ethical Addendum** (see [LICENSE](LICENSE))

The project's original code is released under the MIT License with an additional Ethical Addendum that requires:
- Child protection must remain the primary purpose
- No removal of safety features
- No use for child exploitation or harm
- Transparency with parents and guardians
- Privacy protection for children's data

## Component Licenses

### Main Repository Components

| Component | License | Compatibility | Notes |
|-----------|---------|---------------|-------|
| **Core Source (`src/`)** | MIT + Ethical | ✅ Native | Original code |
| **INTERESTELAR_HEBRON** | MIT + Ethical | ✅ Compatible | Merged component |
| **PROJETO_INTERESTELAR_HEBRON** | MIT + Ethical | ✅ Compatible | Merged project |
| **LEGACY** | MIT + Ethical | ✅ Compatible | Archived code |
| **cosmic-orchestrator** | MIT + Ethical | ✅ Compatible | Orchestration system |
| **Notebooks** | MIT + Ethical | ✅ Compatible | Educational materials |

### Dependencies by License Category

#### ✅ Permissive Licenses (Fully Compatible)

**MIT License** (Most Common - Full Freedom)
- PyYAML, PyJWT, Twisted, click, httplib2, hyperlink, incremental
- jmespath, jsonschema, mdurl, markdown-it-py, netifaces
- pyparsing, pyrsistent, python-magic, pytz, rich, urllib3
- userpath, six, and many more

**BSD License** (Permissive - Requires Attribution)
- Babel, Jinja2, MarkupSafe, PyHamcrest, Pygments, click, colorama
- configobj, dbus-python, idna, jsonpatch, jsonpointer, netaddr
- oauthlib, pyasn1, pyasn1-modules, pyserial, zstandard

**Apache 2.0 License** (Permissive - Patent Grant)
- WALinuxAgent, argcomplete, bcrypt, boto3, botocore, requests
- cryptography, distro, packaging, python-dateutil, s3transfer

**ISC License** (Permissive - Very Similar to MIT)
- pexpect, ptyprocess

**Mozilla Public License 2.0** (Permissive with Copyleft for Changes)
- certifi

**Python Software Foundation License** (Permissive)
- typing_extensions

**Zope Public License** (Permissive)
- zope.interface

#### ⚠️ Copyleft Licenses (Require Special Handling)

**LGPL (Lesser GNU Public License)**
- PyGObject, chardet, systemd-python, launchpadlib, lazr.restfulclient, lazr.uri, wadllib
- **Handling**: Dynamic linking allowed; no source code distribution requirements for our project

**GPL (GNU General Public License)**
- mercurial, pyparted, python-apt, python-debian, sos, ssh-import-id, ubuntu-pro-client, ufw
- **Handling**: System utilities and tools; not distributed with our application; runtime dependencies only

**Dual Licensed (GPL/Apache)**
- cloud-init
- **Handling**: We use under Apache 2.0 terms

## License Compatibility Analysis

### ✅ Compatible Combinations

1. **MIT + BSD**: Fully compatible, requires attribution
2. **MIT + Apache**: Fully compatible, patent grant benefits
3. **MIT + ISC/PSF/Zope/MPL**: Fully compatible
4. **Dynamic Linking with LGPL**: Allowed, no source code obligations
5. **Runtime GPL Dependencies**: Allowed for system utilities

### 🔒 Isolation Strategy

**GPL Components** (when included as dependencies):
- Used only as external system tools
- Not linked into our codebase
- Runtime dependencies only
- No source distribution required for our project

**LGPL Components**:
- Dynamically linked (Python imports)
- No modifications made to LGPL code
- Source code available from original authors
- Compliant with LGPL dynamic linking provisions

## Attribution Requirements

### Required Attributions (BSD/Apache)

All BSD and Apache licensed dependencies require attribution. See [NOTICE.md](NOTICE.md) for complete attribution list.

### Copyright Notices

Preserved in each component:
- Original copyright holders listed
- License texts included
- Attribution maintained in documentation

## Practical Guidelines for Contributors

### ✅ You CAN:
- Use, modify, and distribute the project under MIT terms
- Add new MIT/BSD/Apache licensed dependencies
- Link with LGPL libraries dynamically
- Use system GPL utilities as runtime dependencies
- Distribute binaries under MIT + Ethical Addendum

### ❌ You CANNOT:
- Remove the Ethical Addendum
- Use GPL code directly in the codebase without proper isolation
- Remove attribution for BSD/Apache components
- Violate any child protection provisions
- Remove safety features

## Adding New Dependencies

### Approval Process

1. **Check License**: Verify it's in the compatible list
2. **For Permissive Licenses (MIT/BSD/Apache)**: ✅ Approved automatically
3. **For LGPL**: ✅ Approved if dynamically linked only
4. **For GPL**: ⚠️ Review required - must be external/runtime only
5. **For Proprietary**: ❌ Not allowed
6. **For Unknown/Custom**: ⚠️ Legal review required

### License Compatibility Matrix

| New License | Project License | Compatible? | Notes |
|-------------|----------------|-------------|-------|
| MIT | MIT + Ethical | ✅ Yes | Perfect match |
| BSD | MIT + Ethical | ✅ Yes | Add attribution |
| Apache 2.0 | MIT + Ethical | ✅ Yes | Add attribution |
| ISC/PSF | MIT + Ethical | ✅ Yes | Permissive |
| LGPL | MIT + Ethical | ✅ Yes | Dynamic linking only |
| GPL | MIT + Ethical | ⚠️ Conditional | External/runtime only |
| Proprietary | MIT + Ethical | ❌ No | Not allowed |

## Compliance Verification

### Automated Checks
- GitHub Actions workflow scans dependencies weekly
- License compatibility verified on each PR
- Attribution list automatically updated

### Manual Review
- New dependencies reviewed by maintainers
- License compatibility documented
- Attribution added to NOTICE.md

## Distribution

### Binary Distribution
When distributing compiled/packaged versions:
1. Include LICENSE file
2. Include NOTICE.md with all attributions
3. Include Ethical Addendum
4. Document all dependencies and their licenses

### Source Distribution
When distributing source code:
1. Include all license files
2. Maintain copyright notices
3. Include this harmonization document
4. Preserve Ethical Addendum

## Child Protection Legal Framework

### Additional Considerations
- COPPA (Children's Online Privacy Protection Act)
- GDPR (General Data Protection Regulation) - Children's data
- Regional child protection laws
- Educational use compliance

### Ethical Addendum Legal Standing
The Ethical Addendum is legally binding as part of the MIT License terms. Violation constitutes a material breach of the license agreement.

## Questions and Support

### License Questions
For licensing questions, contact:
- **Project Lead**: Helyton (Hebron)
- **Email**: hebron@trinity-xai.org
- **GitHub Issues**: For public license discussions

### Legal Review
For commercial use or legal review:
- Consult with your legal team
- We provide full transparency
- All licenses are standard OSI-approved

## References

- [Open Source Initiative (OSI)](https://opensource.org/)
- [SPDX License List](https://spdx.org/licenses/)
- [GNU License Compatibility](https://www.gnu.org/licenses/license-compatibility.html)
- [Apache License FAQ](https://www.apache.org/foundation/license-faq.html)

## Changelog

### 2026-02-10
- Initial license harmonization document
- Comprehensive dependency audit
- Attribution compilation
- Compatibility verification

---

**Summary**: All project components and dependencies are properly licensed and fully compatible with the MIT + Ethical Addendum license. The project maintains excellent license hygiene and full compliance with open-source licensing requirements.
