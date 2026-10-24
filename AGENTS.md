Mounting: sshfs pelle@192.168.2.14:/home/pelle/video_analyzer ~/rpi-project \
  -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3


# Development environment

## Architecture
- Codex runs locally on a Linux laptop.
- Project files are stored on a Raspberry Pi and mounted using SSHFS.
- Raspberry Pi SSH address: pelle@192.168.2.14
- Raspberry Pi architecture: Linux ARM64 (aarch64).
- Raspberry Pi has 1 GB RAM.

## File management
- Read and modify project files through the local SSHFS mount.
- All source code must remain on the Raspberry Pi.
- Do not modify files outside the project directory.
- Do not create unnecessary temporary files on the Raspberry Pi.

## Building and testing
- All compilation and testing must execute on the Raspberry Pi.
- Use SSH to run build commands on the Raspberry Pi.
- Do not compile ARM binaries on the laptop.
- Run tests after making code changes when practical.
- Report compilation errors and test failures.

## Resource constraints
- Minimize memory usage on the Raspberry Pi.
- Avoid running multiple builds in parallel.
- Prefer incremental compilation.
- Do not start resource-intensive background processes.

## Security
- Do not reboot the Raspberry Pi without permission.
- Do not install packages without permission.
- Do not modify system configuration without permission.
- Do not use sudo unless explicitly authorized.

## Workflow
1. Inspect the existing project before making changes.
2. Explain significant changes before implementing them.
3. Modify source files through SSHFS.
4. Compile and test on the Raspberry Pi using SSH.
5. Fix errors and rerun tests.
6. Summarize changes and test results.

## Project requirements
- Read SPEC.md before making any code changes.
- SPEC.md defines the functional requirements.
- All implementations must follow SPEC.md.
- Do not change requirements without approval.
- Update README.md when functionality changes.
- Verify implemented functionality against SPEC.md.
