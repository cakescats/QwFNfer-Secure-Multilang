name: Feature request
description: Ask for something the Windows port should do
labels: [enhancement, windows]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem
      description: What can't you do today, and what did you try instead?
    validations:
      required: true
  - type: textarea
    id: proposal
    attributes:
      label: Proposal
      description: What should happen, ideally with the console/API surface it touches
  - type: dropdown
    id: area
    attributes:
      label: Area
      options:
        - Installer / release bundle
        - Console (Serve, Chat, Stats, Log)
        - Engine I/O and tiers on Windows
        - Docs (SETUP.md, BUILD_WINDOWS.md, README)
        - Tool integrations (Claude Code, Open WebUI, other clients)
        - Other
  - type: checkboxes
    id: scope
    attributes:
      label: Scope check
      options:
        - label: This is Windows-port scope (upstream model/Linux behavior goes to Apolog1ze-Dev/QwFNfer)
