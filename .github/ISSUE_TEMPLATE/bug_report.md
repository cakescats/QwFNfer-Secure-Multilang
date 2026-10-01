name: Bug report
description: Something broken on the Windows port
labels: [bug, windows]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting. Upstream Linux/engine issues belong at Apolog1ze-Dev/QwFNfer — this tracker is for the Windows port. Fill in what you can; logs beat adjectives.
  - type: input
    id: version
    attributes:
      label: Version
      description: Release zip name or commit hash (e.g. windows-v1, or the short SHA from qwfnfer's footer)
    validations:
      required: true
  - type: textarea
    id: env
    attributes:
      label: Machine
      description: Windows version, GPU + driver (paste nvidia-smi), RAM, drive type + where the model file sits
      placeholder: |
        Windows 11 24H2, RTX 4070 Ti SUPER 16 GB, driver 581.42, 32 GB RAM, 2 TB NVMe (model on D:\models)
    validations:
      required: true
  - type: textarea
    id: repro
    attributes:
      label: What you did
      description: Quant, tier, exact steps. The Serve banner line names every flag — paste it.
      placeholder: |
        UD-Q4_K_XL, Agentic coding, pressed Auto-tune & start. Banner: qwfn-server ... --ram 13 --ctx 131072 ...
    validations:
      required: true
  - type: textarea
    id: actual
    attributes:
      label: What happened (and what you expected)
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Logs
      description: Tail of %LOCALAPPDATA%\qwfn-console\server.log, /stats output while running, and any popup/dialog text
      render: text
  - type: checkboxes
    id: checks
    attributes:
      label: Pre-flight
      options:
        - label: Only one engine was running (no second qwfn-server, checked Task Manager)
        - label: The model files verify (re-ran the same hf download, it resumed/nothing missing)
        - label: This is a Windows-port issue, not upstream model/Linux behavior
