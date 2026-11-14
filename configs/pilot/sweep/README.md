# Pilot sweep (deliverable 5)

Goal: a no-mitigation setting where retention loss on the protected set is unmistakable but
graded. Plan: three learning rates at 50M tokens/domain (cut from 100M to fit the pilot ceiling), then one 200M run at the chosen lr.
Each config inherits `configs/pilot/base.yaml`; only lr / tokens change. Firm GPU-hour estimates
come from the cuda-check throughput measurement; nothing here runs before that estimate is
approved. Reversed-order robustness run is a later Phase 1 item, not part of the pilot.
