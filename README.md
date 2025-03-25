# CRISPR-MFH
| Model              | Parameters | Model size(KB) | ETA of one epoch | Hardware Required                 | Deployment Feasibility                     |
|--------------------|------------|----------------|------------------|------------------------------------|--------------------------------------------|
| CRISPR-BERT        | 10,175,277 | 119,501        | 13 minutes       | High-end GPU (A100/V100), TPU      | Unsuitable for edge/embedded deployment    |
| CRISPR-M           | 1,196,417  | 15,219         | 2 minutes        | Mid-to-high-end GPU (RTX 3060+)    | Suitable for GPU-based servers             |
| CRISPR-DNT         | 589,190    | 6,503          | 1.2 minutes      | Mid-end GPU (RTX 2060+)            | Suitable for GPU-based servers             |
| CRISPR-MFH (Ours)  | 125,030    | 1,677          | 20 seconds       | CPU / Embedded Devices             | Ideal for lightweight, real-time applications |
