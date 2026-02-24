# Research Pipeline

`run_research_pipeline.sh` is the orchestration entrypoint.  
`pipeline_config.yaml` is the single source of truth for pipeline settings.

## Quick Start

```bash
# From repository root
./run_research_pipeline.sh all
```

## Canonical Defaults

- Dataset path: `plantvillage_ssl/data/plantvillage`
- Split: `20%` labeled, `10%` validation, `10%` test, `60%` SSL stream
- Labeled ratio: `0.20`

## Useful Commands

| Command | Purpose |
|---|---|
| `./run_research_pipeline.sh all` | Full pipeline |
| `./run_research_pipeline.sh train` | Training stages only |
| `./run_research_pipeline.sh ssl` | SSL stage only |
| `./run_research_pipeline.sh benchmark` | Benchmark stage only |
| `./run_research_pipeline.sh config` | Print resolved config |
| `./run_research_pipeline.sh clean` | Remove generated outputs |

## Common Overrides

```bash
./run_research_pipeline.sh all --epochs 20
./run_research_pipeline.sh train --epochs 10 --batch-size 64
./run_research_pipeline.sh all --dry-run --verbose
```

## Notes

- The dataset download stage uses `./download_plantvillage.sh`.
- Generated artifacts are written under `output/research_pipeline/`.
- For detailed knobs, edit `pipeline_config.yaml` directly.
