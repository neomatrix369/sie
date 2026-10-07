# Measure four independent agent stages

Prepare a fixed set of public requests, explicitly run it on any SIE HTTP(S)
endpoint, and score client wall times offline. The four stages are independent
workloads. This example measures each stage; it does not measure a full agent
turn or establish quality parity on a new endpoint.

| Stage | SIE operation | Paired rival | Frozen public workload |
| --- | --- | --- | --- |
| G: guardrails | Qwen3Guard 4B, `chat_completions`, raw user text, temperature 0, 64 output tokens | Haiku 4.5, exact short-verdict user prompt, temperature 0, 10 output tokens, no thinking | Human-labelled ToxicChat and Aegis test prompts from [guardrails](../guardrails) |
| M: caller masking | GLiNER PII + NuNER Zero, `extract`, 36 ordered labels, 300-unit windows/50 overlap, score ≥ 0.6, union, name propagation and bracket masking | Haiku 4.5, recorded PII system prompt and JSON schema, temperature 0, 4096 output tokens, no thinking; exact mentions mapped and masked in the caller | 660 frozen Gretel English documents from [redact](../redact) |
| E: one-query embedding | Qwen3-Embedding 4B, one `Item`, `is_query=True`, dense 2560 dimensions | OpenAI `text-embedding-3-large`, one raw input string, float encoding, 3072 dimensions | 440 held-out questions from [lookalike-search](../lookalike-search); no corpus encoding or retrieval |
| R: reranking | Qwen3-Reranker 4B, `score`, one question and all 20 candidates in their original order; both w2 rules as `instruction` | Cohere `rerank-v4.0-pro`, `Instruction: {rule}\nQuery: {question}`, all 20 documents | 499 frozen questions from [rerank](../rerank), each with final-rule and proposed-rule requests |

[sources.json](sources.json) pins exactly nine files to immutable public
Hugging Face revisions, byte counts and SHA-256 hashes: four evidence
manifests, three evidence input files, and the two guardrails source files.
It also records the request templates and their canonical digests. Downloads
are anonymous. No recorded vectors or rival replies are needed for timing.
ToxicChat is CC-BY-NC-4.0, Aegis is CC-BY-4.0, Gretel is Apache-2.0, and the
rerank evidence records Apache-2.0/public-domain Federal Register provenance.
Embedding-question provenance and source terms are linked in `sources.json`.
Downloaded workload text can contain harmful prompts and synthetic personal
data; these remain workload inputs.

Cohere's historical HTTP body/API version was not retained. This new trial
uses the [documented v2 rerank request](https://docs.cohere.com/v2/reference/rerank)
with explicit `top_n=20` and `max_tokens_per_doc=4096`. The historical redaction
409 ms figure timed two concurrent whole-document extractions. The windowed,
composed masking operation here is a new measurement.

## Run from the repository root

Use the locked workspace SDK and managed tools. Fetch, prepare and score use
only Python's standard library and require no API keys. A localhost SIE
endpoint can run without authentication.

```sh
mise exec -- uv sync --frozen --no-dev --package sie-sdk

mise exec -- python examples/agent-stage-latency/fetch.py \
  --out examples/agent-stage-latency/inputs/trial-1

mise exec -- python examples/agent-stage-latency/prepare.py \
  --inputs examples/agent-stage-latency/inputs/trial-1 \
  --out examples/agent-stage-latency/packets/trial-1.json \
  --phase confirmatory --n 8 --warmup 1 --seed 20261004 \
  --stages G M E R --arms sie \
  --request-timeout 30 --wall-limit 900 --placement-label local-demo

mise exec -- python -m json.tool examples/agent-stage-latency/packets/trial-1.json

mise exec -- uv run --frozen --project . --no-sync --package sie-sdk python \
  examples/agent-stage-latency/run.py \
  --packet examples/agent-stage-latency/packets/trial-1.json \
  --out examples/agent-stage-latency/runs/trial-1.jsonl \
  --sie-base-url http://127.0.0.1:8000 --execute

mise exec -- python examples/agent-stage-latency/score.py \
  --packet examples/agent-stage-latency/packets/trial-1.json \
  --journal examples/agent-stage-latency/runs/trial-1.jsonl \
  --out examples/agent-stage-latency/reports/trial-1.json
```

The illustrative `n=8` is not a statistical recommendation. Before a
confirmatory run, externally choose and freeze n, seed, stages/order, phase,
arm mode, placement label and timing budgets. The placement label must be a
nonsecret name using letters, digits, dots, underscores or hyphens. Supply the
actual endpoint with `--sie-base-url` or `SIE_BASE_URL`; there is no default
endpoint. `SIE_API_KEY` is optional and is passed explicitly to this origin.
URLs with userinfo, query strings or fragments are rejected.

For paired runs prepare with `--arms paired`. Only selected rivals need their
credentials: `ANTHROPIC_API_KEY` for G/M, `OPENAI_API_KEY` for E and
`COHERE_API_KEY` for R. Missing selected credentials fail before inference.
Set keys through your environment, never in packets or command arguments.
Preparation and scoring never contact these providers. Running without
`--execute` fails before constructing a client.

Every output destination must be fresh, including an empty existing download
directory. The example never replaces inputs, packets, journals or reports.
The runner creates `trial-1.jsonl.end.json` beside its journal. Preserve both
files: the exclusive terminal record binds the full journal bytes to the
packet, including any partial final line left by a deadline or interrupt.
Files are flushed and synced before their publication directories; newly
created parent directories and download renames are synced too. Directory
syncing applies on POSIX filesystems that support it. Storage and filesystem
behavior still govern survival after power loss.

## Pilot and disjoint confirmation

A separately prepared pilot can inform an externally selected confirmatory n.
Use a fresh output and exclude the whole earlier packet, including its
warmups. `--exclude-packet` prevents selecting those base identities or semantic
inputs again. Reusing a seed alone does not exclude earlier requests. Multiple
`--exclude-packet` arguments are supported; include every prior packet in this
study.

```sh
mise exec -- python examples/agent-stage-latency/prepare.py \
  --inputs examples/agent-stage-latency/inputs/trial-1 \
  --out examples/agent-stage-latency/packets/pilot-1.json \
  --phase pilot --n 4 --warmup 1 --seed 20261004 \
  --stages G M E R --arms paired \
  --request-timeout 30 --wall-limit 900 --placement-label local-demo

mise exec -- python examples/agent-stage-latency/prepare.py \
  --inputs examples/agent-stage-latency/inputs/trial-1 \
  --out examples/agent-stage-latency/packets/confirm-1.json \
  --exclude-packet examples/agent-stage-latency/packets/pilot-1.json \
  --phase confirmatory --n 16 --warmup 1 --seed 20261004 \
  --stages G M E R --arms paired \
  --request-timeout 30 --wall-limit 1800 --placement-label local-demo
```

Run and score each with its own packet/journal paths using the commands above.
Never extend n by repeatedly inspecting ordinary confidence intervals. If n
exceeds the remaining distinct population, preparation fails before inference.

n counts base evidence units per selected stage. R creates two observations
per question; M creates one extraction per model per window, with no redundant
whole-document call for long texts. The packet prints exact planned semantic
call counts, persists every envelope/window/offset and freezes within-case arm
order using the seed. Cases and paired arms execute serially. M alone runs two
concurrent model paths, with a separate client for each model and serial
windows inside each path. No repetitions or automatic sample extension occur.

## Read the report

The journal syncs metadata and the complete frozen `operation_intent` before
any constituent dispatch. During that operation, a short-lock memory buffer
captures call intents, physical HTTP attempts, replies, stable errors, elapsed
durations and observable SDK retry counts. After the stage timer stops, those
snapshots are replayed in their original interleaving through the fsynced hash
chain. An `operation_result` seals the complete captured event count and is
synced before the next operation. No background writer is used. An SDK per-item
error, malformed verdict, bad vector or invalid ranking remains a failure.
Ordinary failed M operations join both paths and preserve earlier subcalls.
There is no outer retry campaign: capacity waits and OOM retries are disabled.
The SDK can still retry admission/model-loading responses; those attempts and
their full elapsed time remain recorded.

One declared request timeout sets the connect/read/provision component budgets.
The remaining wall budget caps call budgets. A parent-owned process enforces
the total child deadline, including blocked requests and SDK retry sleeps,
then stops/reaps the child. Terminal-record writing and reaping add a small
amount of cleanup time after that deadline. Discovery and client setup are
outside measured operations. Stage timing includes request building/windowing,
transport, SDK processing, reply projection/sanitization/validation, caller
composition/masking and memory event capture. Durable journal replay is outside
stage and call timing but inside the total wall budget. Constituent call timers
start after intent capture and stop after body processing and validation,
before result capture. No calibration value is subtracted from these timers.
Each physical `response.headers_elapsed_s` stops at the response headers,
before consuming the body; it starts after dispatch event capture.
It differs from `call_result.elapsed_s`, which includes body consumption and
SDK processing, and from the complete-stage `operation_result.elapsed_s`.
The scorer requires the frozen next window in each M model path and checks
that the stage duration covers each path's sum of completed call durations,
including failed calls. The two paths may interleave. The timing comparison
allows only 1 microsecond absolute or 1e-9 relative floating-point tolerance.

Discovery must finish before any operation; failed/unsupported or empty
catalogs remain valid discovery outcomes. Missing response model IDs are
allowed. An explicit SIE ID must equal the frozen requested ID. For Anthropic,
the frozen `claude-haiku-4-5` alias may return that ID or a dated snapshot
`claude-haiku-4-5-YYYYMMDD`; other explicit provider IDs must equal their frozen
request ID. This identity check is independent of weights/execution revisions.

Only selected model names, dimensions and safe revision digests are retained
from metadata discovery. Catalog `weights_revision` and response
`execution_revision` are separate fields; the latter may be a gateway bundle
digest. Different or missing revisions are accepted and reported. Responses
also retain model IDs and execution identity/binding digests when supplied.
New endpoint timing does not reproduce historical quality or deployment
identity. Auth headers, arbitrary catalog telemetry and exception messages
are omitted; exact configured credentials echoed in replies are replaced by
`[REDACTED_CREDENTIAL]`.

Scoring validates packet/request bindings, the journal hash chain, observation
and subcall identities, unique terminals, execution order and reply shapes.
It reports planned, observed attempted, successful, failed, unresolved,
unattempted and `attempt_status_unknown` counts per phase/stage/arm, with the
fixed denominator beside latency results. A stop before or during replay can
lose replies, calls or retries that occurred. Missing call intents in a started
operation without a complete seal are attempt-status-unknown; its observed
attempt count is a lower bound when `attempted_count_exact` is false. A sealed
failed operation permits its never-started windows to be known unattempted.
Later operations without durable intent are unattempted. Physical-dispatch and
SDK-retry totals are observed lower bounds whenever their corresponding
`physical_dispatches_exact` or `sdk_retries_exact` flag is false; zero observed
does not establish zero actual activity. Completed captured call records remain
diagnostic evidence even when the operation seal is absent. Capture or replay
errors prevent a seal; any persistence error stops the child before another
operation. A complete seal row can remain visible if its own sync fails, but
the child failure still makes the run nonqualifying.
Successful-operation p50/p90 and all-terminal elapsed p50/p90 are separate.
Partial runs remain incomplete and nonqualifying. A missing earlier outcome
is never repaired into a better history. The versioned timing/checkpoint policy
is bound into `protocol_digest` and included in the report; prepare a fresh
packet after changing the runner. Packets from earlier timing policies are
rejected, so those measurements cannot silently mix.

Confirmatory comparisons use only successful, complete base-case pairs. R's
two rule variants stay together; if any variant fails, that whole base cluster
is excluded. Warmup and pilot observations never enter the estimate. At least
two complete base clusters are required. A seeded paired percentile bootstrap
resamples entire base clusters and estimates `rival median - SIE median` in
seconds and `rival median / SIE median`; a ratio above one favors SIE. Defaults
are 5000 resamples and seed 1729, both frozen in the packet, with linear
quantiles at `(n-1)p`. Excluded pairs/clusters and failure counts accompany
each interval. SIE-only packets have no paired rival estimate.

The paired report also includes each arm's absolute median and its 95%
percentile interval from those same paired base-cluster resamples. These use
the complete paired cohort, distinct from each arm's all-successful p50/p90.
`qualifying` means the run is complete and its comparison is statistically
estimable under this protocol; it does not mean a latency, quality or power
objective was met. No target or win rule is imposed by this example.

Interpret stages independently because their cohorts and placements differ.
Any arithmetic combination is a **sum of stage medians**, with those limits;
it is not a measured median full agent turn, model-compute latency, throughput,
quality parity or cost optimization.

## Validate without inference

After the locked SDK and root dev tools are installed, run from the repository
root. Tests use tiny embedded fixtures and localhost fake HTTP servers; no
weights or paid providers are contacted.

```sh
mise exec -- uv run --frozen --project . --no-sync ruff format --check examples/agent-stage-latency
mise exec -- uv run --frozen --project . --no-sync ruff check --select E,F,I,UP,B examples/agent-stage-latency
mise exec -- uv run --frozen --project . --no-sync ty check examples/agent-stage-latency
mise exec -- uv run --frozen --project . --no-sync pytest -c pyproject.toml -q examples/agent-stage-latency/tests
```

Canonical identities use UTF-8 JSON with sorted keys, compact separators,
unescaped Unicode, no NaN and no newline. Digest fields hash the object before
their own insertion. The packet binds source pins and the checked-in protocol,
preparation and execution code. Retain that code revision to regenerate
byte-identical packets and score an earlier run.
