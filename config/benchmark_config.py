SHARED_SYSTEM_PROMPT_BASE = """

You are the production LLM inference gateway's senior serving engineer.
Your responsibility is to make technically correct, reliable, and measurable
recommendations for deploying language models behind an asynchronous API.

REQUEST ROUTING AND PRIORITIES
Classify requests by their declared priority and validate request parameters
before forwarding them to the inference engine. Preserve request identifiers
across gateway and engine logs. Apply concurrency limits to protect the engine.
High-priority requests may bypass occupancy-based admission rejection, but they
must still obey configured concurrency limits, deadlines, and safety controls.
Never claim that priority guarantees lower latency under every workload.

PREFIX CACHING AND RADIXATTENTION
The gateway receives requests that share a common reference system prompt.
Place the identical system prompt at the beginning of every shared-prefix
request. Keep its wording, whitespace, punctuation, and ordering unchanged.
Append request-specific questions only after the shared prefix. Explain that
RadixAttention can reuse matching cached prefix key-value states when token
prefixes match and cached entries remain available. Do not assume that similar
meaning or similar wording guarantees a cache hit. Distinguish cache reuse from
model-weight caching and distinguish cached tokens from active KV allocations.

KV CACHE AND BACKPRESSURE
KV-cache capacity is a serving resource distinct from physical GPU memory.
Interpret engine metrics according to their documented meanings. If capacity
and available-token metrics are valid, calculate occupancy as one minus the
available-token fraction. Reject standard-priority requests with HTTP 503 when
verified occupancy reaches or exceeds the configured 90 percent threshold.
Return a retry hint when appropriate. If occupancy cannot be verified, apply
the configured fail-closed policy to standard requests and record the reason.
Do not represent missing metrics as zero occupancy. Report high-priority
admission separately from successful inference.

LATENCY AND THROUGHPUT
Define time to first token as the elapsed time until the first generated token.
Inter-token latency measures intervals between successive generated tokens.
Separate queueing delay, prompt prefill, and autoregressive decode when data
allows. Report p50, p95, and p99 latency, request throughput, and generated
token throughput with units and sample counts. Compare equivalent workloads,
including prompt lengths, output limits, concurrency, warmup, and hardware.
Do not infer causality from a single latency observation.

STRUCTURED OUTPUT
For schema-constrained requests, pass the supported JSON schema constraint to
the inference engine. Require a JSON object with the requested fields and types.
Validate the returned text independently before accepting it. Distinguish
engine-side constrained decoding from gateway-side validation. Treat malformed
JSON, missing required fields, unexpected fields, and type mismatches as
validation failures. Never silently repair an invalid result and call it valid.

OBSERVABILITY AND FAILURE HANDLING
Record request identifiers, priorities, response statuses, latency, token
counts when available, cache metrics, and failure reasons. Avoid logging secrets
or unnecessary user data. Apply explicit timeouts and handle engine errors
without leaking internal exceptions. Keep metric names and units consistent.
Distinguish rejected requests, timed-out requests, engine failures, and successful
responses. Avoid retry storms and uncontrolled request amplification.

BENCHMARKING AND EVIDENCE
Use reproducible workloads and preserve raw per-request measurements. Compare
unique-prefix requests against requests with a shared leading prefix. Measure
cache reuse directly where supported and compare latency distributions rather
than relying on isolated examples. Capture GPU memory, KV-cache indicators,
request success rates, and engine versions. State when a metric is unavailable.
Separate observed measurements from hypotheses and expected behavior. Never
invent benchmark results, cache hits, performance improvements, or measurements.

RESPONSE STANDARD
Answer the specific question using concise, technically precise explanations.
Identify relevant assumptions, trade-offs, failure modes, and useful measurements.
Do not confuse a design intention with a verified implementation. If evidence
does not establish a conclusion, state what remains uncertain and how to test it.
"""

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "domain": {"type": "string"},
        "summary": {"type": "string"},
        "score": {"type": "integer"},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["request_id", "domain", "summary", "score", "risks"],
    "additionalProperties": False,
}

DOMAINS = [
    "GPU memory management",
    "continuous batching",
    "KV-cache optimization",
    "quantization and accuracy",
    "latency and throughput SLAs",
]


SCENARIO_A_DOMAINS = [
    ("Astronomy", "Explain how stellar-mass black holes form and how astronomers detect them."),
    ("Cooking", "Diagnose why sourdough bread may have a dense crumb and recommend remedies."),
    ("Personal Finance", "Compare fixed and variable interest rates for a long-term loan."),
    ("Biology", "Explain photosynthesis and the environmental factors that affect its rate."),
    ("Ancient History", "Analyze the economic factors contributing to the decline of the Roman Empire."),
    ("Cybersecurity", "Design a practical approach to identifying phishing emails."),
    ("Agriculture", "Compare drip irrigation and sprinkler irrigation for water conservation."),
    ("Climate Science", "Explain why urban heat islands form and how cities can reduce them."),
    ("Mathematics", "Explain Bayesian probability using a practical medical-testing example."),
    ("Music", "Compare improvisation techniques in jazz and classical music."),
    ("Architecture", "Explain how passive building design reduces energy consumption."),
    ("Marine Biology", "Describe how coral bleaching occurs and its ecological consequences."),
    ("Geology", "Explain how tectonic plate movements produce earthquakes."),
    ("Psychology", "Describe how confirmation bias affects everyday decision-making."),
    ("Education", "Compare retrieval practice with passive rereading for learning."),
    ("Linguistics", "Explain how children acquire grammatical structures."),
    ("Economics", "Describe how inflation affects purchasing power and household budgets."),
    ("Supply Chains", "Explain how demand forecasting can reduce inventory shortages."),
    ("Renewable Energy", "Compare the operational trade-offs of solar and wind power."),
    ("Wildlife Conservation", "Develop a strategy to protect an endangered animal population."),
    ("Public Health", "Explain how vaccination coverage affects disease transmission."),
    ("Nutrition", "Explain the roles of protein, carbohydrates, and fats in a balanced diet."),
    ("Transportation", "Compare electric buses with diesel buses for urban public transport."),
    ("Oceanography", "Explain how ocean currents influence regional climates."),
    ("Photography", "Explain how aperture, shutter speed, and ISO affect an exposure."),
    ("Literature", "Analyze how an unreliable narrator changes a reader's interpretation."),
    ("Archaeology", "Explain how archaeologists establish the age of an artifact."),
    ("Law", "Explain the difference between civil liability and criminal liability."),
    ("Ethics", "Analyze the ethical trade-offs of using facial recognition in public spaces."),
    ("Statistics", "Explain the difference between correlation and causation."),
    ("Botany", "Describe how plants transport water from roots to leaves."),
    ("Meteorology", "Explain the atmospheric conditions that produce thunderstorms."),
    ("Sports Science", "Explain how progressive overload supports strength development."),
    ("Urban Planning", "Recommend measures to reduce traffic congestion in a growing city."),
    ("Geography", "Explain how river erosion shapes valleys over time."),
    ("Robotics", "Compare the uses of sensors and actuators in an autonomous robot."),
    ("Materials Science", "Explain how heat treatment changes the properties of steel."),
    ("Food Safety", "Describe practices that reduce cross-contamination in a kitchen."),
    ("Space Exploration", "Explain the main engineering challenges of a Mars landing."),
    ("Behavioral Economics", "Explain how loss aversion influences consumer decisions."),
    ("Waste Management", "Compare composting with anaerobic digestion for organic waste."),
    ("Genetics", "Explain how dominant and recessive alleles influence inherited traits."),
    ("Cultural Anthropology", "Explain how rituals can reinforce social identity."),
    ("Water Engineering", "Describe the main stages of municipal drinking-water treatment."),
    ("Geophysics", "Explain how seismic waves help scientists study Earth's interior."),
    ("Business Strategy", "Compare market penetration with product diversification."),
    ("Game Theory", "Explain the prisoner's dilemma and its strategic implications."),
    ("Textile Manufacturing", "Explain how fiber selection affects fabric durability."),
    ("Emergency Management", "Develop a practical checklist for preparing for severe flooding."),
    ("Optics", "Explain how lenses form images and why focal length matters."),
]

SCENARIO_B_QUESTIONS = {
    "GPU memory management": [
    "How should GPU memory utilization be monitored during a burst of long-context requests?",
    "What causes GPU out-of-memory errors when request concurrency increases?",
    "How can memory fragmentation affect inference reliability?",
    "How should model weights and KV-cache memory be budgeted?",
    "What signals indicate that GPU memory pressure is approaching an unsafe level?",
    "How can request admission limits reduce memory exhaustion?",
    "What trade-offs arise when increasing the maximum context length?",
    "How should memory headroom be chosen for variable-length requests?",
    "How can operators distinguish persistent memory allocation from active KV-cache usage?",
    "What should the gateway do when available GPU memory falls sharply?",
    "How does batch size affect dynamic KV-cache consumption?",
    "How can memory monitoring reveal inefficient request scheduling?",
    "What are the risks of setting GPU memory utilization too aggressively?",
    "How should memory usage be measured during sustained load?",
    "How can long-running requests affect memory available to new requests?",
    "What evidence is needed to diagnose a memory-related latency spike?",
    "How should a serving system respond to repeated allocation failures?",
    "How can context-length limits improve memory predictability?",
    "What memory metrics should be included in an inference operations dashboard?",
    "How can GPU memory headroom be validated before deployment?",
    ],
    "continuous batching": [
    "How does continuous batching improve utilization when requests finish at different times?",
    "What happens when incoming request rate exceeds serving capacity?",
    "How should batch scheduling balance throughput and latency?",
    "What measurements show that requests are being processed in overlapping batches?",
    "How can long generations affect short requests waiting in the queue?",
    "What trade-offs arise when increasing the maximum running-request count?",
    "How should queue depth and active-request count be interpreted together?",
    "How can variable prompt lengths affect batch efficiency?",
    "What causes throughput to stop increasing as concurrency rises?",
    "How should a scheduler treat requests with different output lengths?",
    "How can request admission control protect batch stability?",
    "What evidence distinguishes client concurrency from engine concurrency?",
    "How can a burst of arrivals affect scheduling delay?",
    "What are the risks of allowing an unbounded request queue?",
    "How should batch performance be evaluated under mixed workloads?",
    "How can short requests avoid excessive waiting behind long generations?",
    "What metrics reveal scheduler saturation?",
    "How should throughput and tail latency be balanced?",
    "How can repeated tests establish a reliable batching baseline?",
    "What operational changes are justified when batching efficiency falls?",
    ],
    "KV-cache optimization": [
    "How does prefix caching reduce repeated prefill computation?",
    "How can cached-token metadata demonstrate actual prefix reuse?",
    "Why might requests with a common textual prefix show few cached tokens?",
    "How can cache eviction affect a burst of shared-prefix requests?",
    "What is the relationship between KV-cache occupancy and admission control?",
    "How should cold-prefix and warm-prefix latency be compared fairly?",
    "How can request-specific text accidentally reduce shared-prefix reuse?",
    "What metrics help distinguish cache reuse from ordinary batching gains?",
    "How does a longer shared prefix affect memory consumption?",
    "What happens when many unique prefixes compete for limited cache space?",
    "How can cache hit-rate gauges be misinterpreted?",
    "What evidence confirms that all requests use identical prefix tokens?",
    "How should cache behavior be tested after an engine restart?",
    "How can prefix caching affect time to first token?",
    "What trade-offs exist between prefix reuse and cache memory pressure?",
    "How can an experiment isolate the effect of shared-prefix caching?",
    "What does a low cached-token count indicate, and what does it not prove?",
    "How should cache metrics be sampled during concurrent traffic?",
    "How can prompt formatting change token-level prefix identity?",
    "What cache measurements should accompany a performance claim?",
    ],
    "quantization and accuracy": [
    "How should quantized inference throughput be compared with a higher-precision baseline?",
    "What quality checks are needed before deploying four-bit quantization?",
    "How can quantization reduce memory requirements?",
    "What performance bottlenecks can remain after model weights are quantized?",
    "How should perplexity changes be interpreted when evaluating quantization?",
    "What is the difference between weight quantization and KV-cache quantization?",
    "How can workload shape affect the measured benefit of quantization?",
    "What evidence is needed to claim that quantization improves throughput?",
    "How should numerical quality regressions be detected?",
    "What trade-offs arise between memory savings and output quality?",
    "How can latency measurements be distorted by inconsistent generation lengths?",
    "What representative prompts should be used in quantization evaluation?",
    "How should quantized model compatibility be validated on target hardware?",
    "What conditions might cause quantization to provide little speedup?",
    "How can GPU memory measurements support a quantization decision?",
    "What should be recorded when comparing two quantization formats?",
    "How can output consistency be evaluated across repeated runs?",
    "Why should model loading time be separated from steady-state inference?",
    "How can throughput and quality be presented together in a report?",
    "What rollback criteria should govern a quantized model deployment?",
    ],
    "latency and throughput SLAs": [
    "How should p50, p95, and p99 latency guide an inference SLA?",
    "What is the difference between time to first token and end-to-end latency?",
    "How should a gateway behave when the engine approaches saturation?",
    "How can queueing delay be distinguished from model execution time?",
    "What evidence is needed to calculate successful request throughput?",
    "How should HTTP 503 responses be included in reliability reporting?",
    "What is the operational value of a Retry-After response header?",
    "How can a burst workload expose tail-latency problems?",
    "What trade-offs arise when tightening request timeouts?",
    "How should an error budget be applied to an inference endpoint?",
    "How can metrics distinguish rejected requests from failed generations?",
    "What measurements are needed to compare warm and cold requests?",
    "How should a serving gateway handle a temporarily unavailable engine?",
    "What is the difference between offered load and completed throughput?",
    "How can client-side concurrency distort performance conclusions?",
    "How should latency histograms be interpreted when buckets are coarse?",
    "What metrics should trigger overload protection?",
    "How can an SLA remain meaningful when output lengths vary?",
    "What evidence supports a claim that backpressure protects the engine?",
    "How should a benchmark report disclose measurement limitations?",
    ],
}