"""Self-contained evaluation package.

The evaluation harness lives here so that the divergence oracle and the
scripted executor cannot share code, state, or assumptions with the protocol
under test (``scp.py`` / ``SemanticVerifier``). The executor and the oracle each
re-implement the operational semantics of the guarantees independently.

This evaluation is a controlled simulation designed to isolate the effect of
semantic verification from other sources of variability. It does not measure
real LLM agent behavior. Extending this to LLM-based agents and network
constraints is left as future work.
"""
