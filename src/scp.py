from __future__ import annotations
import json, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from difflib import SequenceMatcher
import re

# 1. MODELO DE CONTRATOS SEMÁNTICOS

class ContractState(Enum):
    PROPOSED  = "proposed"
    COUNTERED = "countered"
    """Reserved for counter-offers. The present implementation does not
    generate or transition to this state; renegotiation is modeled as
    re-entering the four phases with a revised task (see methodology.tex)."""
    BOUND     = "bound"
    REJECTED  = "rejected"

@dataclass
class SemanticGuarantee:
    """Compromiso explícito sobre el comportamiento de una capacidad"""
    name: str
    value: Any
    description: str = ""

@dataclass
class CapabilityContract:
    """QUÉ ofrece un agente y BAJO QUÉ garantías semánticas"""
    agent_id: str
    capability_name: str
    description: str
    inputs:  dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    guarantees: list[SemanticGuarantee] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state: ContractState = ContractState.PROPOSED

    def guarantee_map(self) -> dict[str, SemanticGuarantee]:
        return {g.name: g for g in self.guarantees}

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "agent_id": self.agent_id,
            "capability_name": self.capability_name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "guarantees": [{"name": g.name, "value": g.value} for g in self.guarantees],
            "constraints": self.constraints,
            "state": self.state.value,
        }

@dataclass
class TaskRequest:
    """Especificación de lo que el agente solicitante requiere"""
    task_description: str
    target_capability: str
    required_guarantees: list[SemanticGuarantee] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_text: str = ""

@dataclass
class Divergence:
    kind: str # missing_guarantee | conflicting_guarantee | constraint_violation
    field: str
    required: str
    offered: str
    severity: str # high | medium | low
    explanation: str = ""

@dataclass
class DivergenceReport:
    task_id: str
    contract_id: str
    divergences: list[Divergence] = field(default_factory=list)
    textual_similarity: float = 0.0

    @property
    def compatible(self) -> bool:
        return not any(d.severity == "high" for d in self.divergences)

    def summary(self) -> str:
        lines = [
            "  ┌- DivergenceReport",
            f"  │  task_id            = {self.task_id}",
            f"  │  contract_id        = {self.contract_id}",
            f"  │  textual_similarity = {self.textual_similarity:.2f}  "
            f"(señal secundaria, NO decisiva)",
            f"  │  compatible         = {self.compatible}",
            f"  │  divergences ({len(self.divergences)}):",
        ]
        for d in self.divergences:
            lines.append(
                f"  │    [{d.severity.upper():6s}] {d.kind}: '{d.field}' "
                f"req={d.required!r} off={d.offered!r}"
            )
            if d.explanation:
                lines.append(f"  │             └- {d.explanation}")
        lines.append("  └-")
        return "\n".join(lines)


# 2. VERIFICADOR SEMÁNTICO  (núcleo del protocolo)

class SemanticVerifier:

    def _values_compatible(self, required: Any, offered: Any) -> bool:
        if isinstance(required, bool):
            return bool(offered) >= bool(required)          
        if isinstance(required, (int, float)) and isinstance(offered, (int, float)):
            return offered >= required                      
        if isinstance(required, str):
            r, o = required.lower(), str(offered).lower()
            return r == o or r in o or o in r
        return required == offered

    def verify(self, task: TaskRequest, contract: CapabilityContract) -> DivergenceReport:
        divs: list[Divergence] = []

        # (a) Garantías semánticas
        req_g = {g.name: g for g in task.required_guarantees}
        off_g = contract.guarantee_map()
        for name, req in req_g.items():
            if name not in off_g:
                divs.append(Divergence(
                    kind="missing_guarantee", field=name,
                    required=str(req.value), offered="<none>", severity="high",
                    explanation=req.description or f"El proveedor no declara '{name}'.",
                ))
            elif not self._values_compatible(req.value, off_g[name].value):
                divs.append(Divergence(
                    kind="conflicting_guarantee", field=name,
                    required=str(req.value), offered=str(off_g[name].value),
                    severity="high", explanation=off_g[name].description,
                ))

        # (b) Restricciones operativas (latencia, idioma, etc.)
        for key, val in task.constraints.items():
            if key not in contract.constraints:
                divs.append(Divergence(
                    kind="missing_constraint", field=key,
                    required=str(val), offered="<none>", severity="medium",
                    explanation=f"El contrato no especifica '{key}'.",
                ))
                continue
            off = contract.constraints[key]
            if key.endswith("_ms") and isinstance(off, (int, float)) \
               and isinstance(val, (int, float)) and off > val:
                divs.append(Divergence(
                    kind="constraint_violation", field=key,
                    required=str(val), offered=str(off), severity="medium",
                    explanation="El proveedor excede el límite declarado",
                ))

        # (c) Similitud textual — solo informativa
        sim = SequenceMatcher(
            None,
            task.task_description.lower(),
            contract.description.lower(),
        ).ratio()

        return DivergenceReport(task.request_id, contract.contract_id, divs, sim)

# 3. AGENTES Y REGISTRO

class Agent:
    def __init__(
        self,
        agent_id: str,
        framework: str,
        model: str,
        latent_profile: Optional[LatentProfile] = None,
    ):
        self.agent_id, self.framework, self.model = agent_id, framework, model
        self.latent_profile = latent_profile
        self._caps: dict[str, CapabilityContract] = {}

    def register_capability(self, c: CapabilityContract) -> None:
        self._caps[c.capability_name] = c

    def get_contract(self, name: str) -> Optional[CapabilityContract]:
        return self._caps.get(name)

    def list_capabilities(self) -> list[str]:
        return list(self._caps.keys())

    def __repr__(self) -> str:
        return f"Agent({self.agent_id} | fw={self.framework} | model={self.model})"


class Registry:
    """Registro de agentes y descubrimiento de capacidades (fase Discovery)"""
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.agent_id] = agent

    def discover(self, capability: str) -> list[Agent]:
        return [a for a in self._agents.values() if capability in a.list_capabilities()]


# 3.5 EJECUTOR GUIADO (reemplaza el stub de ejecución)
# Independiente del SemanticVerifier: produce una salida determinista a partir
# del perfil latente L(p) del proveedor y de la fuente de la tarea.

@dataclass
class LatentProfile:
    """Perfil latente L(p): lo que el proveedor realmente hace al ejecutar.

    Es independiente del contrato declarado y permite simular interpretaciones
    divergentes de una misma capacidad.
    """

    preserve_dates: bool = False
    preserve_glossary: bool = False
    max_tokens: int = 200
    language: str = "en"


_GLOSSARY: list[str] = [
    "clause", "liability", "jurisdiction", "termination",
    "indemnity", "plaintiff", "defendant",
]

_DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"                    # 2025-03-15
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b"                    # 15/03/2025
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"  # March 15, 2025
    r"|\b\d{4}\b"                                      # 2025
)


def _extract_dates(text: str) -> list[str]:
    """Return the date strings found in ``text`` (deterministic)."""
    return _DATE_RE.findall(text)


def _extract_glossary_terms(text: str) -> list[str]:
    """Return the glossary terms that appear in ``text`` (deterministic)."""
    lowered = text.lower()
    return [term for term in _GLOSSARY if term in lowered]


_SUMMARY_INTRO: dict[str, str] = {
    "en": "Summary of the requested task.",
    "es": "Resumen de la tarea solicitada.",
}


def _render_summary(
    profile: LatentProfile, dates: list[str], terms: list[str]
) -> str:
    """Build a deterministic output honoring ``profile``.

    Truncation: if the output exceeds ``max_tokens`` tokens, it is truncated
    (split on whitespace); tokens are never expanded.
    """
    parts: list[str] = [_SUMMARY_INTRO.get(profile.language, _SUMMARY_INTRO["en"])]
    if profile.preserve_dates and dates:
        parts.append("Dates: " + ", ".join(dates) + ".")
    if profile.preserve_glossary and terms:
        parts.append("Terms: " + ", ".join(terms) + ".")
    output = " ".join(parts)
    tokens = output.split()
    if len(tokens) > profile.max_tokens:
        output = " ".join(tokens[: profile.max_tokens])
    return output


_SAMPLE_SOURCE: str = (
    "The plaintiff signed the agreement on 2025-03-15. "
    "The clause limits liability and jurisdiction. "
    "Termination requires written notice."
)


# 4. ORQUESTADOR DEL PROTOCOLO

class ProtocolOrchestrator:
    def __init__(self, registry: Registry, verifier: SemanticVerifier):
        self.registry, self.verifier = registry, verifier

    @staticmethod
    def _log(msg: str) -> None:
        print(f"  [SCP] {msg}")

    def execute(self, requester: Agent, task: TaskRequest, provider_id: str) -> dict:
        print(f"\n{'─' * 72}")
        print(f"SCP EXECUTION  task={task.request_id}")
        print(f"{'─' * 72}")
        print(f"  Requester : {requester}")
        print(f"  Task      : {task.task_description}")
        print(f"  Capability: {task.target_capability}\n")

        # 1) Discovery
        self._log("Fase 1 - Discovery")
        candidates = self.registry.discover(task.target_capability)
        self._log(f"Candidatos: {[c.agent_id for c in candidates]}")

        provider = next((c for c in candidates if c.agent_id == provider_id), None)
        if provider is None:
            return {"status": "provider_not_found"}

        # 2) Proposal
        self._log("Fase 2 · Proposal")
        contract = provider.get_contract(task.target_capability)
        print(f"  Contrato ofrecido (resumen): "
              f"{json.dumps(contract.to_dict(), ensure_ascii=False)}")

        # 3) Verification
        self._log("Fase 3 · Verification")
        report = self.verifier.verify(task, contract)
        print(report.summary())

        # 4) Binding o rechazo
        if report.compatible:
            self._log("Fase 4 · Binding — contrato compatible")
            contract.state = ContractState.BOUND
            self._log(f"BINDING id = {contract.contract_id}")
            return {"status": "executed", "contract": contract,
                    "result": self._execute(provider, task)}

        self._log("Fase 4 · Divergencia — se evita ejecución inconsistente")
        contract.state = ContractState.REJECTED
        return {"status": "rejected", "report": report}

    @staticmethod
    def _execute(provider: Agent, task: TaskRequest) -> dict:
        """Execute the task deterministically from the provider's latent profile."""
        profile: LatentProfile = provider.latent_profile or LatentProfile()
        source: str = task.source_text or task.task_description
        dates: list[str] = _extract_dates(source)
        terms: list[str] = _extract_glossary_terms(source)
        output: str = _render_summary(profile, dates, terms)
        return {
            "executed_by": provider.agent_id,
            "framework": provider.framework,
            "model": provider.model,
            "output": output,
        }


# 5. ESCENARIOS DEMOSTRATIVOS

def build_registry() -> Registry:
    reg = Registry()

    # Agente A — summarizer genérico
    a = Agent(
        "summarizer-A", framework="LangChain", model="gpt-4",
        latent_profile=LatentProfile(
            preserve_dates=False, preserve_glossary=False,
            max_tokens=300, language="es",
        ),
    )
    a.register_capability(CapabilityContract(
        agent_id=a.agent_id,
        capability_name="text_summarization",
        description="Generates concise summaries of long documents for general readers.",
        inputs={"document": "str"}, outputs={"summary": "str"},
        guarantees=[
            SemanticGuarantee("preserve_technical_terms", False,
                              "Los términos técnicos pueden reformularse."),
            SemanticGuarantee("preserve_dates", False,
                              "Las fechas pueden omitirse si no son esenciales."),
            SemanticGuarantee("max_output_tokens", 300, "Resumen de hasta 300 tokens."),
        ],
        constraints={"max_latency_ms": 4000, "language": "es"},
    ))

    # Agente B — summarizer legal
    b = Agent(
        "legal-summarizer-B", framework="AutoGen", model="claude-3.5",
        latent_profile=LatentProfile(
            preserve_dates=True, preserve_glossary=True,
            max_tokens=800, language="es",
        ),
    )
    b.register_capability(CapabilityContract(
        agent_id=b.agent_id,
        capability_name="text_summarization",
        description="Summarizes legal documents preserving clause references and dates.",
        inputs={"document": "str"}, outputs={"summary": "str"},
        guarantees=[
            SemanticGuarantee("preserve_technical_terms", True,
                              "Los términos técnicos se mantienen literalmente."),
            SemanticGuarantee("preserve_dates", True, "Todas las fechas se preservan."),
            SemanticGuarantee("max_output_tokens", 800, "Resumen extenso permitido."),
        ],
        constraints={"max_latency_ms": 9000, "language": "es"},
    ))

    reg.register(a); reg.register(b)
    return reg


def demo_1_divergence() -> None:
    print("\n" + "*" * 72)
    print("DEMO 1 - Divergencia semántica (texto similar, garantías incompatibles)")
    print("*" * 72)
    orch = ProtocolOrchestrator(build_registry(), SemanticVerifier())
    requester = Agent("legal-requester-X", framework="CrewAI", model="gpt-4o")
    task = TaskRequest(
        task_description="Summarize the legal document.",
        target_capability="text_summarization",
        required_guarantees=[
            SemanticGuarantee("preserve_technical_terms", True, "Conservar términos técnicos."),
            SemanticGuarantee("preserve_dates", True, "Conservar todas las fechas."),
        ],
        source_text=_SAMPLE_SOURCE,
        constraints={"max_latency_ms": 5000, "language": "es"},
    )
    r = orch.execute(requester, task, "summarizer-A")
    print(f"\n  >>> RESULTADO: {r['status'].upper()}")


def demo_2_binding() -> None:
    print("\n" + "*" * 72)
    print("DEMO 2 · Contrato compatible → Binding + ejecución")
    print("*" * 72)
    orch = ProtocolOrchestrator(build_registry(), SemanticVerifier())
    requester = Agent("legal-requester-X", framework="CrewAI", model="gpt-4o")
    task = TaskRequest(
        task_description="Summarize the legal document preserving clause references and dates.",
        target_capability="text_summarization",
        required_guarantees=[
            SemanticGuarantee("preserve_technical_terms", True, "Conservar términos técnicos."),
            SemanticGuarantee("preserve_dates", True, "Conservar todas las fechas."),
        ],
        source_text=_SAMPLE_SOURCE,
        constraints={"max_latency_ms": 10000, "language": "es"},
    )
    r = orch.execute(requester, task, "legal-summarizer-B")
    print(f"\n  >>> RESULTADO: {r['status'].upper()}")
    if r["status"] == "executed":
        print(f"  >>> OUTPUT   : {r['result']['output']}")


def demo_3_renegotiation() -> None:
    """El solicitante relaja sus garantías tras la divergencia y re-verifica"""
    print("\n" + "*" * 72)
    print("DEMO 3 · Renegociación semántica (el solicitante relaja requisitos)")
    print("*" * 72)
    orch = ProtocolOrchestrator(build_registry(), SemanticVerifier())
    requester = Agent("casual-requester-Y", framework="CrewAI", model="gpt-4o")
    # no se requiere preservar terminos ni fechas
    task = TaskRequest(
        task_description="Summarize the document briefly.",
        target_capability="text_summarization",
        required_guarantees=[
            SemanticGuarantee("max_output_tokens", 200, "Resumen ≤ 200 tokens."),
        ],
        source_text=_SAMPLE_SOURCE,
        constraints={"max_latency_ms": 5000, "language": "es"},
    )
    r = orch.execute(requester, task, "summarizer-A")
    print(f"\n  >>> RESULTADO: {r['status'].upper()}")


if __name__ == "__main__":
    demo_1_divergence()
    demo_2_binding()
    demo_3_renegotiation()