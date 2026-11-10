"""LLM lifters from real kernel source to IntentIR/SCRC artifacts."""

from importlib import import_module


_LAZY_EXPORTS = {
    "MINISWE_LIFT_BACKEND": ("intentir.extraction.lifters.miniswe", "MINISWE_LIFT_BACKEND"),
    "MINISWE_LIFT_KNOWLEDGE": ("intentir.extraction.lifters.miniswe", "MINISWE_LIFT_KNOWLEDGE"),
    "MiniSWELiftAdapter": ("intentir.extraction.lifters.miniswe", "MiniSWELiftAdapter"),
    "DIRECT_AGENT_LIFT_BACKEND": ("intentir.extraction.lifters.direct_agent", "DIRECT_AGENT_LIFT_BACKEND"),
    "DirectAgentLiftAdapter": ("intentir.extraction.lifters.direct_agent", "DirectAgentLiftAdapter"),
    "ResponsesSCRCLiftAdapter": ("intentir.extraction.lifters.scrc", "ResponsesSCRCLiftAdapter"),
    "ResponsesSCRCRelifter": ("intentir.extraction.lifters.scrc", "ResponsesSCRCRelifter"),
    "SCRCCandidateCheckContext": ("intentir.extraction.lifters.scrc", "SCRCCandidateCheckContext"),
    "SCRC_DECODE_PROTOCOL": ("intentir.extraction.lifters.scrc", "SCRC_DECODE_PROTOCOL"),
    "SCRC_RELIFT_PROTOCOL": ("intentir.extraction.lifters.scrc", "SCRC_RELIFT_PROTOCOL"),
    "SCRCCompilerArgument": ("intentir.extraction.lifters.scrc", "SCRCCompilerArgument"),
    "SCRCCompilerView": ("intentir.extraction.lifters.scrc", "SCRCCompilerView"),
    "SCRCExtraction": ("intentir.extraction.lifters.scrc", "SCRCExtraction"),
    "SCRCProvenance": ("intentir.extraction.lifters.scrc", "SCRCProvenance"),
    "SCRCReliftContext": ("intentir.extraction.lifters.scrc", "SCRCReliftContext"),
    "SCRCReliftRecord": ("intentir.extraction.lifters.scrc", "SCRCReliftRecord"),
    "decode_scrc_lift_artifact": ("intentir.extraction.lifters.scrc", "decode_scrc_lift_artifact"),
    "extract_scrc_artifact": ("intentir.extraction.lifters.scrc", "extract_scrc_artifact"),
    "scrc_candidate_check_context": ("intentir.extraction.lifters.scrc", "scrc_candidate_check_context"),
    "scrc_lift_prompt_sha256": ("intentir.extraction.lifters.scrc", "scrc_lift_prompt_sha256"),
    "scrc_relift_prompt_sha256": ("intentir.extraction.lifters.scrc", "scrc_relift_prompt_sha256"),
    "scrc_lift_core_json_schema": ("intentir.extraction.lifters.scrc_schema", "scrc_lift_core_json_schema"),
    "scrc_lift_core_text_format": ("intentir.extraction.lifters.scrc_schema", "scrc_lift_core_text_format"),
    "scrc_semantic_candidate_text_format": ("intentir.extraction.lifters.scrc_schema", "scrc_semantic_candidate_text_format"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "MINISWE_LIFT_BACKEND",
    "MINISWE_LIFT_KNOWLEDGE",
    "MiniSWELiftAdapter",
    "DIRECT_AGENT_LIFT_BACKEND",
    "DirectAgentLiftAdapter",
    "ResponsesSCRCRelifter",
    "ResponsesSCRCLiftAdapter",
    "SCRCCandidateCheckContext",
    "SCRC_DECODE_PROTOCOL",
    "SCRC_RELIFT_PROTOCOL",
    "SCRCExtraction",
    "SCRCCompilerArgument",
    "SCRCCompilerView",
    "SCRCProvenance",
    "SCRCReliftContext",
    "SCRCReliftRecord",
    "decode_scrc_lift_artifact",
    "extract_scrc_artifact",
    "scrc_candidate_check_context",
    "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_text_format",
    "scrc_semantic_candidate_text_format",
]
