"""SCRC Gate2: source-anchored replay and scalar differential verification.

Keep this package lazy because manifest and policy readers import
``intentir.verify.gate2.policy`` without needing the NumPy-backed execution
adapters.  Runtime consumers still see the same public names.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "GATE2_COMPARISON_PROTOCOL": ("intentir.verify.gate2.compare", "GATE2_COMPARISON_PROTOCOL"),
    "close_q": ("intentir.verify.gate2.compare", "close_q"),
    "compare_outputs": ("intentir.verify.gate2.compare", "compare_outputs"),
    "validate_j_inputs": ("intentir.verify.gate2.compare", "validate_j_inputs"),
    "BoundaryExtent": ("intentir.verify.gate2.inputs", "BoundaryExtent"),
    "GeneratedInputCase": ("intentir.verify.gate2.inputs", "GeneratedInputCase"),
    "InputCaseTemplate": ("intentir.verify.gate2.inputs", "InputCaseTemplate"),
    "InputDistribution": ("intentir.verify.gate2.inputs", "InputDistribution"),
    "TensorInputSpec": ("intentir.verify.gate2.inputs", "TensorInputSpec"),
    "boundary_1d_extents": ("intentir.verify.gate2.inputs", "boundary_1d_extents"),
    "generate_input_matrix": ("intentir.verify.gate2.inputs", "generate_input_matrix"),
    "generate_tensor_input": ("intentir.verify.gate2.inputs", "generate_tensor_input"),
    "materialize_ragged_input_values": (
        "intentir.verify.gate2.inputs",
        "materialize_ragged_input_values",
    ),
    "ExecutionResult": ("intentir.verify.gate2.model", "ExecutionResult"),
    "ExecutionBinding": ("intentir.verify.gate2.model", "ExecutionBinding"),
    "Gate2Result": ("intentir.verify.gate2.model", "Gate2Result"),
    "Gate2Status": ("intentir.verify.gate2.model", "Gate2Status"),
    "Gate2SuiteResult": ("intentir.verify.gate2.model", "Gate2SuiteResult"),
    "OutputAgreement": ("intentir.verify.gate2.model", "OutputAgreement"),
    "PairwiseAgreement": ("intentir.verify.gate2.model", "PairwiseAgreement"),
    "GATE2_EXECUTION_PROTOCOL": ("intentir.verify.gate2.orchestrate", "GATE2_EXECUTION_PROTOCOL"),
    "Gate2Case": ("intentir.verify.gate2.orchestrate", "Gate2Case"),
    "Gate2Runner": ("intentir.verify.gate2.orchestrate", "Gate2Runner"),
    "Gate2Suite": ("intentir.verify.gate2.orchestrate", "Gate2Suite"),
    "CloseRule": ("intentir.verify.gate2.policy", "CloseRule"),
    "ComparisonPolicy": ("intentir.verify.gate2.policy", "ComparisonPolicy"),
    "ComparisonPolicyError": ("intentir.verify.gate2.policy", "ComparisonPolicyError"),
    "DEFAULT_GATE2_POLICY": ("intentir.verify.gate2.policy", "DEFAULT_GATE2_POLICY"),
    "GATE2_POLICY_PROTOCOL": ("intentir.verify.gate2.policy", "GATE2_POLICY_PROTOCOL"),
    "NumericTolerance": ("intentir.verify.gate2.policy", "NumericTolerance"),
    "SCALAR_LOWERING_PROTOCOL": ("intentir.verify.gate2.scalar", "SCALAR_LOWERING_PROTOCOL"),
    "ScalarLoweringRequest": ("intentir.verify.gate2.scalar", "ScalarLoweringRequest"),
    "ScalarLoweringRunner": ("intentir.verify.gate2.scalar", "ScalarLoweringRunner"),
    "ScalarLoweringSpec": ("intentir.verify.gate2.scalar", "ScalarLoweringSpec"),
    "ScalarLoweringUnavailable": (
        "intentir.verify.gate2.scalar",
        "ScalarLoweringUnavailable",
    ),
    "scalar_broadcast_axes": ("intentir.verify.gate2.scalar", "scalar_broadcast_axes"),
    "ArgumentRole": ("intentir.verify.gate2.triton", "ArgumentRole"),
    "TritonArgument": ("intentir.verify.gate2.triton", "TritonArgument"),
    "TritonExecutionSpec": ("intentir.verify.gate2.triton", "TritonExecutionSpec"),
    "TritonRunner": ("intentir.verify.gate2.triton", "TritonRunner"),
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
    "ArgumentRole",
    "BoundaryExtent",
    "CloseRule",
    "ComparisonPolicy",
    "ComparisonPolicyError",
    "DEFAULT_GATE2_POLICY",
    "ExecutionResult",
    "ExecutionBinding",
    "GATE2_COMPARISON_PROTOCOL",
    "GATE2_EXECUTION_PROTOCOL",
    "GATE2_POLICY_PROTOCOL",
    "Gate2Case",
    "Gate2Result",
    "Gate2Runner",
    "Gate2Status",
    "Gate2Suite",
    "Gate2SuiteResult",
    "GeneratedInputCase",
    "InputCaseTemplate",
    "InputDistribution",
    "OutputAgreement",
    "NumericTolerance",
    "PairwiseAgreement",
    "SCALAR_LOWERING_PROTOCOL",
    "ScalarLoweringRequest",
    "ScalarLoweringRunner",
    "ScalarLoweringSpec",
    "ScalarLoweringUnavailable",
    "TensorInputSpec",
    "TritonArgument",
    "TritonExecutionSpec",
    "TritonRunner",
    "boundary_1d_extents",
    "close_q",
    "compare_outputs",
    "generate_input_matrix",
    "generate_tensor_input",
    "materialize_ragged_input_values",
    "scalar_broadcast_axes",
    "validate_j_inputs",
]
