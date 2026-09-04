#ifndef INTENT_DIALECT
#define INTENT_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// IntentIR dialect definition.
//===----------------------------------------------------------------------===//

def Intent_Dialect : Dialect {
  let name = "intent";
  let summary = "IntentIR: certified semantic lifting dialect.";
  let description = [{
    The `intent` dialect is the MLIR form of IntentIR's three-layer
    representation (Layer A algorithmic intent / Layer B portable
    correctness structure / Layer C non-binding schedule hints). Ops are
    tensor-level atomic primitives produced 1:1 from IntentIR JSON.
  }];
  let cppNamespace = "::intent";
  let useDefaultTypePrinterParser = 1;
}

// Kernel IR types are part of the target-neutral exchange boundary emitted by
// the Python frontend.  Keep their grammar in the same dialect TableGen
// source as the operations so the parser, C++ consumers, and frontend use one
// vocabulary.
class Intent_Type<string name, string typeMnemonic>
    : TypeDef<Intent_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Intent_ViewType : Intent_Type<"View", "view"> {
  let summary = "A typed external tensor view at the Kernel IR boundary.";
  let parameters = (ins
    "Type":$tensor,
    StringRefParameter<"view access role">:$role
  );
  let assemblyFormat = "`<` qualified($tensor) `,` $role `>`";
}

def Intent_DomainType : Intent_Type<"Domain", "domain"> {
  let summary = "A Kernel IR iteration-domain descriptor.";
  let parameters = (ins
    StringRefParameter<"domain metadata">:$metadata
  );
  let assemblyFormat = "`<` $metadata `>`";
}

def Intent_LogicalIndexType : Intent_Type<"LogicalIndex", "logical_index"> {
  let summary = "A Kernel IR logical-index descriptor.";
  let parameters = (ins
    StringRefParameter<"logical-index metadata">:$metadata
  );
  let assemblyFormat = "`<` $metadata `>`";
}

// Base class for all IntentIR ops.
class Intent_Op<string mnemonic, list<Trait> traits = []>
    : Op<Intent_Dialect, mnemonic, traits>;

// Base class for the target-neutral Kernel IR exchange operations emitted by
// the canonical-J exporter. These are deliberately separate from the
// tensor-level Intent operations: Kernel IR carries explicit views, domains,
// logical indices, and region structure before a backend binds it to ordinary
// MLIR. Keeping the vocabulary in this TableGen source makes frontend output
// and C++ registration share one grammar.
class Intent_KernelOp<string mnemonic, list<Trait> traits = []>
    : Op<Intent_Dialect, mnemonic, traits>;

#endif // INTENT_DIALECT
