add_mlir_library(CudaTileTransforms
  AlternateReductionLoop.cpp
  FuseFMA.cpp
  LoopSplit.cpp
  SynthesizeDebugInfoScopes.cpp

  DEPENDS
  CudaTileTransformsIncGen

  LINK_LIBS PUBLIC
  CudaTileDialect
  MLIRTransforms

  LINK_COMPONENTS
  Support
)
