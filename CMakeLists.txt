cmake_minimum_required(VERSION 3.20)
project(IntentIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Out-of-tree MLIR dialect project for IntentIR.
# Requires a system / prebuilt MLIR (tested against MLIR 20.1.8 in
# /usr/lib/llvm-20). Configure with:
#   -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
#   -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# LLVM 20's stock mlir_tablegen() emits a depfile whose target is prefixed
# twice when the project is configured out of tree with Ninja (for example
# ``include/Intent/include/Intent/IntentOps.h.inc``).  That makes every
# generated file appear dirty forever and, more importantly, makes the
# generated declaration/definition pair look out of sync to incremental
# consumers.  Keep the same public TableGen targets, but use explicit local
# .td dependencies and an absolute output path so the depfile transform is
# not involved.  System MLIR .td files remain toolchain-owned and are not
# copied into this project.
function(intent_mlir_tablegen ofn)
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})

  if(NOT DEFINED LLVM_TARGET_DEFINITIONS OR
     "${LLVM_TARGET_DEFINITIONS}" STREQUAL "")
    message(FATAL_ERROR
      "LLVM_TARGET_DEFINITIONS must be set before intent_mlir_tablegen()")
  endif()

  if(IS_ABSOLUTE "${LLVM_TARGET_DEFINITIONS}")
    set(_definition "${LLVM_TARGET_DEFINITIONS}")
  else()
    set(_definition "${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS}")
  endif()
  set(_output "${CMAKE_CURRENT_BINARY_DIR}/${ofn}")

  get_directory_property(_includes INCLUDE_DIRECTORIES)
  list(PREPEND _includes "${CMAKE_CURRENT_SOURCE_DIR}")
  list(APPEND _includes ${ARG_EXTRA_INCLUDES})
  list(REMOVE_ITEM _includes "")
  list(REMOVE_DUPLICATES _includes)
  set(_include_flags)
  foreach(_include IN LISTS _includes)
    list(APPEND _include_flags "-I${_include}")
  endforeach()

  # All local .td files are explicit dependencies.  This covers included
  # project definitions such as IntentDialect.td without relying on a
  # generator-specific depfile spelling.
  file(GLOB _local_tds "${CMAKE_CURRENT_SOURCE_DIR}/*.td")
  add_custom_command(
    OUTPUT "${_output}"
    COMMAND $<TARGET_FILE:mlir-tblgen>
            ${ARG_UNPARSED_ARGUMENTS}
            ${_include_flags}
            "${_definition}"
            --write-if-changed
            -o "${_output}"
    DEPENDS "${_definition}" ${_local_tds} ${ARG_DEPENDS} mlir-tblgen
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Building ${ofn}..."
    VERBATIM
  )
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
               "${_output}")
  set_source_files_properties("${_output}" PROPERTIES GENERATED 1)
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${_output}" PARENT_SCOPE)
endfunction()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/Intent)
add_subdirectory(include/Target)
add_subdirectory(lib/Intent)
add_subdirectory(lib/Target)
add_subdirectory(tools/intent-opt)
