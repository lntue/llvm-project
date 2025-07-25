//===- MCTargetOptions.h - MC Target Options --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETOPTIONS_H
#define LLVM_MC_MCTARGETOPTIONS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Compression.h"
#include <string>
#include <vector>

namespace llvm {

enum class EmitDwarfUnwindType {
  Always,          // Always emit dwarf unwind
  NoCompactUnwind, // Only emit if compact unwind isn't available
  Default,         // Default behavior is based on the target
};

class StringRef;

class MCTargetOptions {
public:
  enum AsmInstrumentation {
    AsmInstrumentationNone,
    AsmInstrumentationAddress
  };

  bool MCRelaxAll : 1;
  bool MCNoExecStack : 1;
  bool MCFatalWarnings : 1;
  bool MCNoWarn : 1;
  bool MCNoDeprecatedWarn : 1;
  bool MCNoTypeCheck : 1;
  bool MCSaveTempLabels : 1;
  bool MCIncrementalLinkerCompatible : 1;
  bool FDPIC : 1;
  bool ShowMCEncoding : 1;
  bool ShowMCInst : 1;
  bool AsmVerbose : 1;

  /// Preserve Comments in Assembly.
  bool PreserveAsmComments : 1;

  bool Dwarf64 : 1;

  // Use CREL relocation format for ELF.
  bool Crel = false;

  bool ImplicitMapSyms = false;

  // If true, prefer R_X86_64_[REX_]GOTPCRELX to R_X86_64_GOTPCREL on x86-64
  // ELF.
  bool X86RelaxRelocations = true;

  bool X86Sse2Avx = false;

  std::optional<unsigned> OutputAsmVariant;

  EmitDwarfUnwindType EmitDwarfUnwind;

  int DwarfVersion = 0;

  enum DwarfDirectory {
    // Force disable
    DisableDwarfDirectory,
    // Force enable, for assemblers that support
    // `.file fileno directory filename' syntax
    EnableDwarfDirectory,
    // Default is based on the target
    DefaultDwarfDirectory
  };
  DwarfDirectory MCUseDwarfDirectory;

  // Whether to compress DWARF debug sections.
  DebugCompressionType CompressDebugSections = DebugCompressionType::None;

  std::string ABIName;
  std::string AssemblyLanguage;
  std::string SplitDwarfFile;
  std::string AsSecureLogFile;

  // Used for codeview debug info. These will be set as compiler path and commandline arguments in LF_BUILDINFO
  std::string Argv0;
  std::string CommandlineArgs;

  /// Additional paths to search for `.include` directives when using the
  /// integrated assembler.
  std::vector<std::string> IASSearchPaths;

  // InstPrinter options.
  std::vector<std::string> InstPrinterOptions;

  // Whether to emit compact-unwind for non-canonical personality
  // functions on Darwins.
  bool EmitCompactUnwindNonCanonical : 1;

  // Whether to emit SFrame unwind sections.
  bool EmitSFrameUnwind : 1;

  // Whether or not to use full register names on PowerPC.
  bool PPCUseFullRegisterNames : 1;

  LLVM_ABI MCTargetOptions();

  /// getABIName - If this returns a non-empty string this represents the
  /// textual name of the ABI that we want the backend to use, e.g. o32, or
  /// aapcs-linux.
  LLVM_ABI StringRef getABIName() const;

  /// getAssemblyLanguage - If this returns a non-empty string this represents
  /// the textual name of the assembly language that we will use for this
  /// target, e.g. masm.
  LLVM_ABI StringRef getAssemblyLanguage() const;
};

} // end namespace llvm

#endif // LLVM_MC_MCTARGETOPTIONS_H
