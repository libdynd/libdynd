//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <fstream>

#include "llvm/Support/raw_ostream.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/ValueSymbolTable.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Constants.h"
#include <llvm/IR/Dominators.h>

using namespace std;
using namespace llvm;

struct Hello : public llvm::FunctionPass {
  static char ID;
  //  ofstream myfile;

  Hello() : FunctionPass(ID)
  {
  }

  const char *getPassName() const
  {
    return "Helloxy";
  }

  /*
    bool runOnFunction(llvm::Function &F) override
    {
      llvm::errs() << "Hello: ";
      llvm::errs().write_escaped(F.getName()) << '\n';
      return true;
    }
  */

  /*
  define internal void
  @"_ZN4dynd2nd11base_kernelINS0_10functional17apply_callable_ckIZN20Plugin_Untitled_Test8TestBodyEvE3$_0iNS_13type_sequenceIJiiEEENS_16integer_sequenceImJLm0ELm1EEEENS6_IJEEENS8_ImJEEEEEJEE14single_wrapperEPNS_14ckernel_prefixEPcPKSG_"(%"struct.dynd::ckernel_prefix"*
  %rawself, i8* %dst, i8** %src) #7 align 2 {
    %1 = alloca %"struct.dynd::ckernel_prefix"*, align 8
    %2 = alloca i8*, align 8
    %3 = alloca i8**, align 8
    store %"struct.dynd::ckernel_prefix"* %rawself, %"struct.dynd::ckernel_prefix"** %1, align 8
    store i8* %dst, i8** %2, align 8
    store i8** %src, i8*** %3, align 8
    %4 = load %"struct.dynd::ckernel_prefix"** %1, align 8
    %5 = call %"struct.dynd::nd::functional::apply_callable_ck"*
  @"_ZN4dynd2nd21kernel_prefix_wrapperINS_14ckernel_prefixENS0_10functional17apply_callable_ckIZN20Plugin_Untitled_Test8TestBodyEvE3$_0iNS_13type_sequenceIJiiEEENS_16integer_sequenceImJLm0ELm1EEEENS7_IJEEENS9_ImJEEEEEE8get_selfEPS2_"(%"struct.dynd::ckernel_prefix"*
  %4)
    %6 = load i8** %2, align 8
    %7 = load i8*** %3, align 8
    call void
  @"_ZN4dynd2nd10functional17apply_callable_ckIZN20Plugin_Untitled_Test8TestBodyEvE3$_0iNS_13type_sequenceIJiiEEENS_16integer_sequenceImJLm0ELm1EEEENS5_IJEEENS7_ImJEEEE6singleEPcPKSC_"(%"struct.dynd::nd::functional::apply_callable_ck"*
  %5, i8* %6, i8** %7)
    ret void
  }
  */

  bool doInitialization(llvm::Module &M) override
  {
    auto global_used = M.getNamedGlobal("llvm.compiler.used");
    if (global_used) {
      global_used->dump();
    }

    auto global_annos = M.getNamedGlobal("llvm.global.annotations");
    if (global_annos) {
      auto a = cast<ConstantArray>(global_annos->getOperand(0));
      for (int i = 0; i < a->getNumOperands(); i++) {
        auto e = cast<ConstantStruct>(a->getOperand(i));

        if (Function *fn = dyn_cast<Function>(e->getOperand(0)->getOperand(0))) {
          auto anno = cast<ConstantDataArray>(cast<GlobalVariable>(e->getOperand(1)->getOperand(0))->getOperand(0))
                          ->getAsCString();
          fn->addFnAttr(anno); // <-- add function annotation here
        }
      }
    }

    return true;
  }

  // _ZN4dynd2nd11base_kernelINS0_10functional17apply_callable_ckIZN20Plugin_Untitled_Test8TestBodyEvE3$_0iNS_13type_sequenceIJiiEEENS_16integer_sequenceImJLm0ELm1EEEENS6_IJEEENS8_ImJEEEEEJEE14single_wrapperEPNS_14ckernel_prefixEPcPKSG_
  // _ZN4dynd2nd11base_kernelINS0_10functional17apply_callable_ckIP10test_classINS_7complexIdEEEvNS_13type_sequenceIJS6_S6_EEENS_16integer_sequenceImJLm0ELm1EEEENS9_IJEEENSB_ImJEEEEEJEE14single_wrapperEPNS_14ckernel_prefixEPcPKSJ_

  // linkonce_odr instead of internal
  bool runOnFunction(llvm::Function &F) override
  {
    //    std::string s = "_ZN4dynd2nd11base_kernelINS0_10functional17apply_callable_ckIPFvRdiEvNS_13type_sequenceIJS4_"
    //                  "iEEENS_16integer_sequenceImJLm0ELm1EEEENS7_IJEEENS9_ImJEEEEEJEE14single_wrapperEPNS_14ckernel_"
    //                "prefixEPcPKSH_";
    //  if (F.getName() == s) {
    //  Module *M = F.getParent();
    //    M->dump();
    //}

    std::string tail = "::single_wrapper(dynd::ckernel_prefix*, char*, char* const*)";

    if (F.hasFnAttribute("ir")) {
      const auto &name = F.getName();
      //      llvm::outs() << "F = " << name << "\n";
      //      F.dump();

      std::string var_name = name.drop_back(name.size() - name.rfind("14single_wrapper"));
      var_name = var_name + "9single_irE";

      Module *M = F.getParent();

      GlobalVariable *sym = M->getGlobalVariable(var_name, true);
//      llvm::outs() << name << "\n";
      if (sym != NULL) {
        //  llvm::outs() << sym->getName() << "\n";

        std::string S;
        raw_string_ostream O(S);
        F.print(O);

        auto *ir = ConstantDataArray::getString(M->getContext(), O.str());
        GlobalVariable *var =
            new GlobalVariable(*M, ir->getType(), true, GlobalValue::InternalLinkage, ir, "SourceFile");

        ConstantInt *const_int64_6 = ConstantInt::get(M->getContext(), APInt(64, StringRef("0"), 10));

        // Type::getInt32Ty(M->getContext()), 0)
        Constant *constArray = ConstantExpr::getGetElementPtr(var, const_int64_6);
        Constant *expr = ConstantExpr::getBitCast(constArray, PointerType::getUnqual(Type::getInt8Ty(M->getContext())));

        sym->setInitializer(expr);
      }
    }

    return true;
  }
};

char Hello::ID = 0;

// EP_EarlyAsPossible
// EP_OptimizerLast

static RegisterStandardPasses RegisterMyPass2(PassManagerBuilder::EP_OptimizerLast,
                                              [](const PassManagerBuilder &,
                                                 legacy::PassManagerBase &PM) { PM.add(new Hello()); });
