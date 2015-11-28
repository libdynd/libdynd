//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Bitcode/ReaderWriter.h>

using namespace std;
using namespace llvm;

class EmitLLVMFunctionPass : public FunctionPass {
  static char ID;

public:
  EmitLLVMFunctionPass() : FunctionPass(ID)
  {
  }

  bool doInitialization(Module &M) override
  {
    GlobalVariable *global_annos = M.getNamedGlobal("llvm.global.annotations");
    if (global_annos) {
      auto a = cast<ConstantArray>(global_annos->getOperand(0));
      for (size_t i = 0; i < a->getNumOperands(); i++) {
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

  bool runOnFunction(Function &F) override
  {
    if (F.hasFnAttribute("emit_llvm")) {
      Module *M = F.getParent();

/*
      Module *mod = new Module("test", getGlobalContext());
      ValueToValueMapTy vmap;
      Function *newF = CloneFunction(&F, vmap, false);
      mod->getFunctionList().push_back(newF);
      mod->dump();
*/

      static Regex R("4func.*$");
      GlobalVariable *GV = M->getGlobalVariable(R.sub("2irE", F.getName()), true);
      if (GV != NULL) {
        string S;
        raw_string_ostream SO(S);
        F.print(SO);

        Constant *CDA = ConstantDataArray::getString(M->getContext(), SO.str());
        GV->setInitializer(ConstantExpr::getBitCast(new GlobalVariable(*M, CDA->getType(), true, GV->getLinkage(), CDA),
                                                    Type::getInt8PtrTy(M->getContext())));

        return true;
      }
    }

    return false;
  }
};

char EmitLLVMFunctionPass::ID = 0;

static RegisterStandardPasses RegisterEmitLLVMFunctionPass(PassManagerBuilder::EP_OptimizerLast,
                                                           [](const PassManagerBuilder &, legacy::PassManagerBase &PM) {
  PM.add(new EmitLLVMFunctionPass());
});
