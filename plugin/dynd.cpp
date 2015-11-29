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
#include <llvm/ADT/StringMap.h>

using namespace std;
using namespace llvm;

class EmitLLVMFunctionPass : public FunctionPass {
  static char ID;
  StringMap<Module *> SM;

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

      static Regex R("14single_wrapper.*$");
      GlobalVariable *GV = M->getGlobalVariable(R.sub("2irE", F.getName()), true);

      if (GV != NULL) {
        Module *NewM = new Module("", getGlobalContext());

        ValueToValueMapTy vmap;
        Function *newF = reinterpret_cast<Function *>(NewM->getOrInsertFunction("single_wrapper", F.getFunctionType()));
        Function::arg_iterator DestI = newF->arg_begin();
        for (const Argument &I : F.args()) {
          if (vmap.count(&I) == 0) {
            DestI->setName(I.getName());
            vmap[&I] = &*DestI++;
          }
        }

        SmallVector<ReturnInst *, 3> ret;
        CloneFunctionInto(newF, &F, vmap, false, ret);
        SM[F.getName()] = NewM;

        return true;
      }
    }

    return false;
  }

  bool doFinalization(Module &M) override
  {
    static Regex R("14single_wrapper.*$");
    for (const auto &SME : SM) {
      GlobalVariable *GV = M.getGlobalVariable(R.sub("2irE", SME.getKey()), true);
      if (GV != NULL) {
        Module *NewM = SME.getValue();

        string S;
        raw_string_ostream SO(S);
        NewM->print(SO, NULL);

        Constant *CDA = ConstantDataArray::getString(M.getContext(), SO.str());
        GV->setInitializer(ConstantExpr::getBitCast(new GlobalVariable(M, CDA->getType(), true, GV->getLinkage(), CDA),
                                                    Type::getInt8PtrTy(M.getContext())));

        delete NewM;
      }
    }

    return true;
  }
};

char EmitLLVMFunctionPass::ID = 0;

static RegisterStandardPasses RegisterEmitLLVMFunctionPass(PassManagerBuilder::EP_OptimizerLast,
                                                           [](const PassManagerBuilder &, legacy::PassManagerBase &PM) {
  PM.add(new EmitLLVMFunctionPass());
});
