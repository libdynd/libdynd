//
// Copyright (C) 2011-16 DyND Developers
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
          if (fn->hasFnAttribute("emit_llvm")) {
            fn->addFnAttr("name", anno);
          } else {
            fn->addFnAttr(anno); // <-- add function annotation here
          }
        }
      }
    }

    return true;
  }

  bool runOnFunction(Function &F) override
  {
    if (F.hasFnAttribute("emit_llvm")) {
      StringRef Name = F.getName();

      Module *NewM = new Module("", getGlobalContext());
      StringRef NewName = F.getFnAttribute("name").getValueAsString();
      Function *NewF = reinterpret_cast<Function *>(NewM->getOrInsertFunction(NewName, F.getFunctionType()));

      ValueToValueMapTy vmap;
      Function::arg_iterator DestI = NewF->arg_begin();
      for (const Argument &I : F.args()) {
        if (vmap.count(&I) == 0) {
          DestI->setName(I.getName());
          vmap[&I] = &*DestI++;
        }
      }

      SmallVector<ReturnInst *, 3> ret;
      CloneFunctionInto(NewF, &F, vmap, false, ret);

      StringRef Key = Name.slice(0, Name.find(NewName) - to_string(NewName.size()).size());
      SM[Key] = NewM;
    }

    return false;
  }

  bool doFinalization(Module &M) override
  {
    for (const StringMapEntry<Module *> &SME : SM) {
      GlobalVariable *GV = M.getGlobalVariable(SME.getKey().str() + "2irE", true);

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
