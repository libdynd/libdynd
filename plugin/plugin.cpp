//===- PrintFunctionNames.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which simply prints the names of all the top-level decls
// in the input file.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <clang/CodeGen/CodeGenAction.h>

#include <iostream>
#include <fstream>

#include <clang/AST/CXXInheritance.h>
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"

#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/IR/Module.h>

#include <llvm/IR/LLVMContext.h>

#include <llvm/IR/ValueSymbolTable.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Constants.h"

#include <cxxabi.h>
#include <regex>

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
    return "Hello";
  }

  /*
    bool runOnFunction(llvm::Function &F) override
    {
      llvm::errs() << "Hello: ";
      llvm::errs().write_escaped(F.getName()) << '\n';
      return true;
    }
  */

  bool doInitialization(llvm::Module &M) override
  {
    using namespace llvm;

    auto global_annos = M.getNamedGlobal("llvm.global.annotations");
    if (global_annos) {
      auto a = cast<ConstantArray>(global_annos->getOperand(0));
      for (int i = 0; i < a->getNumOperands(); i++) {
        auto e = cast<ConstantStruct>(a->getOperand(i));

        if (auto fn = dyn_cast<Function>(e->getOperand(0)->getOperand(0))) {
          auto anno = cast<ConstantDataArray>(cast<GlobalVariable>(e->getOperand(1)->getOperand(0))->getOperand(0))
                          ->getAsCString();
          fn->addFnAttr(anno); // <-- add function annotation here
        }
      }
    }

    //    myfile.open("src/dynd/kernels/example.cpp", std::ios_base::app);

    return true;
  }

  bool runOnFunction(llvm::Function &F) override
  {
    std::string tail = "::single_wrapper(dynd::ckernel_prefix*, char*, char* const*)";

    if (F.hasFnAttribute("ir")) {
      const auto &name = F.getName();
      std::string var_name = name.drop_back(name.size() - name.rfind("14single_wrapper"));
      var_name = var_name + "9single_irE";

      llvm::Module *M = F.getParent();
      //      auto *var = M->getNamedAlias(var_name);
      //    if (var != NULL) {
      //    std::exit(-1);
      //}

      //      std::cout << var->hasInitializer() << std::endl;
      //      var->setInitializer(llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 10));

      //      new llvm::GlobalVariable(M, PointerTy_0, true, GlobalValue::CommonLinkage, 0, var_name);

      //      M->dump();
      //      for (auto &g : M->getGlobalList()) {
      //      llvm::outs() << g.getName() << "\n";
      //    if (g.getName().rfind("single_ir") != llvm::StringRef::npos) {
      //  llvm::outs() << g.getName() << "\n";
      //  }

      //        llvm::outs() << g.getName() << "\n";
      //  g.dump();
      //      }

      llvm::GlobalVariable *sym = M->getGlobalVariable(var_name);
      if (sym != NULL) {

        std::string out;
        raw_string_ostream o(out);
        F.print(o);

        auto *ir = ConstantDataArray::getString(M->getContext(), out);
        GlobalVariable *var =
            new GlobalVariable(*M, ir->getType(), true, GlobalValue::PrivateLinkage, ir, "SourceFile", sym);

        ConstantInt *const_int64_6 = ConstantInt::get(M->getContext(), APInt(64, StringRef("0"), 10));

        // Type::getInt32Ty(M->getContext()), 0)
        Constant *constArray = ConstantExpr::getGetElementPtr(var, const_int64_6);
        Constant *expr = ConstantExpr::getBitCast(constArray, PointerType::getUnqual(Type::getInt8Ty(M->getContext())));

        sym->setInitializer(expr);

        //        M->dump();
        //        std::exit(-1);

        //        GlobalVariable *sourceFileStr = new GlobalVariable(
        //          *module, stringConstant->getType(), true, llvm::GlobalValue::InternalLinkage, stringConstant,
        // "SourceFile");

        //        sym->setInitializer(llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 10));
      }
      // else {
      //        llvm::outs() << var_name << "\n";
      //      std::exit(-1);
      //  }
      //      myfile << out << "::ir = 3;" << endl;

      //      free(out);
    }

    //    auto global_annos = M.getNamedGlobal("llvm.global.annotations");

    return true;
  }
};

char Hello::ID = 0;

static void registerMyPass(const llvm::PassManagerBuilder &, llvm::legacy::PassManagerBase &PM)
{
  PM.add(new Hello());
}

// static llvm::RegisterPass<Hello> Y("print-fns", "Hello World Pass", true, true);

static llvm::RegisterStandardPasses RegisterMyPass(llvm::PassManagerBuilder::EP_EarlyAsPossible, registerMyPass);
