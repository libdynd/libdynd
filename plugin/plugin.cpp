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

#include <clang/AST/CXXInheritance.h>
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"

#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/IR/Module.h>

#include <llvm/IR/LLVMContext.h>

#include <clang/CodeGen/ModuleBuilder.h>
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Constants.h"

using namespace std;
using namespace clang;

bool callback(const CXXRecordDecl *BaseDeclaration, void *)
{
  //  std::cout << "BaseDeclaration->getQualifiedNameAsString() = " << BaseDeclaration->getQualifiedNameAsString()
  //          << std::endl;
  //  std::cout << (BaseDeclaration->getNameAsString() != "ckernel_prefix") << std::endl;
  return BaseDeclaration->getNameAsString() == "ckernel_prefix";
}

class Visitor : public RecursiveASTVisitor<Visitor> {
public:
  /*
    bool VisitNamedDecl(clang::NamedDecl *NamedDecl)
    {
      llvm::outs() << "Found " << NamedDecl->getQualifiedNameAsString() << "\n";
      return true;
    }
  */

  /*
    bool VisitCXXRecordDecl(Call *Declaration)
    {
      //    std::cout << "Declaration->getQualifiedNameAsString() = " << Declaration->getQualifiedNameAsString() <<
      // std::endl;
      if (Declaration->hasDefinition()) {
        CXXBasePaths p;
        if (!Declaration->lookupInBases(callback, NULL, p)) {
          std::cout << "Declaration->getQualifiedNameAsString() = " << Declaration->getQualifiedNameAsString()
                    << std::endl;

          //        std::cout << "Success!" << std::endl;
          //        llvm::outs() << Declaration->getQualifiedNameAsString() << "\n";
        }
      }

      return true;
    }
  */

  /*
    bool VisitCallExpr(CallExpr *Expr)
    {
      FunctionDecl *Callee = Expr->getDirectCallee();

      if (Callee != NULL) {
  //      std::cout << Callee->getQualifiedNameAsString() << std::endl;
        if (Callee->getQualifiedNameAsString() == "dynd::nd::callable::make") {
          const TemplateArgumentList *List = Callee->getTemplateSpecializationArgs();
          const TemplateArgument &A = List->get(0);

          switch (A.getKind()) {
          case 1:
            //      std::cout << A.getAsType().getAsString() << std::endl;
            break;
          default:
            //          std::cout << "Kind = " << A.getKind() << std::endl;
            break;
          }
        } else if (Callee->getQualifiedNameAsString() == "dynd::nd::callable::make_all") {
                std::cout << Callee->getQualifiedNameAsString() << std::endl;
        }
      }
      return true;
    }
  */

  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *Func)
  {
    //    std::exit(-1);
    //    std::cout << "visiting" << std::endl;
    if (Func->getQualifiedNameAsString() == "dynd::nd::callable::make") {
      for (const clang::FunctionDecl *Spec : Func->specializations()) {
        const TemplateArgumentList *List = Spec->getTemplateSpecializationArgs();
        const TemplateArgument &A = List->get(0);
        if (A.getKind() == 1) {
          //        std::cout << A.getAsType().getAsString() << std::endl;
        }
      }
    }

    return true;
  }
};

class PrintFunctionsConsumer : public ASTConsumer {
  CompilerInstance &Instance;
  std::set<std::string> ParsedTemplates;

public:
  PrintFunctionsConsumer(CompilerInstance &Instance, std::set<std::string> ParsedTemplates)
      : Instance(Instance), ParsedTemplates(ParsedTemplates)
  {
  }

  void HandleTranslationUnit(ASTContext &Context) override
  {

    Visitor V;
    V.TraverseDecl(Context.getTranslationUnitDecl());

    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts = new clang::DiagnosticOptions();
    clang::TextDiagnosticPrinter *DiagClient = new clang::TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

    //    CodeGenerator *gen =
    //  CreateLLVMCodeGen(Diags, "", Instance.getCodeGenOpts(), Instance.getTargetOpts(), llvm::getGlobalContext());
    // llvm::Module *module = gen->GetModule();

    //    std::cout << "here" << std::endl;
    //  module->dump();
    //    llvm::Function *func = module->getFunction("func");

    //  func->dump();

    //  clang::CodeGenAction *Act = new clang::EmitLLVMOnlyAction(&llvm::getGlobalContext());
    //    Instance.ExecuteAction(*Act);

    /*
        string inputFile = "/home/irwin/Desktop/test.c";


        const char *args[] = {inputFile.c_str()};
        int nargs = sizeof(args) / sizeof(args[0]);
        std::unique_ptr<clang::CompilerInvocation> CI(new clang::CompilerInvocation);
        clang::CompilerInvocation::CreateFromArgs(*CI, args, args + nargs, Diags);
        //  CI->setLangDefaults(clang::IK_CXX, clang::LangStandard::lang_unspecified);
    */

    /*

        std::unique_ptr<llvm::Module> module = Act->takeModule();
        llvm::Function *func = module->getFunction("func");

        func->dump();
    */
  }
};

struct Hello : public llvm::FunctionPass {
  static char ID;

  Hello() : FunctionPass(ID)
  {
  }

  /*
    const char *getPassName() const
    {
      return "Hello";
    }
  */

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
    return true;
  }

  bool runOnFunction(llvm::Function &F) override
  {
    if (F.hasFnAttribute("ir")) {
      llvm::outs() << F.getName() << " has my attribute!\n";
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

class PrintFunctionNamesAction : public PluginASTAction {
  std::set<std::string> ParsedTemplates;

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override
  {
    return llvm::make_unique<PrintFunctionsConsumer>(CI, ParsedTemplates);
  }

  bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override
  {
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      llvm::errs() << "PrintFunctionNames arg = " << args[i] << "\n";

      // Example error handling.
      DiagnosticsEngine &D = CI.getDiagnostics();
      if (args[i] == "-an-error") {
        unsigned DiagID = D.getCustomDiagID(DiagnosticsEngine::Error, "invalid argument '%0'");
        D.Report(DiagID) << args[i];
        return false;
      } else if (args[i] == "-parse-template") {
        if (i + 1 >= e) {
          D.Report(D.getCustomDiagID(DiagnosticsEngine::Error, "missing -parse-template argument"));
          return false;
        }
        ++i;
        ParsedTemplates.insert(args[i]);
      }
    }
    if (!args.empty() && args[0] == "help")
      PrintHelp(llvm::errs());

    return true;
  }
  void PrintHelp(llvm::raw_ostream &ros)
  {
    ros << "Help for PrintFunctionNames plugin goes here\n";
  }

public:
  PrintFunctionNamesAction()
  {
    llvm::PassManagerBuilder::addGlobalExtension(llvm::PassManagerBuilder::EP_EarlyAsPossible, registerMyPass);
  }
};

// static llvm::RegisterPass<Hello> Y("print-fns", "Hello World Pass", true, true);

static llvm::RegisterStandardPasses RegisterMyPass(llvm::PassManagerBuilder::EP_EarlyAsPossible, registerMyPass);

// static FrontendPluginRegistry::Add<PrintFunctionNamesAction> X("print-fns", "print function names");
