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


#include <clang/AST/CXXInheritance.h>

using namespace clang;

namespace {

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

  bool VisitCXXRecordDecl(CXXRecordDecl *Declaration)
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
    //    llvm::errs() << "top-level-decl: \"" << ND->getNameAsString() << "\"\n";

    /*
        struct Visitor : public RecursiveASTVisitor<Visitor> {
          const std::set<std::string> &ParsedTemplates;
          Visitor(const std::set<std::string> &ParsedTemplates) : ParsedTemplates(ParsedTemplates)
          {
          }
          bool VisitFunctionDecl(FunctionDecl *FD)
          {
            if (FD->isLateTemplateParsed() && ParsedTemplates.count(FD->getNameAsString()))
              LateParsedDecls.insert(FD);
            return true;
          }

          std::set<FunctionDecl *> LateParsedDecls;
        } v(ParsedTemplates);
    */

    //  v.TraverseDecl(context.getTranslationUnitDecl());
    //    clang::Sema &sema = Instance.getSema();
    //    for (const FunctionDecl *FD : v.LateParsedDecls) {
    //  }
  }
};

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
};
}

static FrontendPluginRegistry::Add<PrintFunctionNamesAction> X("print-fns", "print function names");
