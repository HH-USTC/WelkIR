// Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
// Department of Computer Science and Engineering, IIT Hyderabad
//
// This software is available under the BSD 4-Clause License. Please see LICENSE
// file in the top-level directory for more details.
//
#ifndef __IR2Vec_FA_H__
#define __IR2Vec_FA_H__

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <unordered_map>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"


namespace IR2Vec {
#define DIM 300
using Vector = llvm::SmallVector<double, DIM>;
} // namespace IR2Vec

using namespace llvm;

// LoopInfo contains a mapping from basic block to the innermost loop. Find
// the outermost loop in the loop nest that contains BB.
static const Loop *getOutermostLoop(const LoopInfo *LI, const BasicBlock *BB) {
  const Loop *L = LI->getLoopFor(BB);
  if (L) {
    while (const Loop *Parent = L->getParentLoop())
      L = Parent;
  }
  return L;
}


bool isPotentiallyReachableFromMany(
    SmallVectorImpl<BasicBlock *> &Worklist, BasicBlock *StopBB,
    const SmallPtrSetImpl<const BasicBlock *> *ExclusionSet,
    const DominatorTree *DT, const LoopInfo *LI) {
  // When the stop block is unreachable, it's dominated from everywhere,
  // regardless of whether there's a path between the two blocks.
  if (DT && !DT->isReachableFromEntry(StopBB))
    DT = nullptr;

  // We can't skip directly from a block that dominates the stop block if the
  // exclusion block is potentially in between.
  if (ExclusionSet && !ExclusionSet->empty())
    DT = nullptr;

  // Normally any block in a loop is reachable from any other block in a loop,
  // however excluded blocks might partition the body of a loop to make that
  // untrue.

  SmallPtrSet<const Loop *, 8> LoopsWithHoles;
  if (LI && ExclusionSet) {
    for (auto BB : *ExclusionSet) {
      if (const Loop *L = getOutermostLoop(LI, BB))
        LoopsWithHoles.insert(L);
    }
  }

  const Loop *StopLoop = LI ? getOutermostLoop(LI, StopBB) : nullptr;

  // Limit the number of blocks we visit. The goal is to avoid run-away
  // compile times on large CFGs without hampering sensible code. Arbitrarily
  // chosen.
  unsigned Limit = 32;

  SmallPtrSet<const BasicBlock *, 32> Visited;
  do {
    BasicBlock *BB = Worklist.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;
    if (BB == StopBB)
      return true;
    if (ExclusionSet && ExclusionSet->count(BB))
      continue;
    if (DT && DT->dominates(BB, StopBB))
      return true;

    const Loop *Outer = nullptr;
    if (LI) {
      Outer = getOutermostLoop(LI, BB);
      // If we're in a loop with a hole, not all blocks in the loop are
      // reachable from all other blocks. That implies we can't simply
      // jump to the loop's exit blocks, as that exit might need to pass
      // through an excluded block. Clear Outer so we process BB's
      // successors.
      if (LoopsWithHoles.count(Outer))
        Outer = nullptr;
      if (StopLoop && Outer == StopLoop)
        return true;
    }

    if (!--Limit) {
      // We haven't been able to prove it one way or the other.
      // Conservatively answer true -- that there is potentially a path.
      return true;
    }

    if (Outer) {
      // All blocks in a single loop are reachable from all other blocks.
      // From any of these blocks, we can skip directly to the exits of
      // the loop, ignoring any other blocks inside the loop body.
      Outer->getExitBlocks(Worklist);
    } else {
      Worklist.append(succ_begin(BB), succ_end(BB));
    }
  } while (!Worklist.empty());

  // We have exhausted all possible paths and are certain that 'To' can not be
  // reached from 'From'.
  return false;
}


bool isPotentiallyReachable(
    const Instruction *A, const Instruction *B,
    const SmallPtrSetImpl<const BasicBlock *> *ExclusionSet,
    const DominatorTree *DT, const LoopInfo *LI) {
  assert(A->getParent()->getParent() == B->getParent()->getParent() &&
         "This analysis is function-local!");

  SmallVector<BasicBlock *, 32> Worklist;

  if (A->getParent() == B->getParent()) {
    // The same block case is special because it's the only time we're
    // looking within a single block to see which instruction comes first.
    // Once we start looking at multiple blocks, the first instruction of
    // the block is reachable, so we only need to determine reachability
    // between whole blocks.
    BasicBlock *BB = const_cast<BasicBlock *>(A->getParent());

    // If the block is in a loop then we can reach any instruction in the
    // block from any other instruction in the block by going around a
    // backedge.
    if (LI && LI->getLoopFor(BB) != nullptr)
      return true;

    // Linear scan, start at 'A', see whether we hit 'B' or the end first.
    for (BasicBlock::const_iterator I = A->getIterator(), E = BB->end(); I != E;
         ++I) {
      if (&*I == B)
        return true;
    }

    // Can't be in a loop if it's the entry block -- the entry block may not
    // have predecessors.
    if (BB == &BB->getParent()->getEntryBlock())
      return false;

    // Otherwise, continue doing the normal per-BB CFG walk.
    Worklist.append(succ_begin(BB), succ_end(BB));

    if (Worklist.empty()) {
      // We've proven that there's no path!
      return false;
    }
  } else {
    Worklist.push_back(const_cast<BasicBlock *>(A->getParent()));
  }

  if (DT) {
    if (DT->isReachableFromEntry(A->getParent()) &&
        !DT->isReachableFromEntry(B->getParent()))
      return false;
    if (!ExclusionSet || ExclusionSet->empty()) {
      if (A->getParent() == &A->getParent()->getParent()->getEntryBlock() &&
          DT->isReachableFromEntry(B->getParent()))
        return true;
      if (B->getParent() == &A->getParent()->getParent()->getEntryBlock() &&
          DT->isReachableFromEntry(A->getParent()))
        return false;
    }
  }

  return isPotentiallyReachableFromMany(
      Worklist, const_cast<BasicBlock *>(B->getParent()), ExclusionSet, DT, LI);
}



class IR2Vec_FA {

private:
  llvm::Module &M;
  std::string res;
  IR2Vec::Vector pgmVector;
  unsigned dataMissCounter;
  unsigned cyclicCounter;

  llvm::SmallDenseMap<llvm::StringRef, unsigned> memWriteOps;
  llvm::SmallDenseMap<const llvm::Instruction *, bool> livelinessMap;
  llvm::SmallDenseMap<llvm::StringRef, unsigned> memAccessOps;

  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128>
      instVecMap;
  llvm::SmallMapVector<const llvm::Function *, IR2Vec::Vector, 16> funcVecMap;

  llvm::SmallMapVector<const llvm::Function *,
                       llvm::SmallVector<const llvm::Function *, 10>, 16>
      funcCallMap;

  llvm::SmallMapVector<const llvm::Instruction *,
                       llvm::SmallVector<const llvm::Instruction *, 10>, 16>
      writeDefsMap;


  llvm::SmallVector<const llvm::Instruction *, 20> instSolvedBySolver;

  llvm::SmallVector<llvm::SmallVector<const llvm::Instruction *, 10>, 10>
      allSCCs;

  llvm::SmallMapVector<const llvm::Instruction *,
                       llvm::SmallVector<llvm::Instruction *, 16>, 16>
      killMap;

  std::map<int, std::vector<int>> SCCAdjList;

  void getAllSCC();

  IR2Vec::Vector getValue(std::string key);
  void collectWriteDefsMap(llvm::Module &M);
  void getTransitiveUse(
      const llvm::Instruction *root, const llvm::Instruction *def,
      llvm::SmallVector<const llvm::Instruction *, 100> &visitedList,
      llvm::SmallVector<const llvm::Instruction *, 10> toAppend = {});
  llvm::SmallVector<const llvm::Instruction *, 10>
  getReachingDefs(const llvm::Instruction *, unsigned i);

  void solveSingleComponent(
      const llvm::Instruction &I,
      llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 16>
          &instValMap);
  void getPartialVec(const llvm::Instruction &I,
                     llvm::SmallMapVector<const llvm::Instruction *,
                                          IR2Vec::Vector, 16> &instValMap);

  void solveInsts(llvm::SmallMapVector<const llvm::Instruction *,
                                       IR2Vec::Vector, 16> &instValMap);
  std::vector<int> topoOrder(int size);

  void topoDFS(int vertex, std::vector<bool> &Visited,
               std::vector<int> &visitStack);

  void inst2Vec(const llvm::Instruction &I,
                llvm::SmallVector<llvm::Function *, 15> &funcStack,
                llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector,
                                     16> &instValMap);
  void traverseRD(const llvm::Instruction *inst,
                  std::unordered_map<const llvm::Instruction *, bool> &Visited,
                  llvm::SmallVector<const llvm::Instruction *, 10> &timeStack);

  void DFSUtil(const llvm::Instruction *inst,
               std::unordered_map<const llvm::Instruction *, bool> &Visited,
               llvm::SmallVector<const llvm::Instruction *, 10> &set);

  void bb2Vec(llvm::BasicBlock &B,
              llvm::SmallVector<llvm::Function *, 15> &funcStack);


  bool isMemOp(llvm::StringRef opcode, unsigned &operand,
               llvm::SmallDenseMap<llvm::StringRef, unsigned> map);
  std::string splitAndPipeFunctionName(std::string s);

  void TransitiveReads(llvm::SmallVector<llvm::Instruction *, 16> &Killlist,
                       llvm::Instruction *Inst, llvm::BasicBlock *ParentBB);
  llvm::SmallVector<llvm::Instruction *, 16>
  createKilllist(llvm::Instruction *Arg, llvm::Instruction *writeInst);

  // For Debugging
  void print(IR2Vec::Vector t, unsigned pos) { llvm::outs() << t[pos]; }

  void updateFuncVecMap(
      llvm::Function *function,
      llvm::SmallSet<const llvm::Function *, 16> &visitedFunctions);

  void updateFuncVecMapWithCallee(const llvm::Function *function);

public:
  IR2Vec_FA(llvm::Module &M) : M{M} {

    pgmVector = IR2Vec::Vector(DIM, 0);
    res = "";

    memWriteOps.try_emplace("store", 1);
    memWriteOps.try_emplace("cmpxchg", 0);
    memWriteOps.try_emplace("atomicrmw", 0);

    memAccessOps.try_emplace("getelementptr", 0);
    memAccessOps.try_emplace("load", 0);

    dataMissCounter = 0;
    cyclicCounter = 0;

    collectWriteDefsMap(M);

    llvm::CallGraph cg = llvm::CallGraph(M);

    for (auto callItr = cg.begin(); callItr != cg.end(); callItr++) {
      if (callItr->first && !callItr->first->isDeclaration()) {
        auto ParentFunc = callItr->first;
        llvm::CallGraphNode *cgn = callItr->second.get();
        if (cgn) {

          for (auto It = cgn->begin(); It != cgn->end(); It++) {

            auto func = It->second->getFunction();
            if (func && !func->isDeclaration()) {
              funcCallMap[ParentFunc].push_back(func);
            }
          }
        }
      }
    }
  }

  std::map<std::string, IR2Vec::Vector> opcMap;

  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128>
  getInstVecMap() {
    return instVecMap;
  }

  llvm::SmallMapVector<const llvm::Function *, IR2Vec::Vector, 16>
  getFuncVecMap() {
    return funcVecMap;
  }

  IR2Vec::Vector getProgramVector() { return pgmVector; }

  void func2Vec(llvm::Function &F);

  llvm::SmallMapVector<const llvm::Instruction *,llvm::SmallVector<const llvm::Instruction *, 10>, 16> instReachingDefsMap;
  // Reverse instReachingDefsMap
  llvm::SmallMapVector<const llvm::Instruction *,llvm::SmallVector<const llvm::Instruction *, 10>, 16> reverseReachingDefsMap;
      
    
};

void IR2Vec_FA::getTransitiveUse(
    const Instruction *root, const Instruction *def,
    SmallVector<const Instruction *, 100> &visitedList,
    SmallVector<const Instruction *, 10> toAppend) {
  unsigned operandNum = 0;
  visitedList.push_back(def);

  for (auto U : def->users()) {
    if (auto use = dyn_cast<Instruction>(U)) {
      if (std::find(visitedList.begin(), visitedList.end(), use) ==
          visitedList.end()) {

        // outs() << "\nDef " << /* def << */ " ";
        // def->print(outs(), true); 
        // outs() << "\n";
        // outs() << "Use " << /* use << */ " ";
        // use->print(outs(), true); outs() << "\n";


        if (isMemOp(use->getOpcodeName(), operandNum, memWriteOps) &&
            use->getOperand(operandNum) == def) {
          writeDefsMap[root].push_back(use);
        } else if (isMemOp(use->getOpcodeName(), operandNum, memAccessOps) &&
                   use->getOperand(operandNum) == def) {
          getTransitiveUse(root, use, visitedList, toAppend);
        }
      }
    }
  }
  return;
}



void IR2Vec_FA::collectWriteDefsMap(Module &M) {
  SmallVector<const Instruction *, 100> visitedList;
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      EliminateUnreachableBlocks(F);
      for (auto &BB : F) {
        for (auto &I : BB) {
          unsigned operandNum = 0;
          if ((isMemOp(I.getOpcodeName(), operandNum, memAccessOps) ||
               isMemOp(I.getOpcodeName(), operandNum, memWriteOps) ||
               strcmp(I.getOpcodeName(), "alloca") == 0) &&
              std::find(visitedList.begin(), visitedList.end(), &I) ==
                  visitedList.end()) {
            if (I.getNumOperands() > 0) {

              // I.print(outs()); outs() << "\n";
              // outs() << "operandnum = " << operandNum << "\n";

              if (auto parent =
                      dyn_cast<Instruction>(I.getOperand(operandNum))) {
                if (std::find(visitedList.begin(), visitedList.end(), parent) ==
                    visitedList.end()) {
                  visitedList.push_back(parent);
                  getTransitiveUse(parent, parent, visitedList);
                }
              }
            }
          }
        }
      }
    }
  }
}


void IR2Vec_FA::TransitiveReads(SmallVector<Instruction *, 16> &Killlist,
                                Instruction *Inst, BasicBlock *ParentBB) {
  assert(Inst != nullptr);
  unsigned operandNum;
  bool isMemAccess = isMemOp(Inst->getOpcodeName(), operandNum, memAccessOps);

  if (!isMemAccess)
    return;
  auto parentI = dyn_cast<Instruction>(Inst->getOperand(operandNum));
  if (parentI == nullptr)
    return;
  if (ParentBB == parentI->getParent())
    Killlist.push_back(parentI);
  TransitiveReads(Killlist, parentI, ParentBB);
}

SmallVector<Instruction *, 16>
IR2Vec_FA::createKilllist(Instruction *Arg, Instruction *writeInst) {

  SmallVector<Instruction *, 16> KillList;
  SmallVector<Instruction *, 16> tempList;
  BasicBlock *ParentBB = writeInst->getParent();

  unsigned opnum;

  for (User *U : Arg->users()) {
    if (Instruction *UseInst = dyn_cast<Instruction>(U)) {
      if (isMemOp(UseInst->getOpcodeName(), opnum, memWriteOps)) {
        Instruction *OpInst = dyn_cast<Instruction>(UseInst->getOperand(opnum));
        if (OpInst && OpInst == Arg)
          tempList.push_back(UseInst);
      }
    }
  }

  for (auto I = tempList.rbegin(); I != tempList.rend(); I++) {
    if (*I == writeInst)
      break;
    if (ParentBB == (*I)->getParent())
      KillList.push_back(*I);
  }

  return KillList;
}

void IR2Vec_FA::func2Vec(Function &F)
{

  instReachingDefsMap.clear();
  allSCCs.clear();
  reverseReachingDefsMap.clear();
  SCCAdjList.clear();

  ReversePostOrderTraversal<Function *> RPOT(&F);

  for (auto *b : RPOT) {
    unsigned opnum;
    SmallVector<Instruction *, 16> lists;
    for (auto &I : *b) {
      lists.clear();
      if (isMemOp(I.getOpcodeName(), opnum, memWriteOps) &&
          dyn_cast<Instruction>(I.getOperand(opnum))) {
        Instruction *argI = cast<Instruction>(I.getOperand(opnum));
        lists = createKilllist(argI, &I);
        TransitiveReads(lists, argI, I.getParent());
        if (argI->getParent() == I.getParent())
          lists.push_back(argI);
        killMap[&I] = lists;
      }
    }
  }

  for (auto *b : RPOT) {
    for (auto &I : *b) {
      for (unsigned int i = 0; i < I.getNumOperands(); i++) {
        if (isa<Instruction>(I.getOperand(i))) {
          auto RD = getReachingDefs(&I, i);
          if (instReachingDefsMap.find(&I) == instReachingDefsMap.end()) {
            instReachingDefsMap[&I] = RD;
          } else {
            auto RDList = instReachingDefsMap[&I];
            RDList.insert(RDList.end(), RD.begin(), RD.end());
            instReachingDefsMap[&I] = RDList;
          }
        }
      }
    }
  }


  // for (auto &Inst: instReachingDefsMap) {
  //   auto RD = Inst.second;
  //   outs() << "(" << Inst.first << ")";
  //   Inst.first->print(outs());
  //   outs() << "\n RD : ";
  //   for (auto defs : RD) {
  //     defs->print(outs());
  //     outs() << "(" << defs << ") ";
  //   }
  //   outs() << "\n";
  // };

  // one time Reversing instReachingDefsMap to be used to calculate SCCs
  for (auto &I : instReachingDefsMap) {
    auto RD = I.second;
    for (auto defs : RD) {
      if (reverseReachingDefsMap.find(defs) == reverseReachingDefsMap.end()) {
        llvm::SmallVector<const llvm::Instruction *, 10> revDefs;
        revDefs.push_back(I.first);
        reverseReachingDefsMap[defs] = revDefs;
      } else {
        auto defVector = reverseReachingDefsMap[defs];
        defVector.push_back(I.first);
        reverseReachingDefsMap[defs] = defVector;
      }
    }
  }
  return;

}




bool IR2Vec_FA::isMemOp(StringRef opcode, unsigned &operand,
                        SmallDenseMap<StringRef, unsigned> map) {
  bool isMemOperand = false;
  auto It = map.find(opcode);
  if (It != map.end()) {
    isMemOperand = true;
    operand = It->second;
  }
  return isMemOperand;
}


SmallVector<const Instruction *, 10>
IR2Vec_FA::getReachingDefs(const Instruction *I, unsigned loc) {
  
      //outs() << "Call to getReachingDefs Started****************************\n";

  auto parent = dyn_cast<Instruction>(I->getOperand(loc));
  if (!parent)
    return {};
  SmallVector<const Instruction *, 10> RD;
  SmallVector<const Instruction *, 10> probableRD;

  // outs() << "Inside RD for : ";
  // I->print(outs()); outs() << "\n";

  if (writeDefsMap[parent].empty()) {
    RD.push_back(parent);
    return RD;
  }

  if (writeDefsMap[parent].size() >= 1) {
    SmallMapVector<const BasicBlock *, SmallVector<const Instruction *, 10>, 16>
        bbInstMap;
    // Remove definitions which don't reach I
    for (auto it : writeDefsMap[parent]) {
      if (it != I && isPotentiallyReachable(it, I)) {

        probableRD.push_back(it);
      }
    }
    probableRD.push_back(parent);

    // outs() << "----PROBABLE RD---" << "\n";
    // for (auto i : probableRD) {
    //   i->print(outs()); outs() << "\n";
    //   bbInstMap[i->getParent()].push_back(i);
    // }

    // outs() << "contents of bbinstmap:\n"; 
    // for (auto i : bbInstMap) {
    //   for (auto j : i.second) {
    //     j->print(outs());
    //     outs() << "\n";
    //   }
    //   outs() << "+++++++++++++++++++++++++\n";
    // }

    // If there is a reachable write within I's basic block only that defn
    // would reach always If there are more than one defn, take the
    // immediate defn before I
    if (!bbInstMap[I->getParent()].empty()) {

      // outs() << "--------Within BB--------\n";
      // I->print(outs()); outs() << "\n";

      auto orderedVec = bbInstMap[I->getParent()];
      const Instruction *probableRD = nullptr;
      for (auto &i : *(I->getParent())) {
        if (&i == I)
          break;
        else {
          if (std::find(orderedVec.begin(), orderedVec.end(), &i) !=
              orderedVec.end())
            probableRD = &i;
        }
      }

      if (probableRD != nullptr) {

      //  outs() << "Returning: ";
      //  probableRD->print(outs()); outs() << "\n";

        RD.push_back(probableRD);
        return RD;
      }
    }

    // outs() << "--------Across BB--------\n";

    SmallVector<const Instruction *, 10> toDelete;
    for (auto it : bbInstMap) {

      // outs() << "--------INSTMAP BEGIN--------\n";
      // it.first->print(outs()); outs() << "\n";


      bool first = true;
      for (auto it1 : bbInstMap[it.first]) {
        if (first) {
          first = false;
          continue;
        }
        toDelete.push_back(it1);

        // it1->print(outs()); outs() << "\n";
      }
      // outs() << "--------INSTMAP END--------\n";
    }
    auto tmp = probableRD;
    probableRD = {};
    for (auto i : tmp) {
      if (std::find(toDelete.begin(), toDelete.end(), i) == toDelete.end())
        probableRD.push_back(i);
    }

    // I->print(outs()); outs() << "\n"; outs() << "probableRD: \n";
    //              for (auto i: probableRD) i->print(outs());
    //              outs() << "\n"; outs() << "-----------------\n";

    SmallPtrSet<const BasicBlock *, 10> bbSet;
    SmallMapVector<const BasicBlock *, const Instruction *, 16> refBBInstMap;

    for (auto i : probableRD) {
      bbSet.insert(i->getParent());
      refBBInstMap[i->getParent()] = i;
      // outs() << i->getParent()->getName().str() << "\n";
    }
    for (auto i : bbSet) {
      // i->print(outs()); outs() << "\n";

      auto exclusionSet = bbSet;
      exclusionSet.erase(i);
      if (isPotentiallyReachable(refBBInstMap[i], I, &exclusionSet, nullptr,
                                 nullptr)) {
        RD.push_back(refBBInstMap[i]);

        // outs() << "refBBInstMap : ";
        // refBBInstMap[i]->print(outs()); outs() << "\n";
      }
    }
    
        // outs() << "****************************\n";
        // outs() << "Reaching defn for "; I->print(outs()); outs() << "\n";
        // for (auto i: RD) 
        //   i->print(outs());
        // outs() << "\n";
        // outs()<< "Call to getReachingDefs Ended****************************\n";

    return RD;
  }

  llvm_unreachable("unreachable");
  return {};
}

#endif
