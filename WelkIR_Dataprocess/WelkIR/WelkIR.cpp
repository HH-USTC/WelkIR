
#include <string>
#include <fstream>
#include <utility>
#include <map>
#include <cassert> 
#include <filesystem> 
#include <iostream>
#include <regex>
#include <unordered_map>
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"



#include "../WelkIR_COMMON/rapidjson/document.h"
#include "../WelkIR_COMMON/rapidjson/stringbuffer.h"
#include "../WelkIR_COMMON/rapidjson/istreamwrapper.h"
#include "../WelkIR_COMMON/structs.h"
#include "../WelkIR_COMMON/utils.h"
#include "../WelkIR_COMMON/FlowAware.h"

using namespace llvm;
#define DEBUG_TYPE "WelkIR"

static cl::opt<std::string> labelFilename("labelFilename", cl::desc("Optionally specify input filename for labels."), cl::value_desc("filename"), cl::init(""));
static cl::opt<std::string> OutputPath("OutputPath", cl::desc("Optionally specify OutputPath for JSON output."), cl::value_desc("OutputPath"), cl::init(""));
static cl::opt<std::string> OutputFile_prefix("OutputFile_prefix", cl::desc("Optionally specify output OutputFile_prefix for JSON output."), cl::value_desc("OutputFile_prefix"), cl::init(""));
static cl::opt<std::string> IRFile_Type("IRFile_Type", cl::desc("IRFile_Type, Vulnerability or Patch"), cl::value_desc("IRFile_Type"), cl::init(""));

namespace
{
	using Label = std::pair<int, int>;
	//add 2024-10-17	
	using WelkFuncMap = std::pair<Function*, WelkFunction*>;
	using WelkBlockMap = std::pair<BasicBlock*, WelkBlock*>;
	using WelkInstMap = std::pair<Instruction*, WelkInstruction*>;

	struct WelkIR : public ModulePass 
  	{
		static char ID;
		WelkIR() : ModulePass(ID) {}

    	bool runOnModule(Module &M) override
		{
			errs() << "WelkIR is starting.\n";	
			std::vector<std::string> patchFilesName;
			for ( llvm::NamedMDNode &NamedMD : M.named_metadata()) 
			{
				if (NamedMD.getName() == "llvm.dbg.cu") 
				{ 
					for (const MDNode *CU : NamedMD.operands())
					{
						if (const DICompileUnit *CompileUnit = dyn_cast<DICompileUnit>(CU)) 
						{
							// 获取源文件信息
							if (const DIFile *File = CompileUnit->getFile()) 
							{
								StringRef Filename = File->getFilename();
								patchFilesName.push_back(extract_filename(Filename.str())); 
								errs() << "Source File: " << Filename.str() << "\n";
							}
						}
					}
				}
			}


			if (patchFilesName.empty()) {
				errs() << "No debug information available.\n";
			}


			// First step: Check to see if we have labels.
			std::vector<std::pair<int, int>> Lines_Vul_labels;
			std::vector<std::pair<int, int>> Func_labels;
			std::multimap<int, std::pair<int, std::string>> func_startline_map; 

			if(labelFilename != "")
			{
				errs() << "Reading labels from: " << labelFilename << "\n";

				// Use rapidjson to read the file into a data structure.
				std::ifstream ifs(labelFilename);
				if ( !ifs.is_open() )
				{
					errs() << "ERROR: Could not open file for reading!\n";
					return false;
				}

				rapidjson::IStreamWrapper isw (ifs);
				rapidjson::Document doc;
				doc.ParseStream(isw);

				// assert(doc.IsObject());
				assert(doc.IsArray());
				for (int i = 0; i < doc.Size(); i++)
				{
					const rapidjson::Value& func_obj = doc[i];
					assert(func_obj.IsObject());

					if (!func_obj.HasMember("func_startline") || !func_obj.HasMember("func_label")  )
					{
						errs() << "ERROR: Function object missing 'name' or not a string!\n";
						continue;
					}
					//|| 
					int func_startline = func_obj["func_startline"].GetInt();
					int func_label = func_obj["func_label"].GetInt();
					std::string func_codefile = func_obj["func_codefile"].GetString();

					Func_labels.push_back(Label(func_startline, func_label));
					func_startline_map.insert({func_startline, {func_label, func_codefile}});


					if (func_label == 1)
					{
						if (!func_obj.HasMember("vul_lines") )
							continue;
						const rapidjson::Value& vul_lines_arr = func_obj["vul_lines"];
						for (int j = 0; j < vul_lines_arr.Size(); j++) 
							Lines_Vul_labels.push_back(Label(vul_lines_arr[j].GetInt(), 1));
					}
				}
			}
			else 
			{ 
	
				errs() << "No labels specified.\n"; 
				// return false;
			}


			std::string IRFileName = processModule(M);
			std::string OutputJsonFile_prefix = OutputFile_prefix + IRFileName;


			//add  2024-10-19
			SetVector<WelkFunction*> WelkFunctions;
			DenseMap<Function*, WelkFunction*> Function_mappings;  

			SetVector<WelkBlock*> WelkBlocks;
			DenseMap<BasicBlock*, WelkBlock*> Block_mappings;  

			SetVector<WelkInstruction*> WelkInstructions;
			DenseMap<Instruction*, WelkInstruction*> Instruction_mappings;  

			// SetVector<Link*>  Functionlinks;
			// SetVector<Link*>  Blocklinks;

			SetVector<Link*>  InstructionLinks;   //CFG
			SetVector<Link*>  DefUse_InstLinks;   //DefUse
			SetVector<Link*>  ReachingDefs_InstLinks; //ReachingDefs
			
			//2024 10-17
			uint Function_id = 0;
			uint Block_id = 0;
			uint Instruction_id = 0;


			errs() << "Analyzing Module " << M.getSourceFileName() << "\n";

			// Second step: Iterate through the functions and instructions in the module, build up nodes.
			for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
			{
				Function* func = &(*F);	
				
				// Check for function headers
				if(func->size() <= 0) 
				{
					//errs() << "  " << func->getName() << " is a function declaration, ignoring.\n";
					continue;
				}
				//Filter  intrinsic functions
				if (func->getIntrinsicID() == llvm::Intrinsic::donothing ||
					func->getIntrinsicID() == llvm::Intrinsic::dbg_declare ||
					func->getIntrinsicID() == llvm::Intrinsic::dbg_label ||
					func->getIntrinsicID() == llvm::Intrinsic::dbg_value ||
					func->getName().startswith("sancov.module_ctor") ||
					func->getName().startswith("asan.module_ctor") ||
					func->getName().startswith("asan.module_dtor")
					)
				{
					//errs() << "  Skipping intrinsic function: " << func->getName() << "\n";
					continue;
				}

				int funcLine = 0;
				if (DISubprogram *Subprogram = func->getSubprogram() ) 
				{
					DIFile *File = Subprogram->getFile();
					StringRef FileName_Full = File->getFilename();
					std::string FileName_Full_Str = FileName_Full.str();
					std::string FileName = extract_filename(FileName_Full_Str);
					funcLine = Subprogram->getLine();

					if (std::find(patchFilesName.begin(), patchFilesName.end(), FileName) != patchFilesName.end()) 
					{
						errs() << "FileName " << FileName << " is in patchFileName list.\n";	
					} 
					else 
					{
						errs() << "FileName " << FileName << " is NOT in patchFileName list.\n";
						continue;  
					}
				}
				else
					continue;

				//2024 10-17 Create a node for this WelkFunction, and create mapping to node for control flow link creation.
				WelkFunction* funcnode = new WelkFunction;
				funcnode->Func = func;
				Function_mappings.insert(WelkFuncMap(func, funcnode));
				funcnode->Funcname = func->getName().str();			
				funcnode->Func_id = Function_id++;
				funcnode->label = "0";  
				funcnode->labelFilename = "";
				funcnode->func_codefile = "";
				funcnode->func_startLine = funcLine;

				for(Label fun_lbl : Func_labels)
				{
					int Func_labels_startline = fun_lbl.first;
					

					if ((Func_labels_startline - funcLine) <= 1 && (funcLine - Func_labels_startline) <= 1) 
					{
						funcnode->label = std::to_string(fun_lbl.second);

						auto range = func_startline_map.equal_range(Func_labels_startline);
						auto it = range.first;
						std::string func_codefile = it->second.second;
						// add 2025 01-17
						funcnode ->labelFilename = labelFilename;
						funcnode ->func_codefile = func_codefile;
					}
				}
				
				// if (funcnode->label == "" )
				// 		continue;
				
				WelkFunctions.insert(funcnode);
				int intra_Block_num = 0;		
				for (Function::iterator BB = funcnode->Func->begin(), BBE = funcnode->Func->end(); BB != BBE; ++BB)
				{
					BasicBlock* block = (&(*BB));
					
					//2024 10-17 Create a node for this WelkBlocks, and create mapping to node for control flow link creation.
					WelkBlock* Blocknode = new WelkBlock;
					Blocknode->Block = block;
					Block_mappings.insert(WelkBlockMap(block, Blocknode));
					
					Blocknode->ParentFunc = funcnode;
					Blocknode->Block_id = Block_id++;
					Blocknode->BlockName = block->getName().str();  
					Blocknode->ExitInstruction = block->getTerminator();
					WelkBlocks.insert(Blocknode);
					funcnode->WelkBlocks.insert(Blocknode);

					intra_Block_num = 0;

					for(BasicBlock::iterator I = Blocknode->Block->begin(), IE = Blocknode->Block->end(); I != IE; ++I)
					{
						Instruction* instr = (&(*I));			

						WelkInstruction* Instnode = new WelkInstruction;
						Instnode->instruction = instr;
						Instruction_mappings.insert(WelkInstMap(instr, Instnode));
						Instnode->ParentBlock = Blocknode;
						Instnode->Inst_id  = Instruction_id++;
						Instnode->Block_id =  Blocknode->Block_id;
						Instnode->intra_Block_num = intra_Block_num;
						intra_Block_num++;
						Instnode->operation = (Operation) instr->getOpcode();
						Instnode->dtype = instr->getType()->getTypeID();
						
						Instnode->current_function = func->getName().str();

						// Get source filename and line number with debug info
						DILocation* loc = instr->getDebugLoc();
						if(loc)
						{
							Instnode->line_number = loc->getLine();
							llvm::StringRef fullPath = loc->getFilename();
							
							Instnode->filename = llvm::sys::path::filename(fullPath).str();
							// errs() << "  filename: " << Instnode->filename << "\n";


							const DISubprogram *subprogram = loc->getScope()->getSubprogram();
							if (subprogram)
							{
								llvm::StringRef originalFuncName = subprogram->getName();
								Instnode->original_function = originalFuncName.str(); 
							}
							else
							{
								Instnode->original_function = ""; 
							}	
						}
						else
						{
							Instnode->line_number = 0;  // Indicates that no debug info was found.
							Instnode->filename = "";
							Instnode->original_function = "";
						}

						// IR source code of the instruction
						std::string irString;
						raw_string_ostream rso(irString);
						instr->print(rso);
						Instnode->irString = irString;

						if(CallBase* cbi = dyn_cast<CallBase>(instr))
						{

							// Record the function call target
							if(cbi->isInlineAsm()) { Instnode->call_func_name = "inline_assembly"; }
							else if (Function* called_func = cbi->getCalledFunction()) { 
								//Filter  intrinsic functions
								if (called_func->getIntrinsicID() == llvm::Intrinsic::donothing ||
									called_func->getIntrinsicID() == llvm::Intrinsic::dbg_declare ||
									called_func->getIntrinsicID() == llvm::Intrinsic::dbg_label ||
									called_func->getIntrinsicID() == llvm::Intrinsic::dbg_value)
								{
									//errs() << "  Skipping intrinsic function: " << called_func->getName() << "\n";
									Instnode->call_func_name = "llvm_Intrinsic";
								}			
								// To reduce node dimensionality we only record target function names for external calls.
								else if(called_func->size() <= 0 && called_func->isDeclaration() ){
									Instnode->call_func_name = called_func->getName().str();
								}
								else { Instnode->call_func_name = ""; }
							}
							else { Instnode->call_func_name = "indirect_call"; }

						}
					
						// Check imported labels for supplementary information, only label tagged instructions that have debug data associated with them
						Instnode->labels = "";
						if(Lines_Vul_labels.size() != 0 && Instnode->line_number != 0)
						{
							for(Label Line_lbl : Lines_Vul_labels)
							{
								if( Instnode->line_number == Line_lbl.first)
								{

									Instnode->labels = "vul";
									funcnode->label = "1";

								}
							}

						}
						//Add insert to WelkInstructions  list
						WelkInstructions.insert(Instnode);
						Blocknode->WelkInstructions.insert(Instnode);
						funcnode->WelkInstructions.insert(Instnode);
					}	
				}	
			}

		errs() << "  Analyzing function Count: " << WelkFunctions.size() << "\n";
		
		int Func_Num = 0;
		for (auto funcnode : WelkFunctions) 
		{
			// Only handle label functions
			// if(funcnode->label == "")
			// 	continue;
			
			errs() << "Func_Num"<< Func_Num << "Funcname: " << funcnode->Funcname << " function blockNodes: " << funcnode->WelkBlocks.size() << "\n";
			Func_Num = Func_Num + 1;

			//ADD Control Flow  
			for (auto blockNode : funcnode->WelkBlocks) {
				
				auto &instVec = blockNode->WelkInstructions;
				//errs() << "WelkBlocks instVec.size(): " << instVec.size() << "\n";
				if (!instVec.empty() ) {

					auto lastInstNode = instVec.back();
					if (lastInstNode == nullptr) 
					{
						errs() << "Error: lastInstNode is null!\n";
						continue;
					}

					// llvm::Instruction* lastInst = lastInstNode->instruction;
					llvm::Instruction* lastInst = blockNode->ExitInstruction;
					if (lastInst == nullptr )
					{
        				errs() << "Error: ExitInstruction is null!\n";
        				continue;
    				}

					assert(lastInst && "Expected a TerminatorInst at the end of the basic block!");
					assert(instVec.back()->instruction == lastInst && "The last instruction in WelkInstructions should be the TerminatorInst of the block.");

					llvm::BasicBlock* BB = lastInst->getParent();
					if (BB == nullptr) {
						errs() << "Error: Parent BasicBlock is null for lastInst!\n";
						continue; 
					}
					llvm::BasicBlock* WelkBlock_Block = blockNode->Block ;

					if(WelkBlock_Block != BB)
					{
						errs() << "WelkBlock_Block != BB" << "\n";
						continue;
					}

					if (BB->empty()) {
						errs() << "Error: BasicBlock is empty for lastInst!\n";
						continue;  
					}

					int succIdx=0;
					llvm::Instruction* backInst = &BB->back();
					if (backInst == nullptr) 
					{
						errs() << "Error: backInst is null for the last instruction!\n";
						continue; 
					}

					if (lastInst == backInst ) {
	
						for (llvm::BasicBlock* SuccBB : llvm::successors(BB)) 
						{
							if (SuccBB == nullptr) {
								errs() << "Error: Successor BasicBlock is null!\n";
								continue; 
							}
							succIdx=0;

							if (!SuccBB->empty()) {
								llvm::Instruction* succInst = &*(SuccBB->begin());
								if (succInst == nullptr) {
                					errs() << "Error: Successor instruction is null!\n";
                					continue;
            					}


								auto mappedIter = Instruction_mappings.find(succInst);					
								if (mappedIter != Instruction_mappings.end()) {

									auto succInstNode = mappedIter->second;
									Link* link = new Link;
									link->source = lastInstNode->Inst_id;
									link->target = succInstNode->Inst_id;
									link->type = ControlFlow;
									// link->dtype = llvm::Type::VoidTyID;
									link->controlType = getControlType(lastInst, succIdx); 
									succIdx++;

									InstructionLinks.insert(link);
								}
							}
						}
					}

					if (instVec.size() < 2)
					{
						errs() << "Ignore adjacent instructions" << "\n";
						continue;
					}				

					for (size_t i = 0; i < instVec.size() - 1; ++i) {
						auto currentInstNode = instVec[i];
						auto nextInstNode = instVec[i + 1];

						Link* link = new Link;
						link->source = currentInstNode->Inst_id;
						link->target = nextInstNode->Inst_id;
						link->type = ControlFlow;
						// link->dtype = llvm::Type::VoidTyID;
						link->controlType = Intra_block; 
						InstructionLinks.insert(link);
					}
				
				}
			}
			errs() << funcnode->Funcname << " finish ControlFlow  InstructionLinks.size(): " << InstructionLinks.size() << "\n";

			//builfd DefUse
			for (auto Instnode : funcnode->WelkInstructions)
			{

				if (Instnode == nullptr)
				{
					errs() << "Error: Instruction for Instnode " << Instnode->Inst_id << " is null!\n";
					continue; 
				}

				if (Instnode->instruction == nullptr)
				{
					errs() << "Error: Instruction for Instnode " << Instnode->Inst_id << " is null!\n";
					continue;  
				}


				if (Instnode->instruction->getParent() != Instnode->ParentBlock->Block) {
					errs() << "Instnode->instruction->getParent() != Instnode->ParentBlock->Block" << "\n";
					continue;
				}

				if (isa<llvm::UnreachableInst>(Instnode->instruction)) {
					// errs() << "DefUse  UnreachableInst" << "\n";
					continue;
				}

				for (User* user : Instnode->instruction->users())
				{
					if (Instruction* i_user = dyn_cast<Instruction>(user))
					{
						if (i_user == nullptr) 
						{
							errs() << "Warning: User is not an Instruction!\n";
							continue; 
						}


						auto mappedIter = Instruction_mappings.find(i_user);
						if (mappedIter != Instruction_mappings.end()) 
						{

							Link* link = new Link;
							link->source = Instnode->Inst_id; 
							link->target = mappedIter->second->Inst_id; 
							link->type = DefUse;

							////////////////////////////add 20241029////////////////////////////
							link->dtype = Instnode->dtype; 
							// link->operation_dtype = mappedIter->second->operation;

							DefUse_InstLinks.insert(link);
						}
						else
						{
							llvm::errs() << "Warning: User instruction not found in Instruction_mappings!\n";
						}
					}
				}
			}
			errs() << "DefUse DefUse_InstLinks.size(): " <<  DefUse_InstLinks.size() << "\n";

			//ADD  ReachingDefs//////////////////
			IR2Vec_FA FA(M);
			Function *func = funcnode->Func;
			if (!func->isDeclaration()) 
			{
				FA.func2Vec(*func);
				size_t mapSize = FA.instReachingDefsMap.size();
				//errs() << "FA.instReachingDefsMap.size() " << mapSize << "\n";

				for (const auto &mapEntry : FA.instReachingDefsMap)
				{
					//const llvm::Instruction *keyInst = mapEntry.first;  
					llvm::Instruction *nonConstkeyInst = const_cast<llvm::Instruction *>(mapEntry.first); 

					const llvm::SmallVector<const llvm::Instruction *, 10> &reachingDefs = mapEntry.second; 
					for (const llvm::Instruction *defInst : reachingDefs)
					{
						llvm::Instruction *nonConstDefInst = const_cast<llvm::Instruction *>(defInst);
						Link* link = new Link;
						link->source = Instruction_mappings[nonConstDefInst]->Inst_id;
						link->target = Instruction_mappings[nonConstkeyInst]->Inst_id;
						link->type = ReachDefinition;
						link->operation_dtype = Instruction_mappings[nonConstDefInst]->operation;
						ReachingDefs_InstLinks.insert(link);
					}
				}
			}
			errs() << "ReachingDefs ReachingDefs_InstLinks.size(): " << ReachingDefs_InstLinks.size() << "\n";


			if(IRFile_Type == "Vulnerability")
			{
				// std::string Vul_OutputPath = OutputPath + "/" + "Vul_Func";
				std::string Vul_OutputPath = OutputPath;
				errs() << "Vul_OutputPath: " << Vul_OutputPath << "\n";
				//outpuGraphJSON
				outpuGraphJSON(Vul_OutputPath, Func_Num, funcnode->WelkInstructions, InstructionLinks, ControlFlow, OutputJsonFile_prefix, funcnode);
				outpuGraphJSON(Vul_OutputPath, Func_Num, funcnode->WelkInstructions, DefUse_InstLinks, DefUse, OutputJsonFile_prefix, funcnode);
				outpuGraphJSON(Vul_OutputPath, Func_Num, funcnode->WelkInstructions, ReachingDefs_InstLinks, ReachDefinition, OutputJsonFile_prefix, funcnode);

			}
			else if (IRFile_Type == "Patch")
			{
				// std::string Patch_OutputPath = OutputPath + "/" + "Patch_Func";
				std::string Patch_OutputPath = OutputPath;
				errs() << "Patch_OutputPath: " << Patch_OutputPath << "\n";
				//outpuGraphJSON
				outpuGraphJSON(Patch_OutputPath, Func_Num, funcnode->WelkInstructions, InstructionLinks, ControlFlow, OutputJsonFile_prefix, funcnode);
				outpuGraphJSON(Patch_OutputPath, Func_Num, funcnode->WelkInstructions, DefUse_InstLinks, DefUse, OutputJsonFile_prefix, funcnode);
				outpuGraphJSON(Patch_OutputPath, Func_Num, funcnode->WelkInstructions, ReachingDefs_InstLinks, ReachDefinition, OutputJsonFile_prefix, funcnode);
			}
				

			InstructionLinks.clear();
			DefUse_InstLinks.clear();
			ReachingDefs_InstLinks.clear();

		}

			
		errs() << "WelkIR is finished.\n" ;
		return false;
	}

	std::string processModule(llvm::Module &M) 
	{

    	std::string ModuleName = M.getModuleIdentifier();
    	size_t lastSlash = ModuleName.find_last_of("/\\");
   		std::string FileNameWithExtension = ModuleName.substr(lastSlash + 1);
    	//llvm::errs() << "File name with extension: " << FileNameWithExtension << "\n";
		return FileNameWithExtension;
  	}

  	};
}

char WelkIR::ID = 0;
static RegisterPass<WelkIR> X("WelkIR", "WelkIR Label Association and Feature Generation Pass");
