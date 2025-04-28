#include <string>
#include <sstream>
#include <string>
#include <fstream>
#include <utility>
#include <map>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <iostream>


#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

enum LinkType { ControlFlow, ControlDependence, DefUse, ReachDefinition, Callgraph };

enum ControlType {
    Intra_block = 0,        
    // Branch instructions
    Br = 1,              
    BrTrue = 2,         
    BrFalse = 3,         
    // Switch instruction
    SwitchDefault = 4,     
    SwitchCase = 5,        
    // Indirect branch instruction
    IndirectBr = 6,       
    // Return instruction
    Ret = 7,               
    // Resume instruction
    Resume = 8,           
    // Unreachable instruction
    Unreachable = 9,      
    // Exception handling instructions
    CleanupRetNormal = 10,       
    CleanupRetUnwind = 11,      
    CatchRet = 12,               
    CatchSwitch = 13,           
    CatchSwitchUnwind = 14,      
    // Invoke instruction
    InvokeNormal = 15,          
    InvokeException = 16,       
    // CallBr instruction
    CallBrNormal = 17,          
    CallBrIndirect = 18,         
    // Function call
    Call = 19, 
    OtherTerminator = 20       
};

// enumeration definitions
enum Operation { // Refer to llvm's Instruction.def. Multiple separate enums in LLVM with complementary ranges, combined here for simplicity.
    // Termination operations
    Operation_Return = 1, Operation_Branch = 2,      Operation_Switch = 3, Operation_IndirectBranch = 4, Operation_Invoke = 5, 
    Operation_Resume = 6, Operation_Unreachable = 7, Operation_CleanupReturn = 8, Operation_CatchReturn = 9, Operation_CatchSwitch = 10,
    Operation_CallBranch = 11,
    // Unary operations
    Operation_Negation = 12,
    // Binary operations
    Operation_Add = 13, Operation_FloatAdd = 14, Operation_Subtract = 15, 
    Operation_FloatSubtract = 16, Operation_Multiply = 17, Operation_FloatMultiply = 18, Operation_UnsignedDivide = 19, Operation_SignedDivide = 20,
    Operation_FloatDivide = 21, Operation_UnsignedModulus = 22, Operation_SignedModulus = 23, Operation_FloatModulus = 24, Operation_ShiftLeft = 25,
    Operation_LogicalShiftRight = 26, Operation_ArithmeticShiftRight = 27, Operation_And = 28, Operation_Or = 29, Operation_Xor = 30, 
    // Memory operations
    Operation_Allocate = 31, Operation_Load = 32, Operation_Store = 33, Operation_GetElementPointer = 34, Operation_Fence = 35, Operation_AtomicCompareExchange = 36, Operation_AtomicReadModifyWrite = 37,
    // Cast operations
    Operation_IntTruncate = 38, Operation_ZeroExtend = 39, Operation_SignExtend = 40, Operation_FloatToUInt = 41, Operation_FloatToSInt = 42, Operation_UIntToFloat = 43, Operation_SIntToFloat = 44,
    Operation_FloatTruncate = 45, Operation_FloatExtend = 46, Operation_PointerToInt = 47, Operation_IntToPointer = 48, Operation_BitCast = 49, Operation_AddressSpaceCast = 50,
    // Pad operations
    Operation_CleanupPad = 51, Operation_CatchPad = 52,
    // Other operations
    Operation_IntCompare = 53, Operation_FloatCompare = 54, Operation_PhiNode = 55, Operation_Call = 56, Operation_Select = 57, Operation_User1 = 58, Operation_User2 = 59, Operation_VarArgument = 60,
    Operation_ExtractElement = 61, Operation_InsertElement = 62, Operation_ShuffleVector = 63, Operation_ExtractValue = 64, Operation_InsertValue = 65, Operation_LandingPad = 66, Operation_Freeze =  67
}; 

struct WelkBlock;
struct WelkInstruction;
struct WelkFunction;

struct CallSite {
    llvm::CallBase* CallInstr;      
    WelkBlock* CallerBlock;         
    WelkInstruction* CallerInstNode; 
    WelkInstruction* NextInstNode;   
};

struct WelkFunction
{
    uint Func_id;  // Unique ID
    // std::vector<const BasicBlock*> AllBBVec;
    llvm::SetVector<WelkBlock*> WelkBlocks;
    llvm::SetVector<WelkInstruction*> WelkInstructions;
    std::string Funcname;
    llvm::Function* Func; // Corresponding LLVM instruciton   
    std::string label;
    /////////////////////////////////////
    std::string labelFilename;
    std::string func_codefile;
    uint func_startLine;
    // std::vector<CallSite> CallSites; 
    // llvm::SetVector<WelkFunction*> CalledFunctions;   // The collection of functions called by the current function
    // llvm::SetVector<WelkFunction*> CallingFunctions;  // Call the function collection of the current function
};

struct WelkBlock
{
    uint Block_id;  // Unique ID
    //std::vector<const Instruction*> AllInstVec;
    llvm::SetVector<WelkInstruction*> WelkInstructions;
    llvm::BasicBlock* Block; // Corresponding LLVM BasicBlock 
    WelkFunction* ParentFunc;
    std::string BlockName;  
    llvm::Instruction* ExitInstruction;
    llvm::SetVector<WelkBlock*> Successors;    // Subsequent basic block set
    // llvm::SetVector<WelkBlock*> Predecessors;  // Collection of precursor basic blocks
};

struct WelkInstruction
{
    uint Inst_id;  // Unique ID
    uint Block_id; // Block_id ID
    uint intra_Block_num;
    llvm::Instruction* instruction; // Corresponding LLVM instruciton 
    WelkBlock*   ParentBlock;
    std::string  irString;          // IR source code of the instruction
    Operation    operation;         // Type of operation performed, obtained via llvm::Instruction::getOpcode.
    llvm::Type::TypeID dtype;       // Here we directly use the LLVM TypeID enum
    int line_number;                // Line in source code. Value of 0 indicates no source line number found in debug information from Clang.
    std::string filename;           // Source code filename. Empty string indicates no source file found in debug information from Clang. 
    std::string call_func_name;           // Name of the target function for function call operations. Empty string indicates no target function. 
    std::string labels;             // Labels associated with line from JSON input.
    std::string current_function; //  function optimized located
    std::string original_function;//  function originally located
};


ControlType getControlType(llvm::Instruction* termInst, unsigned succIdx) {

    unsigned numSuccessors = termInst->getNumSuccessors();
    //llvm::errs() << "ControlFlow  getControlType(): " << numSuccessors << "\n";
    if (succIdx >= numSuccessors) {
        return OtherTerminator; // 如果 succIdx 超出范围，返回默认类型
    }

    if (auto *brInst = llvm::dyn_cast<llvm::BranchInst>(termInst)) {
        if (brInst->isConditional()) {
            return succIdx == 0 ? BrTrue : BrFalse;
        }
        return Br;
    }

    if (auto *swInst = llvm::dyn_cast<llvm::SwitchInst>(termInst)) {
        return succIdx == 0 ? SwitchDefault : SwitchCase;
    }

    if (llvm::isa<llvm::IndirectBrInst>(termInst)) {
        return IndirectBr;
    }

    if (llvm::isa<llvm::ReturnInst>(termInst)) {
        return Ret; 
    }

    if (llvm::isa<llvm::ResumeInst>(termInst)) {
        return Resume; 
    }

    if (llvm::isa<llvm::UnreachableInst>(termInst)) {
        return Unreachable;
    }

    if (auto *invokeInst = llvm::dyn_cast<llvm::InvokeInst>(termInst)) {
        return succIdx == 0 ? InvokeNormal : InvokeException;
    }

    if (auto *cleanupRetInst = llvm::dyn_cast<llvm::CleanupReturnInst>(termInst)) {
        if (cleanupRetInst->hasUnwindDest()) {
            return succIdx == 0 ? CleanupRetNormal : CleanupRetUnwind;
        }
        return CleanupRetNormal; 
    }

    if (llvm::isa<llvm::CatchReturnInst>(termInst)) {
        return CatchRet; // catchret 指令
    }

    if (auto *catchSwitchInst = llvm::dyn_cast<llvm::CatchSwitchInst>(termInst)) {
        unsigned numHandlers = catchSwitchInst->getNumHandlers();
        if (succIdx < numHandlers) {
            return CatchSwitch; 
        } else if (catchSwitchInst->hasUnwindDest() && succIdx == numHandlers) {
            return CatchSwitchUnwind;
        }
    }

    if (auto *callBrInst = llvm::dyn_cast<llvm::CallBrInst>(termInst)) {

        return succIdx == 0 ? CallBrNormal : CallBrIndirect;
    }

    return OtherTerminator;
}

struct Link {
    uint source;              // Unique ID of source node
    uint target;              // Unique ID of the target node
    LinkType  type;            // Kind of edge in the graph
    llvm::Type::TypeID dtype;  //  Type of data (for reach_definition links).
    Operation operation_dtype;  //Type of data (for DefUse links).
    ControlType controlType;  // Type of control 
};

std::string controlTypeToString(ControlType ct) { 
    switch (ct) {
        case Intra_block:          return ", \"controlType\": \"Intra_block\"";
        case Br:                   return ", \"controlType\": \"br\"";
        case BrTrue:               return ", \"controlType\": \"BrTrue\"";
        case BrFalse:              return ", \"controlType\": \"BrFalse\"";
        case SwitchDefault:        return ", \"controlType\": \"SwitchDefault\"";
        case SwitchCase:           return ", \"controlType\": \"SwitchCase\"";
        case IndirectBr:           return ", \"controlType\": \"IndirectBr\"";
        case Ret:                  return ", \"controlType\": \"ret\"";  
        case Resume:               return ", \"controlType\": \"resume\"";
        case Unreachable:          return ", \"controlType\": \"unreachable\"";
        case CleanupRetNormal:     return ", \"controlType\": \"cleanupret.normal\"";
        case CleanupRetUnwind:     return ", \"controlType\": \"cleanupret.unwind\"";
        case CatchRet:             return ", \"controlType\": \"catchret\"";
        case CatchSwitch:          return ", \"controlType\": \"catchswitch\"";
        case CatchSwitchUnwind:    return ", \"controlType\": \"catchswitch.unwind\"";
        case InvokeNormal:         return ", \"controlType\": \"invoke.normal\"";
        case InvokeException:      return ", \"controlType\": \"invoke.exception\"";
        case CallBrNormal:         return ", \"controlType\": \"callbr.normal\"";
        case CallBrIndirect:       return ", \"controlType\": \"callbr.indirect\"";
        case Call:                 return ", \"controlType\": \"call\"";
        case OtherTerminator:      return ", \"controlType\": \"other\"";
        default:                   return ", \"controlType\": \"unknown\"";
    }
}

std::string OperationToString(Operation operation)
{
    
    std::string json = " ,\"operation\": ";
    switch(operation){
        case Operation_Return: json += " \"return\" "; break;
        case Operation_Branch: json += " \"branch\" "; break;
        case Operation_Switch: json += " \"switch\" "; break;
        case Operation_IndirectBranch: json += " \"indirect_branch\" "; break;
        case Operation_Invoke: json += " \"invoke\" "; break;
        case Operation_Resume: json += " \"resume\" "; break;
        case Operation_Unreachable: json += " \"unreachable\" "; break;
        case Operation_CleanupReturn: json += " \"cleanup_return\" "; break;
        case Operation_CatchReturn: json += " \"catch_return\" "; break;
        case Operation_CatchSwitch: json += "catch_switch\" "; break;
        case Operation_CallBranch: json += " \"call_branch\" "; break;
        case Operation_Negation: json += " \"negate\" "; break;
        case Operation_Add: json += " \"add\" "; break;
        case Operation_FloatAdd: json += " \"float_add\" "; break;
        case Operation_Subtract: json += " \"subtract\" "; break;
        case Operation_FloatSubtract: json += " \"float_subtract\" "; break;
        case Operation_Multiply: json += " \"multiply\" "; break;
        case Operation_FloatMultiply: json += " \"float_multiply\" "; break;
        case Operation_UnsignedDivide: json += " \"unsigned_divide\" "; break;
        case Operation_SignedDivide: json += " \"signed_divide\" "; break;
        case Operation_FloatDivide: json += " \"float_divide\" "; break;
        case Operation_UnsignedModulus: json += " \"unsigned_modulus\" "; break;
        case Operation_SignedModulus: json += " \"signed_modulus\" "; break;
        case Operation_FloatModulus: json += " \"float_modulus\" "; break;
        case Operation_ShiftLeft: json += " \"shift_left\" "; break;
        case Operation_LogicalShiftRight: json += " \"logical_shift_right\" "; break;
        case Operation_ArithmeticShiftRight: json += " \"arithmetic_shift_right\" "; break;
        case Operation_And: json += " \"and\" "; break;
        case Operation_Or: json += " \"or\" "; break;
        case Operation_Xor: json += " \"xor\" "; break;
        case Operation_Allocate: json += " \"allocate\" "; break;
        case Operation_Load: json += " \"load\" "; break;
        case Operation_Store: json += " \"store\" "; break;
        case Operation_GetElementPointer: json += " \"get_element_pointer\" "; break;
        case Operation_Fence: json += " \"fence\" "; break;
        case Operation_AtomicCompareExchange: json += " \"atomic_compare_exchange\" "; break;
        case Operation_AtomicReadModifyWrite: json += " \"atomic_read_write_modify\" "; break;
        case Operation_IntTruncate: json += " \"int_truncate\" "; break;
        case Operation_ZeroExtend: json += " \"zero_extend\" "; break;
        case Operation_SignExtend: json += " \"sign_extend\" "; break;
        case Operation_FloatToUInt: json += " \"float_to_uint\" "; break;
        case Operation_FloatToSInt: json += " \"float_to_sint\" "; break;
        case Operation_UIntToFloat: json += " \"uint_to_float\" "; break;
        case Operation_SIntToFloat: json += " \"sint_to_float\" "; break;
        case Operation_FloatTruncate: json += " \"float_truncate\" "; break;
        case Operation_FloatExtend: json += " \"float_extend\" "; break;
        case Operation_PointerToInt: json += " \"pointer_to_int\" "; break;
        case Operation_IntToPointer: json += " \"int_to_pointer\" "; break;
        case Operation_BitCast: json += " \"bit_cast\" "; break;
        case Operation_AddressSpaceCast: json += " \"address_space_cast\" "; break;
        case Operation_CleanupPad: json += " \"cleanup_pad\" "; break;
        case Operation_CatchPad: json += " \"catch_pad\" "; break;
        case Operation_IntCompare: json += " \"int_compare\" "; break;
        case Operation_FloatCompare: json += " \"float_compare\" "; break;
        case Operation_PhiNode: json += " \"phi_node\" "; break;
        case Operation_Call: json += " \"call\" "; break;
        case Operation_Select: json += " \"select\" "; break;
        case Operation_User1: json += " \"user_1\" "; break;
        case Operation_User2: json += " \"user_2\" "; break;
        case Operation_VarArgument: json += " \"var_argument\" "; break;
        case Operation_ExtractElement: json += " \"extract_element\" "; break;
        case Operation_InsertElement: json += " \"insert_element\" "; break;
        case Operation_ShuffleVector: json += " \"shuffle_vector\" "; break;
        case Operation_ExtractValue: json += " \"extract_value\" "; break;
        case Operation_InsertValue: json += " \"insert_value\" "; break;
        case Operation_LandingPad: json += " \"landing_pad\" "; break;
        case Operation_Freeze: json += " \"freeze\" "; break;
    }
    return json;
}


std::string graphToJSON(const std::string& filename, WelkFunction* funcnode) {
    std::string json = "\"graph\": { \"file\": \"";
    json += filename + "\"";
    json += ", \"Funcname\": \"" + funcnode->Funcname + "\"";
    json += ", \"Func_id\": \"" + std::to_string(funcnode->Func_id) + "\"";
    json += ", \"func_startLine\": \"" + std::to_string(funcnode->func_startLine) + "\"";
    json += ", \"labelFilename\": \"" + funcnode->labelFilename + "\"";
    json += ", \"func_codefile\": \"" + funcnode->func_codefile + "\"";
    json += ",\"label\": [" + funcnode->label + "]}";
    return json;
}

std::string dtypeToString(llvm::Type::TypeID dt){
    switch (dt){
        case llvm::Type::VoidTyID: return ", \"dtype\": \"void\"";
        case llvm::Type::HalfTyID:  return ", \"dtype\": \"16bit_float\"";
        case llvm::Type::FloatTyID: return ", \"dtype\": \"32bit_float\"";
        case llvm::Type::DoubleTyID: return ", \"dtype\": \"64bit_float\"";
        case llvm::Type::X86_FP80TyID: return ", \"dtype\": \"80bit_x87_float\"";
        case llvm::Type::FP128TyID: return ", \"dtype\": \"128bit_float\"";
        case llvm::Type::PPC_FP128TyID: return ", \"dtype\": \"128bit_PPC_float\"";
        case llvm::Type::LabelTyID: return ", \"dtype\": \"labels\"";
        case llvm::Type::MetadataTyID: return ", \"dtype\": \"metadata\"";
        case llvm::Type::X86_MMXTyID: return ", \"dtype\": \"64_bit_x86_mmx_vectors\"";
        case llvm::Type::TokenTyID: return ", \"dtype\": \"tokens\"";
        case llvm::Type::IntegerTyID: return ", \"dtype\": \"integers\"";
        case llvm::Type::FunctionTyID: return ", \"dtype\": \"functions\"";
        case llvm::Type::StructTyID: return ", \"dtype\": \"structs\"";
        case llvm::Type::ArrayTyID: return ", \"dtype\": \"arrays\"";
        case llvm::Type::PointerTyID : return ", \"dtype\": \"pointers\"";
        // case llvm::Type::VectorTyID : return ", \"dtype\": \"vectors\"";
        case llvm::Type::BFloatTyID : return ", \"dtype\": \"bfloat\"";
        case llvm::Type::X86_AMXTyID : return ", \"dtype\": \"x86_amx\"";
        case llvm::Type::FixedVectorTyID : return ", \"dtype\": \"FixedVector\"";
        case llvm::Type::ScalableVectorTyID : return ", \"dtype\": \"ScalableVector\"";
        // LLVM_VERSION_MAJOR 16
        // case llvm::Type::TypedPointerTyID : return ", \"dtype\": \"TypedPointer\"";
        // case llvm::Type::TargetExtTyID : return ", \"dtype\": \"TargetExt\"";
    }
    return ", \"dtype\": \"void\"";
}




std::string irStringToJson(const std::string& irString) {

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.String(irString.c_str());
    std::string escapedString = buffer.GetString();
    //llvm::errs() << escapedString << "\n";
    return escapedString; 
}

std::string InstnodeToJSON(WelkInstruction InstNode)
{
    std::string json = "{ \"id\": ";
    json += std::to_string(InstNode.Inst_id);

    json += ",\"Block_id\": ";
    json += std::to_string(InstNode.Block_id);

    json += ",\"intra_Block_num\": ";
    json += std::to_string(InstNode.intra_Block_num);


    std::string irString_buffer = irStringToJson(InstNode.irString);
    json += ", \"irString\":" + irString_buffer;

    json += ", \"filename\":\"" + InstNode.filename + "\"";
    json += ",\"line_number\":";
    json += std::to_string(InstNode.line_number);
    json += ", \"call_func_name\":\"" + InstNode.call_func_name + "\"";
    
    json += OperationToString(InstNode.operation);

    json += dtypeToString(InstNode.dtype);
    json += ",\"label\":\"" + InstNode.labels + "\"}";
    return json;
}

std::string InstlinkToJSON(Link l)
{
    std::string json = "{ \"source\":";
    json += std::to_string(l.source);
    json += ", \"target\":";
    json += std::to_string(l.target);
    json += ", \"type\":";

    switch(l.type){
        case ControlFlow: 
            json += " \"control_flow\" ";
            json += controlTypeToString(l.controlType);
            json += "}";
            return json;
        case ControlDependence: json += " \"control_dependence\" "; break;
        case ReachDefinition: 
            json += " \"reach_definition\" "; 
            json += OperationToString(l.operation_dtype);
            json += "}";
            return json;
        case DefUse: 
            json += " \"def_use\" "; 
            json += dtypeToString(l.dtype);
            json += "}";
            return json;
        case Callgraph: json += " \"callgraph\" "; break;
        default: json += " \"unknown\" "; break;
    }
    return json;

}


//enum LinkType     { ControlFlow, ControlDependence, DefUse, ReachDefinition, Callgraph };
void outpuGraphJSON(const std::string& outputPath, int Func_Num, llvm::SetVector<WelkInstruction*>& WelkInstructions, \
                        llvm::SetVector<Link*>& links, LinkType Graph_Linktype, std::string& IRFileName,  WelkFunction* funcnode)
{
    std::string filename;
    if (Graph_Linktype == ControlFlow)
    {
       filename = outputPath + "/" + IRFileName + "_" + "Func" + std::to_string(Func_Num) + "_ControlFlow.json";
    }
    else if(Graph_Linktype == DefUse)
    {
        filename = outputPath + "/" + IRFileName + "_" + "Func" + std::to_string(Func_Num) + "_DefUse.json";
    }
    else if(Graph_Linktype == ReachDefinition)
    {
        filename = outputPath + "/" + IRFileName + "_" + "Func" + std::to_string(Func_Num) + "_ReachDefinition.json";
    }
    else
    {
        llvm::errs() << "Error Linktype: " << Graph_Linktype << "\n";
        return;
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        llvm::errs() << "Error opening file: " << filename << "\n";
        return;
    }

    outfile << "{ " << graphToJSON(filename, funcnode) << ", \"nodes\": [";


    bool first = true;
    for (const WelkInstruction* n : WelkInstructions) {
        if (!first) {
            outfile << ", ";
        } else {
            first = false;
        }
        outfile << InstnodeToJSON(*n);
    }


    first = true;
    outfile << "], \"links\": [";
    for (const Link* l : links) {
        if (!first) {
            outfile << ", ";
        } else {
            first = false;
        }
        outfile << InstlinkToJSON(*l);
    }

    outfile << "] }\n";
    outfile.close();
}


