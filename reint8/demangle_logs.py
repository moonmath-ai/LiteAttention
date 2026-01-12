#!/usr/bin/env python3
"""
Script to demangle C++ function names in CUDA compilation logs
"""

import argparse
import re
import subprocess
import sys

def demangle_name(mangled_name):
    """Demangle a C++ symbol using cu++filt"""
    try:
        result = subprocess.run(
            ['cu++filt', mangled_name],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # Fallback to c++filt if cu++filt fails
        try:
            result = subprocess.run(
                ['c++filt', mangled_name],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return mangled_name

def get_parameter_annotation(line, context_stack):
    """Get annotation for a parameter based on its context"""
    # CollectiveMainloopFwdSm90 parameters
    mainloop_params = [
        "Pipeline stages",
        "Cluster shape", 
        "Tile shape [Q, K, V]",
        "Head dimension for V",
        "Data type (Element)",
        "Accumulator type (float)",
        "Architecture (Sm90)",
        "Is_causal - Causal masking",
        "Is_local - Local/sliding window attention", 
        "Has_softcap - Softmax capping",
        "Varlen - Variable length sequences",
        "PagedKVNonTMA - Paged KV cache",
        "AppendKV - Append KV tokens",
        "HasQv - Separate Qv tensor",
        "MmaPV_is_RS - P@V register-shared layout",
        "IntraWGOverlap - Intra-warpgroup overlap",
        "PackGQA - Packed GQA layout",
        "Split - Sequence parallelism",
        "V_colmajor - V column-major layout",
        "Is_skipable - Can skip computation",
        "ReverseSkipList - Reverse skip list",
        "Phase - Phase flag",
        "HasMustDoList - Must-do list flag"
    ]
    
    # CollectiveEpilogueFwd parameters  
    epilogue_params = [
        "Output tile shape",
        "Thread block shape",
        "Output data type",
        "Architecture",
        "Epilogue threads",
        "Feature flag 1",
        "Feature flag 2", 
        "Feature flag 3",
        "Feature flag 4"
    ]
    
    # Check if we're in CollectiveMainloopFwdSm90
    if "CollectiveMainloopFwdSm90<" in "".join(context_stack[-3:]):
        # Count parameters seen so far in this template
        param_count = sum(1 for l in context_stack if l.strip().endswith(','))
        if param_count < len(mainloop_params):
            return f"  # {mainloop_params[param_count]}"
    
    # Check if we're in CollectiveEpilogueFwd
    if "CollectiveEpilogueFwd<" in "".join(context_stack[-3:]):
        param_count = sum(1 for l in context_stack if l.strip().endswith(','))
        if param_count < len(epilogue_params):
            return f"  # {epilogue_params[param_count]}"
    
    return ""

def format_demangled_signature(demangled):
    """Format demangled signature with proper indentation and annotations"""
    indent = 0
    result = []
    current_line = ""
    i = 0
    mainloop_depth = 0  # Track nesting depth within mainloop
    epilogue_depth = 0  # Track nesting depth within epilogue
    param_count = 0
    
    # Parameter annotations
    mainloop_params = [
        "Pipeline stages",
        "Cluster shape", 
        "Tile shape [Q, K, V]",
        "Head dimension for V",
        "Data type (Element)",
        "Accumulator type",
        "Architecture",
        "Is_causal",
        "Is_local", 
        "Has_softcap",
        "Varlen",
        "PagedKVNonTMA",
        "AppendKV",
        "HasQv",
        "MmaPV_is_RS",
        "IntraWGOverlap",
        "PackGQA",
        "Split",
        "V_colmajor",
        "Is_skipable",
        "ReverseSkipList",
        "Phase",
        "HasMustDoList"
    ]
    
    epilogue_params = [
        "Output tile shape",
        "Thread block shape",
        "Output data type",
        "Architecture",
        "Epilogue threads",
        "Flag 1",
        "Flag 2", 
        "Flag 3",
        "Flag 4"
    ]
    
    while i < len(demangled):
        char = demangled[i]
        
        if char == '<':
            # Add current content before <, then newline and increase indent
            current_line += char
            
            # Check if entering mainloop or epilogue
            if "CollectiveMainloopFwdSm90<" in current_line:
                mainloop_depth = 1
                epilogue_depth = 0
                param_count = 0
            elif "CollectiveEpilogueFwd<" in current_line:
                epilogue_depth = 1
                mainloop_depth = 0
                param_count = 0
            elif mainloop_depth > 0:
                mainloop_depth += 1
            elif epilogue_depth > 0:
                epilogue_depth += 1
                
            result.append(("  " * indent) + current_line)
            indent += 1
            current_line = ""
        elif char == '>':
            # Finish current line, decrease indent, add >
            if current_line.strip():
                # Add annotation for the last parameter (no comma after it)
                annotation = ""
                if mainloop_depth == 1 and param_count < len(mainloop_params):
                    annotation = f"  # {mainloop_params[param_count]}"
                elif epilogue_depth == 1 and param_count < len(epilogue_params):
                    annotation = f"  # {epilogue_params[param_count]}"
                result.append(("  " * indent) + current_line.strip() + annotation)
            indent -= 1
            
            # Decrease depth counters
            if mainloop_depth > 0:
                mainloop_depth -= 1
                if mainloop_depth == 0:
                    param_count = 0
            if epilogue_depth > 0:
                epilogue_depth -= 1
                if epilogue_depth == 0:
                    param_count = 0
                
            current_line = char
            # Check if this is followed by another > or other chars
            if i + 1 < len(demangled) and demangled[i + 1] in '>,':
                # Keep building this line
                pass
            else:
                # End this line
                result.append(("  " * indent) + current_line)
                current_line = ""
        elif char == ',':
            # End current parameter
            current_line += char
            
            # Add annotation for mainloop or epilogue parameters
            # Only at depth 1 (direct children of mainloop/epilogue)
            annotation = ""
            if mainloop_depth == 1 and param_count < len(mainloop_params):
                annotation = f"  # {mainloop_params[param_count]}"
                param_count += 1
            elif epilogue_depth == 1 and param_count < len(epilogue_params):
                annotation = f"  # {epilogue_params[param_count]}"
                param_count += 1
            
            result.append(("  " * indent) + current_line.strip() + annotation)
            current_line = ""
            # Skip space after comma
            if i + 1 < len(demangled) and demangled[i + 1] == ' ':
                i += 1
        elif char == '(':
            # Keep function call on same line
            paren_count = 1
            current_line += char
            j = i + 1
            while j < len(demangled) and paren_count > 0:
                if demangled[j] == '(':
                    paren_count += 1
                elif demangled[j] == ')':
                    paren_count -= 1
                current_line += demangled[j]
                j += 1
            i = j - 1
        else:
            current_line += char
        
        i += 1
    
    # Add any remaining content
    if current_line.strip():
        result.append(("  " * max(0, indent)) + current_line.strip())
    
    return "\n".join(result)

def should_include_line(line):
    """Determine if a line should be included in the output"""
    # Include lines that are part of compilation output
    include_patterns = [
        r'^\[[\d/]+\]',  # Build progress lines like [1/5]
        r'^ptxas',        # ptxas info lines
        r'bytes stack frame',  # Stack frame info (may not start with ptxas)
        r'DEMANGLED:',    # Our demangled lines
        r'MANGLED',       # Our mangled markers
        r'^=+$',          # Separator lines
    ]
    
    for pattern in include_patterns:
        if re.search(pattern, line):
            return True
    
    return False

def has_zero_spills(lines_buffer):
    """Check if the last few lines indicate zero spills/stack usage"""
    # Look for patterns like "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
    for line in lines_buffer:
        if 'bytes stack frame' in line and 'bytes spill' in line:
            # Extract the numbers
            match = re.search(r'(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads', line)
            if match:
                stack, stores, loads = map(int, match.groups())
                if stack == 0 and stores == 0 and loads == 0:
                    return True
    return False

def process_log_file(input_file, output_file, filter_zero_spills=True):
    """Process the log file and demangle all function names"""
    
    # Pattern to match mangled function names (starting with _ZN)
    mangled_pattern = re.compile(r"'(_Z[^']+)'")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        buffer = []  # Buffer to hold lines for a kernel
        in_kernel_block = False
        current_mangled = None
        current_demangled = None
        
        for line_num, line in enumerate(infile, 1):
            # Check if this is a "Compiling entry function" line
            matches = mangled_pattern.findall(line)
            
            if matches and "Compiling entry function" in line:
                # Start of a new kernel - write previous buffer if valid
                if buffer and current_demangled:
                    if not filter_zero_spills or not has_zero_spills(buffer):
                        # Write the buffered kernel
                        outfile.write("\n" + "="*80 + "\n")
                        outfile.write("DEMANGLED FUNCTION:\n")
                        outfile.write(current_demangled + "\n")
                        outfile.write("="*80 + "\n\n")
                        outfile.write(f"MANGLED: {current_mangled}\n\n")
                        for buf_line in buffer:
                            outfile.write(buf_line)
                
                # Start new buffer
                buffer = []
                in_kernel_block = True
                current_mangled = matches[0]
                demangled = demangle_name(current_mangled)
                if demangled != current_mangled:
                    current_demangled = format_demangled_signature(demangled)
                else:
                    current_demangled = None
            elif in_kernel_block and should_include_line(line):
                # Add to buffer
                buffer.append(line)
                # Check if this is the end of the kernel block
                if "Compile time =" in line:
                    # Keep in block for next iteration to catch the separator
                    pass
            elif should_include_line(line) and not in_kernel_block:
                # Build progress or other relevant lines outside kernel blocks
                outfile.write(line)
        
        # Write last buffer if exists
        if buffer and current_demangled:
            if not filter_zero_spills or not has_zero_spills(buffer):
                outfile.write("\n" + "="*80 + "\n")
                outfile.write("DEMANGLED FUNCTION:\n")
                outfile.write(current_demangled + "\n")
                outfile.write("="*80 + "\n\n")
                outfile.write(f"MANGLED: {current_mangled}\n\n")
                for buf_line in buffer:
                    outfile.write(buf_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demangle C++ function names in CUDA compilation logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Process compile.logs (default)
  %(prog)s my_compile.log               # Process custom log file
  %(prog)s -o output.txt                # Specify output file
  %(prog)s --no-filter                  # Include kernels with zero spills
  %(prog)s input.log -o output.log      # Full customization
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default='compile.logs',
        help='Input compilation log file (default: compile.logs)'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        default='compile_demangled.logs',
        help='Output file for demangled logs (default: compile_demangled.logs)'
    )
    
    parser.add_argument(
        '--no-filter',
        dest='filter_zero_spills',
        action='store_false',
        default=True,
        help='Include kernels with zero spills (by default they are filtered out)'
    )
    
    parser.add_argument(
        '--show-examples',
        action='store_true',
        help='Show example demangled functions after processing'
    )
    
    args = parser.parse_args()
    
    print(f"Processing {args.input_file}...")
    if args.filter_zero_spills:
        print("Filtering out kernels with zero spills...")
    
    try:
        process_log_file(args.input_file, args.output_file, args.filter_zero_spills)
        print(f"✓ Demangled output written to {args.output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
    
    # Show examples if requested
    if args.show_examples:
        print("\n" + "="*80)
        print("Example demangled functions:")
        print("="*80 + "\n")
        
        # Get a few examples from the file
        with open(args.input_file, 'r') as f:
            count = 0
            for line in f:
                if "Compiling entry function" in line and count < 2:
                    match = re.search(r"'(_Z[^']+)'", line)
                    if match:
                        mangled = match.group(1)
                        demangled = demangle_name(mangled)
                        formatted = format_demangled_signature(demangled)
                        print(f"Example {count + 1}:")
                        print(f"\nMangled:\n  {mangled}\n")
                        print(f"Demangled (formatted):")
                        print(formatted)
                        print("\n" + "-"*80 + "\n")
                        count += 1

