# Sequence Validator for Strict Two Workgroup Dependencies

This directory contains a Lean 4 verification system for validating sequences of operations against the dependencies specified in `strict_two_wg.txt`.

## Usage

The main file is `sequence_validator.lean`. It provides functions to check if a sequence of operations satisfies all dependencies.

### Basic Usage

```lean
-- Define a sequence
def mySequence : List Op := [Op.Q, Op.K 0, Op.K 1, Op.QK 0 0]

-- Check if it's valid
#eval isValidSequence mySequence  -- Returns true or false

-- Get detailed validation results
#eval validateSequence mySequence  -- Returns (isValid, violations)
```

### Example Sequences

**Valid sequence:**
```lean
def example1 : List Op := [Op.Q, Op.K 0, Op.K 1, Op.QK 0 0]
-- Valid because QK(0,0) depends on Q and K(0), both appear before it
```

**Invalid sequence:**
```lean
def example2 : List Op := [Op.Q, Op.K 0, Op.V 0, Op.QK 0 0, Op.PV 0 0]
-- Invalid because PV(0,0) depends on V(0) (present) and P(0,0) (missing)
```

### Functions

- `isValidSequence (seq : List Op) : Bool` - Returns `true` if the sequence satisfies all dependencies
- `validateSequence (seq : List Op) : Bool × List (Nat × Op × List Op)` - Returns validation result with detailed violation information
- `printValidation (seq : List Op) : IO Unit` - Prints validation results in a human-readable format

### Operation Types

- `Op.Q` - Query operation
- `Op.K n` - Key operation for timestep n
- `Op.V n` - Value operation for timestep n
- `Op.QK a t` - QK product for inner index a and global index t
- `Op.S a t` - Softmax for inner index a and global index t
- `Op.P a t` - Save to shared memory for inner index a and global index t
- `Op.PV a t` - PV product for inner index a and global index t
- `Op.O a t` - Rescale operation for inner index a and global index t

