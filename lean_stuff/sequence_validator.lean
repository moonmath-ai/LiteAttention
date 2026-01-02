/-
  Sequence Validator for Strict Two Workgroup Dependencies

  This file verifies if a sequence of operations satisfies the dependencies
  specified in strict_two_wg.txt
-/

import Lean

-- Define operation types
inductive Op : Type where
  | Q : Op
  | K : Nat → Op
  | V : Nat → Op
  | QK : Nat → Nat → Op  -- QK(inner_idx, global_idx)
  | S : Nat → Nat → Op   -- S(inner_idx, global_idx)
  | P : Nat → Nat → Op   -- P(inner_idx, global_idx)
  | PV : Nat → Nat → Op  -- PV(inner_idx, global_idx)
  | O : Nat → Nat → Op   -- O(inner_idx, global_idx)

-- Equality for operations
instance : BEq Op where
  beq op1 op2 := match op1, op2 with
    | Op.Q, Op.Q => true
    | Op.K n1, Op.K n2 => n1 == n2
    | Op.V n1, Op.V n2 => n1 == n2
    | Op.QK a1 t1, Op.QK a2 t2 => a1 == a2 && t1 == t2
    | Op.S a1 t1, Op.S a2 t2 => a1 == a2 && t1 == t2
    | Op.P a1 t1, Op.P a2 t2 => a1 == a2 && t1 == t2
    | Op.PV a1 t1, Op.PV a2 t2 => a1 == a2 && t1 == t2
    | Op.O a1 t1, Op.O a2 t2 => a1 == a2 && t1 == t2
    | _, _ => false

-- Get dependencies for an operation
def getDependencies (op : Op) : List Op :=
  match op with
  -- Base cases (n <= 1)
  | Op.Q => []
  | Op.K 0 => []
  | Op.V 0 => []
  | Op.K 1 => []
  | Op.V 1 => []

  -- Base case operations
  | Op.QK 0 0 => [Op.Q, Op.K 0]
  | Op.S 0 0 => [Op.QK 0 0]
  | Op.P 0 0 => [Op.S 0 0]
  | Op.PV 0 0 => [Op.V 0, Op.P 0 0]

  | Op.QK 1 0 => [Op.Q, Op.K 0, Op.P 0 0]
  | Op.S 1 0 => [Op.QK 1 0]
  | Op.P 1 0 => [Op.S 1 0]
  | Op.PV 1 0 => [Op.V 0, Op.P 1 0]

  | Op.QK 0 1 => [Op.Q, Op.K 1, Op.P 1 0]
  | Op.S 0 1 => [Op.QK 0 1]
  | Op.P 0 1 => [Op.S 0 1]
  | Op.O 0 0 => [Op.S 0 1, Op.PV 0 0]
  | Op.PV 0 1 => [Op.V 1, Op.P 0 1, Op.O 0 0]

  | Op.QK 1 1 => [Op.Q, Op.K 1, Op.P 0 1]
  | Op.S 1 1 => [Op.QK 1 1]
  | Op.P 1 1 => [Op.S 1 1]
  | Op.O 1 0 => [Op.S 1 1, Op.PV 1 0]
  | Op.PV 1 1 => [Op.V 1, Op.P 1 1, Op.O 1 0]

  -- Non-base cases (n >= 2)
  -- V(n) depends on PV(0, n-2) and PV(1, n-2) for n >= 2
  | Op.V n =>
    if n >= 2 then [Op.PV 0 (n - 2), Op.PV 1 (n - 2)]
    else []
  -- K(n) depends on QK(0, n-2) and QK(1, n-2) for n >= 2
  | Op.K n =>
    if n >= 2 then [Op.QK 0 (n - 2), Op.QK 1 (n - 2)]
    else []

  -- QK(0, n) depends on P(1, n-1) and K(n-2) for n >= 2
  | Op.QK 0 n =>
    if n >= 2 then [Op.P 1 (n - 1), Op.K (n - 2)]
    else []
  | Op.S 0 n =>
    if n >= 2 then [Op.QK 0 n]
    else []
  | Op.P 0 n =>
    if n >= 2 then [Op.S 0 n]
    else []
  -- O(0, n-1) depends on S(0, n) and PV(0, n-1) when n > 2
  -- Base case: O(0,0) depends on [S(0,1), PV(0,0)]
  -- Non-base: O(0, k) for k > 1 depends on [S(0, k+1), PV(0, k)]
  -- Both follow the pattern: O(0, k) depends on [S(0, k+1), PV(0, k)] for k >= 0
  | Op.O 0 k =>
    if k >= 0 then [Op.S 0 (k + 1), Op.PV 0 k]
    else []
  | Op.PV 0 n =>
    if n >= 2 then [Op.V n, Op.P 0 n, Op.O 0 (n - 1)]
    else []

  -- QK(1, n) depends on P(0, n) and K(n-2) for n >= 2
  | Op.QK 1 n =>
    if n >= 2 then [Op.P 0 n, Op.K (n - 2)]
    else []
  | Op.S 1 n =>
    if n >= 2 then [Op.QK 1 n]
    else []
  | Op.P 1 n =>
    if n >= 2 then [Op.S 1 n]
    else []
  -- O(1, n-1) depends on S(1, n) and PV(1, n-1) when n > 2
  -- Base case: O(1,0) depends on [S(1,1), PV(1,0)]
  -- Non-base: O(1, k) for k > 1 depends on [S(1, k+1), PV(1, k)]
  -- Both follow the pattern: O(1, k) depends on [S(1, k+1), PV(1, k)] for k >= 0
  | Op.O 1 k =>
    if k >= 0 then [Op.S 1 (k + 1), Op.PV 1 k]
    else []
  | Op.PV 1 n =>
    if n >= 2 then [Op.V n, Op.P 1 n, Op.O 1 (n - 1)]
    else []

  | _ => []

-- Helper to safely get an element from a list
def List.getOption {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n + 1 => xs.getOption n

-- Check if an operation appears before a given index in a sequence
def appearsBefore (seq : List Op) (op : Op) (idx : Nat) : Bool :=
  (seq.take idx).elem op

-- Find all indices where an operation appears in a sequence
def findIndices (seq : List Op) (op : Op) : List Nat :=
  (List.range seq.length).filter (fun idx =>
    match seq.getOption idx with
    | some op' => op == op'
    | none => false
  )

-- Check if an operation has already been seen in the accumulator
def alreadySeen (op : Op) (seen : List Op) : Bool :=
  seen.elem op

-- Find all duplicate operations in a sequence (returns list of (operation, list of indices where it appears))
partial def findDuplicates (seq : List Op) : List (Op × List Nat) :=
  match seq with
  | [] => []
  | op :: rest =>
    let indices := findIndices seq op
    let restDuplicates := findDuplicates rest
    if indices.length > 1 && !restDuplicates.any (fun (op', _) => op == op') then
      (op, indices) :: restDuplicates
    else
      restDuplicates

-- Check if all dependencies of an operation appear before it in the sequence
def dependenciesSatisfied (seq : List Op) (idx : Nat) : Bool :=
  match seq.getOption idx with
  | none => false
  | some op =>
    let deps := getDependencies op
    deps.foldl (fun acc dep => acc && appearsBefore seq dep idx) true

-- Check if sequence has no duplicates
def hasNoDuplicates (seq : List Op) : Bool :=
  (findDuplicates seq).isEmpty

-- Validate if a sequence satisfies all dependencies and has no duplicates
def isValidSequence (seq : List Op) : Bool :=
  seq.length > 0 &&
  hasNoDuplicates seq &&
  (List.range seq.length).foldl (fun acc idx => acc && dependenciesSatisfied seq idx) true

-- Helper function to check if a sequence is valid (with better error reporting)
-- Returns: (isValid, dependencyViolations, duplicates)
def validateSequence (seq : List Op) : Bool × List (Nat × Op × List Op) × List (Op × List Nat) :=
  let violations := (List.range seq.length).filterMap (fun idx =>
    match seq.getOption idx with
    | none => none
    | some op =>
      let deps := getDependencies op
      let missing := deps.filter (fun dep => !appearsBefore seq dep idx)
      if missing.isEmpty then none
      else some (idx, op, missing)
  )
  let duplicates := findDuplicates seq
  let isValid := violations.isEmpty && duplicates.isEmpty
  (isValid, violations, duplicates)

-- Convert Nat to String (simple version for small numbers)
partial def natToString : Nat → String
  | 0 => "0"
  | 1 => "1"
  | 2 => "2"
  | 3 => "3"
  | 4 => "4"
  | 5 => "5"
  | 6 => "6"
  | 7 => "7"
  | 8 => "8"
  | 9 => "9"
  | n =>
    let q := n / 10
    let r := n % 10
    if q == 0 then natToString r
    else natToString q ++ natToString r

-- String representation for operations (for debugging)
def Op.toString : Op → String
  | Op.Q => "Q"
  | Op.K n => "K(" ++ natToString n ++ ")"
  | Op.V n => "V(" ++ natToString n ++ ")"
  | Op.QK a t => "QK(" ++ natToString a ++ "," ++ natToString t ++ ")"
  | Op.S a t => "S(" ++ natToString a ++ "," ++ natToString t ++ ")"
  | Op.P a t => "P(" ++ natToString a ++ "," ++ natToString t ++ ")"
  | Op.PV a t => "PV(" ++ natToString a ++ "," ++ natToString t ++ ")"
  | Op.O a t => "O(" ++ natToString a ++ "," ++ natToString t ++ ")"

instance : ToString Op where
  toString := Op.toString

-- Example sequences
def example1 : List Op := [Op.Q, Op.K 0, Op.K 1, Op.QK 0 0]
def example2 : List Op := [Op.Q, Op.K 0, Op.V 0, Op.QK 0 0, Op.PV 0 0]

-- Test cases
#eval isValidSequence example1  -- Should be true
#eval isValidSequence example2  -- Should be false (PV(0,0) depends on P(0,0) which is missing)

-- Helper to print validation results
def printValidation (seq : List Op) : IO Unit := do
  let (isValid, violations, duplicates) := validateSequence seq
  let seqStr := String.join (seq.map (fun op => Op.toString op ++ " "))
  IO.println ("Sequence: " ++ seqStr)
  if isValid then
    IO.println "✓ Sequence is valid (no dependency violations, no duplicates)"
  else
    IO.println "✗ Sequence is invalid"
    if !violations.isEmpty then
      IO.println "  Dependency violations:"
      for (idx, op, missing) in violations do
        let missingStr := String.join (missing.map (fun op => Op.toString op ++ " "))
        IO.println ("    At index " ++ natToString idx ++ ": " ++ Op.toString op ++ " is missing dependencies: " ++ missingStr)
    if !duplicates.isEmpty then
      IO.println "  Duplicate operations:"
      for (op, indices) in duplicates do
        let indicesStr := String.join (indices.map (fun idx => natToString idx ++ " "))
        IO.println ("    " ++ Op.toString op ++ " appears at indices: " ++ indicesStr)

-- Run validation examples
#eval printValidation example1
#eval printValidation example2

open Op

def example3 : List Op := [
  Q, K 0, QK 0 0, S 0 0, P 0 0, -- prologue
  QK 1 0, V 0, PV 0 0, S 1 0, P 1 0,
  K 1, QK 0 1, PV 1 0, S 0 1, P 0 1, O 0 0,
  QK 1 1, V 1, PV 0 1
] -- don't like it becuase we didn't break the dependency of O (being able to compute O, S and P at the same time)

#eval printValidation example3

def example4 : List Op := [
  /- calculate QK -> S -> P for the full tile -/
  Q, K 0, QK 0 0, S 0 0, P 0 0, QK 1 0, S 1 0, P 1 0, -- prologue, also release K(0)
  K 1, QK 0 1, V 0, PV 0 0, S 0 1, P 0 1,
  QK 1 1, PV 1 0, O 0 0, S 1 1, P 1 1, -- release V(0), K(1)
  K 2, QK 0 2, V 1, PV 0 1, O 1 0, S 0 2, P 0 2,
  QK 1 2, PV 1 1, O 0 1, S 1 2, P 1 2, -- release V(1), K(2)
  K 3, QK 0 3, V 2, PV 0 2, O 1 1, S 0 3, P 0 3,
]
#eval printValidation example4
#eval getDependencies (O 0 1)
