#include <torch/csrc/jit/codegen/cuda/expr_simplifier.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool hasSimilarType(DataType t1, DataType t2) {
  if (t1 == t2) {
    return true;
  }
  if (isIntegralType(t1) && isIntegralType(t2)) {
    return true;
  }
  if (isFloatingPointType(t1) && isFloatingPointType(t2)) {
    return true;
  }
  if (isComplexType(t1) && isComplexType(t2)) {
    return true;
  }
  return false;
}

// If `value` is a constant scalar, then evaluate the value of that constant and
// return the evaluated value. Otherwise, returns `value` itself.
Val* foldConstants(Val* value) {
  if (value->isConstScalar()) {
    if (value->isIntegralScalar() && value->isA<Int>()) {
      return IrBuilder::create<Int>(
          value->evaluateInt(), *value->getDataType());
    }
    if (value->isFloatingPointScalar() && value->isA<Double>()) {
      return IrBuilder::create<Double>(
          value->evaluateDouble(), *value->getDataType());
    }
    if (value->isABool() && value->isA<Bool>()) {
      return IrBuilder::create<Bool>(
          value->evaluateBool(), *value->getDataType());
    }
    // TODO: support complex double
  }
  return value;
}

// Get the set of variables that `value` depends on. Items in `variables` are
// considered variables, and items not in `variables` are considered constant.
// For example, if value = a + b + c + d + 3, and `variables` is {a, b, e, f},
// then this function returns {a, b}. All tensors are considered variables.
std::unordered_set<Val*> getSubexprDependency(
    Val* value,
    const std::unordered_set<Val*>& variables) {
  if (value->isOneOf<TensorView, kir::TensorIndex>()) {
    return {value};
  }
  if (variables.count(value) > 0) {
    return {value};
  }
  auto def = value->definition();
  if (def == nullptr) {
    return {};
  }
  std::unordered_set<Val*> result;
  for (auto i : def->inputs()) {
    auto deps = getSubexprDependency(i, variables);
    result.insert(deps.begin(), deps.end());
  }
  return result;
}

bool hasTensor(const std::unordered_set<Val*>& variables) {
  for (auto v : variables) {
    if (v->isOneOf<TensorView, kir::TensorIndex>()) {
      return true;
    }
  }
  return false;
}

} // namespace

namespace assoc_comm_reordering {

// Note: [Reordering associative and commutative operators]
//
// For binary operators that is both associative and commutative, we can freely
// change the order of operands and add/remove parenthesis without changing the
// result. For example, + is both associative and commutative, so we have:
// a + b + c := (a + b) + c = a + (b + c) = (b + a) + c = b + (a + c) = ...
// For these operators, the most convenient way for handling them is to flatten
// them. For example, for the above a + b + c, all we need to know is we are
// adding these three variables together. We don't really care whether we are
// adding a and b first, or adding a and c first, or whether we are adding a to
// c or adding c to a. `FlattenedAssocCommOp` is the class that represents this
// flattened perspective.
//
// The reordering of associative and commutative operators is mostly useful for
// index hoisting. For example, if I have a loop structure and index:
//   FOR i1
//     FOR i2
//       FOR i3
//         index = ((i3 + i2) + i1) + 256
// There is no hoisting opportunity for this index in this loop structure.
// However, if I transform the index into index = ((256 + i1) + i2) + i3,
// then I can hoist the index as
//   FOR i1
//     i4 = (256 + i1)
//     FOR i2
//       i5 = i4 + i2
//       FOR i3
//         index = i5 + i3
// This minimizes the total number of computations.

bool isAssociativeAndCommutative(BinaryOpType type) {
  return type == BinaryOpType::Add || type == BinaryOpType::Mul ||
      type == BinaryOpType::And || type == BinaryOpType::Or ||
      type == BinaryOpType::Xor;
}

// The expression type that represents the flattened ops. For example, if I have
// out = a + b + 3 + c + 5, then I will have:
//   FlattenedAssocCommOp {
//     inputs: [a, b, c]
//     outputs: [out]
//     constant term: 8
//   }
//
// TODO: refactor IR printing so that private exprs like this can be printed.
// This will be very helpful for debugging.
class FlattenedAssocCommOp : public Expr {
 public:
  using Expr::Expr;

  FlattenedAssocCommOp(
      IrBuilderPasskey passkey,
      BinaryOpType op,
      Val* out,
      std::vector<Val*> terms)
      : Expr(passkey) {
    TORCH_CHECK(
        isAssociativeAndCommutative(op),
        "Can only flatten associative and commutative ops");
    addAttribute(
        IrBuilder::create<Attribute<BinaryOpType>>(passkey.ir_container_, op));
    addOutput(out);
    for (auto v : terms) {
      // Note that `addInput` is overriden in this class.
      addInput(v);
    }
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "FlattenedAssocCommOp";
  }

  DataType dtype() const {
    return *output(0)->getDataType();
  }

  BinaryOpType getOpType() const {
    return attribute(0)->as<Attribute<BinaryOpType>>()->value;
  }

  // Constant terms are stored and handled separately from other inputs. For
  // example, if I have a + b + 3 + c + 5, then the constant term will be 8, and
  // a, b, c are all inputs.
  bool hasConstantTerm() const {
    return attributes().size() == 2;
  }

  Val* getConstantTerm() const {
    return attributeVal(1);
  }

  template <typename T>
  T getConstantTermValue() const {
    return *getConstantTerm()->as<Scalar<T>>()->value();
  }

  // Get a vector of inputs, sorted as the order given by `variables`. Note that
  // the sorting key is the rightmost variable that an input depends on. For
  // example, if I have two inputs.
  // v1 = a * c
  // v2 = b
  // and variables is [a, b, c], then v2 < v1 because the rightmost depending
  // variable of v2 is b, and the rightmost depending variable of v1 is c,
  // and b < c. So in this example, this function will return [v2, v1].
  // Tensors are always considered as variables and they are always considered
  // as the rightmost.
  std::vector<Val*> sortedInputs(const std::list<ValInfo>& variables) {
    std::unordered_set<Val*> variables_set;
    variables_set.reserve(variables.size());
    for (auto v : variables) {
      variables_set.emplace(v.variable);
    }
    std::vector<Val*> sorted_inputs(inputs().begin(), inputs().end());
    std::unordered_map<Val*, std::unordered_set<Val*>> dependency;
    dependency.reserve(sorted_inputs.size());
    for (auto v : sorted_inputs) {
      dependency[v] = getSubexprDependency(v, variables_set);
    }
    auto compare = [&](Val* v1, Val* v2) {
      // Find all variables in variables_set that v1 and v2 depends on. The
      // input (v1 or v2) that exclusively has the right most variable in
      // variables_set will be to the right of the other input.
      bool v1_is_left_of_v2 = false;
      auto deps1 = dependency.at(v1);
      auto deps2 = dependency.at(v2);
      if (hasTensor(deps2)) {
        return true;
      }
      if (hasTensor(deps1)) {
        return false;
      }
      for (auto v : variables) {
        if (deps1.count(v.variable) > 0 && deps2.count(v.variable) == 0) {
          v1_is_left_of_v2 = false;
        } else if (
            deps2.count(v.variable) > 0 && deps1.count(v.variable) == 0) {
          v1_is_left_of_v2 = true;
        }
      }
      return v1_is_left_of_v2;
    };
    std::sort(sorted_inputs.begin(), sorted_inputs.end(), compare);
    return sorted_inputs;
  }

 protected:
  // Add a new value as an input of this expression. If `value` is a constant
  // scalar, then evalate this constant and merge it with the constant term.
  // Otherwise, add it as an input.
  void addInput(Val* value) {
    TORCH_CHECK(
        hasSimilarType(dtype(), *value->getDataType()),
        "Input types should be similar, but got: ",
        dtype(),
        ", and ",
        *value->getDataType());
    value = foldConstants(value);
    if (value->isConst()) {
      updateConstantTermWith(value);
    } else {
      Expr::addInput(value);
    }
  }

  // Update the constant term. For example, if getOpType() == "+", and the
  // current constant term is 8, then updateConstantTermWith(7) will change the
  // constant term to 15.
  void updateConstantTermWith(Val* value) {
    if (!hasConstantTerm()) {
      addAttribute(value);
      return;
    }
    auto new_constant_term = IrBuilder::newScalar(dtype());
    IrBuilder::create<BinaryOp>(
        getOpType(), new_constant_term, getConstantTerm(), value);
    new_constant_term = foldConstants(new_constant_term);
    TORCH_INTERNAL_ASSERT(new_constant_term->isConst());
    attributes_.at(1) = new_constant_term;
  }
};

NVFUSER_DEFINE_CLONE_AND_CREATE(FlattenedAssocCommOp)

// Recursively convert expressions like AddOp(AddOp(a, b), AddOp(c, d)) into
// FlattenedAdd(a, b, c, d). This function recursively transforms the entire
// expression, so divOp(AddOp(AddOp(a, b), AddOp(c, d)), addOp(e, f)) will
// become divOp(FlattenAdd(a, b, c, d), FlattenAdd(e, f))
Val* flatten(Val* value) {
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }
  if (isProtectedWithMagicZero(value)) {
    return value;
  }
  value = foldConstants(value);
  if (value->isConst()) {
    return value;
  }

  TORCH_INTERNAL_ASSERT(
      def->outputs().size() == 1,
      "Expressions with multiple output are not supported");

  auto bop = dynamic_cast<BinaryOp*>(def);

  if (bop == nullptr || !isAssociativeAndCommutative(bop->getBinaryOpType())) {
    // Handle non-associative-and-commutative op:
    // Just recursively call flatten on its inputs
    bool changed = false;
    std::vector<Val*> new_inputs;
    new_inputs.reserve(def->inputs().size());
    for (auto v : def->inputs()) {
      new_inputs.emplace_back(flatten(v));
      if (new_inputs.back() != v) {
        changed = true;
      }
    }

    if (!changed) {
      return value;
    }

    Val* output = IrBuilder::newScalar(*value->getDataType());
    auto create_fn = def->newObjectFunc();
    create_fn(
        def->container(), std::move(new_inputs), {output}, def->attributes());
    return output;
  } else {
    // Handle associative-and-commutative op:
    // Convert binary ops into flattened op
    auto output = IrBuilder::newScalar(*value->getDataType());
    std::vector<Val*> inputs;

    auto append_or_merge_inputs = [&](Val* operand) {
      auto op = dynamic_cast<FlattenedAssocCommOp*>(operand->definition());
      if (op != nullptr && op->getOpType() == bop->getBinaryOpType() &&
          hasSimilarType(op->dtype(), *value->getDataType())) {
        inputs.insert(inputs.end(), op->inputs().begin(), op->inputs().end());
        if (op->hasConstantTerm()) {
          inputs.emplace_back(op->getConstantTerm());
        }
      } else {
        inputs.emplace_back(operand);
      }
    };

    append_or_merge_inputs(flatten(bop->lhs()));
    append_or_merge_inputs(flatten(bop->rhs()));

    IrBuilder::create<FlattenedAssocCommOp>(
        bop->getBinaryOpType(), output, std::move(inputs));
    return output;
  }
}

// Recursively convert expressions like FlattenedAdd(a, b, c, d) into
// AddOp(AddOp(AddOp(a, b), c), d))
Val* unflatten(Val* value, const std::list<ValInfo>& variables) {
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }
  if (isProtectedWithMagicZero(value)) {
    return value;
  }

  TORCH_INTERNAL_ASSERT(
      def->outputs().size() == 1,
      "Expressions with multiple output are not supported");

  auto fop = dynamic_cast<FlattenedAssocCommOp*>(def);

  if (fop == nullptr) {
    // Handle ops other than flattened op:
    // Just recursively call `unflatten` on its inputs
    bool changed = false;
    std::vector<Val*> new_inputs;
    new_inputs.reserve(def->inputs().size());
    for (auto v : def->inputs()) {
      new_inputs.emplace_back(unflatten(v, variables));
      if (new_inputs.back() != v) {
        changed = true;
      }
    }

    if (!changed) {
      return value;
    }

    Val* output = IrBuilder::newScalar(*value->getDataType());
    auto create_fn = def->newObjectFunc();
    create_fn(
        def->container(), std::move(new_inputs), {output}, def->attributes());
    return output;
  } else {
    // Handle flattened op:
    // Convert flattened op into original binary ops
    TORCH_INTERNAL_ASSERT(fop->hasConstantTerm() + fop->inputs().size() >= 2);
    auto sorted_inputs = fop->sortedInputs(variables);
    Val* lhs;
    int64_t next;
    if (fop->hasConstantTerm()) {
      lhs = fop->getConstantTerm();
      next = 0;
    } else {
      // We need to recursively unflatten all inputs, because we might have
      // nested flattened expressions like
      // FlattenedAdd(a, b, FlattenedMul(c, d, e))
      lhs = unflatten(sorted_inputs.at(0), variables);
      next = 1;
    }
    while (next < sorted_inputs.size()) {
      auto rhs = unflatten(sorted_inputs.at(next), variables);
      auto output = IrBuilder::newScalar(*value->getDataType());
      IrBuilder::create<BinaryOp>(fop->getOpType(), output, lhs, rhs);
      lhs = output;
      next++;
    }
    return lhs;
  }
}

} // namespace assoc_comm_reordering

Val* simplifyExpr(Val* value, const std::list<ValInfo>& variables) {
  FusionGuard fg(value->fusion());
  auto flattened = assoc_comm_reordering::flatten(value);
  return assoc_comm_reordering::unflatten(flattened, variables);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
