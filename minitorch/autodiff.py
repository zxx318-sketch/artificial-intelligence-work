from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import deque

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1

# Global no_grad flag
_NO_GRAD = False


class no_grad:
    """Context manager that disables gradient computation."""
    def __enter__(self):
        global _NO_GRAD
        _NO_GRAD = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _NO_GRAD
        _NO_GRAD = False
        return False


def is_no_grad() -> bool:
    """Return whether gradient computation is currently disabled."""
    return _NO_GRAD


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # BEGIN Task1.2
    order = []
    visited = set()

    def dfs(v: Variable):
        # 如果已经访问过或者是常量，则跳过
        if v.unique_id in visited :
            return
        visited.add(v.unique_id)
        
        # 先递归访问所有父节点
        # ✅ 访问所有父节点，包括常量（常量会在后续被跳过）
        for parent in v.parents:
            dfs(parent)
        
        # 只将非常量节点加入顺序
        if not v.is_constant():
            order.append(v)

    dfs(variable)
    
    # 我们需要从输出到输入的顺序进行反向传播，所以反转列表
    # 此时 order 变为：输出 -> ... -> 输入
    return reversed(order)
    # END Task1.2

def topological_sort1(variable: Variable) -> Iterable[Variable]:
    import sys
    order = []
    visited = set()

    def dfs(v: Variable):
        # 打印详细信息
        print(f"DFS: id={v.unique_id}, const={v.is_constant()}, leaf={v.is_leaf()}, shape={getattr(v, 'shape', 'N/A')}", file=sys.stderr)
        if hasattr(v, '_tensor') and hasattr(v._tensor, '_storage'):
            storage = v._tensor._storage
            print(f"  storage={list(storage[:4])}... (len={len(storage)})", file=sys.stderr)
        
        if v.unique_id in visited:
            print(f"  SKIP (visited)", file=sys.stderr)
            return
        visited.add(v.unique_id)
        
        parents = list(v.parents)
        print(f"  parents: {[p.unique_id for p in parents]}", file=sys.stderr)
        
        for parent in parents:
            dfs(parent)
        
        if not v.is_constant():
            print(f"  >>> ADD {v.unique_id} to order", file=sys.stderr)
            order.append(v)

    dfs(variable)
    print(f"Final order ids: {[v.unique_id for v in order]}", file=sys.stderr)
    return reversed(order)

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:
        
    """
    # BEGIN Task1.2
    # Map from variable id to accumulated derivative tensor / value.
    grads: dict[int, Any] = {}
    # Initialize gradient at the output node.
    grads[variable.unique_id] = deriv
    # TODO: 
    order=topological_sort(variable)
    # 1. Traverse nodes in topological order
    for node in order:
        if node.unique_id not in grads:
            continue
        grad=grads[node.unique_id]
    # 2. If the node is a leaf, the derivative should be accumulated
        if (node.is_leaf()):
            node.accumulate_derivative(grad)
    # 3. Otherwise, the derivative should be propagated via chain rule
        else:           
            for parent, parent_grad in node.chain_rule(grad):
                if parent_grad is None:
                    continue
                if parent.is_constant():
                    continue
                parent_id = parent.unique_id
                if parent_id in grads:
                    grads[parent_id] = grads[parent_id] + parent_grad
                else:
                    grads[parent_id] = parent_grad
    #raise NotImplementedError("backpropagate not implemented")
    # END Task1.2


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
