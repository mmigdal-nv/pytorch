from ._C import *

class FusionDefinition(_C._FusionDefinition):
    def __enter__(self):
        return self._setup_definition()

    def __exit__(self, type, value, traceback):
        self._finalize_definition()

    def definition(self):
        raise NotImplementedError("definition() should be implemented by child class!")

    def execute(self, inputs):
        # if definition is not previously defined by a context manager try a child class
        if self.id() is None:
            self._setup_definition()
            self.definition()
            self._finalize_definition()

        return self._execute(inputs)

    def from_pytorch(self, tensor) :
        """
        Defines an nvfuser input tensor from a pytorch tensor
        """
        try:
            from .pytorch_utils import torch_dtype_to_nvfuser_dtype
        except ImportError:
            raise ImportError("Unable to import pytorch_utils!")

        if not tensor.is_cuda:
            raise ValueError("Tensor should be on a cuda device!")

        return self.define_tensor(sizes=tensor.size(), strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype))

from .nvfuser_version import __version__

def version():
    r"""returns nvfuser version in format of a string 'm.n.p+git[7d-sha]'.

    We strip the git[7d-sha] and convert the string to
    `packaging.version.Version` for comparison. e.g. you can use it as:
        import nvfuser
        print(nvfuser.version())              # 0.0.1+git21df524
        nvfuser.version() == '0.0.1`          # True
        nvfuser.version() > '0.0.0`           # True

        from packaging.version import Version
        nvfuser.version() < Version('1.0.0')  # True
    """
    return __version__